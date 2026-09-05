"""CPU tests for the native search env: stepping, auto-reset, PPO contract.

No Isaac, no GPU, no assets -- the world is a synthetic raster built in-process
(the test_search.py pattern), so the whole native lane is checked by plain
pytest on a Mac.
"""
import numpy as np
import pytest
import torch

from vesper.lab.ppo import PPO, PPOCfg
from vesper.lab.search_task import PROPRIO_DIM
from vesper.native import NativeSearchEnv, NativeSearchEnvCfg


@pytest.fixture
def world_npz(tmp_path):
    """300 m square: flat, a building block, a wood, a road strip with headings."""
    n, half = 121, 150.0
    cell = 2 * half / (n - 1)
    ground = np.zeros((n, n), np.float32)
    obstacle = ground.copy()
    canopy = ground.copy()
    dens = np.zeros((n, n), np.float32)
    xs = np.linspace(-half, half, n)
    X, Y = np.meshgrid(xs, xs)
    obstacle[(X > 60) & (X < 90) & (Y.__abs__() < 30)] = 30.0
    wood = Y < -60
    canopy[wood] = 12.0
    dens[wood] = 0.7
    drivable = ((obstacle - ground) < 0.5).astype(np.uint8)
    concealed = (drivable & wood).astype(np.uint8)
    road = ((np.abs(Y) < 6) & (drivable > 0)).astype(np.uint8)
    road_yaw = np.zeros((n, n), np.float32)
    p = tmp_path / "w_map.npz"
    np.savez(p, half_m=np.float32(half), cell=np.float32(cell), ground_z=ground,
             obstacle_z=obstacle, canopy_z=canopy, canopy_d=dens,
             drivable=drivable, concealed=concealed, road=road, road_yaw=road_yaw)
    return str(p)


@pytest.fixture
def env(world_npz):
    cfg = NativeSearchEnvCfg()
    cfg.num_envs = 8
    cfg.n_targets = 2
    cfg.episode_length_s = 20.0
    cfg.search = {"arena_half": 120.0, "spawn_alt_min": 30.0, "spawn_alt_max": 50.0}
    cfg.world_map = world_npz
    return NativeSearchEnv(cfg, device="cpu", seed=0)


def test_reset_shapes_and_finiteness(env):
    obs, _ = env.reset()
    assert obs["policy"].shape == (8, PROPRIO_DIM)
    assert obs["privileged"].shape == (8, env.task.obs_dim)
    assert torch.isfinite(obs["policy"]).all() and torch.isfinite(obs["privileged"]).all()
    # drones spawn above the ground, inside the arena
    pos, vel, quat, _ = env.flight_state()
    ground = env.world.ground_at(pos[:, 0], pos[:, 1])
    assert (pos[:, 2] > ground + 20.0).all()
    assert (pos[:, :2].abs() < 120.0).all()
    assert torch.allclose(quat.norm(dim=1), torch.ones(8))
    # vehicles on the ground, one open-role tank guaranteed per set
    assert (env.role == 0).any(dim=1).all()


def test_step_flies_and_stays_finite(env):
    obs = env.ppo_reset()
    for _ in range(60):
        act = torch.randn(8, 3) * 0.3
        act[:, 0] += 0.5
        obs, rew, done, info = env.ppo_step(act)
        assert torch.isfinite(obs).all() and torch.isfinite(rew).all()
    pos, vel, _, _ = env.flight_state()
    assert float(vel.norm(dim=1).max()) > 1.0          # it actually flew somewhere


def test_auto_reset_zeroes_the_episode(env):
    env.ppo_reset()
    # drive one env into the ground: full down-stick until it crashes
    act = torch.zeros(8, 3)
    act[0, 2] = -1.0
    done0 = False
    for i in range(env.max_episode_length):
        obs, rew, done, info = env.ppo_step(act)
        if bool(done[0]):
            done0 = True
            break
    assert done0, "a full-down stick never terminated"
    # the returned observation is the *new* episode's first frame
    assert int(env.episode_length_buf[0]) == 0
    assert not env.task.known[0].any()
    pos, _, _, _ = env.flight_state()
    ground = env.world.ground_at(pos[0:1, 0], pos[0:1, 1])
    assert float(pos[0, 2] - ground) > 20.0            # respawned at altitude


def test_truncation_resets_everyone(env):
    env.ppo_reset()
    hover = torch.zeros(8, 3)
    for _ in range(env.max_episode_length + 1):
        obs, rew, done, info = env.ppo_step(hover)
    assert int(env.episode_length_buf.max()) < env.max_episode_length


def test_groups_share_targets(world_npz):
    cfg = NativeSearchEnvCfg()
    cfg.num_envs = 6
    cfg.n_targets = 2
    cfg.n_groups = 2
    cfg.world_map = world_npz
    cfg.search = {"arena_half": 120.0}
    env = NativeSearchEnv(cfg, device="cpu", seed=1)
    env.reset()
    assert torch.equal(env.target_pos[0], env.target_pos[2])
    assert torch.equal(env.target_pos[1], env.target_pos[3])


def test_ppo_trains_one_iteration(env):
    class Adapter:
        def __init__(self, e):
            self.env, self.num_envs, self.device = e, e.num_envs, e.device
            self.num_obs, self.num_actions = e.num_obs, e.num_actions

        def reset(self):
            return self.env.ppo_reset()

        def step(self, a):
            return self.env.ppo_step(a)

    ppo = PPO(Adapter(env), PPOCfg(horizon=8, minibatches=2, epochs=2), hidden=(32, 32), seed=0)
    hist = ppo.learn(2, log_every=1)
    assert len(hist) == 2
    assert all(np.isfinite(h["pi"]) for h in hist)
