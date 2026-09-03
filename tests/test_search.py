"""CPU tests for the search task: map sampling, occlusion, belief, reward.

No Isaac, no GPU, no assets -- the world is a synthetic raster built in-process,
so every mechanism the policy depends on is checked before a GPU is booted.
"""
import math

import numpy as np
import pytest
import torch

from vesper.lab.search_task import SearchCfg, SearchTask, setpoint, tilt_from_quat
from vesper.worlds.heightmap import WorldMap


@pytest.fixture
def world(tmp_path):
    """200 m square: flat at z=0, a 40 m tower of building at x>40, a wood at y<-40."""
    n, cell, half = 101, 2.0, 100.0
    ground = np.zeros((n, n), np.float32)
    obstacle = ground.copy()
    canopy = ground.copy()
    dens = np.zeros((n, n), np.float32)
    xs = np.linspace(-half, half, n)
    X, Y = np.meshgrid(xs, xs)                       # rows +y, cols +x
    obstacle[(X > 40) & (X < 60)] = 40.0
    wood = Y < -40
    canopy[wood] = 15.0
    dens[wood] = 0.8
    drivable = ((obstacle - ground) < 0.5).astype(np.uint8)
    concealed = (drivable & wood).astype(np.uint8)
    p = tmp_path / "w_map.npz"
    np.savez(p, half_m=np.float32(half), cell=np.float32(cell), ground_z=ground,
             obstacle_z=obstacle, canopy_z=canopy, canopy_d=dens,
             drivable=drivable, concealed=concealed)
    return WorldMap(p)


def test_sampling_and_masks(world):
    assert world.n == 101 and world.half_m == 100.0 and world.cell == 2.0
    z = world.ground_at(torch.tensor([0.0, 30.0]), torch.tensor([0.0, 10.0]))
    assert torch.allclose(z, torch.zeros(2))
    solid = world.solid_at(torch.tensor([50.0]), torch.tensor([0.0]))
    assert solid.item() == pytest.approx(40.0)
    assert world.canopy_at(torch.tensor([0.0]), torch.tensor([-60.0])).item() == pytest.approx(15.0)


def test_line_of_sight_blocked_by_building(world):
    # across the building at 10 m: blocked. over the top at 60 m: clear.
    lo0 = torch.tensor([[0.0, 0.0, 10.0]]); lo1 = torch.tensor([[90.0, 0.0, 10.0]])
    hi0 = torch.tensor([[0.0, 0.0, 60.0]]); hi1 = torch.tensor([[90.0, 0.0, 60.0]])
    assert not world.trace(lo0, lo1)[0].item()
    assert world.trace(hi0, hi1)[0].item()


def test_foliage_accumulates_only_in_the_wood(world):
    over = torch.tensor([[0.0, -60.0, 80.0]])
    under = torch.tensor([[0.0, -60.0, 1.0]])
    clear, foliage = world.trace(over, under)
    assert clear.item()                       # leaves do not block, they attenuate
    assert 8.0 < foliage.item() < 15.0        # ~15 m of canopy at density 0.8
    open_over = torch.tensor([[0.0, 60.0, 80.0]])
    open_under = torch.tensor([[0.0, 60.0, 1.0]])
    assert world.trace(open_over, open_under)[1].item() == pytest.approx(0.0)


def test_mask_sampling_lands_in_the_mask(world):
    g = torch.Generator().manual_seed(0)
    xy, ok = world.sample_mask_xy(world.concealed, 64, 90.0, g)
    assert ok.float().mean() > 0.5
    assert (xy[ok][:, 1] < -40.0).all()        # concealed cells are all in the wood


def _task(world, k=2, **kw):
    cfg = SearchCfg(n_targets=k, arena_half=90.0, grid=4, miss_p=0.0, fix_noise_m=0.0, **kw)
    return SearchTask(world, cfg, num_envs=2, dt=0.04, max_steps=100), cfg


def _state(n=2):
    quat = torch.zeros(n, 4); quat[:, 0] = 1.0
    return quat, torch.zeros(n, 3), torch.zeros(n, 3)


def test_detection_needs_the_target_under_the_cone(world):
    task, cfg = _task(world)
    drone = torch.tensor([[0.0, 0.0, 60.0], [0.0, 0.0, 60.0]])
    # target 0 straight below (in cone), target 1 far to the side (outside 45 deg)
    tgt = torch.tensor([[[0.0, 0.0, 1.0], [80.0, 0.0, 1.0]]] * 2)
    vis, slant = task.detect(drone, tgt)
    assert vis[:, 0].all() and not vis[:, 1].any()


def test_camouflage_shortens_the_range(world):
    task, cfg = _task(world)
    task.contrast[:] = torch.tensor([1.0, 0.2])
    # both directly below at 100 m: plain is inside 220 m, camo's range is 44 m
    drone = torch.tensor([[0.0, 0.0, 100.0]] * 2)
    tgt = torch.tensor([[[0.0, 0.0, 1.0], [0.0, 0.0, 1.0]]] * 2)
    vis, _ = task.detect(drone, tgt)
    assert vis[:, 0].all() and not vis[:, 1].any()


def test_canopy_shortens_the_range(world):
    task, cfg = _task(world)
    # same geometry, one over open ground and one over the wood
    drone = torch.tensor([[0.0, 60.0, 120.0], [0.0, -60.0, 120.0]])
    tgt = torch.tensor([[[0.0, 60.0, 1.0]] * 2, [[0.0, -60.0, 1.0]] * 2])
    vis, _ = task.detect(drone, tgt)
    assert vis[0, 0].item(), "open target at 119 m should be seen"
    assert not vis[1, 0].item(), "the same target under canopy should not be"


def test_belief_latches_and_goes_stale(world):
    task, cfg = _task(world)
    quat, vel, avb = _state()
    drone = torch.tensor([[0.0, 0.0, 60.0]] * 2)
    tgt = torch.tensor([[[0.0, 0.0, 1.0], [0.0, 30.0, 1.0]]] * 2)
    step = torch.zeros(2, dtype=torch.long)
    task.step(drone, vel, quat, avb, tgt, step)
    assert task.known[:, 0].all()
    fix0 = task.fix[:, 0].clone()
    # fly away: the fix is remembered, its age grows
    away = torch.tensor([[0.0, 0.0, 300.0]] * 2)
    for i in range(5):
        task.step(away, vel, quat, avb, tgt, step + i)
    assert task.known[:, 0].all()
    assert torch.allclose(task.fix[:, 0], fix0)
    assert task.fix_age[:, 0].min() > 0.15


def test_reward_ordering_fast_beats_slow_beats_crash(world):
    quat, vel, avb = _state(2)
    tgt = torch.tensor([[[0.0, 0.0, 1.0], [10.0, 0.0, 1.0]]] * 2)

    def run(step_at_reach, crash=False):
        task, cfg = _task(world)
        # the crash case flies into the ground well away from either vehicle:
        # touching one at ground level is a reach, not a crash, by design
        drone = (torch.tensor([[0.0, 0.0, 3.0]] * 2) if not crash
                 else torch.tensor([[-70.0, 70.0, 0.5]] * 2))
        s = torch.full((2,), step_at_reach, dtype=torch.long)
        _, r, term, info = task.step(drone, vel, quat, avb, tgt, s)
        return float(r[0]), bool(term[0])

    fast, _ = run(5)
    slow, _ = run(90)
    crash_r, crash_t = run(5, crash=True)
    assert fast > slow > 0
    assert crash_r < 0 and crash_t


def test_coverage_pays_once_and_grows_with_altitude(world):
    task, cfg = _task(world)
    quat, vel, avb = _state()
    tgt = torch.full((2, 2, 3), 500.0)          # far away, nothing to detect
    step = torch.zeros(2, dtype=torch.long)
    low = torch.tensor([[0.0, 0.0, 12.0]] * 2)
    _, r1, _, _ = task.step(low, vel, quat, avb, tgt, step)
    _, r2, _, _ = task.step(low, vel, quat, avb, tgt, step)
    assert r1[0] > r2[0], "a cell already swept must not pay again"
    covered_low = task.visited.float().mean().item()

    task2, _ = _task(world)
    high = torch.tensor([[0.0, 0.0, 150.0]] * 2)
    task2.step(high, vel, quat, avb, tgt, step)
    assert task2.visited.float().mean().item() > covered_low


def test_observation_never_leaks_an_unseen_target(world):
    task, cfg = _task(world)
    quat, vel, avb = _state()
    drone = torch.tensor([[0.0, 0.0, 60.0]] * 2)
    # one target visible below, one hidden far to the side
    tgt = torch.tensor([[[0.0, 0.0, 1.0], [85.0, 0.0, 1.0]]] * 2)
    step = torch.zeros(2, dtype=torch.long)
    task.step(drone, vel, quat, avb, tgt, step)
    agl = torch.full((2,), 60.0)
    obs = task.observations(drone, vel, quat, avb, agl, torch.zeros(2))
    assert obs.shape == (2, task.obs_dim)
    assert torch.isfinite(obs).all()
    # slot 1 was never seen: every one of its five belief fields must be zero
    base = 12 + 8
    assert obs[:, base:base + 8][:, [0, 3, 4, 5, 6, 7]].abs().max() == 0.0
    assert (obs[:, 12] == 1.0).all()            # slot 0 known


def test_setpoint_is_bounded_by_look_ahead():
    cfg = SearchCfg(look_ahead=25.0)
    pos = torch.zeros(4, 3)
    a = torch.randn(4, 3) * 10
    sp = setpoint(pos, a, cfg)
    assert sp.abs().max() <= 25.0 + 1e-5


def test_tilt_from_quat_upright_and_over():
    upright = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
    assert tilt_from_quat(upright).item() == pytest.approx(0.0, abs=1e-6)
    rolled = torch.tensor([[math.cos(math.pi / 4), math.sin(math.pi / 4), 0.0, 0.0]])
    assert tilt_from_quat(rolled).item() == pytest.approx(math.pi / 2, abs=1e-5)
