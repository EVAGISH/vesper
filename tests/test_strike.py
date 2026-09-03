"""CPU tests for the strike task math and the self-contained PPO."""
import math

import torch

from vesper.lab import strike_task as T
from vesper.lab.ppo import PPO, PPOCfg


CFG = T.StrikeCfg()


def test_sample_targets_within_ring():
    p, v = T.sample_targets(2000, CFG, device="cpu", generator=torch.Generator().manual_seed(0))
    r = p[:, :2].norm(dim=1)
    assert (r >= CFG.target_min_r - 1e-4).all() and (r <= CFG.target_max_r + 1e-4).all()
    assert torch.allclose(p[:, 2], torch.full((2000,), CFG.ground_z + CFG.target_h))
    assert v.shape == (2000, 3)


def test_tilt_from_quat():
    upright = torch.tensor([[1.0, 0, 0, 0]])
    assert T.tilt_from_quat(upright).item() < 1e-6
    flipped = torch.tensor([[0.0, 1.0, 0, 0]])  # 180 deg about x
    assert abs(T.tilt_from_quat(flipped).item() - math.pi) < 1e-5


def test_observation_shape_and_rel():
    n = 8
    dp = torch.randn(n, 3); dv = torch.randn(n, 3)
    q = torch.zeros(n, 4); q[:, 0] = 1
    av = torch.randn(n, 3); tp = torch.randn(n, 3); tv = torch.randn(n, 3)
    obs = T.observations(dp, dv, q, av, tp, tv)
    assert obs.shape == (n, 17)
    assert torch.allclose(obs[:, :3], tp - dp)
    assert torch.allclose(obs[:, -1], (tp - dp).norm(dim=1))


def test_setpoint_bounded():
    pos = torch.zeros(100, 3)
    act = torch.randn(100, 3) * 5
    sp = T.setpoint(pos, act, CFG)
    assert (sp.norm(dim=1) <= CFG.look_ahead * math.sqrt(3) + 1e-4).all()


def test_reward_hit_terminates_with_bonus():
    tp = torch.tensor([[10.0, 0, 1.1]])
    dp = tp.clone()  # right on top of it
    vel = torch.zeros(1, 3); q = torch.tensor([[1.0, 0, 0, 0]])
    r, term, info = T.evaluate(dp, vel, q, tp, prev_dist=torch.tensor([5.0]), cfg=CFG)
    assert info["hit"].item() and term.item()
    assert r.item() > CFG.r_hit - 5


def test_reward_progress_positive_when_closing():
    tp = torch.tensor([[30.0, 0, 1.1]])
    dp = torch.tensor([[20.0, 0, 15.0]])       # 10m nearer than prev
    prev = torch.tensor([[0.0]])  # placeholder
    prev = (tp - torch.tensor([[10.0, 0, 15.0]])).norm(dim=1)  # was farther
    vel = torch.zeros(1, 3); q = torch.tensor([[1.0, 0, 0, 0]])
    r, term, info = T.evaluate(dp, vel, q, tp, prev_dist=prev, cfg=CFG)
    assert not term.item()
    assert r.item() > 0  # closed distance -> positive


def test_reward_ground_crash_penalized():
    tp = torch.tensor([[30.0, 0, 1.1]])
    dp = torch.tensor([[5.0, 0, 0.1]])         # on the ground, far from target
    vel = torch.zeros(1, 3); q = torch.tensor([[1.0, 0, 0, 0]])
    r, term, info = T.evaluate(dp, vel, q, tp, prev_dist=torch.tensor([26.0]), cfg=CFG)
    assert info["ground"].item() and term.item() and r.item() < 0


# ---------- PPO on a toy reach env (whole pipeline, CPU) ----------
class ToyReach:
    """2D point mass chasing a random goal; solvable, to prove PPO learns."""
    num_obs = 2
    num_actions = 2

    def __init__(self, n=256, device="cpu", seed=0):
        self.num_envs = n
        self.device = device
        self.g = torch.Generator(device=device).manual_seed(seed)
        self.x = torch.zeros(n, 2)
        self.goal = torch.zeros(n, 2)
        self.t = torch.zeros(n)

    def _spawn(self, idx):
        self.goal[idx] = (torch.rand(len(idx), 2, generator=self.g) - 0.5) * 8
        self.x[idx] = 0.0
        self.t[idx] = 0.0

    def reset(self):
        self._spawn(torch.arange(self.num_envs))
        return self.goal - self.x

    def step(self, a):
        prev = (self.goal - self.x).norm(dim=1)
        self.x = self.x + 0.3 * torch.tanh(a)
        self.t += 1
        dist = (self.goal - self.x).norm(dim=1)
        hit = dist < 0.3
        done = hit | (self.t >= 30)
        rew = (prev - dist) + 10.0 * hit.float()
        info = {"hit": hit}
        idx = torch.nonzero(done).flatten()
        if len(idx):
            self._spawn(idx)
        return self.goal - self.x, rew, done, info


def test_ppo_learns_toy_reach():
    env = ToyReach(n=256, seed=0)
    ppo = PPO(env, PPOCfg(horizon=32, epochs=4, minibatches=4, lr=3e-3), hidden=(64, 64), seed=0)
    hist = ppo.learn(60)
    finite = [h for h in hist if h["episodes"] > 0]
    early = sum(h["ep_return"] for h in finite[:5]) / 5
    late = sum(h["ep_return"] for h in finite[-5:]) / 5
    assert all(math.isfinite(h["pi"]) and math.isfinite(h["vf"]) for h in hist)
    assert late > early + 1.0, f"no learning: {early:.2f} -> {late:.2f}"
    assert late > 5.0, f"weak final return {late:.2f}"


def _hit_reward(time_frac):
    tp = torch.tensor([[10.0, 0, 1.1]])
    vel = torch.zeros(1, 3); q = torch.tensor([[1.0, 0, 0, 0]])
    r, _, _ = T.evaluate(tp.clone(), vel, q, tp, torch.tensor([5.0]), CFG, time_frac=torch.tensor([time_frac]))
    return r.item()


def _outcome_reward(kind):
    """Reward for a non-hit terminal outcome, far from the target."""
    tp = torch.tensor([[35.0, 0, 1.1]])
    vel = torch.zeros(1, 3)
    q = torch.tensor([[1.0, 0, 0, 0]])
    dp = torch.tensor([[5.0, 0, 20.0]])
    if kind == "ground":
        dp = torch.tensor([[5.0, 0, 0.1]])
    elif kind == "flip":
        q = torch.tensor([[0.0, 1.0, 0, 0]])
    elif kind == "oob":
        dp = torch.tensor([[CFG.arena_radius + CFG.oob_margin + 5.0, 0, 20.0]])
    r, _, _ = T.evaluate(dp, vel, q, tp, torch.tensor([30.0]), CFG, time_frac=torch.tensor([0.5]))
    return r.item()


def test_faster_hit_is_worth_more():
    early, late = _hit_reward(0.05), _hit_reward(0.95)
    assert early > late, f"early {early} should beat late {late}"
    assert early - late > 50, "speed bonus should be a real incentive"


def test_outcome_ordering_fast_hit_beats_slow_hit_beats_crash():
    fast, slow = _hit_reward(0.05), _hit_reward(0.95)
    worst_timeout_cost = -CFG.w_time * 750          # a full 15 s episode of time penalty
    for kind in ("ground", "oob", "flip"):
        crash = _outcome_reward(kind)
        assert slow > crash, f"slow hit {slow} must beat {kind} {crash}"
        assert worst_timeout_cost > crash, (
            f"timing out ({worst_timeout_cost}) must beat {kind} ({crash}), "
            "or the policy learns to die early to stop the time bleed")
    assert fast > slow > worst_timeout_cost
