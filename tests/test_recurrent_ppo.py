"""The recurrent trainer runs end to end on a fake camera env, and learns something.

No Isaac: a stand-in env with the same observation contract as ChaseEnv. The
point is that the loop is wired correctly -- sequences stay in order, the
hidden state carries and resets, the belief head is supervised only where a
target was in frame, and the gradients actually move the policy.
"""
import torch

from vesper.lab.recurrent_ppo import RPPOCfg, RecurrentPPO, load_vision_policy


class FakeCamEnv:
    """A 1-D task through a camera: a bright column in the frame marks the target's
    bearing; going toward it pays. Small enough to train in seconds on a CPU."""

    def __init__(self, num_envs=8, res=16, k=1, device="cpu", seed=0):
        self.num_envs, self.res, self.k = num_envs, res, k
        self.device = device
        self.num_actions = 3
        self.g = torch.Generator().manual_seed(seed)

        class C:
            observation_space = 11
            state_space = 12 + 6 * k + 1
        self.cfg = C()
        self.bearing = torch.zeros(num_envs)
        self.t = torch.zeros(num_envs)

    def _obs(self):
        n, R = self.num_envs, self.res
        px = torch.zeros(n, R, R, 3, dtype=torch.uint8)
        col = ((self.bearing * 0.5 + 0.5) * (R - 1)).long().clamp(0, R - 1)
        px[torch.arange(n), :, col] = 255
        depth = torch.rand(n, R, R, 1, generator=self.g)
        pr = torch.zeros(n, 11); pr[:, 0] = self.t / 20.0
        pv = torch.zeros(n, self.cfg.state_space); pv[:, 0] = self.bearing
        return {"pixels": px, "depth": depth, "policy": pr, "privileged": pv}

    def vision_reset(self):
        self.bearing = torch.rand(self.num_envs, generator=self.g) * 2 - 1
        self.t = torch.zeros(self.num_envs)
        return self._obs()

    def vision_step(self, a):
        # steering left/right closes the bearing; reward is closing it
        prev = self.bearing.abs()
        self.bearing = (self.bearing - 0.3 * torch.tanh(a[:, 1])).clamp(-1, 1)
        rew = prev - self.bearing.abs()
        self.t += 1
        done = (self.bearing.abs() < 0.05) | (self.t >= 20)
        rew = rew + 5.0 * (self.bearing.abs() < 0.05).float()
        info = {"hit": (self.bearing.abs() < 0.05).float(),
                "time_to_hit": torch.where(self.bearing.abs() < 0.05, self.t,
                                           torch.full_like(self.t, float("nan"))),
                "belief_target": torch.stack([self.bearing, torch.zeros(self.num_envs),
                                              torch.zeros(self.num_envs)], dim=1),
                "belief_ok": torch.ones(self.num_envs, dtype=torch.bool)}
        if done.any():
            idx = torch.nonzero(done).flatten()
            self.bearing[idx] = torch.rand(len(idx), generator=self.g) * 2 - 1
            self.t[idx] = 0.0
        return self._obs(), rew, done, info


def _avg(rows, key):
    v = [r[key] for r in rows if r[key] == r[key]]
    return sum(v) / len(v) if v else float("nan")


def test_recurrent_ppo_trains_and_round_trips(tmp_path):
    env = FakeCamEnv(num_envs=16, res=16)
    ppo = RecurrentPPO(env, RPPOCfg(horizon=8, epochs=2, env_minibatches=2, lr=3e-3),
                       device="cpu", seed=0, res=16)
    rows = []
    hist = ppo.learn(40, log_every=1, on_log=rows.append)
    assert len(hist) == 40 and len(rows) == 40
    for k in ("ep_return", "hit_rate", "time_to_hit", "episodes", "pi", "vf", "ent", "aux"):
        assert k in rows[-1]
    assert _avg(hist[-8:], "ep_return") > _avg(hist[:8], "ep_return") + 1.0, "the policy is not learning"
    assert _avg(hist[-8:], "hit_rate") > 0.9, "the toy task should be solved"
    assert _avg(hist[-8:], "aux") < _avg(hist[:8], "aux"), "the belief head is not being fitted"

    ppo.save(tmp_path / "v.pt")
    ck = torch.load(tmp_path / "v.pt")
    assert ck["kind"] == "vision" and ck["res"] == 16
    ac, norm, pnorm = load_vision_policy(ck)
    obs = env.vision_reset()
    h = ac.initial_state(env.num_envs)
    assert h.shape == (16, 256)
    with torch.no_grad():
        mean, _, _, h2 = ac(obs["pixels"], obs["depth"], norm(obs["policy"]), h)
    assert mean.shape == (16, 3) and torch.isfinite(mean).all() and h2.shape == (16, 256)


def test_trainer_refuses_an_env_with_no_camera():
    env = FakeCamEnv(num_envs=4, res=16)
    env.vision_reset = lambda: {"policy": torch.zeros(4, 11), "privileged": torch.zeros(4, 19)}
    ppo = RecurrentPPO(env, RPPOCfg(horizon=4), device="cpu", res=16)
    try:
        ppo.learn(1)
    except ValueError as e:
        assert "pixels" in str(e)
    else:
        raise AssertionError("training with no camera should not silently proceed")
