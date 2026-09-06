"""Self-contained PPO for the throughput lane.

Deliberately free of rsl_rl / skrl: it talks to any vectorized env exposing
`num_envs`, `device`, `num_obs`, `num_actions`, `reset() -> obs[N,obs]` and
`step(act[N,act]) -> (obs, reward[N], done[N])` with Isaac-style auto-reset.
Gaussian policy, GAE, clipped surrogate, running observation normalization.
Small enough to read; runs on CPU (tests) or GPU (droplet).
"""
from dataclasses import dataclass

import torch
import torch.nn as nn


class RunningNorm(nn.Module):
    """Welford running mean/var for observation normalization (frozen at eval)."""

    def __init__(self, dim, eps=1e-4):
        super().__init__()
        self.register_buffer("mean", torch.zeros(dim))
        self.register_buffer("var", torch.ones(dim))
        self.register_buffer("count", torch.tensor(eps))

    def update(self, x):
        bmean = x.mean(0)
        bvar = x.var(0, unbiased=False)
        bcount = torch.tensor(float(x.shape[0]), device=x.device)
        delta = bmean - self.mean
        tot = self.count + bcount
        self.mean += delta * bcount / tot
        m_a = self.var * self.count
        m_b = bvar * bcount
        self.var = (m_a + m_b + delta * delta * self.count * bcount / tot) / tot
        self.count = tot

    def forward(self, x):
        return (x - self.mean) / (self.var.sqrt() + 1e-5)


def mlp(sizes, act=nn.ELU):
    layers = []
    for i in range(len(sizes) - 1):
        layers += [nn.Linear(sizes[i], sizes[i + 1])]
        if i < len(sizes) - 2:
            layers += [act()]
    return nn.Sequential(*layers)


class ActorCritic(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden=(256, 256, 128), init_std=1.0):
        super().__init__()
        self.hidden = tuple(int(h) for h in hidden)
        self.actor = mlp([obs_dim, *hidden, act_dim])
        self.critic = mlp([obs_dim, *hidden, 1])
        self.log_std = nn.Parameter(torch.ones(act_dim) * torch.tensor(init_std).log())

    def dist(self, obs):
        mean = self.actor(obs)
        return torch.distributions.Normal(mean, self.log_std.exp())

    def act(self, obs):
        d = self.dist(obs)
        a = d.sample()
        return a, d.log_prob(a).sum(-1), self.critic(obs).squeeze(-1)

    def evaluate(self, obs, act):
        d = self.dist(obs)
        return (d.log_prob(act).sum(-1), d.entropy().sum(-1), self.critic(obs).squeeze(-1))


@dataclass
class PPOCfg:
    horizon: int = 32          # steps per env per update
    epochs: int = 5
    minibatches: int = 4
    gamma: float = 0.99
    lam: float = 0.95
    clip: float = 0.2
    lr: float = 3e-4
    ent_coef: float = 0.005
    vf_coef: float = 1.0
    max_grad_norm: float = 1.0
    init_std: float = 1.0
    # extra per-episode scalars to average into the log row. Any info key whose
    # value is a [num_envs] tensor works; the search task reports how much of the
    # map got swept and how many vehicles were found and reached.
    track: tuple = ()


class PPO:
    def __init__(self, env, cfg: PPOCfg = PPOCfg(), hidden=(256, 256, 128), device=None, seed=0):
        self.env = env
        self.cfg = cfg
        self.device = device or env.device
        torch.manual_seed(seed)
        self.ac = ActorCritic(env.num_obs, env.num_actions, hidden, cfg.init_std).to(self.device)
        self.norm = RunningNorm(env.num_obs).to(self.device)
        self.opt = torch.optim.Adam(self.ac.parameters(), lr=cfg.lr)

    def _rollout(self, obs):
        c, N, dev = self.cfg, self.env.num_envs, self.device
        T = c.horizon
        obs_buf = torch.zeros(T, N, self.env.num_obs, device=dev)
        act_buf = torch.zeros(T, N, self.env.num_actions, device=dev)
        logp_buf = torch.zeros(T, N, device=dev)
        val_buf = torch.zeros(T, N, device=dev)
        rew_buf = torch.zeros(T, N, device=dev)
        done_buf = torch.zeros(T, N, device=dev)
        ep_ret = torch.zeros(N, device=dev)
        rets, intercepts, tti = [], [], []
        extra = {k: [] for k in self.cfg.track}
        for t in range(T):
            self.norm.update(obs)
            nobs = self.norm(obs)
            with torch.no_grad():
                a, logp, v = self.ac.act(nobs)
            nxt, rew, done, info = self.env.step(a)
            obs_buf[t], act_buf[t], logp_buf[t], val_buf[t] = obs, a, logp, v
            rew_buf[t], done_buf[t] = rew, done.float()
            ep_ret += rew
            for i in torch.nonzero(done).flatten().tolist():
                rets.append(ep_ret[i].item())
                if info is not None and "intercept" in info:
                    h = float(info["intercept"][i].item())
                    intercepts.append(h)
                    if h and "time_to_intercept" in info:
                        v = float(info["time_to_intercept"][i].item())
                        if v == v:                      # not NaN
                            tti.append(v)
                for k in extra:
                    if info is not None and k in info:
                        v = float(info[k][i].item())
                        if v == v:
                            extra[k].append(v)
                ep_ret[i] = 0.0
            obs = nxt
        with torch.no_grad():
            last_v = self.ac.critic(self.norm(obs)).squeeze(-1)
        return (obs, (obs_buf, act_buf, logp_buf, val_buf, rew_buf, done_buf, last_v),
                rets, intercepts, tti, extra)

    def _gae(self, val, rew, done, last_v):
        c = self.cfg
        T, N = rew.shape
        adv = torch.zeros(T, N, device=self.device)
        gae = torch.zeros(N, device=self.device)
        for t in reversed(range(T)):
            nextv = last_v if t == T - 1 else val[t + 1]
            nonterm = 1.0 - done[t]
            delta = rew[t] + c.gamma * nextv * nonterm - val[t]
            gae = delta + c.gamma * c.lam * nonterm * gae
            adv[t] = gae
        return adv, adv + val

    def update(self, batch):
        c = self.cfg
        obs_b, act_b, logp_b, val_b, rew_b, done_b, last_v = batch
        adv, ret = self._gae(val_b, rew_b, done_b, last_v)
        T, N = rew_b.shape
        obs = self.norm(obs_b.reshape(T * N, -1))
        act = act_b.reshape(T * N, -1)
        logp_old = logp_b.reshape(-1)
        adv = adv.reshape(-1)
        ret = ret.reshape(-1)
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)
        idx = torch.randperm(T * N, device=self.device)
        mb = (T * N) // c.minibatches
        stats = {"pi": 0.0, "vf": 0.0, "ent": 0.0}
        for _ in range(c.epochs):
            for s in range(0, T * N, mb):
                j = idx[s:s + mb]
                logp, ent, v = self.ac.evaluate(obs[j], act[j])
                ratio = (logp - logp_old[j]).exp()
                a = adv[j]
                pi_loss = -torch.min(ratio * a, ratio.clamp(1 - c.clip, 1 + c.clip) * a).mean()
                vf_loss = (v - ret[j]).pow(2).mean()
                ent_loss = ent.mean()
                loss = pi_loss + c.vf_coef * vf_loss - c.ent_coef * ent_loss
                self.opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.ac.parameters(), c.max_grad_norm)
                self.opt.step()
                stats["pi"] += pi_loss.item(); stats["vf"] += vf_loss.item(); stats["ent"] += ent_loss.item()
        n = c.epochs * max(1, (T * N) // mb)
        return {k: v / n for k, v in stats.items()}

    def learn(self, iterations, log_every=10, on_log=None):
        obs = self.env.reset()
        if obs.dim() != 2:
            raise ValueError(f"PPO expects a flat [N, obs] observation, got {tuple(obs.shape)}; "
                             "the pixel policy (vesper.lab.vision) needs the recurrent trainer, not this one")
        history = []
        for it in range(iterations):
            obs, batch, rets, intercepts, tti, extra = self._rollout(obs)
            stats = self.update(batch)
            row = {"iter": it,
                   "ep_return": sum(rets) / len(rets) if rets else float("nan"),
                   "intercept_rate": sum(intercepts) / len(intercepts) if intercepts else float("nan"),
                   "time_to_intercept": sum(tti) / len(tti) if tti else float("nan"),
                   "episodes": len(rets),
                   **{k: (sum(v) / len(v) if v else float("nan")) for k, v in extra.items()},
                   **stats}
            history.append(row)
            if on_log and (it % log_every == 0 or it == iterations - 1):
                on_log(row)
        return history

    def save(self, path):
        torch.save({"ac": self.ac.state_dict(), "norm": self.norm.state_dict(),
                    "obs_dim": self.env.num_obs, "act_dim": self.env.num_actions,
                    "hidden": list(self.ac.hidden)}, path)

    @torch.no_grad()
    def action(self, obs, deterministic=True):
        nobs = self.norm(obs)
        return self.ac.actor(nobs) if deterministic else self.ac.dist(nobs).sample()


def load_policy(ck, device="cpu"):
    """(ActorCritic, RunningNorm) rebuilt from a checkpoint dict, whatever width it was.

    The network shape lives in the checkpoint ("hidden"); older files without
    it were all trained at the class default. Every script that flies, scores
    or serves a policy goes through here rather than guessing the width.
    """
    hidden = tuple(ck.get("hidden", ActorCritic.__init__.__defaults__[0]))
    ac = ActorCritic(ck["obs_dim"], ck["act_dim"], hidden).to(device)
    ac.load_state_dict(ck["ac"]); ac.eval()
    norm = RunningNorm(ck["obs_dim"]).to(device)
    norm.load_state_dict(ck["norm"])
    return ac, norm
