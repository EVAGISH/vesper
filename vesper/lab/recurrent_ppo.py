"""PPO for a recurrent policy over camera frames.

vesper.lab.ppo shuffles individual steps into minibatches. That is correct for
a feed-forward MLP and wrong for a GRU: the hidden state only means anything if
the steps arrive in order, so this trainer keeps time intact and minibatches
over *environments* instead.

  rollout   T steps x N envs, storing the observation the actor saw (pixels,
            depth, proprio), the privileged vector for the critic, and the
            hidden state at the start of the rollout. The state carries across
            rollouts and is zeroed per environment on a done, exactly as at
            inference.
  update    for each minibatch of environments: replay the T steps in order
            from the stored initial state, backpropagating through the whole
            window (truncated BPTT). Advantages are GAE over the same window.
  aux       the belief head is supervised on the true relative vector to the
            nearest forklift, but only on steps where one was actually in
            frame -- otherwise it would be asked to hallucinate.

Frames dominate memory: at 96 px a step costs ~45 kB per environment (uint8
RGB + fp16 depth), so a 32 x 256 rollout is ~370 MB. Both are kept on the GPU
in the dtype the env produced them in and converted per minibatch.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from vesper.lab.ppo import RunningNorm
from vesper.lab.vision import VisionActorCritic


@dataclass
class RPPOCfg:
    horizon: int = 32          # steps per env per update (also the BPTT window)
    epochs: int = 3            # fewer than feed-forward PPO: each epoch replays every sequence
    env_minibatches: int = 4   # environments per gradient step = num_envs / this
    gamma: float = 0.995
    lam: float = 0.95
    clip: float = 0.2
    lr: float = 3e-4
    ent_coef: float = 0.003
    vf_coef: float = 1.0
    aux_coef: float = 0.5      # weight on the belief head's supervised loss
    max_grad_norm: float = 1.0
    init_std: float = 0.5
    track: tuple = ()


class RecurrentPPO:
    def __init__(self, env, cfg: RPPOCfg = RPPOCfg(), device=None, seed=0, res=96,
                 priv_dim=None, proprio_dim=None):
        self.env, self.cfg = env, cfg
        self.device = device or env.device
        torch.manual_seed(seed)
        self.res = int(res)
        priv_dim = int(priv_dim if priv_dim is not None else env.cfg.state_space)
        proprio_dim = int(proprio_dim if proprio_dim is not None else env.cfg.observation_space)
        self.ac = VisionActorCritic(proprio_dim, env.num_actions, priv_dim=priv_dim,
                                    res=self.res, init_std=cfg.init_std).to(self.device)
        self.norm = RunningNorm(proprio_dim).to(self.device)
        self.priv_norm = RunningNorm(priv_dim).to(self.device)
        self.opt = torch.optim.Adam(self.ac.parameters(), lr=cfg.lr)
        self.h = None

    # ---------------------------------------------------------------- rollout
    def _rollout(self, obs, done):
        c, T = self.cfg, self.cfg.horizon
        N, dev = self.env.num_envs, self.device
        R = self.res
        buf = {
            "px": torch.zeros(T, N, R, R, 3, dtype=torch.uint8, device=dev),
            "dp": torch.zeros(T, N, R, R, 1, dtype=torch.float16, device=dev),
            "pr": torch.zeros(T, N, self.norm.mean.numel(), device=dev),
            "pv": torch.zeros(T, N, self.priv_norm.mean.numel(), device=dev),
            "act": torch.zeros(T, N, self.env.num_actions, device=dev),
            "logp": torch.zeros(T, N, device=dev),
            "val": torch.zeros(T, N, device=dev),
            "rew": torch.zeros(T, N, device=dev),
            "done": torch.zeros(T, N, device=dev),
            "tgt": torch.zeros(T, N, 3, device=dev),      # belief target (relative, scaled)
            "tgt_ok": torch.zeros(T, N, device=dev),      # ...supervise only when one was in frame
        }
        h0 = self.h.clone()
        ep_ret = torch.zeros(N, device=dev)
        rets, touches, ttt = [], [], []
        extra = {k: [] for k in c.track}
        for t in range(T):
            px, dp, pr, pv = self._split(obs)
            self.norm.update(pr); self.priv_norm.update(pv)
            npr, npv = self.norm(pr), self.priv_norm(pv)
            buf["px"][t], buf["dp"][t] = px, dp.half()
            buf["pr"][t], buf["pv"][t] = pr, pv
            buf["done"][t] = done.float()
            with torch.no_grad():
                mean, val, _, self.h = self.ac(px, dp, npr, self.h, priv=npv, done=done)
                dist = self.ac.dist(mean)
                a = dist.sample()
                buf["logp"][t] = dist.log_prob(a).sum(-1)
            buf["act"][t], buf["val"][t] = a, val
            obs, rew, done, info = self.env.vision_step(a)
            buf["rew"][t] = rew
            tgt, ok = self._belief_target(info)
            buf["tgt"][t], buf["tgt_ok"][t] = tgt, ok
            ep_ret += rew
            for i in torch.nonzero(done).flatten().tolist():
                rets.append(ep_ret[i].item())
                if info is not None and "touch" in info:
                    touches.append(float(info["touch"][i].item()))
                    v = float(info["time_to_touch"][i].item())
                    if v == v:
                        ttt.append(v)
                for k in extra:
                    if info is not None and k in info:
                        v = float(info[k][i].item())
                        if v == v:
                            extra[k].append(v)
                ep_ret[i] = 0.0
        with torch.no_grad():
            px, dp, pr, pv = self._split(obs)
            _, last_v, _, _ = self.ac(px, dp, self.norm(pr), self.h, priv=self.priv_norm(pv), done=done)
        return obs, done, buf, h0, last_v, rets, touches, ttt, extra

    def _split(self, obs):
        px = obs["pixels"]
        dp = obs["depth"].float()
        return px, dp, obs["policy"], obs["privileged"]

    def _belief_target(self, info):
        """(relative vector to the nearest *visible* forklift / 100 m, mask)."""
        N, dev = self.env.num_envs, self.device
        if info is None or "belief_target" not in info:
            return torch.zeros(N, 3, device=dev), torch.zeros(N, device=dev)
        t, ok = info["belief_target"], info["belief_ok"]
        return t.to(dev), ok.float().to(dev)

    # ---------------------------------------------------------------- update
    def _gae(self, val, rew, done, last_v, last_done):
        """`done[t]` is the flag *entering* step t, so step t's successor is
        terminal exactly when done[t+1] is set -- and after the last step that
        flag is the rollout's trailing done, not zero."""
        c = self.cfg
        T, N = rew.shape
        adv = torch.zeros(T, N, device=self.device)
        gae = torch.zeros(N, device=self.device)
        for t in reversed(range(T)):
            nextv = last_v if t == T - 1 else val[t + 1]
            nonterm = 1.0 - (done[t + 1] if t + 1 < T else last_done)
            delta = rew[t] + c.gamma * nextv * nonterm - val[t]
            gae = delta + c.gamma * c.lam * nonterm * gae
            adv[t] = gae
        return adv, adv + val

    def update(self, buf, h0, last_v, last_done):
        c = self.cfg
        T, N = buf["rew"].shape
        adv, ret = self._gae(buf["val"], buf["rew"], buf["done"], last_v, last_done)
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)
        mb = max(1, N // c.env_minibatches)
        stats = {"pi": 0.0, "vf": 0.0, "ent": 0.0, "aux": 0.0}
        steps = 0
        for _ in range(c.epochs):
            order = torch.randperm(N, device=self.device)
            for s in range(0, N, mb):
                j = order[s:s + mb]
                h = h0[j]
                lp, vs, aux, ents = [], [], [], []
                for t in range(T):
                    npr = self.norm(buf["pr"][t, j])
                    npv = self.priv_norm(buf["pv"][t, j])
                    mean, v, belief, h = self.ac(buf["px"][t, j], buf["dp"][t, j].float(), npr, h,
                                                 priv=npv, done=buf["done"][t, j] > 0.5)
                    d = self.ac.dist(mean)
                    lp.append(d.log_prob(buf["act"][t, j]).sum(-1))
                    ents.append(d.entropy().sum(-1))
                    vs.append(v)
                    aux.append(((belief - buf["tgt"][t, j]) ** 2).mean(dim=1) * buf["tgt_ok"][t, j])
                logp = torch.stack(lp); v = torch.stack(vs)
                ratio = (logp - buf["logp"][:, j]).exp()
                a = adv[:, j]
                pi_loss = -torch.min(ratio * a, ratio.clamp(1 - c.clip, 1 + c.clip) * a).mean()
                vf_loss = (v - ret[:, j]).pow(2).mean()
                ent = torch.stack(ents).mean()
                ok = buf["tgt_ok"][:, j].sum().clamp(min=1.0)
                aux_loss = torch.stack(aux).sum() / ok
                loss = pi_loss + c.vf_coef * vf_loss - c.ent_coef * ent + c.aux_coef * aux_loss
                self.opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.ac.parameters(), c.max_grad_norm)
                self.opt.step()
                stats["pi"] += pi_loss.item(); stats["vf"] += vf_loss.item()
                stats["ent"] += ent.item(); stats["aux"] += aux_loss.item()
                steps += 1
        return {k: v / max(1, steps) for k, v in stats.items()}

    # ---------------------------------------------------------------- loop
    def learn(self, iterations, log_every=10, on_log=None):
        obs = self.env.vision_reset()
        for key in ("pixels", "depth", "policy", "privileged"):
            if key not in obs:
                raise ValueError(f"the env gives no '{key}': the vision trainer needs a rendered camera "
                                 "(ChaseEnvCfg.camera = True, launched with --enable_cameras)")
        N = self.env.num_envs
        self.h = self.ac.initial_state(N, self.device)
        done = torch.zeros(N, dtype=torch.bool, device=self.device)
        history = []
        for it in range(iterations):
            obs, done, buf, h0, last_v, rets, touches, ttt, extra = self._rollout(obs, done)
            stats = self.update(buf, h0, last_v, done.float())
            self.h = self.h.detach()
            row = {"iter": it,
                   "ep_return": sum(rets) / len(rets) if rets else float("nan"),
                   "touch_rate": sum(touches) / len(touches) if touches else float("nan"),
                   "time_to_touch": sum(ttt) / len(ttt) if ttt else float("nan"),
                   "episodes": len(rets),
                   **{k: (sum(v) / len(v) if v else float("nan")) for k, v in extra.items()},
                   **stats}
            history.append(row)
            if on_log and (it % log_every == 0 or it == iterations - 1):
                on_log(row)
        return history

    def save(self, path):
        torch.save({"ac": self.ac.state_dict(), "norm": self.norm.state_dict(),
                    "priv_norm": self.priv_norm.state_dict(),
                    "kind": "vision", "res": self.res,
                    "proprio_dim": self.norm.mean.numel(), "priv_dim": self.priv_norm.mean.numel(),
                    "act_dim": self.env.num_actions}, path)


def load_vision_policy(ck, device="cpu"):
    """(VisionActorCritic, proprio RunningNorm, privileged RunningNorm) from a checkpoint."""
    ac = VisionActorCritic(ck["proprio_dim"], ck["act_dim"], priv_dim=ck["priv_dim"],
                           res=ck.get("res", 96)).to(device)
    ac.load_state_dict(ck["ac"]); ac.eval()
    norm = RunningNorm(ck["proprio_dim"]).to(device); norm.load_state_dict(ck["norm"])
    pnorm = RunningNorm(ck["priv_dim"]).to(device); pnorm.load_state_dict(ck["priv_norm"])
    return ac, norm, pnorm
