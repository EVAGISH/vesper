"""Score a search policy and say *how* it fails, not just how often.

    /isaac-sim/python.sh scripts/eval_search.py --policy runs/<id>/search.pt \
        --num_envs 256 --episodes 400 --headless

A single "cleared" number cannot distinguish a policy that never finds anything
from one that finds everything and then flies into a building on the way down.
This reports the whole funnel -- swept, found, reached, and which terminal
condition ended each episode -- plus how long the drone had a target in hand
before it either reached it or lost it.
"""
import argparse
import json

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--policy", required=True)
parser.add_argument("--num_envs", type=int, default=256)
parser.add_argument("--episodes", type=int, default=400)
parser.add_argument("--targets", type=int, default=3)
parser.add_argument("--arena", type=float, default=300.0)
parser.add_argument("--episode_s", type=float, default=90.0)
parser.add_argument("--seed", type=int, default=11)
parser.add_argument("--stochastic", action="store_true")
parser.add_argument("--groups", type=int, default=0)
parser.add_argument("--camera", action="store_true", help="sightings from rendered pixels")
parser.add_argument("--out", default=None)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.headless = True
if args.camera:
    args.enable_cameras = True
app = AppLauncher(args).app

import torch  # noqa: E402

from vesper.lab.ppo import load_policy  # noqa: E402
from vesper.lab.search_env import SearchEnv, SearchEnvCfg  # noqa: E402

cfg = SearchEnvCfg()
cfg.scene.num_envs = args.num_envs
cfg.scene.env_spacing = 0.0
cfg.n_targets = args.targets
cfg.episode_length_s = args.episode_s
cfg.search = {"arena_half": args.arena}
cfg.n_groups = args.groups
cfg.camera = args.camera
env = SearchEnv(cfg, seed=args.seed)

ck = torch.load(args.policy, map_location=env.device)
ac, norm = load_policy(ck, env.device)
if ck["obs_dim"] != env.num_obs:
    raise SystemExit(f"policy expects {ck['obs_dim']}-wide observations, env gives {env.num_obs}")


@torch.no_grad()
def policy(o):
    n = norm(o)
    return ac.dist(n).sample() if args.stochastic else ac.actor(n)


obs = env.ppo_reset()
dev = env.device
N = env.num_envs
tally = {k: 0.0 for k in ("episodes", "crash", "oob", "flip", "timeout", "all_cleared")}
sums = {k: 0.0 for k in ("found", "cleared", "coverage", "t_clear", "agl", "steps")}
n_clear = 0
# per-episode accumulators
held = torch.zeros(N, device=dev)          # steps with at least one live known target
alive = torch.zeros(N, device=dev)
agl_sum = torch.zeros(N, device=dev)
held_total = 0.0
dt = cfg.sim.dt * cfg.decimation
max_steps = int(args.episode_s / dt)

while tally["episodes"] < args.episodes:
    act = policy(obs)
    obs, rew, done, info = env.ppo_step(act)
    live = (env.task.known & ~env.task.reached).any(dim=1).float()
    held += live
    alive += 1.0
    agl_sum += info["agl"]
    d = torch.nonzero(done).flatten()
    if len(d):
        crash = info["crash"][d]; oob = info["oob"][d]; flip = info["flip"][d]
        allc = info["intercept"][d]
        timeout = ~(crash | oob | flip | allc)
        tally["episodes"] += len(d)
        tally["crash"] += float(crash.sum()); tally["oob"] += float(oob.sum())
        tally["flip"] += float(flip.sum()); tally["all_cleared"] += float(allc.sum())
        tally["timeout"] += float(timeout.sum())
        sums["found"] += float(info["found"][d].sum())
        sums["cleared"] += float(info["cleared"][d].sum())
        sums["coverage"] += float(info["coverage"][d].sum())
        sums["steps"] += float(alive[d].sum())
        sums["agl"] += float((agl_sum[d] / alive[d].clamp(min=1)).sum())
        held_total += float((held[d] / alive[d].clamp(min=1)).sum())
        t = info["time_to_intercept"][d]
        good = t == t
        sums["t_clear"] += float(t[good].sum()); n_clear += int(good.sum())
        held[d] = 0.0; alive[d] = 0.0; agl_sum[d] = 0.0

e = max(1.0, tally["episodes"])
report = {
    "policy": args.policy, "episodes": int(e), "targets": args.targets, "arena_m": 2 * args.arena,
    "found_frac": round(sums["found"] / e, 3),
    "cleared_frac": round(sums["cleared"] / e, 3),
    "coverage_frac": round(sums["coverage"] / e, 3),
    "all_cleared_rate": round(tally["all_cleared"] / e, 3),
    "time_to_clear_s": round(sums["t_clear"] / n_clear, 1) if n_clear else None,
    "mean_agl_m": round(sums["agl"] / e, 1),
    "mean_episode_s": round(sums["steps"] / e * dt, 1),
    "frac_of_episode_holding_a_target": round(held_total / e, 3),
    "ended_by": {k: round(tally[k] / e, 3) for k in ("crash", "oob", "flip", "timeout", "all_cleared")},
}
print("EVAL " + json.dumps(report, indent=1), flush=True)
if args.out:
    open(args.out, "w").write(json.dumps(report, indent=1))
env.close()
app.close()
