"""Train the search-and-reach policy on the Cornell world.

The drone starts somewhere random over campus and does not know where any of the
forklifts are. It has to sweep ground, find them with a downward camera that
terrain, buildings and foliage can deny, and then run each one down -- fastest
total clearance wins.

    /isaac-sim/python.sh scripts/train_search.py --num_envs 1024 --iters 1500 --headless

Writes runs/<id>/search.pt (best policy + obs-norm), search_last.pt, curve.jsonl.
"""
import argparse
import json
import time

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--num_envs", type=int, default=1024)
parser.add_argument("--iters", type=int, default=1500)
parser.add_argument("--horizon", type=int, default=48)
parser.add_argument("--targets", type=int, default=3)
parser.add_argument("--arena", type=float, default=300.0, help="search box half-extent (m)")
parser.add_argument("--arena_start", type=float, default=None,
                    help="curriculum: begin here and grow linearly to --arena over "
                         "--arena_iters, so the first reaches are found in a small box")
parser.add_argument("--arena_iters", type=int, default=600)
parser.add_argument("--episode_s", type=float, default=90.0)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--lr", type=float, default=3e-4)
parser.add_argument("--gamma", type=float, default=0.997,
                    help="search needs a long horizon; 0.997 is ~13 s at 25 Hz")
parser.add_argument("--world", default=None, help="world USD (default the Cornell site)")
parser.add_argument("--map", default=None, help="world map npz (default beside the USD)")
parser.add_argument("--vehicle", default=None, help="forklift | cart | path to a USD")
parser.add_argument("--resume", default=None, help="checkpoint to warm-start from")
parser.add_argument("--tag", default="search-train")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.headless = True
app = AppLauncher(args).app

import torch  # noqa: E402

from vesper.capture import RunCapture  # noqa: E402
from vesper.lab.ppo import PPO, PPOCfg  # noqa: E402
from vesper.lab.search_env import SearchEnv, SearchEnvCfg  # noqa: E402


class Adapter:
    """PPO <-> SearchEnv (gym-style) bridge."""

    def __init__(self, env):
        self.env = env
        self.num_envs = env.num_envs
        self.device = env.device
        self.num_obs = env.num_obs
        self.num_actions = env.num_actions

    def reset(self):
        return self.env.ppo_reset()

    def step(self, a):
        return self.env.ppo_step(a)


cfg = SearchEnvCfg()
cfg.scene.num_envs = args.num_envs
cfg.scene.env_spacing = 0.0            # one world, shared by every environment
cfg.n_targets = args.targets
cfg.episode_length_s = args.episode_s
cfg.search = {"arena_half": args.arena}
cfg.vehicle_model = args.vehicle
if args.world:
    cfg.world_usd = args.world
if args.map:
    cfg.world_map = args.map

env = SearchEnv(cfg, seed=args.seed)
adapter = Adapter(env)

track = ("found", "cleared", "coverage", "oob", "flip", "crash")
ppo = PPO(adapter, PPOCfg(horizon=args.horizon, lr=args.lr, gamma=args.gamma, track=track),
          device=env.device, seed=args.seed)
if args.resume:
    ck = torch.load(args.resume, map_location=env.device)
    ppo.ac.load_state_dict(ck["ac"]); ppo.norm.load_state_dict(ck["norm"])
    print(f"resumed from {args.resume}", flush=True)

cap = RunCapture(args.tag)
cap.note(num_envs=args.num_envs, iters=args.iters, targets=args.targets, arena=args.arena,
         episode_s=args.episode_s, gamma=args.gamma, world=cfg.world_usd, seed=args.seed)
curve = open(cap.dir / "curve.jsonl", "w")
print(f"train: {args.num_envs} envs x {args.iters} iters, {args.targets} targets in a "
      f"{2*args.arena:.0f} m box, obs {env.num_obs} -> {cap.dir}", flush=True)

t0 = time.time()
best = -1e9


def log(row):
    global best
    curve.write(json.dumps(row) + "\n"); curve.flush()
    steps = (row["iter"] + 1) * args.horizon * args.num_envs
    sps = steps / (time.time() - t0 + 1e-9)
    print(f"it {row['iter']:5d} | ret {row['ep_return']:8.1f} | found {row['found']:.2f} "
          f"| cleared {row['cleared']:.2f} | swept {row['coverage']:.2f} "
          f"| all {row['intercept_rate']:.2f} | t {row['time_to_intercept']:5.1f}s "
          f"| box {2*env.tcfg.arena_half:.0f}m "
          f"| oob {row['oob']:.2f} flip {row['flip']:.2f} crash {row['crash']:.2f} "
          f"| eps {row['episodes']:5d} | {sps/1e3:.0f}k step/s", flush=True)
    # "cleared" is the score that matters: the fraction of vehicles reached
    score = row["cleared"]
    if score == score and score >= best:
        best = score
        ppo.save(cap.dir / "search.pt")


if args.arena_start:
    # The arena is the curriculum knob that matters: in a 300 m box a policy that
    # cannot yet search still trips over a forklift often enough to learn that
    # reaching one pays, and the observation scales positions by arena_half, so
    # what it learns carries over as the box grows.
    def grow(row):
        f = min(1.0, (row["iter"] + 1) / max(1, args.arena_iters))
        env.task.set_arena(args.arena_start + f * (args.arena - args.arena_start))
        log(row)

    print(f"curriculum: arena {args.arena_start:.0f} -> {args.arena:.0f} m over "
          f"{args.arena_iters} iters", flush=True)
    env.task.set_arena(args.arena_start)
    hist = ppo.learn(args.iters, log_every=10, on_log=grow)
else:
    hist = ppo.learn(args.iters, log_every=10, on_log=log)
ppo.save(cap.dir / "search_last.pt")
curve.close()
wall = time.time() - t0
final = [h for h in hist if h["episodes"] > 0][-20:] or [{}]


def avg(key):
    v = [h[key] for h in final if h.get(key) == h.get(key)]
    return round(sum(v) / len(v), 3) if v else float("nan")


summary = {"iters": args.iters, "wall_s": round(wall), "envs": args.num_envs,
           "mean_return": avg("ep_return"), "found": avg("found"), "cleared": avg("cleared"),
           "coverage": avg("coverage"), "all_cleared_rate": avg("intercept_rate"),
           "time_to_clear_s": avg("time_to_intercept"), "best_cleared": round(best, 3),
           "oob": avg("oob"), "flip": avg("flip"), "crash": avg("crash"),
           "policy": str(cap.dir / "search.pt")}
print("DONE " + json.dumps(summary), flush=True)
(cap.dir / "summary.json").write_text(json.dumps(summary, indent=1))
env.close()
app.close()
