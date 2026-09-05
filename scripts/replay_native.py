"""Backfill after-action artifacts for an existing checkpoint -- no retrain.

    .venv/bin/python scripts/replay_native.py runs/<id>/search.pt \
        --map assets/kramatorsk/kramatorsk_map.npz [--run runs/<id>]

Rolls the policy for one deterministic episode on the native sim and writes
trajectory.parquet + replay.json into the run dir (default: the checkpoint's
own dir), so the Runs tab shows the trajectory and the video renderer has data.
"""
import argparse
from pathlib import Path

import torch

from vesper.lab.ppo import load_policy
from vesper.native import NativeSearchEnv, NativeSearchEnvCfg
from vesper.native.replay import log_episode

parser = argparse.ArgumentParser()
parser.add_argument("checkpoint")
parser.add_argument("--map", default=None, help="world map npz (default the Cornell site)")
parser.add_argument("--run", default=None, help="run dir to write into (default: checkpoint's dir)")
parser.add_argument("--targets", type=int, default=3)
parser.add_argument("--arena", type=float, default=300.0)
parser.add_argument("--seed", type=int, default=7)
parser.add_argument("--device", default="cpu")
parser.add_argument("--reach_radius", type=float, default=None,
                    help="widen the kill sphere for this eval so the clip ends in a neutralization")
parser.add_argument("--episode_s", type=float, default=None,
                    help="episode length for the eval (default the env's ~75s)")
args = parser.parse_args()

ck = torch.load(args.checkpoint, map_location=args.device)
cfg = NativeSearchEnvCfg()
cfg.num_envs = 16
cfg.n_targets = args.targets
cfg.n_groups = 1                       # one shared target set -- a single mission to watch
cfg.search = {"arena_half": args.arena}
if args.map:
    cfg.world_map = args.map
env = NativeSearchEnv(cfg, device=args.device, seed=args.seed)
if ck["obs_dim"] != env.num_obs:
    raise SystemExit(f"policy expects {ck['obs_dim']}-wide obs, env gives {env.num_obs}")

ac, norm = load_policy(ck, args.device)


class Pol:
    @torch.no_grad()
    def action(self, obs, deterministic=True):
        return ac.actor(norm(obs))


run_dir = Path(args.run) if args.run else Path(args.checkpoint).parent
run_dir.mkdir(parents=True, exist_ok=True)
world_name = Path(cfg.world_map).stem.replace("_map", "")
max_steps = int(args.episode_s / env._dt) if args.episode_s else None
n = log_episode(env, Pol(), run_dir, world_name, max_steps=max_steps, reach_radius=args.reach_radius)
print(f"wrote {run_dir/'trajectory.parquet'} and {run_dir/'replay.json'} ({n} frames)")
