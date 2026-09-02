"""Benchmark an Isaac Lab task headless: env-steps/s at a given env count.

    /isaac-sim/python.sh scripts/bench_lab.py --task Isaac-Quadcopter-Direct-v0 \
        --num_envs 4096 --steps 1000 --headless

Zero actions: quadcopters free-fall and hit episode resets -- deliberately, so
the number includes reset cost (the honest sweep-workload figure).
"""
import argparse
import json
import time

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--task", default="Isaac-Quadcopter-Direct-v0")
parser.add_argument("--num_envs", type=int, default=2048)
parser.add_argument("--steps", type=int, default=1000)
parser.add_argument("--out", default=None)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app = AppLauncher(args).app

import gymnasium as gym
import torch

import isaaclab_tasks  # noqa: F401  (registers tasks)
from isaaclab_tasks.utils import parse_env_cfg

cfg = parse_env_cfg(args.task, device=args.device, num_envs=args.num_envs)
env = gym.make(args.task, cfg=cfg)
env.reset()
act_dim = gym.spaces.flatdim(env.unwrapped.single_action_space)
action = torch.zeros((args.num_envs, act_dim), device=env.unwrapped.device)

for _ in range(50):  # warmup
    env.step(action)
torch.cuda.synchronize()
t0 = time.time()
for _ in range(args.steps):
    env.step(action)
torch.cuda.synchronize()
wall = time.time() - t0

result = {
    "task": args.task, "num_envs": args.num_envs, "steps": args.steps,
    "wall_s": round(wall, 2),
    "env_steps_per_s": round(args.num_envs * args.steps / wall),
}
print(f"BENCH {json.dumps(result)}", flush=True)
if args.out:
    with open(args.out, "a") as f:
        f.write(json.dumps(result) + "\n")
env.close()
app.close()
