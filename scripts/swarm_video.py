"""Render a grid of quadcopter envs to MP4 -- the 'many drones at once' gate.

    /isaac-sim/python.sh scripts/swarm_video.py --num_envs 16 --headless --enable_cameras
"""
import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--num_envs", type=int, default=16)
parser.add_argument("--seconds", type=int, default=8)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app = AppLauncher(args).app

import gymnasium as gym
import torch

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import parse_env_cfg

from vesper.capture import RunCapture

TASK = "Isaac-Quadcopter-Direct-v0"
cfg = parse_env_cfg(TASK, device=args.device, num_envs=args.num_envs)
env = gym.make(TASK, cfg=cfg, render_mode="rgb_array")
env.reset()
act_dim = gym.spaces.flatdim(env.unwrapped.single_action_space)

cap = RunCapture("swarm")
cap.note(scene=f"{TASK} x{args.num_envs}, random actions")
fps = 30
for i in range(args.seconds * fps):
    action = (torch.rand((args.num_envs, act_dim), device=env.unwrapped.device) - 0.3)
    env.step(action)
    frame = env.render()
    if frame is not None:
        cap.add_frame(frame)
print(f"wrote {cap.finish(fps=fps)}", flush=True)
env.close()
app.close()
