"""VesperQuad at scale: SE3 controller flies per-env waypoint squares.

Bench:  /isaac-sim/python.sh scripts/fly_swarm.py --num_envs 4096 --headless --bench
Video:  /isaac-sim/python.sh scripts/fly_swarm.py --num_envs 16 --headless \
            --enable_cameras --video --seconds 20
"""
import argparse
import json
import time

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--num_envs", type=int, default=1024)
parser.add_argument("--seconds", type=float, default=20.0)
parser.add_argument("--bench", action="store_true")
parser.add_argument("--video", action="store_true")
parser.add_argument("--out", default=None)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app = AppLauncher(args).app

import torch

from vesper.capture import RunCapture
from vesper.control import SE3Controller
from vesper.lab.vesper_quad import VesperQuadEnv, VesperQuadEnvCfg
from vesper.record import TrajectoryWriter

cfg = VesperQuadEnvCfg()
cfg.scene.num_envs = args.num_envs
env = VesperQuadEnv(cfg, render_mode="rgb_array" if args.video else None)
env.reset()
ctrl = SE3Controller(env.params, args.num_envs, device=env.device)

# per-env square circuit, corners relative to each env origin
corners = torch.tensor([[1.5, 0, 2.0], [1.5, 1.5, 2.0], [0, 1.5, 2.0], [0, 0, 2.0]], device=env.device)
wp_idx = torch.zeros(args.num_envs, dtype=torch.long, device=env.device)

cap = RunCapture("vesperquad") if args.video else None
traj = TrajectoryWriter(cap.dir) if cap else None
steps = int(args.seconds / (cfg.sim.dt * cfg.decimation))
t0 = None
for i in range(steps):
    pos, vel, quat, avb = env.flight_state()
    target = corners[wp_idx]
    omega = ctrl.compute(pos, vel, quat, avb, target)
    env.step(omega / env.params.omega_max)
    reached = (pos - target).norm(dim=1) < 0.4
    wp_idx = torch.where(reached, (wp_idx + 1) % 4, wp_idx)
    if i == 49:
        torch.cuda.synchronize(); t0 = time.time()  # warmup boundary
    if cap and i % 2 == 0:
        traj.append(i * cfg.sim.dt * cfg.decimation, pos[0].cpu().numpy(), quat[0].cpu().numpy())
        frame = env.render()
        if frame is not None:
            cap.add_frame(frame)
torch.cuda.synchronize()
if args.bench and t0:
    wall = time.time() - t0
    r = {"task": "VesperQuad+SE3", "num_envs": args.num_envs,
         "steps": steps - 50, "wall_s": round(wall, 2),
         "env_steps_per_s": round(args.num_envs * (steps - 50) / wall)}
    print(f"BENCH {json.dumps(r)}", flush=True)
    if args.out:
        open(args.out, "a").write(json.dumps(r) + "\n")
if cap:
    traj.close()
    print(f"wrote {cap.finish()}", flush=True)
pos, _, _, _ = env.flight_state()
print(f"final mean dist to circuit: {(pos - corners[wp_idx]).norm(dim=1).mean():.2f}m", flush=True)
env.close()
app.close()
