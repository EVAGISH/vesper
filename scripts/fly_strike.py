"""Fly the trained strike policy and film it: a drone dives onto a moving tank.

    /isaac-sim/python.sh scripts/fly_strike.py --policy runs/<id>/strike.pt \
        --num_envs 12 --seconds 24 --headless --enable_cameras

One cinematic chase on env 0 (camera sits behind the drone along the drone->target
axis, so both stay framed through the terminal dive). Writes runs/<id>/*.mp4.
"""
import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--policy", required=True)
parser.add_argument("--num_envs", type=int, default=12)
parser.add_argument("--seconds", type=float, default=24.0)
parser.add_argument("--target_speed", type=float, default=4.0)
parser.add_argument("--stochastic", action="store_true")
parser.add_argument("--tag", default="strike")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.enable_cameras = True
app = AppLauncher(args).app

import numpy as np  # noqa: E402
import torch  # noqa: E402

from isaacsim.core.utils.viewports import set_camera_view  # noqa: E402

from vesper.capture import RunCapture  # noqa: E402
from vesper.lab.ppo import ActorCritic, RunningNorm  # noqa: E402
from vesper.lab.strike_env import StrikeEnv, StrikeEnvCfg  # noqa: E402

cfg = StrikeEnvCfg()
cfg.scene.num_envs = args.num_envs
cfg.scene.env_spacing = 12.0
cfg.spawn_targets = True
cfg.strike = {"target_speed": args.target_speed}
env = StrikeEnv(cfg, render_mode="rgb_array", seed=1)

ck = torch.load(args.policy, map_location=env.device)
ac = ActorCritic(ck["obs_dim"], ck["act_dim"]).to(env.device)
ac.load_state_dict(ck["ac"]); ac.eval()
norm = RunningNorm(ck["obs_dim"]).to(env.device)
norm.load_state_dict(ck["norm"])


@torch.no_grad()
def policy(obs):
    n = norm(obs)
    return ac.dist(n).sample() if args.stochastic else ac.actor(n)


obs = env.ppo_reset()
cap = RunCapture(args.tag)
dt = cfg.sim.dt * cfg.decimation
steps = int(args.seconds / dt)
o0 = env.scene.env_origins[0]
hits = 0
for i in range(steps):
    act = policy(obs)
    obs, rew, done, info = env.ppo_step(act)
    hits += int(info["hit"][0].item()) if "hit" in info else 0
    if i % 2 == 0:
        env.update_target_visuals()
        pos, _, _, _ = env.flight_state()
        d = (pos[0] + o0).cpu().numpy()
        t = (env.target_pos[0] + o0).cpu().numpy()
        to_t = t - d
        n = np.linalg.norm(to_t) + 1e-6
        back = to_t / n
        eye = d - back * 9.0 + np.array([0.0, 0.0, 4.0])
        look = d + back * (0.5 * n)
        set_camera_view(tuple(eye.tolist()), tuple(look.tolist()))
        frame = env.render()
        if frame is not None:
            cap.add_frame(frame)

path = cap.finish()
print(f"strikes on env0: {hits}", flush=True)
print(f"wrote {path}", flush=True)
env.close()
app.close()
