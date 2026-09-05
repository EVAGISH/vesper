"""Fly the trained pursuit policy and film it: a drone runs down a moving ground vehicle.

    /isaac-sim/python.sh scripts/fly_pursuit.py --policy runs/<id>/pursuit.pt \
        --num_envs 8 --seconds 20 --headless --enable_cameras

Films env 0 with a cinematic chase camera that frames both the drone and its
target through the final approach (isaacsim Camera sensor + get_rgba, the same
capture path fly_mission uses). Writes runs/<id>/pursuit.mp4.
"""
import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--policy", required=True)
parser.add_argument("--num_envs", type=int, default=8)
parser.add_argument("--seconds", type=float, default=20.0)
parser.add_argument("--target_speed", type=float, default=4.0)
parser.add_argument("--hfov", type=float, default=70.0)
parser.add_argument("--stochastic", action="store_true")
parser.add_argument("--vehicle", default=None, help="tank | path to a custom USD")
parser.add_argument("--tag", default="pursuit")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.enable_cameras = True
app = AppLauncher(args).app

import numpy as np  # noqa: E402
import torch  # noqa: E402

# Isaac Lab's AppLauncher experience file doesn't load the camera sensor
# extension; enable it before importing Camera (fly_mission gets it free via
# the plain SimulationApp).
from isaacsim.core.utils.extensions import enable_extension  # noqa: E402
enable_extension("isaacsim.sensors.camera")
from isaacsim.sensors.camera import Camera  # noqa: E402
import isaacsim.core.utils.numpy.rotations as rot_utils  # noqa: E402

from vesper.capture import RunCapture  # noqa: E402
from vesper.lab.ppo import load_policy  # noqa: E402
from vesper.lab.pursuit_env import PursuitEnv, PursuitEnvCfg  # noqa: E402

cfg = PursuitEnvCfg()
cfg.scene.num_envs = args.num_envs
cfg.scene.env_spacing = 90.0   # keep the independent envs visually separate
cfg.pursuit = {"target_speed": args.target_speed}
cfg.vehicle_model = args.vehicle
env = PursuitEnv(cfg, render_mode="rgb_array", seed=1)

ck = torch.load(args.policy, map_location=env.device)
ac, norm = load_policy(ck, env.device)


@torch.no_grad()
def policy(obs):
    n = norm(obs)
    return ac.dist(n).sample() if args.stochastic else ac.actor(n)


cam = Camera(prim_path="/World/chase_cam", position=np.array([0.0, 0.0, 20.0]), resolution=(1280, 720))
obs = env.ppo_reset()
cam.initialize()
ap = cam.get_horizontal_aperture()
cam.set_focal_length(ap / (2.0 * np.tan(np.radians(args.hfov) / 2.0)))
cam.set_clipping_range(0.05, 3000.0)

cap = RunCapture(args.tag)
dt = cfg.sim.dt * cfg.decimation
steps = int(args.seconds / dt)
o0 = env.scene.env_origins[0].cpu().numpy()
intercepts = 0


def frame_camera(drone, target):
    sep = float(np.linalg.norm(target - drone)) + 1e-6
    fwd = (target - drone)
    fwd_xy = fwd[:2] / (np.linalg.norm(fwd[:2]) + 1e-6)
    back = np.array([fwd_xy[0], fwd_xy[1], 0.0])
    cam_pos = drone - (7.0 + 0.25 * sep) * back + np.array([0.0, 0.0, 4.0 + 0.12 * sep])
    look = drone + 0.5 * fwd
    d = look - cam_pos
    yaw = np.degrees(np.arctan2(d[1], d[0]))
    pitch = np.degrees(np.arctan2(-d[2], np.linalg.norm(d[:2])))
    cam.set_world_pose(cam_pos, rot_utils.euler_angles_to_quats(np.array([0.0, pitch, yaw]), degrees=True))


for i in range(steps):
    act = policy(obs)
    obs, rew, done, info = env.ppo_step(act)
    intercepts += int(info["intercept"][0].item()) if "intercept" in info else 0
    if i % 2 == 0:
        pos, _, _, _ = env.flight_state()
        drone = pos[0].cpu().numpy() + o0
        target = env.target_pos[0].cpu().numpy() + o0
        frame_camera(drone, target)
        env.sim.render()
        rgba = cam.get_rgba()
        if rgba is not None and rgba.size:
            cap.add_frame(rgba[:, :, :3])

path = cap.finish()
print(f"intercepts on env0: {intercepts}", flush=True)
print(f"wrote {path}", flush=True)
env.close()
app.close()
