"""Render two artefacts that show the site, the airframe and the tank.

  tank.png    a stationary tank photographed from a three-quarter view
  flight.mp4  15 s of basic flight: a straight, descending approach toward the
              tank, filmed from the drone's own lens and from a chase camera

This exists to verify that vehicle geometry actually reaches the renderer. The
tank used to be authored from UsdGeom.Cube/Cylinder implicit primitives, which
PhysX collided against but the RTX path never drew, so the strike smoke suite
passed while every frame was empty. vesper.worlds.vehicle now emits meshes.

    /isaac-sim/python.sh scripts/render_demo.py --headless --enable_cameras
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--seconds", type=float, default=15.0)
parser.add_argument("--fps", type=int, default=25)
parser.add_argument("--seed", type=int, default=23)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.enable_cameras = True
app = AppLauncher(args).app

import carb  # noqa: E402

settings = carb.settings.get_settings()
settings.set("/rtx-transient/resourcemanager/enableTextureStreaming", False)
settings.set("/rtx/post/aa/op", 2)
settings.set("/rtx/post/dlss/execMode", 0)
settings.set("/rtx/post/motionblur/enabled", False)

import numpy as np  # noqa: E402
import torch  # noqa: E402
from PIL import Image, ImageDraw  # noqa: E402

from isaacsim.core.utils.extensions import enable_extension  # noqa: E402

enable_extension("isaacsim.sensors.camera")
from isaacsim.sensors.camera import Camera  # noqa: E402
import isaacsim.core.utils.numpy.rotations as rot_utils  # noqa: E402
from pxr import Usd, UsdGeom  # noqa: E402

from vesper.capture import RunCapture  # noqa: E402
from vesper.lab.chase_env import ChaseEnv, ChaseEnvCfg  # noqa: E402
from vesper.lab.frames import sensor_pose, yaw_from_quat  # noqa: E402

cfg = ChaseEnvCfg()
cfg.scene.num_envs = 1
cfg.scene.env_spacing = 0.0
cfg.n_targets = 1
cfg.episode_length_s = max(args.seconds + 10.0, 30.0)
cfg.camera = False                  # external cameras render; no tiled sensor needed
cfg.vehicle_cycle_s = 10_000.0      # the tank stays where this script puts it
# This script only renders. Tree colliders change physics through foliage, which
# nothing here touches: the approach lane is chosen to stay 9 m clear of every
# trunk. Training still needs a map rebuilt with colliders; this does not.
cfg.require_tree_colliders = False
try:
    cfg.sim.use_fabric = False      # reliable readback for sequential camera capture
except Exception:
    pass
env = ChaseEnv(cfg, render_mode="rgb_array", seed=args.seed)
dt = cfg.sim.dt * cfg.decimation
steps = int(args.seconds / dt)


def make_camera(path, hfov, resolution):
    cam = Camera(prim_path=path, position=np.array([0.0, 0.0, 200.0]), resolution=resolution)
    cam.initialize()
    aperture = cam.get_horizontal_aperture()
    cam.set_focal_length(aperture / (2.0 * np.tan(np.radians(hfov) / 2.0)))
    cam.set_clipping_range(0.05, 6000.0)
    return cam


fpv = make_camera("/World/demo_fpv", 110.0, (640, 640))
chase = make_camera("/World/demo_chase", 60.0, (640, 640))
portrait = make_camera("/World/demo_portrait", 45.0, (1280, 960))


def look_at(cam, pos, target):
    delta = np.asarray(target, dtype=np.float64) - np.asarray(pos, dtype=np.float64)
    yaw = np.degrees(np.arctan2(delta[1], delta[0]))
    pitch = np.degrees(np.arctan2(-delta[2], np.linalg.norm(delta[:2]) + 1e-6))
    cam.set_world_pose(np.asarray(pos, dtype=np.float64),
                       rot_utils.euler_angles_to_quats(np.array([0.0, pitch, yaw]), degrees=True))


def tree_positions(usd_path):
    stage = Usd.Stage.Open(str(usd_path))
    root = stage.GetPrimAtPath("/World/trees")
    cache = UsdGeom.XformCache()
    pts = [tuple(cache.GetLocalToWorldTransform(p).ExtractTranslation())[:2]
           for p in root.GetChildren()]
    return np.asarray(pts, dtype=np.float32) if pts else np.zeros((0, 2), np.float32)


def clear_point(xy):
    p = torch.as_tensor(np.asarray(xy, dtype=np.float32)).view(1, 2)
    return bool(env.world.is_drivable(p[:, 0], p[:, 1])[0]) and not bool(
        env.world.in_safe(p[:, 0], p[:, 1])[0])


def ground(xy):
    p = torch.as_tensor(np.asarray(xy, dtype=np.float32), device=env.device).view(1, 2)
    return float(env.world.ground_at(p[:, 0], p[:, 1])[0])


def find_stand(trees):
    """A central, drivable, unprotected, tree-free spot with 50 m of clear run-in."""
    world = env.world
    mask = (world.drivable > 0.5) & (world.safe < 0.5) & (world.tree_z <= world.ground_z + 0.5)
    cells = torch.nonzero(mask).cpu().numpy()
    centre = np.array([world.n / 2, world.n / 2])
    order = np.argsort(((cells - centre) ** 2).sum(axis=1))
    for row, col in cells[order][:: max(1, len(cells) // 3000)]:
        tank = np.array([col * world.cell - world.half_m, row * world.cell - world.half_m],
                        dtype=np.float32)
        for direction in np.array([[1, 0], [0, 1], [-1, 0], [0, -1]], dtype=np.float32):
            probes = [tank + direction * d for d in np.linspace(-4.0, 50.0, 20)]
            if len(trees) and any(
                    float(np.linalg.norm(trees - p, axis=1).min()) < 9.0 for p in probes):
                continue
            if all(clear_point(p) for p in probes):
                return tank, direction
    raise RuntimeError("no clear stand for the tank found on this site")


trees = tree_positions(cfg.world_usd)
tank_xy, approach = find_stand(trees)
tank_z = ground(tank_xy)
tank_centre = np.array([tank_xy[0], tank_xy[1], tank_z + 1.4], dtype=np.float32)
# The tank faces across the approach lane so the flight sees its flank and front.
tank_yaw = math.atan2(float(approach[1]), float(approach[0])) + math.pi / 2


def place(drone_xy, drone_z, drone_yaw):
    env.vision_reset()
    drone = torch.tensor([[float(drone_xy[0]), float(drone_xy[1]), float(drone_z),
                           math.cos(drone_yaw / 2), 0.0, 0.0, math.sin(drone_yaw / 2)]],
                         device=env.device)
    tank = torch.tensor([[float(tank_xy[0]), float(tank_xy[1]), tank_z + 0.08,
                          math.cos(tank_yaw / 2), 0.0, 0.0, math.sin(tank_yaw / 2)]],
                        device=env.device)
    env._robot.write_root_pose_to_sim(drone)
    env._robot.write_root_velocity_to_sim(torch.zeros(1, 6, device=env.device))
    env._vehicles.write_root_pose_to_sim(tank)
    env._vehicles.write_root_velocity_to_sim(torch.zeros(1, 6, device=env.device))
    env.yaw_des[:] = drone_yaw
    env.driver.heading[:] = tank_yaw
    env.driver.cruise[:] = 0.0          # a stationary tank
    env.driver.speed_cmd[:] = 0.0
    env.driver.goal[:] = torch.as_tensor(tank_xy, device=env.device)
    env.driver.goal_age[:] = 0.0
    env.driver.on_road[:] = False
    env.vision_step(torch.tensor([[0.0, 0.0, 0.0, -1.0]], device=env.device))


def warm(n=12):
    """RTX needs several frames before geometry and textures are resident."""
    for _ in range(n):
        env.sim.render()


start_xy = tank_xy + approach * 46.0
# Cornell slopes, so every altitude in this script is height over the ground
# directly below, not an absolute z. An absolute ramp flies into the hill.
START_AGL, END_AGL = 13.0, 4.2
start_alt = ground(start_xy) + START_AGL
heading = math.atan2(float(-approach[1]), float(-approach[0]))   # pointing at the tank
place(start_xy, start_alt, heading)
warm(24)

out = Path("runs") / "render-demo"
out.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------- tank portrait
side = np.array([-approach[1], approach[0]], dtype=np.float32)
cam_xy = tank_xy + approach * 9.5 + side * 7.5
look_at(portrait, np.array([cam_xy[0], cam_xy[1], tank_z + 5.0]), tank_centre)
warm(16)
Image.fromarray(np.asarray(portrait.get_rgba()[..., :3], dtype=np.uint8)).save(out / "tank.png")
print(f"wrote {out/'tank.png'}", flush=True)

# ---------------------------------------------------------------- 15 s flight
capture = RunCapture("render-demo-flight")
telemetry = []
hold_xy = tank_xy + approach * 9.0
for i in range(steps):
    t = i * dt
    frac = min(1.0, t / max(args.seconds - 3.0, 1e-3))
    goal_xy = start_xy + (hold_xy - start_xy) * frac
    goal_alt = ground(goal_xy) + START_AGL + (END_AGL - START_AGL) * frac

    d = env._robot.data
    pos = d.root_pos_w[0]
    delta = torch.tensor([float(goal_xy[0]), float(goal_xy[1]), float(goal_alt)],
                         device=env.device) - pos
    yaw = yaw_from_quat(d.root_quat_w)[0]
    cy, sy = torch.cos(yaw), torch.sin(yaw)
    axes = torch.stack([(cy * delta[0] + sy * delta[1]) / 9.0,
                        (-sy * delta[0] + cy * delta[1]) / 9.0,
                        delta[2] / 5.0]).clamp(-1.0, 1.0)
    action = torch.cat([axes, torch.tensor([-1.0], device=env.device)]).view(1, 4)

    drone = d.root_pos_w[0].cpu().numpy()
    tank_now = env.target_pos[0, 0].cpu().numpy()
    rng = float(np.linalg.norm(tank_now - drone))
    cp, cq = sensor_pose(d.root_pos_w[:1], d.root_quat_w[:1],
                         env.task.cfg.cam_pitch_deg, tuple(cfg.cam_offset))
    fpv.set_world_pose(cp[0].cpu().numpy(), cq[0].cpu().numpy())
    back = drone[:2] - tank_now[:2]
    back = back / (np.linalg.norm(back) + 1e-6)
    look_at(chase, np.array([drone[0] + back[0] * 11.0, drone[1] + back[1] * 11.0, drone[2] + 4.5]),
            0.5 * (drone + tank_now))
    env.sim.render()

    left = Image.fromarray(np.asarray(fpv.get_rgba()[..., :3], dtype=np.uint8))
    right = Image.fromarray(np.asarray(chase.get_rgba()[..., :3], dtype=np.uint8))
    frame = Image.new("RGB", (1280, 640))
    frame.paste(left, (0, 0)); frame.paste(right, (640, 0))
    draw = ImageDraw.Draw(frame, "RGBA")
    draw.rectangle((0, 0, 300, 60), fill=(0, 0, 0, 190))
    draw.text((10, 8), f"APPROACH  t={t:05.2f}s", fill=(255, 255, 255))
    draw.text((10, 26), f"agl {drone[2] - ground(drone[:2]):5.1f} m   tank range {rng:5.1f} m",
              fill=(125, 220, 255))
    draw.rectangle((640, 0, 780, 22), fill=(0, 0, 0, 170))
    draw.text((648, 5), "CHASE CAMERA", fill=(255, 255, 255))
    draw.text((8, 620), "DRONE LENS (110 deg)", fill=(255, 255, 255))
    capture.add_frame(np.asarray(frame), "flight")

    telemetry.append({"t": round(t, 3), "drone": drone.round(3).tolist(),
                      "tank": tank_now.round(3).tolist(), "range_m": round(rng, 3)})
    _, _, done, info = env.vision_step(action)
    if bool(done[0]):
        print(f"episode ended early at t={t:.2f}s: "
              + json.dumps({k: bool(info[k][0]) for k in ("crash", "oob", "flip")}), flush=True)
        break

capture.note(kind="render_demo", seconds=args.seconds,
             tank_xy=[round(float(v), 2) for v in tank_xy],
             frames=len(telemetry))
(capture.dir / "telemetry.jsonl").write_text(
    "".join(json.dumps(r) + "\n" for r in telemetry))
run_dir = capture.dir
capture.finish(fps=args.fps)
print("RENDER_DEMO " + json.dumps({"portrait": str(out / "tank.png"),
                                   "flight_run": str(run_dir),
                                   "frames": len(telemetry)}), flush=True)
env.close()
app.close()
