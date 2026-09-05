"""Render deterministic teleop smoke runs for the tank-strike ChaseEnv.

Runs inside the Isaac container and writes four short FPV/overview videos:
tree contact, successful detonation, missed detonation, and moving-tank follow.
The pilot is a scripted operator over the exact four ChaseEnv action channels;
it does not use a policy or privileged observation.

    /isaac-sim/python.sh scripts/teleop_strike_smoke.py --headless --enable_cameras
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--seconds", type=float, default=12.0)
parser.add_argument("--seed", type=int, default=23)
parser.add_argument("--fps", type=int, default=25)
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
from vesper.capture.explosion import explosion_frame  # noqa: E402
from vesper.lab.chase_env import ChaseEnv, ChaseEnvCfg  # noqa: E402
from vesper.lab.frames import sensor_pose, yaw_from_quat  # noqa: E402


cfg = ChaseEnvCfg()
cfg.scene.num_envs = 1
cfg.scene.env_spacing = 0.0
cfg.n_targets = 1
cfg.episode_length_s = max(args.seconds + 8.0, 25.0)
cfg.camera = False                  # external FPV camera is enough for teleop smoke
cfg.vehicle_cycle_s = 10_000.0
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


fpv = make_camera("/World/teleop_fpv", 110.0, (640, 640))
overview = make_camera("/World/teleop_overview", 72.0, (960, 540))


def look_at(cam, pos, target):
    delta = target - pos
    yaw = np.degrees(np.arctan2(delta[1], delta[0]))
    pitch = np.degrees(np.arctan2(-delta[2], np.linalg.norm(delta[:2]) + 1e-6))
    cam.set_world_pose(pos, rot_utils.euler_angles_to_quats(
        np.array([0.0, pitch, yaw]), degrees=True))


def xy_at(world, row, col):
    return np.array([col * world.cell - world.half_m,
                     row * world.cell - world.half_m], dtype=np.float32)


def clear_point(world, xy):
    p = torch.as_tensor(xy).view(1, 2)
    return bool(world.is_drivable(p[:, 0], p[:, 1])[0]) and not bool(
        world.in_safe(p[:, 0], p[:, 1])[0])


def tree_positions(usd_path):
    stage = Usd.Stage.Open(str(usd_path))
    root = stage.GetPrimAtPath("/World/trees")
    cache = UsdGeom.XformCache()
    points = []
    for prim in root.GetChildren():
        t = cache.GetLocalToWorldTransform(prim).ExtractTranslation()
        points.append((float(t[0]), float(t[1])))
    if not points:
        raise RuntimeError(f"no tree transforms found in {usd_path}")
    return np.asarray(points, dtype=np.float32)


def find_tree_lane(world, trees):
    directions = np.array([[1, 0], [0, 1], [-1, 0], [0, -1],
                           [0.707, 0.707], [0.707, -0.707]], dtype=np.float32)
    order = np.argsort((trees * trees).sum(axis=1))
    for tree in trees[order]:
        p = torch.as_tensor(tree).view(1, 2)
        if bool(world.in_safe(p[:, 0], p[:, 1])[0]):
            continue
        for direction in directions:
            # Start close enough that a straight stick input reaches the trunk
            # before controller drift can turn this into a navigation test.
            start = tree - direction * 6.0
            tank = tree + direction * 8.0
            if clear_point(world, start) and clear_point(world, tank):
                return tree, start, tank, direction
    raise RuntimeError("could not find an unprotected tree with a clear approach lane")


def find_clear_lane(world, trees):
    mask = (world.drivable > 0.5) & (world.safe < 0.5) & (world.tree_z <= world.ground_z + 0.5)
    cells = torch.nonzero(mask).cpu().numpy()
    directions = np.array([[1, 0], [0, 1], [-1, 0], [0, -1]], dtype=np.float32)
    centre = np.array([world.n / 2, world.n / 2])
    order = np.argsort(((cells - centre) ** 2).sum(axis=1))
    for row, col in cells[order][::max(1, len(cells) // 2500)]:
        tank = xy_at(world, row, col)
        for direction in directions:
            start = tank - direction * 18.0
            # Clear for the strike approach and for 12 seconds of tank driving.
            checks = [tank + direction * d for d in np.linspace(-18.0, 42.0, 16)]
            tree_clear = all(float(np.linalg.norm(trees - p, axis=1).min()) > 8.0 for p in checks)
            if tree_clear and all(clear_point(world, p) for p in checks):
                return start, tank, direction
    raise RuntimeError("could not find a clear unprotected strike lane")


trees = tree_positions(cfg.world_usd)
tree_xy, tree_start, tree_tank, tree_dir = find_tree_lane(env.world, trees)
strike_start, strike_tank, strike_dir = find_clear_lane(env.world, trees)


def ground(xy):
    p = torch.as_tensor(xy, device=env.device).view(1, 2)
    return float(env.world.ground_at(p[:, 0], p[:, 1])[0])


def teleport(drone_xy, drone_z, tank_xy, heading, moving=False):
    env.vision_reset()
    yaw = math.atan2(float(heading[1]), float(heading[0]))
    drone_pose = torch.tensor([[float(drone_xy[0]), float(drone_xy[1]), float(drone_z),
                                math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2)]], device=env.device)
    tank_yaw = yaw
    tank_pose = torch.tensor([[float(tank_xy[0]), float(tank_xy[1]), ground(tank_xy) + 0.08,
                               math.cos(tank_yaw / 2), 0.0, 0.0, math.sin(tank_yaw / 2)]],
                             device=env.device)
    env._robot.write_root_pose_to_sim(drone_pose)
    env._robot.write_root_velocity_to_sim(torch.zeros(1, 6, device=env.device))
    env._vehicles.write_root_pose_to_sim(tank_pose)
    env._vehicles.write_root_velocity_to_sim(torch.zeros(1, 6, device=env.device))
    env.yaw_des[:] = yaw
    env.driver.heading[:] = tank_yaw
    env.driver.goal[:] = torch.as_tensor(tank_xy, device=env.device) + torch.as_tensor(heading, device=env.device) * 35
    env.driver.goal_age[:] = 0.0
    env.driver.speed_cmd[:] = 0.0
    env.driver.cruise[:] = 3.4 if moving else 0.0
    env.driver.on_road[:] = False
    # One controlled step refreshes Isaac buffers after the direct pose writes.
    action = torch.tensor([[0.0, 0.0, 0.0, -1.0]], device=env.device)
    obs, reward, done, info = env.vision_step(action)
    if bool(done[0]):
        raise RuntimeError("scenario placement immediately terminated")
    return obs


def teleop_to(world_goal, altitude, detonate=False):
    d = env._robot.data
    pos = d.root_pos_w[0]
    goal = torch.tensor([float(world_goal[0]), float(world_goal[1]), float(altitude)], device=env.device)
    delta = goal - pos
    yaw = yaw_from_quat(d.root_quat_w)[0]
    cy, sy = torch.cos(yaw), torch.sin(yaw)
    forward = cy * delta[0] + sy * delta[1]
    left = -sy * delta[0] + cy * delta[1]
    axes = torch.stack([forward / 7.0, left / 7.0, delta[2] / 4.0]).clamp(-1.0, 1.0)
    return torch.cat([axes, torch.tensor([1.0 if detonate else -1.0], device=env.device)]).view(1, 4)


def annotate(frame, scenario, t, action, distance, status="TELEOP"):
    image = Image.fromarray(frame)
    draw = ImageDraw.Draw(image, "RGBA")
    draw.rectangle((0, 0, 390, 78), fill=(0, 0, 0, 190))
    draw.text((10, 8), f"{scenario.upper()}   t={t:05.2f}s", fill=(255, 255, 255))
    draw.text((10, 28),
              f"stick f/l/u {action[0]:+.2f} {action[1]:+.2f} {action[2]:+.2f}  fuse {action[3]:+.0f}",
              fill=(125, 220, 255))
    draw.text((10, 48), f"tank range {distance:5.2f} m   {status}",
              fill=(120, 255, 155) if "HIT" in status else (255, 215, 105))
    return np.asarray(image)


def render_pair(scenario, t, action, status="TELEOP"):
    drone = env._robot.data.root_pos_w[0].cpu().numpy()
    quat = env._robot.data.root_quat_w[:1]
    tank = env.target_pos[0, 0].cpu().numpy()
    distance = float(np.linalg.norm(tank - drone))
    cp, cq = sensor_pose(env._robot.data.root_pos_w[:1], quat,
                         env.task.cfg.cam_pitch_deg, tuple(cfg.cam_offset))
    fpv.set_world_pose(cp[0].cpu().numpy(), cq[0].cpu().numpy())
    horizontal = tank[:2] - drone[:2]
    if np.linalg.norm(horizontal) < 1.0:
        horizontal = np.array([1.0, 0.0])
    horizontal = horizontal / (np.linalg.norm(horizontal) + 1e-6)
    midpoint = 0.45 * drone + 0.55 * tank
    cam_pos = midpoint - np.array([horizontal[0], horizontal[1], 0.0]) * 16.0 + np.array([0, 0, 9.0])
    look_at(overview, cam_pos, midpoint)
    env.sim.render()
    a = action[0].detach().cpu().numpy()
    return (annotate(np.asarray(fpv.get_rgba()[..., :3], dtype=np.uint8), scenario, t, a, distance, status),
            annotate(np.asarray(overview.get_rgba()[..., :3], dtype=np.uint8), scenario, t, a, distance, status),
            drone, tank, distance)


def run_scenario(name, start, tank_xy, heading, mode):
    if mode == "tree":
        altitude = ground(tree_xy) + 2.6
    elif mode == "tank":
        altitude = ground(tank_xy) + 15.0
    else:
        altitude = ground(tank_xy) + 2.7
    teleport(start, altitude, tank_xy, heading, moving=mode == "tank")
    capture = RunCapture(f"teleop-{name}")
    telemetry = []
    outcome = "timeout"
    event_info = {}
    last_fpv = last_overview = None
    tank_initial = env.target_pos[0, 0].cpu().numpy().copy()
    scenario_steps = min(steps, int(4.0 / dt)) if mode == "tank" else steps
    for i in range(scenario_steps):
        t = i * dt
        tank_now = env.target_pos[0, 0].cpu().numpy()
        drone_now = env._robot.data.root_pos_w[0].cpu().numpy()
        distance = float(np.linalg.norm(tank_now - drone_now))
        if mode == "tree":
            # Deliberately hold a straight forward stick into the trunk.  The
            # altitude channel only cancels small settling error.
            up = float(np.clip((altitude - drone_now[2]) / 4.0, -0.35, 0.35))
            action = torch.tensor([[0.55, 0.0, up, -1.0]], device=env.device)
        elif mode == "hit":
            action = teleop_to(tank_now[:2], altitude, detonate=distance <= 3.75)
        elif mode == "miss":
            action = teleop_to(tank_now[:2], altitude, detonate=distance <= 5.25)
        else:
            # The operator holds position above the lane while observing the
            # autonomous tank traverse it.
            action = teleop_to(drone_now[:2], altitude)

        last_fpv, last_overview, drone_now, tank_now, distance = render_pair(name, t, action)
        capture.add_frame(last_fpv, "fpv")
        capture.add_frame(last_overview, "overview")
        telemetry.append({"t": round(t, 3), "drone": drone_now.round(3).tolist(),
                          "tank": tank_now.round(3).tolist(), "range_m": round(distance, 3),
                          "tree_range_m": round(float(np.linalg.norm(drone_now[:2] - tree_xy)), 3),
                          "action": action[0].cpu().numpy().round(3).tolist()})
        _, reward, done, info = env.vision_step(action)

        if bool(info["detonated"][0]):
            hit = bool(info["hit"][0])
            outcome = "hit" if hit else "miss"
            event_info = {"reward": round(float(reward[0]), 3),
                          "range_m": round(distance, 3), "hit": hit,
                          "blast_radius_m": env.task.cfg.blast_radius}
            status = "DETONATION — HIT" if hit else "DETONATION — MISS"
            a = action[0].cpu().numpy()
            hit_fpv = annotate(last_fpv, name, t, a, distance, status)
            hit_overview = annotate(last_overview, name, t, a, distance, status)
            for j in range(18):
                capture.add_frame(explosion_frame(hit_fpv, None, j / 17, 290), "fpv")
                capture.add_frame(explosion_frame(hit_overview, (480, 270), j / 17, 125), "overview")
            break
        if bool(info["crash"][0]):
            outcome = "tree_contact" if mode == "tree" else "crash"
            event_info = {"reward": round(float(reward[0]), 3), "range_m": round(distance, 3)}
            a = action[0].cpu().numpy()
            contact_fpv = annotate(last_fpv, name, t, a, distance, "TREE CONTACT")
            contact_overview = annotate(last_overview, name, t, a, distance, "TREE CONTACT")
            for _ in range(10):
                capture.add_frame(contact_fpv, "fpv")
                capture.add_frame(contact_overview, "overview")
            break
        if bool(done[0]):
            outcome = "terminated"
            event_info = {k: bool(info[k][0]) for k in ("crash", "oob", "flip")}
            break

    if mode == "tank" and outcome == "timeout":
        tank_final = env.target_pos[0, 0].cpu().numpy()
        displacement = float(np.linalg.norm(tank_final[:2] - tank_initial[:2]))
        outcome = "tank_motion" if displacement >= 8.0 else "tank_stalled"
        event_info = {"duration_s": round(scenario_steps * dt, 3),
                      "tank_displacement_m": round(displacement, 3)}

    expected = {"tree": "tree_contact", "hit": "hit", "miss": "miss",
                "tank": "tank_motion"}[mode]
    passed = outcome == expected
    capture.note(scenario=name, control="scripted_teleop", expected=expected,
                 outcome=outcome, passed=passed, event=event_info,
                 blast_radius_m=env.task.cfg.blast_radius)
    (capture.dir / "telemetry.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in telemetry))
    run_dir = str(capture.dir)
    capture.finish(fps=args.fps)
    return {"scenario": name, "run": run_dir, "expected": expected,
            "outcome": outcome, "passed": passed, **event_info}


results = []
exit_code = 1
try:
    results.append(run_scenario("tree", tree_start, tree_tank, tree_dir, "tree"))
    results.append(run_scenario("hit", strike_start, strike_tank, strike_dir, "hit"))
    results.append(run_scenario("miss", strike_start, strike_tank, strike_dir, "miss"))
    results.append(run_scenario("tank-follow", strike_start, strike_tank, strike_dir, "tank"))
    summary = {"pass": all(r["passed"] for r in results), "results": results,
               "placements": {"tree": tree_xy.round(2).tolist(),
                               "strike_tank": strike_tank.round(2).tolist()}}
    summary_path = Path("runs") / "teleop-strike-smoke-summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    print("TELEOP_SMOKE " + json.dumps(summary), flush=True)
    exit_code = 0 if summary["pass"] else 1
finally:
    env.close()
    app.close()

raise SystemExit(exit_code)
