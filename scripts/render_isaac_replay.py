"""Photoreal after-action replay: replay.json -> isaac.mp4 (+ isaac_fpv.mp4).

Reads the after-action log a native run carries (vesper.native.replay) and
re-films it in the real Isaac RTX world -- the same USD site, tank models and
camera lenses the Isaac lane trains with -- purely kinematically: every drone
and tank prim is placed exactly where the log says it was, frame by frame.
No physics is stepped, so the render can never disagree with the trajectory.

Runs inside the Isaac container on the droplet:

    /isaac-sim/python.sh scripts/render_isaac_replay.py runs/<id> [--world cornell]
        [--fps 24] [--stride 2] [--cam both|chase|fpv] [--max_frames N]

Writes isaac.mp4 (chase of drone 0, the same trailing shot warm_session
publishes as /overview.mjpeg) and isaac_fpv.mp4 (drone 0's own nadir-cone
lens via sensor_pose) into the run dir, where the Runs tab picks up any
*.mp4 automatically.
"""
import argparse
import json
import math
import os
import shutil
import subprocess
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("run_dir", help="run directory containing replay.json")
parser.add_argument("--world", default=None, help="override the world named in replay.json")
parser.add_argument("--fps", type=int, default=24, help="playback framerate of the mp4")
parser.add_argument("--stride", type=int, default=2, help="render 1 of every N logged frames")
parser.add_argument("--cam", choices=["fpv", "chase", "both"], default="both")
parser.add_argument("--hfov", type=float, default=75.0, help="chase camera horizontal FOV (deg)")
parser.add_argument("--spf", type=int, default=2,
                    help="RTX render passes per output frame (temporal convergence)")
parser.add_argument("--settle", type=int, default=30, help="warmup renders before the first frame")
parser.add_argument("--max_frames", type=int, default=0, help="stop after N output frames (0 = all)")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.headless = True
args.enable_cameras = True
app = AppLauncher(args).app

import carb  # noqa: E402
# Same RTX workarounds as warm_session/fly_search on the 16k-tree worlds:
# texture streaming off, and no fabric (the CUDA illegal-access crash faults
# inside omni.physx.fabric's GPU sync). We never step physics here, but the
# stage still owns a physics scene, so keep both belts on.
carb.settings.get_settings().set("/rtx-transient/resourcemanager/enableTextureStreaming", False)

import numpy as np  # noqa: E402
import torch  # noqa: E402
import isaacsim.core.utils.numpy.rotations as rot_utils  # noqa: E402
from isaacsim.core.api import World  # noqa: E402
from isaacsim.core.utils.stage import add_reference_to_stage  # noqa: E402
from isaacsim.core.utils.extensions import enable_extension  # noqa: E402
from PIL import Image  # noqa: E402

enable_extension("isaacsim.sensors.camera")
from isaacsim.sensors.camera import Camera  # noqa: E402

from vesper.lab.frames import sensor_pose  # noqa: E402
from vesper.worlds.heightmap import WorldMap  # noqa: E402
from vesper.worlds.vehicle import write_tank_usd  # noqa: E402

try:  # single-prim wrapper was renamed across isaacsim releases
    from isaacsim.core.prims import SingleXFormPrim as XPrim
except ImportError:
    from isaacsim.core.prims import XFormPrim as XPrim

REPO = Path(__file__).resolve().parents[1]
IRIS_USD = os.environ.get(
    "VESPER_IRIS_USD",
    "/pegasus/extensions/pegasus.simulator/pegasus/simulator/assets/Robots/Iris/iris.usd")
# the trained task's lens: SearchCfg cam_pitch_deg / fov_half_deg, SearchEnvCfg cam_offset
CAM_PITCH_DEG, FOV_HALF_DEG = 40.0, 55.0
CAM_OFFSET = (0.12, 0.0, -0.04)
TANK_CLEARANCE = 0.08

run_dir = Path(args.run_dir)
replay = json.loads((run_dir / "replay.json").read_text())
world_name = args.world or replay["world"]
world_usd = REPO / "assets" / world_name / f"{world_name}.usd"
world_map = REPO / "assets" / world_name / f"{world_name}_map.npz"
frames = replay["frames"][:: max(args.stride, 1)]
if args.max_frames:
    frames = frames[: args.max_frames]
n_drones = len(frames[0]["d"])
n_targets = int(replay.get("targets", len(frames[0]["tg"])))
print(f"[replay] {run_dir.name}: world {world_name}, {len(frames)} frames "
      f"(stride {args.stride}), {n_drones} drones, {n_targets} targets", flush=True)

# ---------------------------------------------------------------- scene
try:
    world = World(stage_units_in_meters=1.0, sim_params={"use_fabric": False})
except TypeError:
    world = World(stage_units_in_meters=1.0)
add_reference_to_stage(str(world_usd), "/World/ground")

tank_usd = REPO / "assets" / "vehicles" / "tank.usd"
if not tank_usd.exists():
    write_tank_usd(tank_usd)
tanks = []
for i in range(n_targets):
    add_reference_to_stage(str(tank_usd), f"/World/Tank_{i}")
    tanks.append(XPrim(f"/World/Tank_{i}"))
drones = []
for i in range(n_drones):
    add_reference_to_stage(IRIS_USD, f"/World/Drone_{i}")
    drones.append(XPrim(f"/World/Drone_{i}"))

want_fpv = args.cam in ("fpv", "both")
want_chase = args.cam in ("chase", "both")
fpv = Camera(prim_path="/World/fpv_cam", position=np.array([0.0, 0.0, 200.0]),
             resolution=(900, 900)) if want_fpv else None
chase = Camera(prim_path="/World/chase_cam", position=np.array([0.0, 0.0, 200.0]),
               resolution=(1280, 720)) if want_chase else None

world.reset()
for cam, hf in ((fpv, 2.0 * FOV_HALF_DEG), (chase, args.hfov)):
    if cam is None:
        continue
    cam.initialize()
    ap = cam.get_horizontal_aperture()
    cam.set_focal_length(ap / (2.0 * np.tan(np.radians(hf) / 2.0)))
    cam.set_clipping_range(0.05, 6000.0)

wmap = WorldMap(str(world_map), device="cpu")


def yaw_quat(yaw):
    return np.array([math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2)])


def look_at(cam, pos, target):
    d = target - pos
    yaw = np.degrees(np.arctan2(d[1], d[0]))
    pitch = np.degrees(np.arctan2(-d[2], np.linalg.norm(d[:2]) + 1e-6))
    cam.set_world_pose(pos, rot_utils.euler_angles_to_quats(
        np.array([0.0, pitch, yaw]), degrees=True))


def grab(cam):
    rgba = cam.get_rgba()
    if rgba is None or not getattr(rgba, "size", 0):
        return None
    return np.asarray(rgba)[..., :3].astype(np.uint8)


# tanks sit on the real terrain; heading comes from their logged motion
txy = np.array([[t[0], t[1]] for t in frames[0]["tg"]])
tank_yaw = np.zeros(n_targets)
tank_z = wmap.ground_at(torch.tensor(txy[:, 0]), torch.tensor(txy[:, 1])).numpy() + TANK_CLEARANCE

streams = {}
if want_chase:
    streams["isaac"] = run_dir / "_frames_isaac"
if want_fpv:
    streams["isaac_fpv"] = run_dir / "_frames_isaac_fpv"
for d in streams.values():
    shutil.rmtree(d, ignore_errors=True)
    d.mkdir(parents=True)

chase_dir = np.array([1.0, 0.0])
prev_d0 = np.array(frames[0]["d"][0], dtype=float)

for _ in range(args.settle):
    world.render()

for fi, fr in enumerate(frames):
    dpos = np.array(fr["d"], dtype=float)
    hdg = fr["hdg"]
    for i, prim in enumerate(drones):
        prim.set_world_pose(dpos[i], yaw_quat(hdg[i]))
    for i, prim in enumerate(tanks):
        x, y = fr["tg"][i][0], fr["tg"][i][1]
        if abs(x - txy[i, 0]) > 0.05 or abs(y - txy[i, 1]) > 0.05:
            tank_yaw[i] = math.atan2(y - txy[i, 1], x - txy[i, 0])
            txy[i] = (x, y)
            tank_z[i] = float(wmap.ground_at(torch.tensor([x]), torch.tensor([y]))[0]) + TANK_CLEARANCE
        prim.set_world_pose(np.array([x, y, tank_z[i]]), yaw_quat(tank_yaw[i]))

    d0 = dpos[0]
    if want_fpv:
        # drone 0's own camera: body-fixed, pitched forward-down, the task's lens
        p = torch.tensor(d0, dtype=torch.float32).unsqueeze(0)
        q = torch.tensor(yaw_quat(hdg[0]), dtype=torch.float32).unsqueeze(0)
        cp, cq = sensor_pose(p, q, CAM_PITCH_DEG, CAM_OFFSET)
        fpv.set_world_pose(cp[0].numpy(), cq[0].numpy())
    if want_chase:
        # trail 10 m behind, 3 m above, along the smoothed direction of travel
        v_xy = d0[:2] - prev_d0[:2]
        speed = np.linalg.norm(v_xy)
        if speed > 1e-3:
            chase_dir = 0.9 * chase_dir + 0.1 * (v_xy / speed)
            chase_dir /= np.linalg.norm(chase_dir) + 1e-6
        look_at(chase, d0 + np.array([-10.0 * chase_dir[0], -10.0 * chase_dir[1], 3.0]), d0)
    prev_d0 = d0

    for _ in range(max(args.spf, 1)):
        world.render()
    if want_chase:
        rgb = grab(chase)
        if rgb is not None:
            Image.fromarray(rgb).save(streams["isaac"] / f"{fi:06d}.png", compress_level=1)
    if want_fpv:
        rgb = grab(fpv)
        if rgb is not None:
            Image.fromarray(rgb).save(streams["isaac_fpv"] / f"{fi:06d}.png", compress_level=1)
    if fi % 50 == 0:
        print(f"[replay] frame {fi}/{len(frames)} t={fr['t']:.1f}s", flush=True)

# ---------------------------------------------------------------- encode
for name, d in streams.items():
    out = run_dir / f"{name}.mp4"
    subprocess.run(
        # -nostdin: same guard as RunCapture -- ffmpeg must not eat the
        # caller's stdin when this script is piped over ssh
        ["ffmpeg", "-y", "-nostdin", "-loglevel", "error", "-framerate", str(args.fps),
         "-i", str(d / "%06d.png"),
         "-c:v", "libx264", "-pix_fmt", "yuv420p", str(out)],
        check=True,
    )
    shutil.rmtree(d, ignore_errors=True)
    print(f"[replay] wrote {out}", flush=True)

app.close()
