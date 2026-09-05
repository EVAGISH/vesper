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

import omni.usd  # noqa: E402
from pxr import UsdGeom, UsdLux, UsdShade, Sdf, Gf  # noqa: E402

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

# ------------------------------------------------------------ neutralization FX
# When a target's `reached` flag flips 0->1 we play a lightweight, purely
# kinematic destruction: an emissive fireball sphere that swells then fades, a
# dark smoke puff that keeps expanding and lingers, a one-shot light flash, and
# a swap of the intact tank for a charred wreck box. All keyed to `age` = output
# frames since impact (the mp4 plays at args.fps, so age/fps is wall-clock secs
# regardless of --stride). No particle sim -- just per-frame scale/emissive/
# opacity edits, which never fault the RTX crash path the way physics does.
stage = omni.usd.get_context().get_stage()
FPS = float(max(args.fps, 1))


def _clamp(v, lo, hi):
    return max(lo, min(hi, v))


def _material(path, diffuse, emissive=(0.0, 0.0, 0.0), opacity=1.0):
    mat = UsdShade.Material.Define(stage, path)
    sh = UsdShade.Shader.Define(stage, path + "/S")
    sh.CreateIdAttr("UsdPreviewSurface")
    sh.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*diffuse))
    sh.CreateInput("emissiveColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*emissive))
    sh.CreateInput("opacity", Sdf.ValueTypeNames.Float).Set(float(opacity))
    sh.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.9)
    sh.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(0.0)
    mat.CreateSurfaceOutput().ConnectToSource(sh.ConnectableAPI(), "surface")
    return mat, sh


def _sphere(path, mat):
    UsdGeom.Xform.Define(stage, path)
    geo = UsdGeom.Sphere.Define(stage, path + "/geo")
    geo.CreateRadiusAttr(1.0)
    UsdShade.MaterialBindingAPI(geo.GetPrim()).Bind(mat)
    return XPrim(path)


def _box(path, mat, half):
    UsdGeom.Xform.Define(stage, path)
    geo = UsdGeom.Cube.Define(stage, path + "/geo")
    geo.CreateSizeAttr(1.0)  # unit cube spans -0.5..0.5; XPrim scale sets full dims
    UsdShade.MaterialBindingAPI(geo.GetPrim()).Bind(mat)
    xp = XPrim(path)
    xp.set_local_scale(np.array(half))
    return xp


def _set_visible(path, vis):
    prim = stage.GetPrimAtPath(path)
    if prim and prim.IsValid():
        UsdGeom.Imageable(prim).GetVisibilityAttr().Set(
            UsdGeom.Tokens.inherited if vis else UsdGeom.Tokens.invisible)


def _set_emissive(shader, color):
    shader.GetInput("emissiveColor").Set(Gf.Vec3f(*color))


def _set_opacity(shader, o):
    shader.GetInput("opacity").Set(float(o))


def _set_diffuse(shader, color):
    shader.GetInput("diffuseColor").Set(Gf.Vec3f(*color))


fx = []
for i in range(n_targets):
    fire_mat, fire_sh = _material(f"/World/Fx_{i}/fire_mat", (1.0, 0.45, 0.08),
                                  emissive=(9.0, 3.2, 0.5))
    smoke_mat, smoke_sh = _material(f"/World/Fx_{i}/smoke_mat", (0.05, 0.05, 0.05),
                                    opacity=0.85)
    fire = _sphere(f"/World/Fx_{i}/fire", fire_mat)
    smoke = _sphere(f"/World/Fx_{i}/smoke", smoke_mat)
    flash = UsdLux.SphereLight.Define(stage, f"/World/Fx_{i}/flash")
    flash.CreateRadiusAttr(0.4)
    flash.CreateColorAttr(Gf.Vec3f(1.0, 0.6, 0.25))
    flash.CreateIntensityAttr(0.0)
    flash_xp = XPrim(f"/World/Fx_{i}/flash")
    for p in (f"/World/Fx_{i}/fire", f"/World/Fx_{i}/smoke"):
        _set_visible(p, False)
    fx.append(dict(fire=fire, fire_sh=fire_sh, smoke=smoke, smoke_sh=smoke_sh,
                   flash=flash, flash_xp=flash_xp))

impact_fi = [None] * n_targets      # output-frame index at which each target died
impact_pos = [None] * n_targets     # tank position captured at impact
strike_done = False                 # drone 0 (kamikaze) hidden after first kill
post_strike = False                 # after impact: ease the fpv up off the blast
strike_cam = None                   # fpv position at the instant of impact
PULLBACK_H = 13.0                   # fpv settles this high, framing the wreck


def play_fx(i, fi):
    """Drive target i's destruction prims for the current output frame."""
    e = fx[i]
    age = fi - impact_fi[i]
    cx, cy, cz = impact_pos[i]

    # fireball: swells 0.4 -> ~3 m over ~0.3 s, emissive fades out by ~1 s
    if age < 26:
        r = 0.4 + 2.6 * _clamp(age / 6.0, 0.0, 1.0) + 0.4 * _clamp((age - 6) / 12.0, 0.0, 1.0)
        bright = 1.0 if age < 6 else _clamp(1.0 - (age - 6) / 16.0, 0.0, 1.0)
        e["fire"].set_world_pose(np.array([cx, cy, cz + 0.8 + 0.04 * age]),
                                 np.array([1.0, 0.0, 0.0, 0.0]))
        e["fire"].set_local_scale(np.array([r, r, r]))
        # cools orange -> deep red as it fades
        _set_emissive(e["fire_sh"], (9.0 * bright, 3.2 * bright * bright, 0.5 * bright ** 3))
        _set_opacity(e["fire_sh"], _clamp(0.4 + 0.6 * bright, 0.0, 1.0))
        _set_visible(f"/World/Fx_{i}/fire", True)
    else:
        _set_visible(f"/World/Fx_{i}/fire", False)

    # light flash: one hard pulse over the first ~0.2 s
    inten = 8.0e5 * _clamp(1.0 - age / 5.0, 0.0, 1.0) if age <= 5 else 0.0
    e["flash_xp"].set_world_pose(np.array([cx, cy, cz + 1.5]),
                                 np.array([1.0, 0.0, 0.0, 0.0]))
    e["flash"].GetIntensityAttr().Set(float(inten))

    # smoke: starts ~0.1 s in, expands 1 -> 6 m, rises, fades but lingers
    if age >= 3:
        a = age - 3
        rs = 1.0 + 5.0 * _clamp(a / 36.0, 0.0, 1.0)
        zs = cz + 1.0 + 3.0 * _clamp(a / 36.0, 0.0, 1.0)
        op = _clamp(0.85 * (1.0 - a / 70.0), 0.12, 0.85)
        g = _clamp(0.05 + 0.06 * (a / 70.0), 0.05, 0.14)  # greys out as it thins
        e["smoke"].set_world_pose(np.array([cx, cy, zs]), np.array([1.0, 0.0, 0.0, 0.0]))
        e["smoke"].set_local_scale(np.array([rs, rs, rs]))
        _set_diffuse(e["smoke_sh"], (g, g, g))
        _set_opacity(e["smoke_sh"], op)
        _set_visible(f"/World/Fx_{i}/smoke", True)


# ----------------------------------------------------------- kamikaze dive
# The logged drone 0 is a high-altitude searcher: at the `reached` flip it sits
# ~20 m up and several metres out, and never actually descends onto the tank --
# so the raw flip detonates in mid-air and reads as a fly-past, not a strike.
# Override drone 0's final DIVE_FRAMES to plunge from its logged path straight
# down onto the struck tank, holding the fpv locked on the target as it drops,
# and fire the explosion at contact. MESH_H = where the drone body ends up (on
# the tank); CAM_H = where the fpv freezes (safely above the fireball).
DIVE_FRAMES = 22
MESH_H, CAM_H = 1.4, 6.0
strike_fi = strike_ti = None
for _fi, _fr in enumerate(frames):
    _hit = next((i for i in range(n_targets)
                 if len(_fr["tg"][i]) > 3 and _fr["tg"][i][3] >= 0.5), None)
    if _hit is not None:
        strike_fi, strike_ti = _fi, _hit
        break
strike_contact = None
if strike_fi is not None:
    _tx, _ty = frames[strike_fi]["tg"][strike_ti][0], frames[strike_fi]["tg"][strike_ti][1]
    _tz = float(wmap.ground_at(torch.tensor([_tx]), torch.tensor([_ty]))[0]) + TANK_CLEARANCE
    strike_contact = np.array([_tx, _ty, _tz])
    dive_start_fi = max(0, strike_fi - DIVE_FRAMES)
    dive_p0 = np.array(frames[dive_start_fi]["d"][0], dtype=float)
    print(f"[replay] kamikaze dive: drone 0 -> target {strike_ti}, "
          f"frames {dive_start_fi}..{strike_fi} (impact t={frames[strike_fi]['t']:.1f}s)",
          flush=True)


def dive_alpha(fi):
    """Ease-in progress [0,1] through the dive; accelerates into the target."""
    return (_clamp((fi - dive_start_fi) / max(strike_fi - dive_start_fi, 1), 0.0, 1.0)) ** 1.6


for _ in range(args.settle):
    world.render()

for fi, fr in enumerate(frames):
    dpos = np.array(fr["d"], dtype=float)
    hdg = fr["hdg"]
    diving = (strike_fi is not None and not strike_done
              and dive_start_fi <= fi <= strike_fi)
    if diving:
        # drone 0 body plunges from its logged path onto the tank
        dpos[0] = (1 - dive_alpha(fi)) * dive_p0 + dive_alpha(fi) * (
            strike_contact + np.array([0.0, 0.0, MESH_H]))
    for i, prim in enumerate(drones):
        prim.set_world_pose(dpos[i], yaw_quat(hdg[i]))
    for i, prim in enumerate(tanks):
        x, y = fr["tg"][i][0], fr["tg"][i][1]
        if abs(x - txy[i, 0]) > 0.05 or abs(y - txy[i, 1]) > 0.05:
            tank_yaw[i] = math.atan2(y - txy[i, 1], x - txy[i, 0])
            txy[i] = (x, y)
            tank_z[i] = float(wmap.ground_at(torch.tensor([x]), torch.tensor([y]))[0]) + TANK_CLEARANCE
        prim.set_world_pose(np.array([x, y, tank_z[i]]), yaw_quat(tank_yaw[i]))

    # detect the neutralization: tg = [x, y, found, reached]; reached 0->1
    for i in range(n_targets):
        reached = len(fr["tg"][i]) > 3 and fr["tg"][i][3] >= 0.5
        if reached and impact_fi[i] is None:
            impact_fi[i] = fi
            impact_pos[i] = np.array([txy[i, 0], txy[i, 1], tank_z[i]])
            _set_visible(f"/World/Tank_{i}", False)      # tank destroyed; smoke covers it
            if not strike_done:                          # drone 0 detonated on it
                _set_visible("/World/Drone_0", False)
                strike_done = True
            print(f"[replay] neutralization: target {i} at frame {fi} t={fr['t']:.1f}s",
                  flush=True)
    for i in range(n_targets):
        if impact_fi[i] is not None:
            play_fx(i, fi)

    d0 = dpos[0]
    if want_fpv:
        # drone 0's own camera: body-fixed, pitched forward-down, the task's lens.
        # During the dive the camera plunges toward the tank locked on the target;
        # at contact it freezes just above the blast so the fireball, smoke and
        # wreck linger instead of the log's continued drone-0 path swinging away.
        if post_strike:
            # ease up off the fireball to frame the burning wreck on the road
            b = _clamp((fi - strike_fi) / 14.0, 0.0, 1.0)
            b = b * b * (3.0 - 2.0 * b)  # smoothstep
            cam_pos = (1 - b) * strike_cam + b * (strike_contact + np.array([0.0, 0.0, PULLBACK_H]))
            look_at(fpv, cam_pos, strike_contact + np.array([0.0, 0.0, 0.6]))
        elif diving:
            a = dive_alpha(fi)
            cam_pos = (1 - a) * dive_p0 + a * (strike_contact + np.array([0.0, 0.0, CAM_H]))
            look_at(fpv, cam_pos, strike_contact + np.array([0.0, 0.0, 0.6]))
            if fi == strike_fi:
                post_strike = True
                strike_cam = cam_pos
        else:
            p = torch.tensor(d0, dtype=torch.float32).unsqueeze(0)
            q = torch.tensor(yaw_quat(hdg[0]), dtype=torch.float32).unsqueeze(0)
            cp, cq = sensor_pose(p, q, CAM_PITCH_DEG, CAM_OFFSET)
            pose = (cp[0].numpy(), cq[0].numpy())
            fpv.set_world_pose(*pose)
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
