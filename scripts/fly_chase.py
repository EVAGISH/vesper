"""Fly a trained chase policy on the site and film it.

    /isaac-sim/python.sh scripts/fly_chase.py --policy runs/<id>/chase.pt \
        --seconds 60 --headless --enable_cameras

Writes into runs/<id>/:
  fpv.mp4      the drone's own view: the body-fixed camera, pitched forward-down,
               with the policy's real input tensor (RGB and depth, 96 px) inset,
               and a cross where the policy's belief head thinks the forklift is.
               What is in this frame is what the policy had to work with.
  chase.mp4    the same moment from behind, framed on whatever it is going for.
  track.png    top-down over the site's ground texture: drone path, forklift
               paths, the launch zone, the safe zones, where each touch happened.
  events.json  the timeline (first sighting, touch, crash).

Handles both checkpoint kinds: a vision policy (recurrent, camera in) and a
privileged teacher (flat MLP on the state vector). Only environment 0 is filmed.
"""
import argparse
import json
import os

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--policy", required=True)
parser.add_argument("--num_envs", type=int, default=4)
parser.add_argument("--seconds", type=float, default=60.0)
parser.add_argument("--targets", type=int, default=6)
parser.add_argument("--arena", type=float, default=300.0)
parser.add_argument("--hfov", type=float, default=75.0, help="chase camera lens")
parser.add_argument("--seed", type=int, default=7)
parser.add_argument("--stochastic", action="store_true")
parser.add_argument("--every", type=int, default=2, help="capture one frame every N control steps")
parser.add_argument("--world", default=None)
parser.add_argument("--map", default=None)
parser.add_argument("--zones", default=None)
parser.add_argument("--vehicle", default=None)
parser.add_argument("--tag", default="chase")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.enable_cameras = True
app = AppLauncher(args).app

import carb  # noqa: E402

_s = carb.settings.get_settings()
_s.set("/rtx-transient/resourcemanager/enableTextureStreaming", False)
_s.set("/rtx/post/aa/op", 2)
_s.set("/rtx/post/dlss/execMode", 0)
_s.set("/rtx/post/motionblur/enabled", False)

import numpy as np  # noqa: E402
import torch  # noqa: E402
from PIL import Image, ImageDraw  # noqa: E402

from isaacsim.core.utils.extensions import enable_extension  # noqa: E402
enable_extension("isaacsim.sensors.camera")
from isaacsim.sensors.camera import Camera  # noqa: E402
import isaacsim.core.utils.numpy.rotations as rot_utils  # noqa: E402

from vesper.capture import RunCapture  # noqa: E402
from vesper.control.se3 import quat_to_rot  # noqa: E402
from vesper.lab.chase_env import ChaseEnv, ChaseEnvCfg  # noqa: E402
from vesper.lab.frames import sensor_pose  # noqa: E402
from vesper.lab.ppo import load_policy  # noqa: E402
from vesper.lab.recurrent_ppo import load_vision_policy  # noqa: E402

ck = torch.load(args.policy, map_location="cpu")
is_vision = ck.get("kind") == "vision"

cfg = ChaseEnvCfg()
# The RTX-render crash on this tree-heavy world faults inside omni.physx.fabric's
# GPU sync. Fabric is a throughput optimization a small render run does not need.
if not os.environ.get("VESPER_KEEP_FABRIC"):
    try:
        cfg.sim.use_fabric = False
    except Exception:
        pass
cfg.scene.num_envs = args.num_envs
cfg.scene.env_spacing = 0.0
cfg.n_targets = args.targets
cfg.episode_length_s = args.seconds
cfg.chase = {"arena_half": args.arena}
cfg.vehicle_model = args.vehicle
cfg.camera = True                       # the film is the policy's own lens either way
cfg.cam_res = ck.get("res", 96)
if args.world:
    cfg.world_usd = args.world
if args.map:
    cfg.world_map = args.map
if args.zones:
    cfg.zones = args.zones
env = ChaseEnv(cfg, render_mode="rgb_array", seed=args.seed)

if is_vision:
    ac, norm, pnorm = load_vision_policy(ck, env.device)
    hidden = ac.initial_state(env.num_envs, env.device)
    print(f"vision policy: {ac.n_params(deployed=True)/1e6:.2f}M on the airframe", flush=True)
else:
    ac, norm = load_policy(ck, env.device)
    if ck["obs_dim"] != env.cfg.state_space:
        raise SystemExit(f"teacher expects {ck['obs_dim']}-wide state, env gives {env.cfg.state_space}")
    print("privileged teacher policy (state vector, no camera)", flush=True)

belief_xyz = np.zeros(3)


@torch.no_grad()
def policy(obs, done):
    global hidden, belief_xyz
    if is_vision:
        mean, _, belief, hidden = ac(obs["pixels"], obs["depth"], norm(obs["policy"]), hidden,
                                     done=done)
        belief_xyz = belief[0].cpu().numpy() * 100.0
        return ac.dist(mean).sample() if args.stochastic else mean
    n = norm(obs["privileged"])
    return ac.dist(n).sample() if args.stochastic else ac.actor(n)


cap = RunCapture(args.tag)
dt = cfg.sim.dt * cfg.decimation
steps = int(args.seconds / dt)


def make_cam(path, hfov, res=(1280, 720)):
    return Camera(prim_path=path, position=np.array([0.0, 0.0, 200.0]), resolution=res), hfov


sensor_fov = 2.0 * env.task.cfg.fov_half_deg
fpv = make_cam("/World/fpv_cam", sensor_fov, res=(900, 900))
chase = make_cam("/World/chase_cam", args.hfov)
for cam, hf in (fpv, chase):
    cam.initialize()
    ap = cam.get_horizontal_aperture()
    cam.set_focal_length(ap / (2.0 * np.tan(np.radians(hf) / 2.0)))
    cam.set_clipping_range(0.05, 6000.0)
print(f"sensor lens: {sensor_fov:.0f} deg full FOV, pitched {env.task.cfg.cam_pitch_deg:.0f} deg down",
      flush=True)

AMBER, GREEN, CYAN, WHITE, RED = ((255, 200, 70), (110, 255, 150), (120, 220, 255),
                                  (255, 255, 255), (230, 90, 90))


def look_at(camhf, pos, target):
    cam, hfov = camhf
    d = target - pos
    yaw = np.degrees(np.arctan2(d[1], d[0]))
    pitch = np.degrees(np.arctan2(-d[2], np.linalg.norm(d[:2]) + 1e-6))
    q = rot_utils.euler_angles_to_quats(np.array([0.0, pitch, yaw]), degrees=True)
    cam.set_world_pose(pos, q)
    return pos, quat_to_rot(torch.as_tensor(q, dtype=torch.float32).view(1, 4))[0].numpy(), hfov


def body_camera(camhf):
    """Put the render camera exactly where the policy's camera is."""
    cam, hfov = camhf
    d = env._robot.data
    cp, cq = sensor_pose(d.root_pos_w[:1], d.root_quat_w[:1], env.task.cfg.cam_pitch_deg,
                         tuple(cfg.cam_offset))
    cam.set_world_pose(cp[0].cpu().numpy(), cq[0].cpu().numpy())
    return cp[0].cpu().numpy(), quat_to_rot(cq)[0].cpu().numpy(), hfov


def project(pos, R, hfov, p, w, h):
    """World point -> pixel for a camera at `pos` with rotation `R` (columns:
    forward, left, up -- Isaac's 'world' camera convention)."""
    f, left, up = R[:, 0], R[:, 1], R[:, 2]
    d = p - pos
    z = float(d @ f)
    if z <= 0.5:
        return None
    fx = (w / 2.0) / np.tan(np.radians(hfov) / 2.0)
    return float(w / 2 - (d @ left) / z * fx), float(h / 2 - (d @ up) / z * fx), z, fx


def annotate(rgb, cam, t, agl, vis, prot, tp, drone, sensor=False, inset=None, depth=None):
    img = Image.fromarray(rgb)
    d = ImageDraw.Draw(img)
    w, h = img.size
    pos, R, hfov = cam

    if sensor:
        d.line([w / 2 - 12, h / 2, w / 2 + 12, h / 2], fill=WHITE, width=1)
        d.line([w / 2, h / 2 - 12, w / 2, h / 2 + 12], fill=WHITE, width=1)
        fx = (w / 2.0) / np.tan(np.radians(hfov) / 2.0)
        hy = h / 2 - fx * np.tan(np.radians(env.task.cfg.cam_pitch_deg))
        if 0 < hy < h:
            d.line([w / 2 - 40, hy, w / 2 + 40, hy], fill=(200, 200, 200), width=1)

    # only forklifts the camera can actually see get a box: drawing the rest
    # would show the viewer a hunt the policy is not running
    for k in range(len(tp)):
        if not vis[k]:
            continue
        pr = project(pos, R, hfov, tp[k], w, h)
        if not pr:
            continue
        x, y, z, fx = pr
        col = CYAN if prot[k] else AMBER
        s = float(np.clip(fx * 3.0 / z, 10.0, 200.0))
        d.rectangle([x - s, y - s * 0.6, x + s, y + s * 0.6], outline=col, width=3)
        d.text((x - s, y - s * 0.6 - 14), f"{'PROTECTED' if prot[k] else 'V'}{k}  {z:.0f} m", fill=col)

    # where the policy's own belief head puts the nearest forklift
    if sensor and is_vision and np.linalg.norm(belief_xyz) > 1.0:
        pr = project(pos, R, hfov, drone + belief_xyz, w, h)
        if pr:
            x, y, _, _ = pr
            d.line([x - 10, y, x + 10, y], fill=GREEN, width=2)
            d.line([x, y - 10, x, y + 10], fill=GREEN, width=2)
            d.text((x + 12, y - 6), "belief", fill=GREEN)

    if inset is not None:
        s = 2
        tile = Image.fromarray(inset).resize((inset.shape[1] * s, inset.shape[0] * s), Image.NEAREST)
        img.paste(tile, (w - tile.width - 8, h - tile.height - 8))
        d.rectangle([w - tile.width - 9, h - tile.height - 9, w - 8, h - 8], outline=CYAN, width=2)
        d.text((w - tile.width - 8, h - tile.height - 24), "POLICY RGB", fill=CYAN)
    if depth is not None:
        s = 2
        dm = (np.clip(depth[..., 0], 0, 1) * 255).astype(np.uint8)
        tile = Image.fromarray(np.stack([dm] * 3, -1)).resize((dm.shape[1] * s, dm.shape[0] * s),
                                                              Image.NEAREST)
        img.paste(tile, (w - tile.width - 8, h - 2 * tile.height - 40))
        d.rectangle([w - tile.width - 9, h - 2 * tile.height - 41, w - 8, h - tile.height - 40],
                    outline=CYAN, width=2)
        d.text((w - tile.width - 8, h - 2 * tile.height - 56), "POLICY DEPTH", fill=CYAN)

    d.rectangle([0, 0, 260, 74], fill=(0, 0, 0))
    d.text((10, 8), f"t {t:5.1f} s    AGL {agl:4.0f} m", fill=WHITE)
    d.text((10, 26), f"in frame {int(vis.sum())}/{len(vis)}", fill=CYAN)
    d.text((10, 44), "VISION POLICY" if is_vision else "STATE TEACHER", fill=GREEN)
    if vis.any():
        d.rectangle([0, 0, w - 1, h - 1], outline=AMBER, width=6)
        d.text((w - 150, 12), "CONTACT", fill=AMBER)
    return np.asarray(img)


obs = env.vision_reset()
done = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
drone_path, veh_paths = [], [[] for _ in range(args.targets)]
events, touch_xy = [], []
smooth = None
peak_seen = 0

for i in range(steps):
    act = policy(obs, done)
    obs, rew, done, info = env.vision_step(act)
    t = i * dt
    d0 = env._robot.data.root_pos_w[0].cpu().numpy()
    tp = env.vehicle_pos.cpu().numpy()
    vis = info["visible"][0].cpu().numpy()
    prot = env.protected()[0].cpu().numpy()
    drone_path.append(d0.copy())
    peak_seen = max(peak_seen, int(vis.sum()))
    for k in range(args.targets):
        veh_paths[k].append(tp[k].copy())
    for k in range(args.targets):
        if vis[k] and not any(e["k"] == k and e["kind"] == "seen" for e in events):
            events.append({"t": round(t, 2), "k": int(k), "kind": "seen",
                           "xy": [round(float(tp[k][0]), 1), round(float(tp[k][1]), 1)]})
            print(f"t={t:6.2f}s  forklift {k} in frame", flush=True)
    if bool(info["touch"][0].item()) or bool(info["touch_protected"][0].item()):
        kind = "touch" if bool(info["touch"][0].item()) else "touch_protected"
        events.append({"t": round(t, 2), "kind": kind,
                       "xy": [round(float(d0[0]), 1), round(float(d0[1]), 1)]})
        touch_xy.append(d0.copy())
        print(f"t={t:6.2f}s  {kind.upper()}", flush=True)
    for k in ("crash", "oob", "flip"):
        if bool(info[k][0].item()):
            events.append({"t": round(t, 2), "kind": k})
            print(f"t={t:6.2f}s  {k}", flush=True)
    if bool(done[0].item()):
        print(f"episode 0 ended at t={t:.2f}s", flush=True)

    if i % args.every != 0:
        continue

    # frame the chase on whatever the drone can see, else on its own heading
    live = np.flatnonzero(vis)
    if live.size:
        k = live[int(np.argmin(np.linalg.norm(tp[live, :2] - d0[:2], axis=1)))]
        goal = tp[k]
    else:
        v = env._robot.data.root_lin_vel_w[0].cpu().numpy()
        n = np.linalg.norm(v[:2])
        fwd = v[:2] / n if n > 1.0 else np.array([1.0, 0.0])
        smooth = fwd if smooth is None else 0.90 * smooth + 0.10 * fwd
        f2 = smooth / (np.linalg.norm(smooth) + 1e-6)
        goal = d0 + np.array([f2[0], f2[1], 0.0]) * 70.0 - np.array([0.0, 0.0, 30.0])

    cam = body_camera(fpv)
    env.sim.render()
    rgba = fpv[0].get_rgba()
    agl = float(info["agl"][0].item())
    px, dp = env.pixels(), env.depth()
    inset = px[0].cpu().numpy() if px is not None else None
    dins = dp[0].cpu().numpy() if dp is not None else None
    if rgba is not None and rgba.size:
        cap.add_frame(annotate(np.asarray(rgba[:, :, :3], dtype=np.uint8), cam, t, agl, vis, prot,
                               tp, d0, sensor=True, inset=inset, depth=dins), stream="fpv")

    to = goal - d0
    sep = float(np.linalg.norm(to))
    axis = to[:2] / (np.linalg.norm(to[:2]) + 1e-6)
    back = np.array([axis[0], axis[1], 0.0])
    pull = float(np.clip(8.0 + 0.22 * sep, 10.0, 26.0))
    rise = float(np.clip(3.5 + 0.10 * sep, 4.5, 13.0))
    cam = look_at(chase, d0 - pull * back + np.array([0.0, 0.0, rise]), d0 + 0.45 * to)
    rgba = chase[0].get_rgba()
    if rgba is not None and rgba.size:
        cap.add_frame(annotate(np.asarray(rgba[:, :, :3], dtype=np.uint8), cam, t, agl, vis, prot,
                               tp, d0), stream="chase")

# ---------------------------------------------------------------- track plot
try:
    ground_png = os.path.join(os.path.dirname(cfg.world_usd), "ground.png")
    Image.MAX_IMAGE_PIXELS = None
    W = 1400
    base = (Image.open(ground_png).convert("RGB").resize((W, W), Image.BILINEAR)
            if os.path.exists(ground_png) else Image.new("RGB", (W, W), (30, 34, 30)))
    half = env.world.half_m

    def to_px(x, y):
        return ((x + half) / (2 * half) * W, (half - y) / (2 * half) * W)

    dr = ImageDraw.Draw(base, "RGBA")
    a = args.arena
    dr.rectangle([*to_px(-a, a), *to_px(a, -a)], outline=WHITE, width=2)
    if env.zones_path:
        from vesper.worlds.zones import Zones
        z = Zones.load(env.zones_path)
        if z.launch:
            dr.polygon([to_px(x, y) for x, y in z.launch], fill=(60, 200, 90, 60), outline=GREEN)
        for poly in z.safe:
            dr.polygon([to_px(x, y) for x, y in poly], fill=(120, 220, 255, 60), outline=CYAN)
    cols = [(255, 120, 120), (255, 200, 90), (150, 200, 255), (200, 150, 255), (150, 255, 200),
            (255, 255, 150)]
    for k, p in enumerate(veh_paths):
        pts = [to_px(q[0], q[1]) for q in p]
        if len(pts) > 1:
            dr.line(pts, fill=cols[k % len(cols)], width=3)
        dr.ellipse([pts[-1][0] - 7, pts[-1][1] - 7, pts[-1][0] + 7, pts[-1][1] + 7],
                   outline=cols[k % len(cols)], width=3)
    pts = [to_px(q[0], q[1]) for q in drone_path]
    if len(pts) > 1:
        dr.line(pts, fill=WHITE, width=3)
    for q in touch_xy:
        x, y = to_px(q[0], q[1])
        dr.ellipse([x - 12, y - 12, x + 12, y + 12], outline=GREEN, width=4)
    base.save(cap.dir / "track.png")
except Exception as e:                                          # noqa: BLE001
    print(f"track plot skipped: {e}", flush=True)

(cap.dir / "events.json").write_text(json.dumps(events, indent=1))
cap.note(policy=args.policy, kind="vision" if is_vision else "teacher", targets=args.targets,
         peak_in_frame=peak_seen, touches=len(touch_xy), events=events)
print(f"DONE {json.dumps({'touches': len(touch_xy), 'peak_in_frame': peak_seen, 'run': str(cap.dir)})}",
      flush=True)
cap.finish()
env.close()
app.close()
