"""Fly the trained search policy on the Cornell world and film it.

    /isaac-sim/python.sh scripts/fly_search.py --policy runs/<id>/search.pt \
        --seconds 90 --headless --enable_cameras

Writes into runs/<id>/:
  fpv.mp4       the drone's own sensor view: the body-fixed camera, pitched
                forward-down, with the task's lens (2 x fov_half_deg) -- the
                same pose and intrinsics as the policy's TiledCamera, so the
                frame is what the policy sees, at video resolution. With
                --camera the actual 128 px policy tensor is inset. A box marks
                any vehicle the policy currently holds a fix on.
  chase.mp4     the same moment from behind the drone, framed against whatever it
                is going for -- the nearest vehicle it has a fix on, or its own
                heading while it is still sweeping.
  track.png     top-down plot over the site's own ground texture: drone path,
                vehicle paths, where each vehicle was first seen and reached
  events.json   the timeline (first sighting and reach per vehicle)

Only environment 0 is filmed; the rest run alongside it for context.
"""
import argparse
import json
import os

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--policy", required=True)
parser.add_argument("--num_envs", type=int, default=4)
parser.add_argument("--seconds", type=float, default=90.0)
parser.add_argument("--targets", type=int, default=3)
parser.add_argument("--arena", type=float, default=300.0)
parser.add_argument("--hfov", type=float, default=75.0)
parser.add_argument("--seed", type=int, default=7)
parser.add_argument("--stochastic", action="store_true")
parser.add_argument("--every", type=int, default=2, help="capture one frame every N control steps")
parser.add_argument("--vehicle", default=None)
parser.add_argument("--groups", type=int, default=0, help="vehicle sets shared by groups of envs")
parser.add_argument("--camera", action="store_true",
                    help="render the policy's own tiled camera too (pixel sightings + inset)")
parser.add_argument("--tag", default="search")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.enable_cameras = True
app = AppLauncher(args).app

import carb  # noqa: E402

# Full RTX settings copied from fly_mission.py, which renders this same 16k-tree
# world without the cubric HtoD / CUDA illegal-access crash the Lab render path
# hits. Textures up front (no streaming), DLSS off (execMode 0 -> no DLSS buffer
# allocation), FXAA instead of temporal AA, motion blur off.
_s = carb.settings.get_settings()
_s.set("/rtx-transient/resourcemanager/enableTextureStreaming", False)
_s.set("/rtx/post/aa/op", 2)
_s.set("/rtx/post/dlss/execMode", 0)
_s.set("/rtx/post/motionblur/enabled", False)
_s.set("/rtx/post/motionblur/maxBlurDiameterFraction", 0.0)

import numpy as np  # noqa: E402
import torch  # noqa: E402
from PIL import Image, ImageDraw  # noqa: E402

from isaacsim.core.utils.extensions import enable_extension  # noqa: E402
enable_extension("isaacsim.sensors.camera")
from isaacsim.sensors.camera import Camera  # noqa: E402
import isaacsim.core.utils.numpy.rotations as rot_utils  # noqa: E402

from vesper.capture import RunCapture  # noqa: E402
from vesper.lab.ppo import load_policy  # noqa: E402
from vesper.lab.search_env import SearchEnv, SearchEnvCfg  # noqa: E402
from vesper.lab.search_task import sensor_pose  # noqa: E402
from vesper.control.se3 import quat_to_rot  # noqa: E402

cfg = SearchEnvCfg()
# The RTX-render crash on the 16k-tree world faults inside omni.physx.fabric's
# GPU sync (DirectGpuHelper). Fabric is a throughput optimization we don't need
# for a small render run -- turning it off avoids that path entirely.
if not os.environ.get("VESPER_KEEP_FABRIC"):
    try:
        cfg.sim.use_fabric = False
    except Exception:                                      # older/newer cfg shape
        pass
cfg.scene.num_envs = args.num_envs
cfg.scene.env_spacing = 0.0
cfg.n_targets = args.targets
cfg.episode_length_s = args.seconds
cfg.search = {"arena_half": args.arena}
cfg.vehicle_model = args.vehicle
cfg.n_groups = args.groups
cfg.camera = args.camera
env = SearchEnv(cfg, render_mode="rgb_array", seed=args.seed)

ck = torch.load(args.policy, map_location=env.device)
ac, norm = load_policy(ck, env.device)
if ck["obs_dim"] != env.num_obs:
    raise SystemExit(f"policy expects {ck['obs_dim']}-wide observations, env gives {env.num_obs}")


@torch.no_grad()
def policy(obs):
    n = norm(obs)
    return ac.dist(n).sample() if args.stochastic else ac.actor(n)


def make_cam(path, hfov, res=(1280, 720)):
    return Camera(prim_path=path, position=np.array([0.0, 0.0, 200.0]), resolution=res), hfov


# the sensor lens is not a cinematography choice: it is the task's own cone
sensor_fov = 2.0 * env.task.cfg.fov_half_deg
# square frame so the whole cone is inscribed -- it reads as a sensor scope
fpv = make_cam("/World/fpv_cam", sensor_fov, res=(900, 900))
chase = make_cam("/World/chase_cam", args.hfov)
obs = env.ppo_reset()
for c, hf in (fpv, chase):
    c.initialize()
    ap = c.get_horizontal_aperture()
    c.set_focal_length(ap / (2.0 * np.tan(np.radians(hf) / 2.0)))
    c.set_clipping_range(0.05, 6000.0)
print(f"sensor lens: {sensor_fov:.0f} deg full FOV", flush=True)

cap = RunCapture(args.tag)
dt = cfg.sim.dt * cfg.decimation
steps = int(args.seconds / dt)

drone_path, veh_paths = [], [[] for _ in range(args.targets)]
events = []
smooth = None
# Peak belief, sampled every step. Reading env.task.known at the end reports
# zeros: DirectRLEnv auto-resets the moment the episode ends and the reset
# clears the belief, so the final state is the *next* episode's, not this one's.
peak_found = peak_reached = 0


def look_at(camhf, pos, target):
    """Aim a free camera (roll zero) and return (pos, R, hfov) for projection."""
    cam, hfov = camhf
    d = target - pos
    yaw = np.degrees(np.arctan2(d[1], d[0]))
    pitch = np.degrees(np.arctan2(-d[2], np.linalg.norm(d[:2]) + 1e-6))
    q = rot_utils.euler_angles_to_quats(np.array([0.0, pitch, yaw]), degrees=True)
    cam.set_world_pose(pos, q)
    R = quat_to_rot(torch.as_tensor(q, dtype=torch.float32).view(1, 4))[0].numpy()
    return pos, R, hfov


def body_camera(camhf):
    """Put the render camera exactly where the policy's camera is: on env 0's
    airframe, pitched forward-down, rolling and pitching with the hull."""
    cam, hfov = camhf
    d = env._robot.data
    cp, cq = sensor_pose(d.root_pos_w[:1], d.root_quat_w[:1], env.task.cfg.cam_pitch_deg,
                         tuple(cfg.cam_offset))
    pos, q = cp[0].cpu().numpy(), cq[0].cpu().numpy()
    cam.set_world_pose(pos, q)
    return pos, quat_to_rot(cq)[0].cpu().numpy(), hfov


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


AMBER, GREEN, CYAN, WHITE = (255, 200, 70), (110, 255, 150), (120, 220, 255), (255, 255, 255)


def annotate(rgb, cam, t, agl, known, reached, vis, tp, drone, sensor=False, inset=None):
    """Burn in what the policy knows.  Only vehicles it has actually detected get
    a marker -- drawing the others would show the viewer a hunt the policy is not
    running."""
    img = Image.fromarray(rgb)
    d = ImageDraw.Draw(img)
    w, h = img.size
    pos, R, hfov = cam

    if sensor:
        # the whole frame is the lens now: a crosshair on the camera axis and a
        # horizon tick mark where level would be (cam_pitch_deg above centre)
        d.line([w / 2 - 12, h / 2, w / 2 + 12, h / 2], fill=WHITE, width=1)
        d.line([w / 2, h / 2 - 12, w / 2, h / 2 + 12], fill=WHITE, width=1)
        fx = (w / 2.0) / np.tan(np.radians(hfov) / 2.0)
        hy = h / 2 - fx * np.tan(np.radians(env.task.cfg.cam_pitch_deg))
        if 0 < hy < h:
            d.line([w / 2 - 40, hy, w / 2 + 40, hy], fill=(200, 200, 200), width=1)
    if inset is not None:
        # the policy's actual input tensor, scaled up, in the corner
        s = 2
        tile = Image.fromarray(inset).resize((inset.shape[1] * s, inset.shape[0] * s), Image.NEAREST)
        img.paste(tile, (w - tile.width - 8, h - tile.height - 8))
        d.rectangle([w - tile.width - 9, h - tile.height - 9, w - 8, h - 8], outline=CYAN, width=2)
        d.text((w - tile.width - 8, h - tile.height - 24), "POLICY INPUT", fill=CYAN)

    for k in range(len(known)):
        if not known[k]:
            continue
        pr = project(pos, R, hfov, tp[k], w, h)
        if not pr:
            continue
        x, y, z, fx = pr
        col = GREEN if reached[k] else AMBER
        s = float(np.clip(fx * 5.0 / z, 14.0, 200.0))
        d.rectangle([x - s, y - s * 0.6, x + s, y + s * 0.6], outline=col, width=3)
        label = f"V{k}  REACHED" if reached[k] else f"V{k}  {z:.0f} m"
        d.text((x - s, y - s * 0.6 - 14), label, fill=col)

    d.rectangle([0, 0, 250, 92], fill=(0, 0, 0))
    d.text((10, 8), f"t {t:5.1f} s    AGL {agl:4.0f} m", fill=WHITE)
    d.text((10, 26), f"found   {int(known.sum())}/{len(known)}", fill=CYAN)
    d.text((10, 44), f"reached {int(reached.sum())}/{len(reached)}", fill=GREEN)
    for i in range(len(known)):
        c = (170, 60, 60) if not known[i] else (GREEN if reached[i] else AMBER)
        d.rectangle([10 + 26 * i, 64, 30 + 26 * i, 84], fill=c)
    if vis.any():
        d.rectangle([0, 0, w - 1, h - 1], outline=AMBER, width=6)
        d.text((w - 150, 12), "SENSOR CONTACT", fill=AMBER)
    return np.asarray(img)


for i in range(steps):
    act = policy(obs)
    obs, rew, done, info = env.ppo_step(act)
    t = i * dt
    d = env._robot.data.root_pos_w[0].cpu().numpy()
    tp = env.target_pos[0].cpu().numpy()
    known = env.task.known[0].cpu().numpy().copy()
    reached = env.task.reached[0].cpu().numpy().copy()
    vis = info["visible"][0].cpu().numpy()
    drone_path.append(d.copy())
    peak_found = max(peak_found, int(known.sum()))
    peak_reached = max(peak_reached, int(reached.sum()))
    for k in range(args.targets):
        veh_paths[k].append(tp[k].copy())
    for k in range(args.targets):
        tag = None
        if known[k] and not any(e["k"] == k and e["kind"] == "found" for e in events):
            tag = "found"
        if reached[k] and not any(e["k"] == k and e["kind"] == "reached" for e in events):
            tag = "reached"
        if tag:
            events.append({"t": round(t, 2), "k": int(k), "kind": tag,
                           "xy": [round(float(tp[k][0]), 1), round(float(tp[k][1]), 1)]})
            print(f"t={t:6.2f}s  vehicle {k} {tag}", flush=True)
    if bool(done[0].item()):
        print(f"episode 0 ended at t={t:.2f}s", flush=True)
        break

    if i % args.every != 0:
        continue

    # What is the drone going for right now?  A fix it holds and has not reached,
    # else its own heading.  The camera frames the drone against that, so the
    # shot is the hunt rather than the scenery.
    live = np.flatnonzero(known & ~reached)
    if live.size:
        fix = env.task.fix[0].cpu().numpy()
        k = live[int(np.argmin(np.linalg.norm(fix[live, :2] - d[:2], axis=1)))]
        goal = fix[k]
    else:
        v = env._robot.data.root_lin_vel_w[0].cpu().numpy()
        n = np.linalg.norm(v[:2])
        fwd = v[:2] / n if n > 1.0 else np.array([1.0, 0.0])
        smooth = fwd if smooth is None else 0.90 * smooth + 0.10 * fwd
        f2 = smooth / (np.linalg.norm(smooth) + 1e-6)
        goal = d + np.array([f2[0], f2[1], 0.0]) * 70.0 - np.array([0.0, 0.0, 30.0])

    # the drone's own sensor view: on the airframe, pitched forward-down, the task's lens
    cam = body_camera(fpv)
    env.sim.render()
    rgba = fpv[0].get_rgba()
    agl = float(info["agl"][0].item())
    px = env.pixels()
    inset = px[0].cpu().numpy() if px is not None else None
    if rgba is not None and rgba.size:
        cap.add_frame(annotate(np.asarray(rgba[:, :, :3], dtype=np.uint8), cam,
                               t, agl, known, reached, vis, tp, d, sensor=True, inset=inset),
                      stream="fpv")

    to = goal - d
    sep = float(np.linalg.norm(to))
    axis = to[:2] / (np.linalg.norm(to[:2]) + 1e-6)
    back = np.array([axis[0], axis[1], 0.0])
    pull = float(np.clip(8.0 + 0.22 * sep, 10.0, 26.0))
    rise = float(np.clip(3.5 + 0.10 * sep, 4.5, 13.0))
    cam = look_at(chase, d - pull * back + np.array([0.0, 0.0, rise]), d + 0.45 * to)
    rgba = chase[0].get_rgba()
    if rgba is not None and rgba.size:
        cap.add_frame(annotate(np.asarray(rgba[:, :, :3], dtype=np.uint8), cam,
                               t, agl, known, reached, vis, tp, d), stream="chase")

# ---------------------------------------------------------------- track plot
try:
    import os
    from vesper.lab.search_env import CORNELL_USD
    ground_png = os.path.join(os.path.dirname(CORNELL_USD), "ground.png")
    Image.MAX_IMAGE_PIXELS = None
    W = 1400
    base = (Image.open(ground_png).convert("RGB").resize((W, W), Image.BILINEAR)
            if os.path.exists(ground_png) else Image.new("RGB", (W, W), (30, 34, 30)))
    half = env.world.half_m

    def to_px(x, y):
        return ((x + half) / (2 * half) * W, (half - y) / (2 * half) * W)

    dr = ImageDraw.Draw(base)
    a = args.arena
    dr.rectangle([*to_px(-a, a), *to_px(a, -a)], outline=(255, 255, 255), width=2)
    cols = [(255, 120, 120), (255, 200, 90), (150, 200, 255)]
    for k, p in enumerate(veh_paths):
        pts = [to_px(q[0], q[1]) for q in p]
        if len(pts) > 1:
            dr.line(pts, fill=cols[k % 3], width=3)
        dr.ellipse([pts[-1][0] - 7, pts[-1][1] - 7, pts[-1][0] + 7, pts[-1][1] + 7],
                   outline=cols[k % 3], width=3)
    pts = [to_px(q[0], q[1]) for q in drone_path]
    if len(pts) > 1:
        dr.line(pts, fill=(80, 255, 140), width=3)
    for e in events:
        px, py = to_px(*e["xy"])
        r = 12 if e["kind"] == "found" else 18
        col = (255, 255, 90) if e["kind"] == "found" else (90, 255, 120)
        dr.ellipse([px - r, py - r, px + r, py + r], outline=col, width=4)
        dr.text((px + r + 4, py - 8), f'{e["kind"]} v{e["k"]} @{e["t"]:.0f}s', fill=col)
    base.save(cap.dir / "track.png")
    print(f"wrote {cap.dir/'track.png'}", flush=True)
except Exception as exc:                                   # a plot is never worth the run
    print(f"track plot skipped: {exc}", flush=True)

(cap.dir / "events.json").write_text(json.dumps(
    {"events": events, "targets": args.targets, "seconds": args.seconds,
     "found": peak_found, "reached": peak_reached}, indent=1))
cap.note(found=peak_found, reached=peak_reached, events=events)
path = cap.finish()
print(f"found {peak_found}/{args.targets}, reached {peak_reached}/{args.targets}", flush=True)
print(f"wrote {path}", flush=True)
env.close()
app.close()
