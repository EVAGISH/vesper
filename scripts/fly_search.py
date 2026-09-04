"""Fly the trained search policy on the Cornell world and film it.

    /isaac-sim/python.sh scripts/fly_search.py --policy runs/<id>/search.pt \
        --seconds 90 --headless --enable_cameras

Writes into runs/<id>/:
  chase.mp4     a camera behind and above the drone, tilted down the way the
                sensor looks, with a HUD showing what the policy currently knows
  overview.mp4  a high camera holding the whole search box, so the sweep pattern
                and the moment each vehicle is found are visible
  track.png     top-down plot over the site's own ground texture: drone path,
                vehicle paths, where each vehicle was first seen and reached
  events.json   the timeline (first sighting and reach per vehicle)

Only environment 0 is filmed; the rest run alongside it for context.
"""
import argparse
import json

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
parser.add_argument("--tag", default="search")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.enable_cameras = True
app = AppLauncher(args).app

import numpy as np  # noqa: E402
import torch  # noqa: E402
from PIL import Image, ImageDraw  # noqa: E402

from isaacsim.core.utils.extensions import enable_extension  # noqa: E402
enable_extension("isaacsim.sensors.camera")
from isaacsim.sensors.camera import Camera  # noqa: E402
import isaacsim.core.utils.numpy.rotations as rot_utils  # noqa: E402

from vesper.capture import RunCapture  # noqa: E402
from vesper.lab.ppo import ActorCritic, RunningNorm  # noqa: E402
from vesper.lab.search_env import SearchEnv, SearchEnvCfg  # noqa: E402

cfg = SearchEnvCfg()
cfg.scene.num_envs = args.num_envs
cfg.scene.env_spacing = 0.0
cfg.n_targets = args.targets
cfg.episode_length_s = args.seconds
cfg.search = {"arena_half": args.arena}
cfg.vehicle_model = args.vehicle
env = SearchEnv(cfg, render_mode="rgb_array", seed=args.seed)

ck = torch.load(args.policy, map_location=env.device)
ac = ActorCritic(ck["obs_dim"], ck["act_dim"]).to(env.device)
ac.load_state_dict(ck["ac"]); ac.eval()
norm = RunningNorm(ck["obs_dim"]).to(env.device)
norm.load_state_dict(ck["norm"])


@torch.no_grad()
def policy(obs):
    n = norm(obs)
    return ac.dist(n).sample() if args.stochastic else ac.actor(n)


def make_cam(path, res=(1280, 720)):
    c = Camera(prim_path=path, position=np.array([0.0, 0.0, 200.0]), resolution=res)
    return c


chase = make_cam("/World/chase_cam")
over = make_cam("/World/over_cam")
obs = env.ppo_reset()
for c in (chase, over):
    c.initialize()
    ap = c.get_horizontal_aperture()
    c.set_focal_length(ap / (2.0 * np.tan(np.radians(args.hfov) / 2.0)))
    c.set_clipping_range(0.05, 6000.0)

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


def look_at(cam, pos, target, up_bias=0.0):
    d = target - pos
    yaw = np.degrees(np.arctan2(d[1], d[0]))
    pitch = np.degrees(np.arctan2(-d[2], np.linalg.norm(d[:2]) + 1e-6)) + up_bias
    cam.set_world_pose(pos, rot_utils.euler_angles_to_quats(
        np.array([0.0, pitch, yaw]), degrees=True))


def hud(rgb, t, known, reached, agl, visible):
    """Burn the policy's own belief into the frame -- otherwise a viewer cannot
    tell a lucky pass over a forklift from an actual detection."""
    img = Image.fromarray(rgb)
    d = ImageDraw.Draw(img)
    d.rectangle([0, 0, 330, 96], fill=(0, 0, 0))
    d.text((10, 8), f"t {t:5.1f}s   AGL {agl:4.0f} m", fill=(255, 255, 255))
    d.text((10, 26), f"found   {int(known.sum())}/{len(known)}", fill=(120, 220, 255))
    d.text((10, 44), f"reached {int(reached.sum())}/{len(reached)}", fill=(140, 255, 140))
    for i in range(len(known)):
        col = (255, 90, 90) if not known[i] else ((140, 255, 140) if reached[i] else (255, 220, 90))
        d.rectangle([10 + 26 * i, 66, 30 + 26 * i, 86], fill=col)
    if visible.any():
        d.rectangle([0, 0, img.width - 1, img.height - 1], outline=(255, 220, 90), width=6)
        d.text((img.width - 150, 12), "TARGET SEEN", fill=(255, 220, 90))
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

    if i % args.every == 0:
        # chase: behind the drone along its own velocity, looking down its sensor axis
        v = env._robot.data.root_lin_vel_w[0].cpu().numpy()
        h = v[:2]
        n = np.linalg.norm(h)
        fwd = h / n if n > 1.0 else np.array([1.0, 0.0])
        smooth = fwd if smooth is None else 0.92 * smooth + 0.08 * fwd
        f = smooth / (np.linalg.norm(smooth) + 1e-6)
        cpos = d - np.array([f[0], f[1], 0.0]) * 22.0 + np.array([0.0, 0.0, 11.0])
        look_at(chase, cpos, d + np.array([f[0], f[1], 0.0]) * 45.0 - np.array([0.0, 0.0, 20.0]))
        env.sim.render()
        rgba = chase.get_rgba()
        if rgba is not None and rgba.size:
            agl = float(info["agl"][0].item())
            cap.add_frame(hud(np.asarray(rgba[:, :, :3], dtype=np.uint8), t, known, reached, agl, vis),
                          stream="chase")
        # overview: fixed high camera holding the whole search box
        gz = float(env.world.ground_at(torch.tensor(0.0), torch.tensor(0.0)))
        look_at(over, np.array([-args.arena * 1.15, -args.arena * 1.15, gz + args.arena * 1.5]),
                np.array([0.0, 0.0, gz]))
        rgba = over.get_rgba()
        if rgba is not None and rgba.size:
            cap.add_frame(np.asarray(rgba[:, :, :3], dtype=np.uint8), stream="overview")

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
