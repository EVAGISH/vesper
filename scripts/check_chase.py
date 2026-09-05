"""Smoke-test the chase environment on the GPU before spending hours training in it.

    /isaac-sim/python.sh scripts/check_chase.py --num_envs 16 --camera --headless --enable_cameras

Everything asserted here is something that silently ruins a training run: a
camera that renders nothing (or renders blank), tanks that sink through the
terrain or never move, a contact sensor that never fires, drones launching
outside the launch zone, a safe zone that does not protect, and observations
that leak the truth into the actor's vector. It also reports the two numbers
that decide whether vision training is affordable at all: env-steps per second
and peak VRAM with the cameras on.

Exits non-zero if any check fails.
"""
import argparse
import time

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--num_envs", type=int, default=16)
parser.add_argument("--steps", type=int, default=200)
parser.add_argument("--targets", type=int, default=12)
parser.add_argument("--arena", type=float, default=590.0)
parser.add_argument("--vehicle", default=None)
parser.add_argument("--camera", action="store_true", help="render the body-fixed camera (implies --enable_cameras)")
parser.add_argument("--res", type=int, default=96)
parser.add_argument("--world", default=None)
parser.add_argument("--map", default=None)
parser.add_argument("--zones", default=None)
parser.add_argument("--geofence", action="store_true",
                    help="append the signed distance to the nearest safe zone to the actor's vector")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.headless = True
if args.camera:
    args.enable_cameras = True
app = AppLauncher(args).app

import carb  # noqa: E402

# Same RTX settings the other render scripts use on this 16k-tree world: textures
# up front (no streaming), DLSS off, FXAA instead of temporal AA, no motion blur.
_s = carb.settings.get_settings()
_s.set("/rtx-transient/resourcemanager/enableTextureStreaming", False)
_s.set("/rtx/post/aa/op", 2)
_s.set("/rtx/post/dlss/execMode", 0)
_s.set("/rtx/post/motionblur/enabled", False)

import torch  # noqa: E402

from vesper.lab.chase_env import ChaseEnv, ChaseEnvCfg  # noqa: E402
from vesper.lab.frames import PROPRIO_DIM, yaw_from_quat  # noqa: E402
from vesper.lab.vehicles import ROLES  # noqa: E402

cfg = ChaseEnvCfg()
cfg.scene.num_envs = args.num_envs
cfg.scene.env_spacing = 0.0
cfg.n_targets = args.targets
cfg.chase = {"arena_half": args.arena}
cfg.vehicle_model = args.vehicle
cfg.camera = args.camera
cfg.cam_res = args.res
if args.world:
    cfg.world_usd = args.world
if args.map:
    cfg.world_map = args.map
if args.zones:
    cfg.zones = args.zones
cfg.geofence = args.geofence
env = ChaseEnv(cfg, seed=0)
obs = env.vision_reset()

fails = []


def check(name, ok, detail=""):
    print(f"[{'PASS' if ok else 'FAIL'}] {name} {detail}", flush=True)
    if not ok:
        fails.append(name)


w = env.world
check("world map loaded", w.n > 100 and w.half_m > 100, f"{w.n}x{w.n} @ {w.cell} m, +-{w.half_m} m")
check("every env shares one origin", float(env.scene.env_origins.abs().max()) == 0.0)
check("actor observation is the proprio vector",
      obs["policy"].shape[1] == PROPRIO_DIM + (1 if args.geofence else 0),
      str(tuple(obs["policy"].shape)) + (" (+geofence)" if args.geofence else ""))
check("privileged observation is present", obs["privileged"].shape[1] == env.cfg.state_space)
check("observations are finite", bool(torch.isfinite(obs["policy"]).all()
                                     and torch.isfinite(obs["privileged"]).all()))

# --- zones
print(f"       zones: {env.zones_path or 'none (launch anywhere, nothing protected)'}", flush=True)
if env.zones_path:
    check("launch zone is a real subset of the site", 0 < float(w.launch.mean()) < 0.9,
          f"{100*float(w.launch.mean()):.1f}% of the site")
    d0 = env._robot.data.root_pos_w
    r, c = w.nearest_cell(d0[:, 0], d0[:, 1])
    check("drones launch inside the launch zone", float((w.launch[r, c] > 0.5).float().mean()) > 0.99,
          f"{100*float((w.launch[r, c] > 0.5).float().mean()):.0f}%")
    check("drones spawn spread over the pad, not at one point",
          float(d0[:, 0].std()) > 3.0 or float(d0[:, 1].std()) > 3.0,
          f"spread {float(d0[:, 0].std()):.0f} x {float(d0[:, 1].std()):.0f} m")
    if float(w.safe.sum()) > 0:
        check("the launch pad is friendly ground", bool(w.in_safe(d0[:, 0], d0[:, 1]).all()),
              "the drone starts friendly and has to leave")
        check("the friendly zone's distance field is built",
              float(w.safe_in.max()) > 5.0 and float(w.safe_out[w.safe > 0.5].max()) == 0.0,
              f"deepest point {float(w.safe_in.max()):.0f} m inside")
    else:
        print("[SKIP] no friendly zone in this world's zones file", flush=True)

# --- camera
if args.camera:
    px, dp = obs.get("pixels"), obs.get("depth")
    check("camera renders a tile per env",
          px is not None and tuple(px.shape) == (env.num_envs, args.res, args.res, 3),
          "none" if px is None else f"{tuple(px.shape)} {px.dtype}")
    check("depth renders alongside it",
          dp is not None and tuple(dp.shape) == (env.num_envs, args.res, args.res, 1),
          "none" if dp is None else f"{tuple(dp.shape)} in [{float(dp.min()):.2f}, {float(dp.max()):.2f}]")
    if px is not None:
        check("frames are not blank", float(px.float().std()) > 2.0, f"std {float(px.float().std()):.1f}")
    if dp is not None:
        check("depth is normalised and has near returns", float(dp.max()) <= 1.0 and float(dp.min()) >= 0.0)
    seen = env._seen_px()
    check("tank segmentation labels are present",
          seen is not None and env._seg_table is not None and int((env._seg_table >= 0).sum()) >= args.targets,
          f"{0 if env._seg_table is None else int((env._seg_table >= 0).sum())} tank ids")
    print(f"       VRAM after the first render: {torch.cuda.max_memory_allocated()/1e9:.2f} GB", flush=True)

# --- tanks start on the ground, on their layers
vp = env.vehicle_pos
gz = w.ground_at(vp[:, 0], vp[:, 1])
check("tanks start on drivable ground",
      float(w.is_drivable(vp[:, 0], vp[:, 1]).float().mean()) > 0.8,
      f"{100*float(w.is_drivable(vp[:, 0], vp[:, 1]).float().mean()):.0f}%")
check("tanks start inside the arena", float(vp[:, :2].abs().max()) <= args.arena + 1.0)
print(f"       roles: {[ROLES[int(i)][0] for i in env.driver.role.tolist()]}", flush=True)

# --- step it under a random policy, biased forward so the drones actually fly
zs, spd, agl, detonations, hits, crashes, nose_err, seen_frac, in_safe, exits = (
    [], [], [], 0, 0, 0, [], [], 0, [])
was_in = None
torch.manual_seed(0)
t0 = time.time()
act = torch.zeros(env.num_envs, env.num_actions, device=env.device)
for i in range(args.steps):
    act = 0.85 * act + 0.15 * torch.randn_like(act) * 0.5
    act[:, 0] += 0.5
    obs, rew, done, info = env.vision_step(act.clamp(-1, 1))
    if not torch.isfinite(obs["policy"]).all():
        check("observations stay finite", False, f"NaN at step {i}")
        break
    if i > 40:
        v = env.vehicle_pos
        zs.append((v[:, 2] - w.ground_at(v[:, 0], v[:, 1])).clone())
        spd.append(env._vehicles.data.root_lin_vel_w[:, :2].norm(dim=1).clone())
        agl.append(info["agl"].clone())
        detonations += int(info["detonated"].sum())
        hits += int(info["hit"].sum())
        crashes += int(info["crash"].sum())
        dv = env._robot.data.root_lin_vel_w
        fast = dv[:, :2].norm(dim=1) > 4.0
        if fast.any():
            yaw = yaw_from_quat(env._robot.data.root_quat_w)[fast]
            head = torch.atan2(dv[fast, 1], dv[fast, 0])
            nose_err.append(torch.atan2(torch.sin(head - yaw), torch.cos(head - yaw)).abs())
        seen_frac.append(info["visible"].float().mean().clone())
        in_safe += int(info["in_safe"].sum())
        if was_in is not None:
            exits.append(int((was_in & ~info["in_safe"]).sum()))
        was_in = info["in_safe"].clone()
else:
    check("observations stay finite", True)
wall = time.time() - t0
print(f"       {args.steps * env.num_envs / wall / 1e3:.2f}k env-steps/s over {args.steps} steps "
      f"({env.num_envs} envs{f', camera {args.res}px' if args.camera else ''})", flush=True)
if args.camera:
    print(f"       peak VRAM: {torch.cuda.max_memory_allocated()/1e9:.2f} GB", flush=True)

z = torch.cat(zs)
check("tanks ride the terrain", -0.5 < float(z.mean()) < 3.5 and float(z.std()) < 2.0,
      f"height above ground mean {float(z.mean()):.2f} std {float(z.std()):.2f}")
check("no tank falls through or launches", float(z.min()) > -3.0 and float(z.max()) < 25.0,
      f"[{float(z.min()):.1f}, {float(z.max()):.1f}] m")
s = torch.cat(spd)
check("driving tanks move", float(s.max()) > 2.0, f"max ground speed {float(s.max()):.1f} m/s")
if nose_err:
    ne = torch.cat(nose_err)
    check("the nose follows the velocity", float(ne.median()) < 0.5,
          f"median heading error {float(ne.median()):.2f} rad")
else:
    print("[SKIP] never fast enough to judge the heading follow", flush=True)
check("detonations and contacts produce events", detonations + crashes > 0,
      f"{detonations} detonations, {hits} tank hits, {crashes} crashes under a random policy")
v = float(torch.stack(seen_frac).mean())
check("the camera denies most tanks most of the time", 0.0 <= v < 0.5,
      f"{100*v:.1f}% of tank-steps in frame (this is a search, not a chase)")
print(f"       drone AGL over the run: mean {float(torch.cat(agl).mean()):.0f} m", flush=True)

if float(w.safe.sum()) > 0:
    print(f"       friendly ground: {in_safe} drone-steps over it, {sum(exits)} exits "
          f"(the policy's first job is to leave)", flush=True)

# --- the actor's vector must carry no world position
pr = obs["policy"]
check("the actor vector is bounded and carries no world position",
      float(pr.abs().max()) < 50.0 and pr.shape[1] == env.cfg.observation_space,
      f"max |value| {float(pr.abs().max()):.2f}")

print("\n=== " + ("ALL CHECKS PASSED" if not fails else f"FAILED: {fails}"), flush=True)
env.close()
app.close()
raise SystemExit(1 if fails else 0)
