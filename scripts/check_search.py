"""Smoke-test the search environment on the GPU before spending hours training in it.

    /isaac-sim/python.sh scripts/check_search.py --num_envs 32 --headless

Everything asserted here is something that silently ruins a training run: a world
that loads at the wrong scale, vehicles spawning inside buildings or sinking
through terrain, environments that were never actually collision-filtered apart,
a sensor that reports every target from anywhere (making it a chase, not a
search), and observations that carry a target the drone has not seen. Exits
non-zero if any check fails.
"""
import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--num_envs", type=int, default=32)
parser.add_argument("--steps", type=int, default=200)
parser.add_argument("--targets", type=int, default=3)
parser.add_argument("--arena", type=float, default=300.0)
parser.add_argument("--vehicle", default=None)
parser.add_argument("--groups", type=int, default=0, help="vehicle sets shared by groups of envs")
parser.add_argument("--camera", action="store_true",
                    help="render the body-fixed TiledCamera and check pixel sightings (the "
                         "smoke test for vision: does it render on this world, how fast, how much VRAM)")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.headless = True
if args.camera:
    args.enable_cameras = True
app = AppLauncher(args).app

import time  # noqa: E402

import torch  # noqa: E402

from vesper.lab.search_env import ROLES, SearchEnv, SearchEnvCfg  # noqa: E402
from vesper.lab.search_task import PROPRIO_DIM, yaw_from_quat  # noqa: E402

cfg = SearchEnvCfg()
cfg.scene.num_envs = args.num_envs
cfg.scene.env_spacing = 0.0
cfg.n_targets = args.targets
cfg.search = {"arena_half": args.arena}
cfg.vehicle_model = args.vehicle
cfg.n_groups = args.groups
cfg.camera = args.camera
env = SearchEnv(cfg, seed=0)
obs = env.ppo_reset()
full = env._get_observations()

fails = []


def check(name, ok, detail=""):
    print(f"[{'PASS' if ok else 'FAIL'}] {name} {detail}", flush=True)
    if not ok:
        fails.append(name)


w = env.world
check("world map loaded", w.n > 100 and w.half_m > 100,
      f"{w.n}x{w.n} @ {w.cell} m, +-{w.half_m} m")
check("every env shares one origin", float(env.scene.env_origins.abs().max()) == 0.0,
      f"max |origin| = {float(env.scene.env_origins.abs().max()):.3f}")
check("observation width matches the task", obs.shape[1] == env.num_obs, str(tuple(obs.shape)))
check("first observation is finite", bool(torch.isfinite(obs).all()))
check("actor observation is the proprio vector", full["policy"].shape[1] == PROPRIO_DIM,
      str(tuple(full["policy"].shape)))
check("privileged observation is present", full["privileged"].shape[1] == env.cfg.state_space)
if args.camera:
    px = full.get("pixels")
    check("camera renders a tile per env",
          px is not None and tuple(px.shape) == (env.num_envs, cfg.cam_res, cfg.cam_res, 3),
          "none" if px is None else f"{tuple(px.shape)} {px.dtype}")
    if px is not None:
        check("frames are not blank", float(px.float().std()) > 2.0, f"std {float(px.float().std()):.1f}")
    check("vehicle segmentation labels are present",
          env._seen_px() is not None and env._seg_table is not None and env._seg_table.numel() > 1,
          f"{0 if env._seg_table is None else int((env._seg_table >= 0).sum())} vehicle ids")
    print(f"       VRAM after first render: {torch.cuda.max_memory_allocated()/1e9:.2f} GB", flush=True)

# --- vehicles: on the ground they were placed on, and on drivable ground
tp = env.target_pos
gz = w.ground_at(tp[..., 0], tp[..., 1])
check("vehicles start on drivable ground",
      float(w.is_drivable(tp[..., 0], tp[..., 1]).float().mean()) > 0.9,
      f"{100*float(w.is_drivable(tp[..., 0], tp[..., 1]).float().mean()):.0f}%")
check("vehicles start inside the arena", float(tp[..., :2].abs().max()) <= args.arena + 1.0,
      f"max |xy| = {float(tp[..., :2].abs().max()):.0f} m")
check("every env hunts a live vehicle set", bool((env.group < env.G).all()),
      f"{env.G} sets for {env.num_envs} envs")

# --- placement by role: roads and parking strips, when the map has them
vp = env.vehicle_pos[: env.G]
role = env.role[: env.G]
layer_of = [r[3] for r in ROLES]
for li, name in enumerate(layer_of):
    if name == "drivable":
        continue
    sel = role == li
    if not sel.any():
        continue
    mask = getattr(w, name)
    if float(mask.sum()) == 0:
        print(f"[SKIP] no '{name}' cells in this map (old export?)", flush=True)
        continue
    r, c = w.nearest_cell(vp[..., 0][sel], vp[..., 1][sel])
    frac = float((mask[r, c] > 0.5).float().mean())
    check(f"'{ROLES[li][0]}' vehicles start on the '{name}' layer", frac > 0.8, f"{100*frac:.0f}%")

# --- roles actually differ
print(f"       roles per env (first 4): {env.role[:4].tolist()}", flush=True)
print(f"       speeds  (first 4): {env.veh_speed[:4].round(decimals=1).tolist()}", flush=True)
print(f"       contrast(first 4): {env.task.contrast[:4].round(decimals=2).tolist()}", flush=True)
check("roles are shuffled across envs", int(env.role.unique().numel()) > 1,
      f"{sorted(set(env.role.flatten().tolist()))}")
check("some targets are camouflaged", float(env.task.contrast.min()) < 0.5,
      f"min contrast {float(env.task.contrast.min()):.2f}")
check("some targets are parked or crawling", float(env.veh_speed.min()) < 1.5,
      f"min speed {float(env.veh_speed.min()):.2f} m/s")

# --- step it under a random policy
zs, above, vis_frac, spd, nose_err, px_hits = [], [], [], [], [], []
torch.manual_seed(0)
t_start = time.time()
for i in range(args.steps):
    act = torch.randn(env.num_envs, env.num_actions, device=env.device) * 0.3
    act[:, 0] += 0.8                      # bias forward, so the heading has something to follow
    obs, rew, done, info = env.ppo_step(act)
    if not torch.isfinite(obs).all():
        check("observations stay finite", False, f"NaN at step {i}")
        break
    if i > 40:
        tp = env.target_pos
        g = w.ground_at(tp[..., 0], tp[..., 1])
        zs.append((tp[..., 2] - g).flatten().clone())
        vis_frac.append(info["visible"].float().mean().clone())
        spd.append(torch.stack([v.data.root_lin_vel_w[:, :2].norm(dim=1) for v in env._vehicles], 1)[: env.G].flatten())
        above.append(info["agl"].clone())
        dv = env._robot.data.root_lin_vel_w
        fast = dv[:, :2].norm(dim=1) > 4.0
        if fast.any():
            yaw = yaw_from_quat(env._robot.data.root_quat_w)[fast]
            head = torch.atan2(dv[fast, 1], dv[fast, 0])
            nose_err.append(torch.atan2(torch.sin(head - yaw), torch.cos(head - yaw)).abs())
        if args.camera:
            px_hits.append((env._seen_px() > 0).float().mean().clone())
else:
    check("observations stay finite", True)
wall = time.time() - t_start
print(f"       {args.steps * env.num_envs / wall / 1e3:.1f}k env-steps/s over {args.steps} steps "
      f"({env.num_envs} envs{', camera on' if args.camera else ''})", flush=True)
if args.camera:
    print(f"       peak VRAM: {torch.cuda.max_memory_allocated()/1e9:.2f} GB", flush=True)

z = torch.cat(zs)
check("vehicles ride the terrain", -0.5 < float(z.mean()) < 3.0 and float(z.std()) < 1.5,
      f"height above ground mean {float(z.mean()):.2f} std {float(z.std()):.2f}")
check("no vehicle falls through or launches", float(z.min()) > -3.0 and float(z.max()) < 25.0,
      f"[{float(z.min()):.1f}, {float(z.max()):.1f}] m")

v = float(torch.stack(vis_frac).mean())
check("the sensor denies most targets most of the time", 0.0 < v < 0.6,
      f"{100*v:.1f}% of target-steps visible (a chase would be ~100%)")

s = torch.cat(spd)
check("moving vehicles move", float(s.max()) > 2.0, f"max ground speed {float(s.max()):.1f} m/s")
a = torch.cat(above)
print(f"       drone AGL over the run: mean {float(a.mean()):.0f} m", flush=True)
if nose_err:
    ne = torch.cat(nose_err)
    check("the nose follows the velocity", float(ne.median()) < 0.5,
          f"median heading error {float(ne.median()):.2f} rad on {ne.numel()} fast samples")
else:
    print("[SKIP] never fast enough to judge the heading follow", flush=True)
if args.camera and px_hits:
    ph = float(torch.stack(px_hits).mean())
    check("some vehicles land in some frames", ph > 0.0, f"{100*ph:.1f}% of target-steps with pixels")

# --- the belief must never carry an unseen target
never = ~env.task.known
priv = env._get_observations()["privileged"]
if never.any():
    base = 12
    blk = priv[:, base:base + 8 * env.k].view(env.num_envs, env.k, 8)
    leak = blk[never][:, [0, 3, 4, 5, 6, 7]].abs().max()
    check("unseen targets leak nothing into the observation", float(leak) == 0.0, f"max {float(leak):.3f}")
else:
    print("[SKIP] every target was seen; no unseen slot to check", flush=True)

print("\n=== " + ("ALL CHECKS PASSED" if not fails else f"FAILED: {fails}"), flush=True)
env.close()
app.close()
raise SystemExit(1 if fails else 0)
