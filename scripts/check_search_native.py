"""Smoke-test the NATIVE search environment -- no Isaac, runs on a Mac.

    .venv/bin/python scripts/check_search_native.py --num_envs 32

The same guarantees check_search.py asserts under Isaac, minus the ones that
only exist there (env origins, collision filtering, rendered cameras): world
scale, spawn placement by role, a sensor that denies most targets, vehicles
that ride the terrain and actually move, a nose that follows the velocity, and
a belief that never leaks an unseen target. Also prints full-loop env-steps/s,
the number that decides whether the state-teacher lane can train on this
machine. Exits non-zero if any check fails.
"""
import argparse
import time

import torch

from vesper.lab.ground import ROLES
from vesper.lab.search_task import PROPRIO_DIM, yaw_from_quat
from vesper.native import NativeSearchEnv, NativeSearchEnvCfg

parser = argparse.ArgumentParser()
parser.add_argument("--num_envs", type=int, default=32)
parser.add_argument("--steps", type=int, default=200)
parser.add_argument("--targets", type=int, default=3)
parser.add_argument("--arena", type=float, default=300.0)
parser.add_argument("--groups", type=int, default=0)
parser.add_argument("--map", default=None)
parser.add_argument("--device", default="cpu", help="cpu | mps | cuda")
args = parser.parse_args()

cfg = NativeSearchEnvCfg()
cfg.num_envs = args.num_envs
cfg.n_targets = args.targets
cfg.search = {"arena_half": args.arena}
cfg.n_groups = args.groups
if args.map:
    cfg.world_map = args.map
env = NativeSearchEnv(cfg, device=args.device, seed=0)
obs = env.ppo_reset()
full = env._observations()

fails = []


def check(name, ok, detail=""):
    print(f"[{'PASS' if ok else 'FAIL'}] {name} {detail}", flush=True)
    if not ok:
        fails.append(name)


w = env.world
check("world map loaded", w.n > 100 and w.half_m > 100,
      f"{w.n}x{w.n} @ {w.cell} m, +-{w.half_m} m")
check("observation width matches the task", obs.shape[1] == env.num_obs, str(tuple(obs.shape)))
check("first observation is finite", bool(torch.isfinite(obs).all()))
check("actor observation is the proprio vector", full["policy"].shape[1] == PROPRIO_DIM,
      str(tuple(full["policy"].shape)))
check("privileged observation is present", full["privileged"].shape[1] == env.task.obs_dim)

# --- vehicles: on the ground they were placed on, and on drivable ground
tp = env.target_pos
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
check("roles are shuffled across envs", int(env.role.unique().numel()) > 1,
      f"{sorted(set(env.role.flatten().tolist()))}")
check("some targets are camouflaged", float(env.task.contrast.min()) < 0.5,
      f"min contrast {float(env.task.contrast.min()):.2f}")
check("some targets are parked or crawling", float(env.veh_speed.min()) < 1.5,
      f"min speed {float(env.veh_speed.min()):.2f} m/s")

# --- step it under a random policy
zs, above, vis_frac, spd, nose_err = [], [], [], [], []
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
        spd.append(env.veh_vel[: env.G].norm(dim=2).flatten().clone())
        above.append(info["agl"].clone())
        dpos, dv, dquat, _ = env.flight_state()
        fast = dv[:, :2].norm(dim=1) > 4.0
        if fast.any():
            yaw = yaw_from_quat(dquat)[fast]
            head = torch.atan2(dv[fast, 1], dv[fast, 0])
            nose_err.append(torch.atan2(torch.sin(head - yaw), torch.cos(head - yaw)).abs())
else:
    check("observations stay finite", True)
wall = time.time() - t_start
print(f"       {args.steps * env.num_envs / wall / 1e3:.1f}k env-steps/s over {args.steps} steps "
      f"({env.num_envs} envs, device {args.device})", flush=True)

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

# --- the belief must never carry an unseen target
never = ~env.task.known
priv = env._observations()["privileged"]
if never.any():
    base = 12
    blk = priv[:, base:base + 8 * env.k].view(env.num_envs, env.k, 8)
    leak = blk[never][:, [0, 3, 4, 5, 6, 7]].abs().max()
    check("unseen targets leak nothing into the observation", float(leak) == 0.0, f"max {float(leak):.3f}")
else:
    print("[SKIP] every target was seen; no unseen slot to check", flush=True)

print("\n=== " + ("ALL CHECKS PASSED" if not fails else f"FAILED: {fails}"), flush=True)
raise SystemExit(1 if fails else 0)
