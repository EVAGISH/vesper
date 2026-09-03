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
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.headless = True
app = AppLauncher(args).app

import torch  # noqa: E402

from vesper.lab.search_env import SearchEnv, SearchEnvCfg  # noqa: E402

cfg = SearchEnvCfg()
cfg.scene.num_envs = args.num_envs
cfg.scene.env_spacing = 0.0
cfg.n_targets = args.targets
cfg.search = {"arena_half": args.arena}
cfg.vehicle_model = args.vehicle
env = SearchEnv(cfg, seed=0)
obs = env.ppo_reset()

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

# --- vehicles: on the ground they were placed on, and on drivable ground
tp = env.target_pos
gz = w.ground_at(tp[..., 0], tp[..., 1])
check("vehicles start on drivable ground",
      float(w.is_drivable(tp[..., 0], tp[..., 1]).float().mean()) > 0.9,
      f"{100*float(w.is_drivable(tp[..., 0], tp[..., 1]).float().mean()):.0f}%")
check("vehicles start inside the arena", float(tp[..., :2].abs().max()) <= args.arena + 1.0,
      f"max |xy| = {float(tp[..., :2].abs().max()):.0f} m")

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
zs, above, vis_frac, spd = [], [], [], []
torch.manual_seed(0)
for i in range(args.steps):
    act = torch.randn(env.num_envs, env.num_actions, device=env.device) * 0.3
    obs, rew, done, info = env.ppo_step(act)
    if not torch.isfinite(obs).all():
        check("observations stay finite", False, f"NaN at step {i}")
        break
    if i > 40:
        tp = env.target_pos
        g = w.ground_at(tp[..., 0], tp[..., 1])
        zs.append((tp[..., 2] - g).flatten().clone())
        vis_frac.append(info["visible"].float().mean().clone())
        spd.append(torch.stack([v.data.root_lin_vel_w[:, :2].norm(dim=1) for v in env._vehicles], 1).flatten())
        above.append(info["agl"].clone())
else:
    check("observations stay finite", True)

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

# --- the belief must never carry an unseen target
never = ~env.task.known
if never.any():
    k = env.tcfg.grid
    base = 12
    blk = obs[:, base:base + 8 * env.k].view(env.num_envs, env.k, 8)
    leak = blk[never][:, [0, 3, 4, 5, 6, 7]].abs().max()
    check("unseen targets leak nothing into the observation", float(leak) == 0.0, f"max {float(leak):.3f}")
else:
    print("[SKIP] every target was seen; no unseen slot to check", flush=True)

print("\n=== " + ("ALL CHECKS PASSED" if not fails else f"FAILED: {fails}"), flush=True)
env.close()
app.close()
raise SystemExit(1 if fails else 0)
