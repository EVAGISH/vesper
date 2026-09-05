"""Verify a pursuit target actually loads and behaves in PhysX. Needs a GPU.

    /isaac-sim/python.sh scripts/check_vehicle.py --vehicle tank --headless
    /isaac-sim/python.sh scripts/check_vehicle.py --vehicle cart --headless

Exits non-zero if any check fails, so it works as a smoke test. Everything it
asserts is something that has actually been broken here: a stock prop with no
RigidBodyAPI refusing to spawn, a collider whose floor sat above the model
origin leaving the wheels buried, and a deg/s-vs-rad/s mix-up on
max_angular_velocity that clamped the hull so it could not steer and made it
crab sideways across the terrain. None of it is visible without stepping PhysX.
"""
import argparse
import math

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--vehicle", default="tank")
parser.add_argument("--num_envs", type=int, default=16)
parser.add_argument("--steps", type=int, default=250)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.headless = True
app = AppLauncher(args).app

import torch  # noqa: E402
from isaacsim.core.utils.stage import get_current_stage  # noqa: E402
from pxr import UsdGeom, Usd  # noqa: E402

from vesper.lab.pursuit_env import PursuitEnv, PursuitEnvCfg, resolve_vehicle  # noqa: E402

spec = resolve_vehicle(args.vehicle)
print(f"\n=== spec: {spec}\n", flush=True)

cfg = PursuitEnvCfg()
cfg.scene.num_envs = args.num_envs
cfg.scene.env_spacing = 110.0
cfg.vehicle_model = args.vehicle
env = PursuitEnv(cfg, seed=0)
env.reset()

fails = []


def check(name, ok, detail=""):
    print(f"[{'PASS' if ok else 'FAIL'}] {name} {detail}", flush=True)
    if not ok:
        fails.append(name)


# --- 1. the prim exists and is the model we asked for
stage = get_current_stage()
prim = stage.GetPrimAtPath("/World/envs/env_0/Vehicle")
check("vehicle prim exists", prim.IsValid(), str(prim.GetPath()))
kids = [c.GetName() for c in prim.GetChildren()]
print(f"       children: {kids}", flush=True)
bb = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_, UsdGeom.Tokens.render])
size = bb.ComputeWorldBound(prim).ComputeAlignedRange().GetSize()
check("bbox is vehicle-sized", all(0.3 < s < 8.0 for s in size), f"{tuple(round(s,2) for s in size)}")

# --- 2. PhysX mass is what we asked for, not a density guess
masses = env._vehicle.root_physx_view.get_masses()
m = float(masses[0].item())
print(f"       physx mass = {m:.1f} kg", flush=True)
if spec.get("mass"):
    check("mass override applied", abs(m - spec["mass"]) < 1.0, f"{m:.1f} vs {spec['mass']}")
else:
    check("mass is plausible", 100.0 < m < 20000.0, f"{m:.1f} kg")

# --- 3. step it: must settle on the ground, drive at target_speed, face its travel
zs, speeds, yaw_err = [], [], []
act = torch.zeros(env.num_envs, env.num_actions, device=env.device)
for i in range(args.steps):
    env.step(act)
    if i > 60:                      # let it settle after the spawn drop
        v = env._vehicle.data
        zs.append((v.root_pos_w[:, 2] - env.scene.env_origins[:, 2]).clone())
        sp = v.root_lin_vel_w[:, :2].norm(dim=1)
        speeds.append(sp.clone())
        q = v.root_quat_w
        yaw = torch.atan2(2 * (q[:, 0] * q[:, 3] + q[:, 1] * q[:, 2]),
                          1 - 2 * (q[:, 2] ** 2 + q[:, 3] ** 2))
        nose = yaw - env._veh_yaw_offset          # model nose, undoing the spec offset
        travel = torch.atan2(v.root_lin_vel_w[:, 1], v.root_lin_vel_w[:, 0])
        d = torch.atan2(torch.sin(nose - travel), torch.cos(nose - travel))
        yaw_err.append(d[sp > 0.5].abs().clone())

z = torch.cat(zs); sp = torch.cat(speeds); ye = torch.cat(yaw_err)
zmean, zstd = float(z.mean()), float(z.std())
print(f"       height above ground: mean {zmean:.2f} m, std {zstd:.3f}", flush=True)
check("rests on the terrain", -0.2 < zmean < 2.5 and zstd < 0.35, f"mean {zmean:.2f} std {zstd:.3f}")
check("no fall-through / explosion", bool(z.min() > -1.0) and bool(z.max() < 12.0),
      f"min {float(z.min()):.2f} max {float(z.max()):.2f}")

want = env.tcfg.target_speed
print(f"       ground speed: mean {float(sp.mean()):.2f} m/s (target {want})", flush=True)
check("drives at target_speed", abs(float(sp.mean()) - want) < 1.0, f"{float(sp.mean()):.2f} vs {want}")

deg = math.degrees(float(ye.mean()))
print(f"       nose-vs-travel error: mean {deg:.1f} deg", flush=True)
check("yaw_offset points the nose forward", deg < 20.0, f"{deg:.1f} deg")

print("\n=== " + ("ALL CHECKS PASSED" if not fails else f"FAILED: {fails}"), flush=True)
env.close()
app.close()
raise SystemExit(1 if fails else 0)
