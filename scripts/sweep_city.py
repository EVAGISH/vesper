"""Step 7 sweep: N seeded variants of the city corridor, vectorized in VesperQuad.

    /isaac-sim/python.sh scripts/sweep_city.py --variants 512 --headless

Per-env wind (OU gusts), visibility, range-noise, spawn offset. The system
under test is the conventional SE(3) waypoint loop (policies plug in at Step 8).
Outputs into runs/<id>/: results.jsonl, report.json, rays.parquet (env 0),
path_fail_*.parquet (up to 20 failures), scenario.json (base).
"""
import argparse
import json
import math
import time

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--variants", type=int, default=512)
parser.add_argument("--seconds", type=float, default=45.0)
parser.add_argument("--seed", type=int, default=3)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app = AppLauncher(args).app

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch

from vesper.capture import RunCapture
from vesper.control import SE3Controller
from vesper.dynamics import GustField
from vesper.eval import bin_success, findings
from vesper.lab.vesper_quad import VesperQuadEnv, VesperQuadEnvCfg
from vesper.scenario.randomizer import sample_variants
from vesper.scenario.spec import city_scenario
from vesper.sensors import PrismRayCaster, RangeNoise

N = args.variants
base = city_scenario(seed=args.seed)
variants = sample_variants(base, N)

cfg = VesperQuadEnvCfg()
cfg.scene.num_envs = N
cfg.scene.env_spacing = 0.0            # everyone flies the same (one) city
cfg.city_buildings = base.buildings
env = VesperQuadEnv(cfg)
dev = env.device

env.spawn_offsets = torch.zeros(N, 3, device=dev)
env.spawn_offsets[:, 0] = torch.tensor([v.spawn_east for v in variants], device=dev)
env.reset()

ctrl = SE3Controller(env.params, N, device=dev)
dt = cfg.sim.dt * cfg.decimation
gen = torch.Generator(device=dev); gen.manual_seed(args.seed)
wind_dirs = torch.tensor([[math.cos(math.radians(v.wind_dir_deg)),
                           math.sin(math.radians(v.wind_dir_deg)), 0.0] for v in variants], device=dev)
wind_mean = wind_dirs * torch.tensor([[v.wind_speed_ms] for v in variants], device=dev)
gust_std = torch.tensor([[0.25 * v.wind_speed_ms] for v in variants], device=dev)
gusts = GustField(N, wind_mean, gust_std=1.0, dt=dt, device=dev, generator=gen)
gusts.sigma = gusts.sigma * gust_std  # per-env gust intensity scales with wind
vis = torch.tensor([v.visibility_m for v in variants], device=dev)
noise_std = torch.tensor([[v.range_noise_std] for v in variants], device=dev)

rays = PrismRayCaster(base.buildings, num_rays=16, max_range=50.0, device=dev)
rnoise = RangeNoise(std=noise_std, dropout_p=0.02, generator=gen)

# spec (north,east,alt) -> world (x=e, y=n, z=alt)
wps = torch.tensor([[e, n, a] for n, e, a in base.waypoints], device=dev)
wp_idx = torch.zeros(N, dtype=torch.long, device=dev)
done_ok = torch.zeros(N, dtype=torch.bool, device=dev)
collided = torch.zeros(N, dtype=torch.bool, device=dev)
t_done = torch.full((N,), float("nan"), device=dev)

# inflated AABBs for collision termination
lo, hi = rays.box_lo.clone(), rays.box_hi.clone()
lo[:, :2] -= 0.15; hi[:, :2] += 0.15; hi[:, 2] += 0.15

cap = RunCapture("sweep")
base.save(cap.dir / "scenario.json")
ray_log, path_log = [], []
steps = int(args.seconds / dt)
t0 = time.time()
for i in range(steps):
    pos, vel, quat, avb = env.flight_state()
    env.wind_world = gusts.step()
    target = wps[wp_idx.clamp(max=len(wps) - 1)]
    omega = ctrl.compute(pos, vel, quat, avb, target)
    active = ~(done_ok | collided)
    omega[~active] = 0.0
    env.step(omega / env.params.omega_max)

    t = i * dt
    inside = ((pos.unsqueeze(1) >= lo) & (pos.unsqueeze(1) <= hi)).all(dim=2).any(dim=1)
    newly_hit = inside & active
    collided |= newly_hit
    t_done = torch.where(newly_hit, torch.full_like(t_done, t), t_done)

    reached = ((pos - target).norm(dim=1) < 0.6) & active
    wp_idx = torch.where(reached, wp_idx + 1, wp_idx)
    finished = (wp_idx >= len(wps)) & active
    done_ok |= finished
    t_done = torch.where(finished, torch.full_like(t_done, t), t_done)

    yaw = torch.atan2(2 * (quat[:, 0] * quat[:, 3] + quat[:, 1] * quat[:, 2]),
                      1 - 2 * (quat[:, 2] ** 2 + quat[:, 3] ** 2))
    if i % 5 == 0:
        r = rnoise.apply(rays.cast(pos, yaw, visibility=vis), rays.max_range)
        ray_log.append((t, pos[0].cpu().numpy().copy(), float(yaw[0]), r[0].cpu().numpy().copy()))
        path_log.append(pos.cpu().numpy().copy())
    if bool((done_ok | collided).all()):
        break
print(f"swept {N} variants x {t:.0f}s sim in {time.time()-t0:.0f}s wall", flush=True)

results = []
for j, v in enumerate(variants):
    ok = bool(done_ok[j])
    results.append({
        "variant": j, "success": 1.0 if ok else 0.0,
        "failure": None if ok else ("collision" if bool(collided[j]) else "timeout"),
        "t_done": None if torch.isnan(t_done[j]) else round(float(t_done[j]), 1),
        "wind_speed_ms": v.wind_speed_ms, "wind_dir_deg": v.wind_dir_deg,
        "visibility_m": v.visibility_m, "range_noise_std": v.range_noise_std,
        "spawn_east": v.spawn_east,
    })
with open(cap.dir / "results.jsonl", "w") as f:
    f.writelines(json.dumps(r) + "\n" for r in results)

report = {"findings": findings(results),
          "bins": {d: bin_success(results, d)
                   for d in ["wind_speed_ms", "visibility_m", "range_noise_std", "spawn_east"]}}
(cap.dir / "report.json").write_text(json.dumps(report, indent=2))
for line in report["findings"]:
    print("FINDING:", line, flush=True)

# artifacts: env-0 rays + up to 20 failure paths
ts = [r[0] for r in ray_log]
pq.write_table(pa.table({
    "t": ts,
    **{k: [r[1][ax] for r in ray_log] for ax, k in enumerate(["px", "py", "pz"])},
    "yaw": [r[2] for r in ray_log],
    **{f"r{k}": [r[3][k] for r in ray_log] for k in range(16)},
}), cap.dir / "rays.parquet")
paths = np.stack(path_log)  # [T, N, 3]
fails = [r["variant"] for r in results if not r["success"]][:20]
for j in fails:
    pq.write_table(pa.table({"t": ts, "px": paths[:, j, 0], "py": paths[:, j, 1], "pz": paths[:, j, 2],
                             "qw": np.ones(len(ts)), "qx": np.zeros(len(ts)),
                             "qy": np.zeros(len(ts)), "qz": np.zeros(len(ts))}),
                   cap.dir / f"path_fail_{j:04d}.parquet")
cap.note(kind="sweep", variants=N, findings=report["findings"])
cap.finish()
env.close()
app.close()
