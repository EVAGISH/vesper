"""Probe a (possibly remote) world USD inside the container and write its collider wrapper.

    /isaac-sim/python.sh scripts/probe_world.py https://.../rivermark.usd assets/rivermark --spawn 0 0

Opens the stage with Isaac's USD (so omniverse:// and https:// payloads resolve),
writes <out_dir>/<name>_world.usda (meters, Z-up, static triangle colliders on every
mesh), prints bounds, mesh count and the ground height under the spawn, and the
`terrain` block for a ScenarioSpec. No SimulationApp needed for the USD work, but
Isaac's python is required for the asset resolver.
"""
import argparse
import json
import time
from pathlib import Path

from vesper.worlds.prepare import ground_height, inspect, write_wrapper

ap = argparse.ArgumentParser()
ap.add_argument("usd")
ap.add_argument("out_dir")
ap.add_argument("--spawn", type=float, nargs=2, default=(0.0, 0.0))
ap.add_argument("--name", default=None)
a = ap.parse_args()

out_dir = Path(a.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
name = a.name or Path(a.usd.split("?")[0]).stem
t0 = time.time()
rep = inspect(a.usd)
print(f"opened in {time.time() - t0:.0f}s: {rep.meshes} meshes, {rep.instancers} instancers ({rep.instances} instances), "
      f"units {rep.meters_per_unit} m, up {rep.up_axis}; bounds {rep.bounds_min_m} -> {rep.bounds_max_m}", flush=True)
rep = write_wrapper(a.usd, out_dir / f"{name}_world.usda")
print(f"wrapper: {rep.wrapper} ({rep.collided} colliders)", flush=True)
x, y = a.spawn
z = ground_height(rep.wrapper, x, y)
print(f"ground at ({x}, {y}): {z}", flush=True)
if z is not None:
    print("terrain:", json.dumps({"usd": rep.wrapper, "translation": [-x, -y, -z], "rotation_xyz_deg": [0, 0, 0], "scale": 1.0}))
