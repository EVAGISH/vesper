"""Time how long a built world takes to become flyable in Isaac Sim, stage by stage.

    docker compose run --rm -T sim /isaac-sim/python.sh scripts/isaac_load_time.py \
        assets/<site>/<site>.usd [--out runs/<name>] [--no-physics]

Reports (stdout + <out>/report.json):
  app_start_s    SimulationApp boot (shader cache warm: ~13 s; cold box: minutes)
  reference_s    composing the site onto the stage (tens of thousands of tree instances)
  physics_s      World.reset(): PhysX scene + cooking every mesh collider
                 (tree colliders are analytic cylinders/cones: nothing to cook)
  first_frame_s  first RTX frames (BVH build over the whole site)
  counts         meshes / instances / colliders by prim type
Trimmed from dig-twin's isaac/load_world.py (no tour render).
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("usd")
ap.add_argument("--out", default=None)
ap.add_argument("--no-physics", action="store_true")
a = ap.parse_args()
out = Path(a.out or f"runs/{Path(a.usd).stem}_load"); out.mkdir(parents=True, exist_ok=True)

T0 = time.time()
from isaacsim import SimulationApp  # noqa: E402

app = SimulationApp({"headless": True, "renderer": "RayTracedLighting", "width": 1280, "height": 720})
report: dict = {"usd": str(Path(a.usd).resolve()), "app_start_s": round(time.time() - T0, 1)}

import carb  # noqa: E402
import numpy as np  # noqa: E402

_s = carb.settings.get_settings()
_s.set("/rtx-transient/resourcemanager/enableTextureStreaming", False)
_s.set("/rtx/sceneDb/allowedDynamicMeshBytes", 2 ** 31)

import omni.usd  # noqa: E402
from isaacsim.core.api import World  # noqa: E402
from isaacsim.core.utils.stage import add_reference_to_stage  # noqa: E402
from isaacsim.sensors.camera import Camera  # noqa: E402
from pxr import Usd, UsdGeom, UsdPhysics  # noqa: E402

try:                                   # what this PhysX build can do (heightfields? cooked data in USD?)
    from pxr import PhysxSchema
    report["physx_schema"] = {"heightfield": [n for n in dir(PhysxSchema) if "Height" in n],
                              "cooked": [n for n in dir(PhysxSchema) if "Cook" in n]}
except ImportError:
    report["physx_schema"] = None


def log(msg):
    print(f"[load +{time.time() - T0:6.1f}s] {msg}", flush=True)


log(f"app up in {report['app_start_s']} s; physx schema: {report['physx_schema']}")
world = World(stage_units_in_meters=1.0, physics_dt=1.0 / 60.0, rendering_dt=1.0 / 30.0)
stage = omni.usd.get_context().get_stage()

t = time.time()
add_reference_to_stage(str(Path(a.usd).resolve()), "/World/site")
site = stage.GetPrimAtPath("/World/site")
report["reference_s"] = round(time.time() - t, 1)
log(f"referenced in {report['reference_s']} s")

# never descend into instance proxies (57k trees x 150 prims = 8M prims, minutes of Python)
counts = {"meshes": 0, "instanceable": 0, "prototypes": len(stage.GetPrototypes()), "colliders": {}, "collider_verts": 0}
it = iter(Usd.PrimRange(site))
for prim in it:
    if prim.IsInstance():
        counts["instanceable"] += 1; it.PruneChildren(); continue
    if prim.IsA(UsdGeom.Mesh):
        counts["meshes"] += 1
    if prim.HasAPI(UsdPhysics.CollisionAPI):
        k = prim.GetTypeName() or "?"
        counts["colliders"][k] = counts["colliders"].get(k, 0) + 1
        if prim.IsA(UsdGeom.Mesh):
            pts = UsdGeom.Mesh(prim).GetPointsAttr().Get()
            counts["collider_verts"] += len(pts) if pts else 0
report["counts"] = counts
log(f"counts {json.dumps(counts)}")

if not a.no_physics:
    t = time.time()
    world.reset()
    report["physics_s"] = round(time.time() - t, 1)
    log(f"physics ready (PhysX scene + cooking) in {report['physics_s']} s")
else:
    report["physics_s"] = None

cam = Camera(prim_path="/World/tcam", position=np.array([0.0, -1500.0, 800.0]), resolution=(1280, 720))
cam.initialize()
t = time.time()
for _ in range(3):
    world.step(render=True)
report["first_frame_s"] = round(time.time() - t, 1)
log(f"first frames in {report['first_frame_s']} s")
report["total_s"] = round(time.time() - T0, 1)
(out / "report.json").write_text(json.dumps(report, indent=1))
log(f"TOTAL {report['total_s']} s -> {out / 'report.json'}")
app.close()
