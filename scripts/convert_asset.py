"""Convert a mesh (glTF/OBJ/FBX/STL) to a collidable USD, headless, inside the container.

    /isaac-sim/python.sh scripts/convert_asset.py assets/baylands/baylands.gltf --yup --headless

Writes <same dir>/<stem>.usd with a static triangle-mesh collider (exact mesh;
fine for static terrain, PhysX supports it for non-moving bodies), then prints the
world-space bounds so the spec's terrain placement can be checked. Isaac's
converter does NOT re-orient Y-up sources; pass --yup for glTF.
"""
import argparse
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("mesh")
parser.add_argument("--collision", default="triangle", choices=["triangle", "convexHull", "convexDecomposition"],
                    help="triangle = exact static mesh collider (terrain); convex* for movable props")
parser.add_argument("--instanceable", action="store_true")
parser.add_argument("--yup", action="store_true",
                    help="source is Y-up (glTF always is): bake a +90deg X rotation so 'up' lands on the stage's +Z")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app = AppLauncher(args).app

import isaaclab.sim as sim_utils
from isaaclab.sim.converters import MeshConverter, MeshConverterCfg
from isaaclab.sim.schemas import schemas_cfg

MESH_COLLISION = {
    "triangle": schemas_cfg.TriangleMeshPropertiesCfg,
    "convexHull": schemas_cfg.ConvexHullPropertiesCfg,
    "convexDecomposition": schemas_cfg.ConvexDecompositionPropertiesCfg,
}
from pxr import Usd, UsdGeom

src = Path(args.mesh).resolve()
cfg = MeshConverterCfg(
    asset_path=str(src), usd_dir=str(src.parent), usd_file_name=src.stem + ".usd",
    force_usd_conversion=True, make_instanceable=args.instanceable,
    collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
    mesh_collision_props=MESH_COLLISION[args.collision](),
    rotation=(0.7071068, 0.7071068, 0.0, 0.0) if args.yup else (1.0, 0.0, 0.0, 0.0),
)
conv = MeshConverter(cfg)
out = conv.usd_path
print(f"wrote {out}", flush=True)

stage = Usd.Stage.Open(out)
print("stage upAxis:", UsdGeom.GetStageUpAxis(stage), "metersPerUnit:", UsdGeom.GetStageMetersPerUnit(stage))
bbox = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_]).ComputeWorldBound(stage.GetDefaultPrim())
r = bbox.ComputeAlignedRange()
print("bounds min:", [round(v, 2) for v in r.GetMin()], "max:", [round(v, 2) for v in r.GetMax()], flush=True)
n_mesh = sum(1 for p in stage.Traverse() if p.IsA(UsdGeom.Mesh))
print("meshes:", n_mesh, flush=True)
app.close()
