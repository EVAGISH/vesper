"""Prepare an imported world USD (Unreal/Blender/converter export) for the sim.

Pure pxr (usd-core on the Mac, Isaac's USD in the container) -- no Isaac imports,
so it is CPU-testable. Non-destructive: the source file is never edited. Instead a
small wrapper stage is written that references the export under /World/level and
lays `over` opinions on top:

  * PhysX static collision on every Mesh (exact triangle mesh), including
    PointInstancer prototypes (trees/rocks scattered by foliage tools)
  * a root xform that converts the export's units to meters and fixes Y-up
    (Unreal exports are Z-up; glTF/Blender-default are Y-up)

The wrapper is what ScenarioSpec.terrain["usd"] points at.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics


@dataclass
class WorldReport:
    source: str
    meters_per_unit: float
    up_axis: str
    meshes: int
    instancers: int
    instances: int
    bounds_min_m: list = field(default_factory=list)   # in meters, after unit/axis fix
    bounds_max_m: list = field(default_factory=list)
    collided: int = 0
    wrapper: str = ""


def _mesh_prims(stage: Usd.Stage, skip_instancer_protos: bool = False):
    """All Mesh prims; optionally prune PointInstancer subtrees (prototypes are not
    drawn where they sit, so they must not count as world geometry)."""
    it = iter(Usd.PrimRange(stage.GetPseudoRoot()))
    for prim in it:
        if skip_instancer_protos and prim.IsA(UsdGeom.PointInstancer):
            it.PruneChildren()
            continue
        if prim.IsA(UsdGeom.Mesh):
            yield prim


def _is_url(p) -> bool:
    return str(p).startswith(("http://", "https://", "omniverse://"))


def inspect(usd_path: str | Path) -> WorldReport:
    stage = Usd.Stage.Open(str(usd_path))
    mpu = UsdGeom.GetStageMetersPerUnit(stage)
    up = UsdGeom.GetStageUpAxis(stage)
    meshes = sum(1 for _ in _mesh_prims(stage))
    inst_prims = [p for p in stage.Traverse() if p.IsA(UsdGeom.PointInstancer)]
    n_inst = 0
    for p in inst_prims:
        ids = UsdGeom.PointInstancer(p).GetProtoIndicesAttr().Get()
        n_inst += len(ids) if ids else 0
    bbox = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_, UsdGeom.Tokens.render])
    rng = Gf.Range3d()
    for prim in stage.GetPseudoRoot().GetChildren():
        rng.UnionWith(bbox.ComputeWorldBound(prim).ComputeAlignedRange())
    lo, hi = (np.array(rng.GetMin()) * mpu, np.array(rng.GetMax()) * mpu) if not rng.IsEmpty() else (np.zeros(3), np.zeros(3))
    if up == "Y":  # report in Z-up: (x, y_up, z) -> (x, -z, y)
        lo, hi = np.array([lo[0], -hi[2], lo[1]]), np.array([hi[0], -lo[2], hi[1]])
    return WorldReport(str(usd_path), mpu, up, meshes, len(inst_prims), n_inst,
                       [round(float(v), 2) for v in lo], [round(float(v), 2) for v in hi])


def write_wrapper(usd_path: str | Path, out_path: str | Path | None = None,
                  collision: bool = True) -> WorldReport:
    """Write <stem>_world.usda next to the source: /World/level references the export
    (scaled to meters, rotated to Z-up) and every mesh gets a static triangle-mesh
    collider via `over` opinions. Returns the inspection report with the wrapper path."""
    if _is_url(usd_path):
        if out_path is None:
            raise ValueError("out_path is required for URL sources")
        src_id, rel = str(usd_path), str(usd_path)
    else:
        usd_path = Path(usd_path).resolve()
        src_id, rel = str(usd_path), None
    out_path = Path(out_path) if out_path else usd_path.with_name(usd_path.stem + "_world.usda")
    rep = inspect(src_id)

    stage = Usd.Stage.CreateNew(str(out_path))
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    world = UsdGeom.Xform.Define(stage, "/World")
    stage.SetDefaultPrim(world.GetPrim())
    level = UsdGeom.Xform.Define(stage, "/World/level")
    if rep.up_axis == "Y":
        level.AddRotateXOp().Set(90.0)
    if rep.meters_per_unit != 1.0:
        s = rep.meters_per_unit
        level.AddScaleOp().Set(Gf.Vec3f(s, s, s))
    # reference the export's default prim (or its root children if it has none)
    src = Usd.Stage.Open(src_id)
    if rel is None:
        import os
        rel = os.path.relpath(usd_path, out_path.parent)
    default = src.GetDefaultPrim()
    if default:
        level.GetPrim().GetReferences().AddReference(str(rel))
        src_roots = [default]
    else:
        src_roots = list(src.GetPseudoRoot().GetChildren())
        for r in src_roots:
            child = UsdGeom.Xform.Define(stage, f"/World/level/{r.GetName()}")
            child.GetPrim().GetReferences().AddReference(str(rel), r.GetPath())

    # collision overs: path under the wrapper mirrors the source path below the referenced root
    if collision:
        n = 0
        for mesh in _mesh_prims(src):
            if default:
                if not mesh.GetPath().HasPrefix(default.GetPath()):
                    continue
                sub = mesh.GetPath().MakeRelativePath(default.GetPath())
                dst = Sdf.Path("/World/level").AppendPath(sub) if str(sub) != "." else Sdf.Path("/World/level")
            else:
                dst = Sdf.Path("/World/level").AppendPath(mesh.GetPath().MakeRelativePath("/"))
            over = stage.OverridePrim(dst)
            UsdPhysics.CollisionAPI.Apply(over)
            mca = UsdPhysics.MeshCollisionAPI.Apply(over)
            mca.CreateApproximationAttr().Set(UsdPhysics.Tokens.none)  # exact triangles (static only)
            n += 1
        rep.collided = n
    stage.GetRootLayer().Save()
    rep.wrapper = str(out_path)
    return rep


def _triangles_world(stage: Usd.Stage):
    """Yield (N,3,3) world-space triangle arrays per mesh (polygons fan-triangulated)."""
    cache = UsdGeom.XformCache(Usd.TimeCode.Default())
    for prim in _mesh_prims(stage, skip_instancer_protos=True):
        mesh = UsdGeom.Mesh(prim)
        pts, counts, idx = mesh.GetPointsAttr().Get(), mesh.GetFaceVertexCountsAttr().Get(), mesh.GetFaceVertexIndicesAttr().Get()
        if not pts or not counts:
            continue
        m = np.array(cache.GetLocalToWorldTransform(prim))
        w = np.asarray(pts, dtype=float) @ m[:3, :3] + m[3, :3]
        idx = np.asarray(idx); tris = []; k = 0
        for c in counts:
            f = idx[k:k + c]; k += c
            for j in range(1, c - 1):
                tris.append((f[0], f[j], f[j + 1]))
        if tris:
            yield w[np.array(tris)]


def ground_height(usd_path: str | Path, x: float, y: float) -> float | None:
    """Top surface height (m, Z-up world) under a vertical ray at (x, y), or None if
    nothing is below. Exact ray/triangle test on the composed meshes (numpy; fine for
    a few million triangles)."""
    stage = Usd.Stage.Open(str(usd_path))
    best = None
    for T in _triangles_world(stage):
        a, b, c = T[:, 0], T[:, 1], T[:, 2]
        # 2D point-in-triangle via barycentrics, then interpolate z
        d = (b[:, 1] - c[:, 1]) * (a[:, 0] - c[:, 0]) + (c[:, 0] - b[:, 0]) * (a[:, 1] - c[:, 1])
        ok = np.abs(d) > 1e-12
        l1 = np.where(ok, ((b[:, 1] - c[:, 1]) * (x - c[:, 0]) + (c[:, 0] - b[:, 0]) * (y - c[:, 1])) / np.where(ok, d, 1), -1)
        l2 = np.where(ok, ((c[:, 1] - a[:, 1]) * (x - c[:, 0]) + (a[:, 0] - c[:, 0]) * (y - c[:, 1])) / np.where(ok, d, 1), -1)
        l3 = 1 - l1 - l2
        inside = ok & (l1 >= -1e-9) & (l2 >= -1e-9) & (l3 >= -1e-9)
        if inside.any():
            z = (l1 * a[:, 2] + l2 * b[:, 2] + l3 * c[:, 2])[inside].max()
            best = float(z) if best is None else max(best, float(z))
    return best
