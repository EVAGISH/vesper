"""CPU test: prepare an Unreal-style USD export (cm units, Z-up, landscape mesh +
foliage PointInstancer) into a collidable, meter-scaled wrapper."""
from pathlib import Path

import numpy as np
from pxr import Gf, Usd, UsdGeom, UsdPhysics, Vt

from vesper.worlds.prepare import ground_height, inspect, write_wrapper


def _make_ue_export(path: Path) -> None:
    stage = Usd.Stage.CreateNew(str(path))
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 0.01)          # Unreal default: 1 unit = 1 cm
    root = UsdGeom.Xform.Define(stage, "/Level")
    stage.SetDefaultPrim(root.GetPrim())
    # landscape: 200 m x 200 m quad at z = 150 cm (in cm units)
    land = UsdGeom.Mesh.Define(stage, "/Level/Landscape/Mesh")
    land.CreatePointsAttr([(-10000, -10000, 150), (10000, -10000, 150), (10000, 10000, 150), (-10000, 10000, 150)])
    land.CreateFaceVertexCountsAttr([4]); land.CreateFaceVertexIndicesAttr([0, 1, 2, 3])
    # tree prototype + instancer with 3 trees
    proto = UsdGeom.Mesh.Define(stage, "/Level/Foliage/Prototypes/Tree")
    proto.CreatePointsAttr([(0, 0, 0), (100, 0, 0), (0, 0, 1500)])
    proto.CreateFaceVertexCountsAttr([3]); proto.CreateFaceVertexIndicesAttr([0, 1, 2])
    inst = UsdGeom.PointInstancer.Define(stage, "/Level/Foliage/Instancer")
    inst.CreatePrototypesRel().SetTargets([proto.GetPath()])
    inst.CreateProtoIndicesAttr(Vt.IntArray([0, 0, 0]))
    inst.CreatePositionsAttr(Vt.Vec3fArray([Gf.Vec3f(0, 0, 150), Gf.Vec3f(500, 0, 150), Gf.Vec3f(0, 500, 150)]))
    stage.GetRootLayer().Save()


def test_wrapper_scales_rotates_and_collides(tmp_path):
    src = tmp_path / "level.usda"
    _make_ue_export(src)
    rep = inspect(src)
    assert rep.meters_per_unit == 0.01 and rep.up_axis == "Z"
    assert rep.meshes == 2 and rep.instancers == 1 and rep.instances == 3
    assert rep.bounds_min_m[0] == -100.0 and rep.bounds_max_m[0] == 100.0   # reported in meters

    rep = write_wrapper(src)
    assert rep.collided == 2
    w = Usd.Stage.Open(rep.wrapper)
    assert UsdGeom.GetStageMetersPerUnit(w) == 1.0
    land = w.GetPrimAtPath("/World/level/Landscape/Mesh")
    assert land and land.HasAPI(UsdPhysics.CollisionAPI) and land.HasAPI(UsdPhysics.MeshCollisionAPI)
    assert UsdPhysics.MeshCollisionAPI(land).GetApproximationAttr().Get() == "none"
    tree = w.GetPrimAtPath("/World/level/Foliage/Prototypes/Tree")
    assert tree.HasAPI(UsdPhysics.CollisionAPI)
    # composed world bounds are in meters now
    bb = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_]).ComputeWorldBound(land).ComputeAlignedRange()
    assert np.allclose(bb.GetMin(), (-100, -100, 1.5)) and np.allclose(bb.GetMax(), (100, 100, 1.5))
    # landscape top at the origin is 1.5 m
    assert abs(ground_height(rep.wrapper, 0.0, 0.0) - 1.5) < 1e-6


def test_wrapper_fixes_y_up(tmp_path):
    src = tmp_path / "gltfish.usda"
    stage = Usd.Stage.CreateNew(str(src))
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.y)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    root = UsdGeom.Xform.Define(stage, "/Root"); stage.SetDefaultPrim(root.GetPrim())
    m = UsdGeom.Mesh.Define(stage, "/Root/Ground")
    m.CreatePointsAttr([(-5, 2, -5), (5, 2, -5), (5, 2, 5), (-5, 2, 5)])   # y-up plane at height 2
    m.CreateFaceVertexCountsAttr([4]); m.CreateFaceVertexIndicesAttr([0, 1, 2, 3])
    stage.GetRootLayer().Save()
    rep = write_wrapper(src)
    assert rep.up_axis == "Y" and rep.bounds_min_m[2] == 2.0 and rep.bounds_max_m[2] == 2.0
    assert abs(ground_height(rep.wrapper, 0.0, 0.0) - 2.0) < 1e-6
