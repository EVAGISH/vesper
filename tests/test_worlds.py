from vesper.scenario.spec import crash_scenario
from vesper.worlds import sample_city_block


def test_layout_deterministic():
    assert sample_city_block(5) == sample_city_block(5)
    assert sample_city_block(5) != sample_city_block(6)


def test_corridor_clear():
    for b in sample_city_block(0):
        cy, d = b["center"][1], b["size"][1]
        assert abs(cy) - d / 2 > 0.5  # corridor stays flyable, now tight


def test_crash_scenario_blocks_corridor():
    spec = crash_scenario(0)
    blocker = spec.buildings[-1]
    assert blocker["center"] == [8.0, 0.0]
    assert spec.max_sim_s == 75.0
    # round trip with buildings
    from vesper.scenario import ScenarioSpec
    assert ScenarioSpec.from_dict(spec.to_dict()) == spec


def test_custom_tank_is_a_drivable_rigid_body(tmp_path):
    """The project tank: one rigid body, one cheap collider, nose +X.

    assets/ is gitignored, so this proxy is generated at run time -- a broken
    generator would otherwise only surface on the GPU box.
    """
    from pxr import Usd, UsdGeom, UsdPhysics
    from vesper.worlds.vehicle import write_tank_usd

    out = write_tank_usd(tmp_path / "tank.usd")
    stage = Usd.Stage.Open(str(out))
    root = stage.GetDefaultPrim()

    assert UsdGeom.GetStageUpAxis(stage) == UsdGeom.Tokens.z
    assert UsdGeom.GetStageMetersPerUnit(stage) == 1.0
    assert root.HasAPI(UsdPhysics.RigidBodyAPI)
    assert UsdPhysics.MassAPI(root).GetMassAttr().Get() > 0

    colliders = [p for p in stage.Traverse() if p.HasAPI(UsdPhysics.CollisionAPI)]
    assert len(colliders) == 1, "one box collider keeps 4096 clones cheap"
    for part in ("track_left", "track_right", "turret", "barrel"):
        assert stage.GetPrimAtPath(f"/Vehicle/{part}").IsValid()

    box = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_])
    r = box.ComputeWorldBound(root).ComputeAlignedRange()
    size = r.GetSize()
    assert size[0] > size[1], "longest axis is X, so the model's nose is +X"
    assert 8.0 < size[0] < 10.0 and 3.0 < size[1] < 4.0 and 2.0 < size[2] < 3.0
    assert abs(r.GetMin()[2]) < 0.05, "tracks sit on the ground plane, not sunk or floating"


def _fake_converted_mesh(path, extent, bow_axis=2):
    """Stand in for convert_asset.py's output: one collidable box mesh.

    The real BTR-80A cannot be committed (it is a gitignored CC-BY download), so
    the wrapper is tested against a mesh with the same awkward properties: not
    metre-scale, not centred, not resting on z=0, and carrying a collider the
    wrapper is supposed to switch off.
    """
    from pxr import Gf, Usd, UsdGeom, UsdPhysics

    stage = Usd.Stage.CreateNew(str(path))
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    root = UsdGeom.Xform.Define(stage, "/mesh")
    stage.SetDefaultPrim(root.GetPrim())
    m = UsdGeom.Mesh.Define(stage, "/mesh/geom")
    lo = Gf.Vec3f(100.0, 200.0, 300.0)                  # nowhere near the origin
    hi = Gf.Vec3f(*(lo[i] + extent[i] for i in range(3)))
    pts = [Gf.Vec3f(x, y, z) for x in (lo[0], hi[0]) for y in (lo[1], hi[1]) for z in (lo[2], hi[2])]
    m.CreatePointsAttr(pts)
    m.CreateFaceVertexCountsAttr([4])
    m.CreateFaceVertexIndicesAttr([0, 1, 3, 2])
    m.CreateExtentAttr([lo, hi])
    UsdPhysics.CollisionAPI.Apply(m.GetPrim())
    stage.GetRootLayer().Save()
    return path


def test_mesh_vehicle_wrapper_normalises_a_downloaded_model(tmp_path):
    """A fetched mesh becomes a rigid body posed the way the env assumes.

    Nose on +X, wheels on z=0, centred, scaled to real metres, and physics
    unchanged from the generated tank: exactly one box collider, authored mass.
    """
    from pxr import Usd, UsdGeom, UsdPhysics
    from vesper.worlds.vehicle import BTR80_HULL, BTR80_LENGTH_M, write_mesh_vehicle_usd

    # 780 long / 300 wide / 287 tall, bow on +Z: the BTR-80A's real proportions
    # and units, which are centimetre-ish and lie along the wrong axis.
    mesh = _fake_converted_mesh(tmp_path / "mesh.usd", (300.0, 287.0, 780.0))
    out = write_mesh_vehicle_usd(mesh, tmp_path / "veh.usd", length_m=BTR80_LENGTH_M,
                                 nose_yaw_deg=90.0, mass_kg=13_600.0, hull=BTR80_HULL)

    stage = Usd.Stage.Open(str(out))
    root = stage.GetDefaultPrim()
    assert root.HasAPI(UsdPhysics.RigidBodyAPI)
    assert UsdPhysics.MassAPI(root).GetMassAttr().Get() == 13_600.0

    live = [p for p in stage.Traverse()
            if p.HasAPI(UsdPhysics.CollisionAPI)
            and UsdPhysics.CollisionAPI(p).GetCollisionEnabledAttr().Get() is not False]
    assert len(live) == 1, "the referenced mesh's own colliders must stay off"
    assert live[0].GetPath().pathString == "/Vehicle/collision"

    r = UsdGeom.BBoxCache(Usd.TimeCode.Default(),
                          [UsdGeom.Tokens.default_]).ComputeWorldBound(root).ComputeAlignedRange()
    size, lo = r.GetSize(), r.GetMin()
    # the 780-unit axis was Z in the source; the yaw brings it onto X at 7.65 m
    assert abs(size[0] - BTR80_LENGTH_M) < 0.01
    assert abs(lo[2]) < 1e-4, "wheels rest on z=0"
    assert abs(lo[0] + size[0] / 2) < 0.01 and abs(lo[1] + size[1] / 2) < 0.01, "centred"


def test_mesh_vehicle_wrapper_can_drop_physics(tmp_path):
    """physics=False leaves geometry only.

    The synthetic-data renderer places vehicles by writing their USD transform
    every frame. A rigid body's pose belongs to PhysX the moment a physics
    scene exists, so those writes would move nothing and every frame would be
    an empty field -- the wrapper has to be able to omit the body entirely.
    """
    from pxr import Usd, UsdPhysics
    from vesper.worlds.vehicle import write_mesh_vehicle_usd

    mesh = _fake_converted_mesh(tmp_path / "mesh.usd", (300.0, 287.0, 780.0))
    out = write_mesh_vehicle_usd(mesh, tmp_path / "visual.usd", length_m=7.65,
                                 nose_yaw_deg=90.0, physics=False)
    stage = Usd.Stage.Open(str(out))
    assert not stage.GetDefaultPrim().HasAPI(UsdPhysics.RigidBodyAPI)
    assert not stage.GetPrimAtPath("/Vehicle/collision").IsValid()
    assert all(UsdPhysics.CollisionAPI(p).GetCollisionEnabledAttr().Get() is False
               for p in stage.Traverse() if p.HasAPI(UsdPhysics.CollisionAPI))
