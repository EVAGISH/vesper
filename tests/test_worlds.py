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


def test_vehicle_proxy_is_a_drivable_rigid_body(tmp_path):
    """The offline pursuit target: one rigid body, one cheap collider, nose +X.

    assets/ is gitignored, so this proxy is generated at run time -- a broken
    generator would otherwise only surface on the GPU box.
    """
    from pxr import Usd, UsdGeom, UsdPhysics
    from vesper.worlds.vehicle import write_vehicle_usd

    out = write_vehicle_usd(tmp_path / "utility_cart.usd")
    stage = Usd.Stage.Open(str(out))
    root = stage.GetDefaultPrim()

    assert UsdGeom.GetStageUpAxis(stage) == UsdGeom.Tokens.z
    assert UsdGeom.GetStageMetersPerUnit(stage) == 1.0
    assert root.HasAPI(UsdPhysics.RigidBodyAPI)
    assert UsdPhysics.MassAPI(root).GetMassAttr().Get() > 0

    colliders = [p for p in stage.Traverse() if p.HasAPI(UsdPhysics.CollisionAPI)]
    assert len(colliders) == 1, "one box collider keeps 4096 clones cheap"

    box = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_])
    r = box.ComputeWorldBound(root).ComputeAlignedRange()
    size = r.GetSize()
    assert size[0] > size[1], "longest axis is X, so the model's nose is +X"
    assert 2.0 < size[0] < 6.0 and 1.0 < size[1] < 3.0 and 1.0 < size[2] < 3.0
    assert abs(r.GetMin()[2]) < 0.05, "wheels sit on the ground plane, not sunk or floating"


def test_no_military_framing_in_source():
    """This is a civil autonomy project; the pursuit target is a ground vehicle."""
    import pathlib
    import re

    root = pathlib.Path(__file__).resolve().parents[1]
    banned = re.compile(
        r"\b(tank|armou?r(ed)?|military|battlefield|weapon|missile|warhead|munition|"
        r"combat|enemy|hostile)\b", re.I)
    offenders = []
    for path in list(root.glob("vesper/**/*.py")) + list(root.glob("scripts/*.py")):
        for n, line in enumerate(path.read_text().splitlines(), 1):
            if banned.search(line):
                offenders.append(f"{path.relative_to(root)}:{n}: {line.strip()}")
    assert not offenders, "military framing crept back in:\n" + "\n".join(offenders)
