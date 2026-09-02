from vesper.scenario.spec import city_scenario, crash_scenario
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
