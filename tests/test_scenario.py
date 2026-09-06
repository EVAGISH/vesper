from vesper.scenario import ScenarioSpec
from vesper.scenario.spec import square_scenario


def test_round_trip(tmp_path):
    spec = square_scenario(seed=7)
    p = spec.save(tmp_path / "s.json")
    loaded = ScenarioSpec.load(p)
    assert loaded == spec
    assert loaded.waypoints[0] == [4, 0, 3.0]


def test_unknown_fields_ignored():
    spec = ScenarioSpec.from_dict({"seed": 3, "not_a_field": 1})
    assert spec.seed == 3
