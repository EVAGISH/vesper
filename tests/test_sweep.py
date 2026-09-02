from vesper.eval import bin_success, findings
from vesper.scenario.randomizer import sample_variants
from vesper.scenario.spec import city_scenario


def test_variants_deterministic_and_varied():
    base = city_scenario(seed=3)
    a, b = sample_variants(base, 8), sample_variants(base, 8)
    assert [v.to_dict() for v in a] == [v.to_dict() for v in b]
    assert len({v.wind_speed_ms for v in a}) > 4
    assert all(v.waypoints == base.waypoints for v in a)


def test_findings_flag_worst_dimension():
    results = [{"wind_speed_ms": w, "visibility_m": 500, "range_noise_std": 0.1,
                "success": 1.0 if w < 5 else 0.0,
                "failure": None if w < 5 else "collision"}
               for w in [1, 1, 3, 3, 5, 6, 7, 7]]
    f = findings(results)
    assert any("wind_speed_ms" in line for line in f)
    rows = bin_success(results, "wind_speed_ms")
    assert rows[-1]["success"] == 0.0 and rows[0]["success"] == 1.0
