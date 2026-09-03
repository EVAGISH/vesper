"""Emit a scenario JSON.

    python3 scripts/make_scenario.py {square|city|crash} <seed> <out>
    python3 scripts/make_scenario.py imported <seed> <out> --usd assets/x/x_world.usda --spawn X Y --ground Z
"""
import argparse

from vesper.scenario.spec import city_scenario, crash_scenario, imported_scenario, square_scenario

ap = argparse.ArgumentParser()
ap.add_argument("kind", choices=["square", "city", "crash", "imported"])
ap.add_argument("seed", type=int)
ap.add_argument("out")
ap.add_argument("--usd", help="imported: wrapper USD from scripts/prepare_world.py (repo-relative)")
ap.add_argument("--spawn", type=float, nargs=2, default=(0.0, 0.0))
ap.add_argument("--ground", type=float, default=0.0)
ap.add_argument("--alt", type=float, default=15.0)
ap.add_argument("--loop", type=float, default=30.0)
a = ap.parse_args()

if a.kind == "imported":
    spec = imported_scenario(a.usd, tuple(a.spawn), a.ground, a.seed, a.alt, a.loop)
else:
    spec = {"square": square_scenario, "city": city_scenario, "crash": crash_scenario}[a.kind](a.seed)
print(spec.save(a.out))
