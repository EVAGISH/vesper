"""Emit a scenario JSON:  python3 scripts/make_scenario.py {square|city|crash|baylands} <seed> <out>"""
import sys

from vesper.scenario.spec import baylands_scenario, city_scenario, crash_scenario, square_scenario

kind, seed, out = sys.argv[1], int(sys.argv[2]), sys.argv[3]
spec = {"square": square_scenario, "city": city_scenario, "crash": crash_scenario, "baylands": baylands_scenario}[kind](seed)
print(spec.save(out))
