"""Seeded scenario variants for stress-test sweeps (pure python)."""
import numpy as np

from .spec import ScenarioSpec


def sample_variants(base: ScenarioSpec, n: int, seed: int | None = None) -> list[ScenarioSpec]:
    """n seeded copies of `base` varying spawn, wind, visibility, sensor noise.
    Variant i is reproducible from (base, seed, i)."""
    seed = base.seed if seed is None else seed
    out = []
    for i in range(n):
        rng = np.random.default_rng([seed, i])
        v = ScenarioSpec.from_dict(base.to_dict())
        v.seed = int(rng.integers(0, 2**31))
        v.spawn_east = round(float(rng.uniform(-1.5, 1.5)), 3)
        v.wind_speed_ms = round(float(rng.uniform(0.0, 8.0)), 2)
        v.wind_dir_deg = round(float(rng.uniform(0, 360)), 1)
        v.visibility_m = round(float(rng.choice([50.0, 120.0, 300.0, 1000.0]))
                               * float(rng.uniform(0.8, 1.2)), 1)
        v.range_noise_std = round(float(rng.uniform(0.02, 0.4)), 3)
        out.append(v)
    return out
