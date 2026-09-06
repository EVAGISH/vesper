"""Seeded building layouts (pure python -- runs anywhere, CPU-testable).

A building is {"center": [north, east], "size": [d_north, d_east], "height": h}, meters,
matching the (north, east) frame of ScenarioSpec waypoints.
This is the placeholder generator until Pipeline A (aerial image -> footprints)
produces real geometry; the spec format is the same either way.
"""
import numpy as np


def sample_city_block(seed: int, corridor_len: float = 14.0, rows: int = 4) -> list[dict]:
    """Buildings flanking a corridor running north at east=0, corridor kept clear."""
    rng = np.random.default_rng(seed)
    buildings = []
    for i in range(rows):
        x = 3.0 + i * (corridor_len - 2.0) / max(rows - 1, 1)
        for side in (-1.0, 1.0):
            y = side * float(rng.uniform(2.6, 4.2))
            w, d = float(rng.uniform(2.0, 3.5)), float(rng.uniform(2.0, 3.5))
            h = float(rng.uniform(3.0, 9.0))
            buildings.append({"center": [round(x, 2), round(y, 2)],
                              "size": [round(w, 2), round(d, 2)],
                              "height": round(h, 2)})
    return buildings


def blocking_building(x: float, height: float = 6.0) -> dict:
    """A prism square in the corridor's path -- the deliberate-crash target."""
    return {"center": [x, 0.0], "size": [3.0, 3.0], "height": height}
