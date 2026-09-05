"""Operator-drawn zones on a site: where drones launch, where targets are off limits.

A zones file sits next to a world (assets/<site>/zones.json, or a repo-root
<site>_zones.json for a tracked default) and holds polygons in site metres:

    {"launch": [[x, y], ...],                 one polygon: drones spawn inside it
     "safe":   [[[x, y], ...], ...]}          any number: a vehicle inside one is
                                              protected -- no sighting bonus, no
                                              hit reward

Polygons rasterise onto the world map's grid so the GPU side asks a mask, not
a geometry library. Pure numpy + PIL, no shapely.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


@dataclass
class Zones:
    launch: list | None = None                  # [[x, y], ...] or None = anywhere
    safe: list = field(default_factory=list)    # [[[x, y], ...], ...]

    @classmethod
    def load(cls, path) -> "Zones":
        d = json.loads(Path(path).read_text())
        return cls(launch=d.get("launch"), safe=list(d.get("safe") or []))

    def save(self, path) -> Path:
        p = Path(path)
        p.write_text(json.dumps({"launch": self.launch, "safe": self.safe}, indent=1))
        return p

    def masks(self, n: int, half: float, cell: float):
        """(launch uint8 [n,n], safe uint8 [n,n]) on the map grid (row +y, col +x)."""
        launch = (rasterize([self.launch], n, half, cell) if self.launch
                  else np.ones((n, n), np.uint8))
        safe = rasterize(self.safe, n, half, cell) if self.safe else np.zeros((n, n), np.uint8)
        return launch, safe


def rasterize(polys, n: int, half: float, cell: float) -> np.ndarray:
    img = Image.new("L", (n, n), 0)
    d = ImageDraw.Draw(img)
    for poly in polys:
        pts = [((x + half) / cell, (y + half) / cell) for x, y in poly]
        if len(pts) >= 3:
            d.polygon(pts, fill=1, outline=1)
    return np.asarray(img, dtype=np.uint8).copy()


def find_zones(world_map_path, repo_root=None) -> Path | None:
    """assets/<site>/zones.json beside the map, else <site>_zones.json at the repo root."""
    m = Path(world_map_path)
    beside = m.with_name("zones.json")
    if beside.exists():
        return beside
    site = m.stem.replace("_map", "")
    root = Path(repo_root) if repo_root else m.resolve().parents[2]
    tracked = root / f"{site}_zones.json"
    return tracked if tracked.exists() else None
