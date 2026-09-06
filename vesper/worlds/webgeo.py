"""World geometry for the browser/three.js render lane (web_world.json).

Everything is derived from data the sim itself flies against, so the view and
the task agree: terrain + tree placement from <world>_map.npz, crisp building
footprints from osm.json with heights read back off the obstacle raster (the
truth the sensor raymarches). Built once, cached beside the assets, rebuilt
when the map is re-exported.

Shared by web/server/app.py's /api/world3d endpoint (live view) and
scripts/render_three_replay.py (offline replay export) — one source of truth.
"""
from __future__ import annotations

import json
from pathlib import Path


def build_world3d(world: str, assets: Path) -> dict:
    """The /api/world3d payload for `world` (assets/<world>/<world>_map.npz)."""
    import numpy as np

    d = assets / world
    npz = d / f"{world}_map.npz"
    m = np.load(npz)
    half = float(m["half_m"]); cell = float(m["cell"])
    ground = np.asarray(m["ground_z"], np.float32)
    obstacle = np.asarray(m["obstacle_z"], np.float32)
    canopy = np.asarray(m["canopy_z"], np.float32)
    trunks = np.asarray(m["trunks"], np.float32) if "trunks" in m.files else np.zeros_like(ground)
    n = ground.shape[0]

    stride = max(1, (n - 1) // 200)
    zg = ground[::stride, ::stride]

    def at(field, x, y):
        c = np.clip(((np.asarray(x) + half) / cell).round().astype(int), 0, n - 1)
        r = np.clip(((np.asarray(y) + half) / cell).round().astype(int), 0, n - 1)
        return field[r, c]

    # --- buildings: OSM footprints, heights from the sim's own obstacle raster
    buildings = []
    meta = json.loads((d / "dem_meta.json").read_text())
    la0, lo0, la1, lo1 = meta["bbox"]
    try:
        from vesper.worlds.geo import GeoSite, parse_osm
        site = GeoSite(lat0=(la0 + la1) / 2, lon0=(lo0 + lo1) / 2, half_m=half)
        osm = json.loads((d / "osm.json").read_text())
        rng = np.random.default_rng(0)
        for poly, tags in parse_osm(site, osm)["buildings"]:
            xy = np.asarray(poly.exterior.coords, np.float32)[:-1]
            if len(xy) < 3 or np.abs(xy).max() > half:
                continue
            cx, cy = float(xy[:, 0].mean()), float(xy[:, 1].mean())
            h = float((at(obstacle, xy[:, 0], xy[:, 1]) - at(ground, xy[:, 0], xy[:, 1])).max())
            if h < 2.0:                     # not in the raster (edge case): typical height
                h = float(rng.uniform(4.5, 8.0))
            buildings.append({"p": [[round(float(x), 1), round(float(y), 1)] for x, y in xy],
                              "h": round(h, 1), "z": round(float(at(ground, cx, cy)), 1)})
    except Exception as e:                                   # noqa: BLE001
        print(f"[world3d] building parse failed for {world}: {e}", flush=True)

    # --- trees: one entry per trunk, jittered deterministically inside its cell
    trees = []
    rng = np.random.default_rng(7)
    rows, cols = np.nonzero(trunks > 0)
    for r, c in zip(rows.tolist(), cols.tolist()):
        k = int(min(trunks[r, c], 3))
        x0 = c * cell - half
        y0 = r * cell - half
        hgt = float(np.clip(canopy[r, c] - ground[r, c], 4.0, 26.0))
        for _ in range(k):
            x = x0 + float(rng.uniform(-0.5, 0.5)) * cell
            y = y0 + float(rng.uniform(-0.5, 0.5)) * cell
            trees.append([round(x, 1), round(y, 1), round(float(at(ground, x, y)), 1),
                          round(hgt * float(rng.uniform(0.8, 1.15)), 1)])
    if len(trees) > 30000:
        idx = np.random.default_rng(1).choice(len(trees), 30000, replace=False)
        trees = [trees[i] for i in sorted(idx.tolist())]

    out = {"world": world, "half_m": half,
           "ground": f"/site/{world}/ground" if ((d / "ground.png").exists() or (d / "ground.jpg").exists()) else None,
           "terrain": {"n": int(zg.shape[0]), "step": cell * stride,
                       "z": [round(float(v), 1) for v in zg.reshape(-1)]},
           "buildings": buildings, "trees": trees}

    # --- static RF coverage (optional layer, absent on maps baked before it):
    # the full-AO signal field + station positions for a connectivity underlay
    if "comms" in m.files:
        comms = np.asarray(m["comms"], np.float32)
        cs = max(1, (n - 1) // 96)
        cz = comms[::cs, ::cs]
        stations = (np.asarray(m["comms_stations"], np.float32).tolist()
                    if "comms_stations" in m.files else [])
        jammers = (np.asarray(m["comms_jammers"], np.float32).tolist()
                   if "comms_jammers" in m.files else [])
        out["comms"] = {"n": int(cz.shape[0]), "step": cell * cs,
                        "v": [round(float(v), 2) for v in cz.reshape(-1)],
                        "stations": [[round(float(x), 1) for x in s] for s in stations],
                        "jammers": [[round(float(x), 1) for x in s] for s in jammers]}
    return out


def ensure_world3d(world: str, assets: Path) -> Path:
    """Return the cached web_world.json for `world`, (re)building it when the
    baked map is newer. Raises FileNotFoundError when the world has no map."""
    d = assets / world
    npz = d / f"{world}_map.npz"
    if not npz.is_file():
        raise FileNotFoundError(f"world has no baked map: {npz}")
    cache = d / "web_world.json"
    if not (cache.exists() and cache.stat().st_mtime >= npz.stat().st_mtime):
        cache.write_text(json.dumps(build_world3d(world, assets), separators=(",", ":")))
    return cache


def ensure_ground_jpg(world: str, assets: Path) -> Path | None:
    """Web-friendly downscale of the ground ortho (the source can be 30 MB),
    cached in the system temp dir — the same file /site/<world>/ground serves."""
    import tempfile

    src = assets / world / "ground.png"
    if not src.is_file():
        pre = assets / world / "ground.jpg"                 # mirrored preview of a box build
        return pre if pre.is_file() else None
    cache = Path(tempfile.gettempdir()) / f"vesper_{world}_ground.jpg"
    if not cache.exists() or cache.stat().st_mtime < src.stat().st_mtime:
        try:
            from PIL import Image
            Image.MAX_IMAGE_PIXELS = None
            img = Image.open(src).convert("RGB")
            img.thumbnail((4096, 4096), Image.BILINEAR)
            img.save(cache, "JPEG", quality=85)
        except ImportError:
            return src
    return cache
