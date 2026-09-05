"""Export a geo world USD to the torch-side map the search task needs.

The search policy's reward, the vehicles' driver and the geometric sensor all
reason about terrain, buildings, tree cover and roads without loading USD:
where the ground is, what blocks line of sight, what attenuates it, where a
tank can drive and park. All of that is baked here, once, into rasters on
the terrain's own grid.

    python3 scripts/export_world_map.py assets/cornell/cornell.usd

Writes <world>_map.npz (float32 rasters) + <world>_map.json (metadata):
  ground_z    (n,n)  terrain height, world z
  obstacle_z  (n,n)  top of solid obstacles (buildings), world z; = ground_z where none
  canopy_z    (n,n)  top of tree canopy, world z; = ground_z where none
  canopy_d    (n,n)  canopy density 0..1 (how much of the cell a crown covers)
  drivable    (n,n)  uint8: gentle slope, no building, no water -- where a vehicle spawns
  concealed   (n,n)  uint8: drivable AND under canopy -- where a *hidden* vehicle spawns
  road        (n,n)  uint8: paved road a tank would use (from the site's OSM dump)
  road_yaw    (n,n)  road direction, radians in [0, pi)
  parking     (n,n)  uint8: drivable strip a few metres off a building wall
  park_yaw    (n,n)  heading parallel to that wall, radians in [0, pi)
  trunks      (n,n)  trunks per cell
  tree_z      (n,n)  top of the hard tree (conservative: the crown disc up to the
                     tree top) -- only written when the world's tree species carry
                     colliders; the contact sensor is what decides a real crash
  comms       (n,n)  effective RF connectivity 0..1: base-station signal
                     (terrain/building LOS x range falloff at a nominal drone
                     altitude) suppressed by any jammer with line of sight --
                     the usable pockets under jamming are the terrain shadows.
                     Static given fixed emitters + fixed terrain, so it is
                     baked once here; per-step is a lookup.
  comms_stations (s,3) antenna positions (x, y, z world) of the base stations
  jam         (n,n)  raw jammer interference 0..1 (only with --jammer)
  comms_jammers (j,3) jammer antenna positions (only with --jammer)

Only pxr + numpy + shapely, so it runs on the Mac; the rasters are what ship
to the GPU.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
from pxr import Usd, UsdGeom

from vesper.worlds.rasters import (DEFAULT_CROWN, DEFAULT_H, SPECIES_CROWN, SPECIES_H,
                                   parking_from_buildings, rasterize_roads, splat_canopy,
                                   splat_tree_solids)


def terrain_grid(stage):
    """Terrain mesh -> (ground_z [n,n], half_m, cell). The mesh is a regular grid
    laid out row-major in y, so the points reshape straight into a raster."""
    import math
    mesh = UsdGeom.Mesh(stage.GetPrimAtPath("/World/terrain"))
    if not mesh:
        raise SystemExit("no /World/terrain mesh in this stage")
    P = np.asarray(mesh.GetPointsAttr().Get(), dtype=np.float64)
    n = int(round(math.sqrt(len(P))))
    if n * n != len(P):
        raise SystemExit(f"terrain is not a square grid ({len(P)} points)")
    xs = P[:n, 0]
    half = float(abs(xs[0]))
    cell = float(xs[1] - xs[0])
    if not np.allclose(P[:, 0].reshape(n, n)[0], xs):
        raise SystemExit("terrain grid is not row-major in y")
    return P[:, 2].reshape(n, n).astype(np.float32), half, cell


def rasterize_mesh_tops(stage, path, n, half, cell, out):
    """Splat every face's max z over its xy footprint. Walls and roofs together
    give the building top over its plan; nothing else in the scene needs solids."""
    mesh = UsdGeom.Mesh(stage.GetPrimAtPath(path))
    if not mesh:
        return 0
    P = np.asarray(mesh.GetPointsAttr().Get(), dtype=np.float64)
    counts = np.asarray(mesh.GetFaceVertexCountsAttr().Get(), dtype=np.int64)
    idx = np.asarray(mesh.GetFaceVertexIndicesAttr().Get(), dtype=np.int64)
    starts = np.concatenate([[0], np.cumsum(counts)[:-1]])
    hit = 0
    for s, c in zip(starts, counts):
        f = P[idx[s:s + c]]
        x0, y0 = f[:, 0].min(), f[:, 1].min()
        x1, y1 = f[:, 0].max(), f[:, 1].max()
        zmax = float(f[:, 2].max())
        c0 = int(np.floor((x0 + half) / cell)); c1 = int(np.ceil((x1 + half) / cell))
        r0 = int(np.floor((y0 + half) / cell)); r1 = int(np.ceil((y1 + half) / cell))
        c0, c1 = max(0, c0), min(n - 1, c1)
        r0, r1 = max(0, r0), min(n - 1, r1)
        if c1 < c0 or r1 < r0:
            continue
        sub = out[r0:r1 + 1, c0:c1 + 1]
        np.maximum(sub, zmax, out=sub)
        hit += 1
    return hit


def rasterize_mesh_mask(stage, path, n, half, cell, out):
    mesh = UsdGeom.Mesh(stage.GetPrimAtPath(path))
    if not mesh:
        return 0
    P = np.asarray(mesh.GetPointsAttr().Get(), dtype=np.float64)
    counts = np.asarray(mesh.GetFaceVertexCountsAttr().Get(), dtype=np.int64)
    idx = np.asarray(mesh.GetFaceVertexIndicesAttr().Get(), dtype=np.int64)
    starts = np.concatenate([[0], np.cumsum(counts)[:-1]])
    for s, c in zip(starts, counts):
        f = P[idx[s:s + c]]
        c0 = max(0, int(np.floor((f[:, 0].min() + half) / cell)))
        c1 = min(n - 1, int(np.ceil((f[:, 0].max() + half) / cell)))
        r0 = max(0, int(np.floor((f[:, 1].min() + half) / cell)))
        r1 = min(n - 1, int(np.ceil((f[:, 1].max() + half) / cell)))
        if c1 >= c0 and r1 >= r0:
            out[r0:r1 + 1, c0:c1 + 1] = 1
    return int(out.sum())


def read_trees(stage):
    """[(x, y, height_m, crown_radius_m)] for every tree instance in /World/trees.

    Each tree is an Xform referencing a prepared species layer with a uniform
    scale of (per-tree variation x the species' cm->m conversion). The variation
    is what changes the tree's real height, so it is recovered as the tree's
    scale over the median scale of its own species -- no need to open the
    (instanceable, slow) species assets to measure them.
    """
    trees = stage.GetPrimAtPath("/World/trees")
    if not trees:
        return np.zeros((0, 4), np.float32), set()
    rows = []
    for prim in trees.GetChildren():
        x = UsdGeom.Xformable(prim)
        t = s = None
        for op in x.GetOrderedXformOps():
            if op.GetOpType() == UsdGeom.XformOp.TypeTranslate:
                t = op.Get()
            elif op.GetOpType() == UsdGeom.XformOp.TypeScale:
                s = op.Get()
        if t is None:
            continue
        name, asset = "?", None
        refs = prim.GetMetadata("references")
        if refs and refs.prependedItems:
            asset = str(refs.prependedItems[0].assetPath)
            name = Path(asset).stem
        rows.append((float(t[0]), float(t[1]), float(t[2]), float(s[0]) if s else 1.0, name, asset))
    if not rows:
        return np.zeros((0, 4), np.float32), set()
    names = np.array([r[4] for r in rows])
    scales = np.array([r[3] for r in rows])
    out = np.zeros((len(rows), 4), np.float32)
    for i, r in enumerate(rows):
        med = float(np.median(scales[names == r[4]]))
        rel = r[3] / med if med > 0 else 1.0
        h = SPECIES_H.get(r[4], DEFAULT_H) * rel
        out[i] = (r[0], r[1], h, SPECIES_CROWN.get(r[4], DEFAULT_CROWN) * h)
    return out, {r[5] for r in rows if r[5]}


def species_have_colliders(usd: Path, species_assets: set) -> bool:
    """True when every referenced species layer authors a trunk collider (see
    vesper.worlds.geo._prepare_species_usd). Then the trees are hard in PhysX
    and the map must say so too."""
    if not species_assets:
        return False
    for rel in species_assets:
        p = (usd.parent / rel).resolve()
        if not p.exists():
            return False
        st = Usd.Stage.Open(str(p))
        root = st.GetPrimAtPath("/Tree")
        if not root or not st.GetPrimAtPath("/Tree/trunk_col"):
            return False
        from pxr import UsdGeom, UsdPhysics
        if not any(pr.IsA(UsdGeom.Mesh) and pr.HasAPI(UsdPhysics.CollisionAPI) for pr in Usd.PrimRange(root)):
            return False
    return True


def emitter_field(solid, ground, n, half, cell, emitters_xy, mast_m, rx_agl_m,
                  range_m, graze_m, samples=48):
    """Received strength [0..1] per cell from the best of a set of emitters.

    First-order RF, same LOS idea as WorldMap.trace but in numpy over the whole
    grid at once: a cell receives when the ray from the emitter antenna
    (ground + mast) to the cell at a nominal drone altitude (ground + rx_agl)
    clears the solid raster, with a soft penalty when it grazes within
    `graze_m` of terrain/buildings, times a quadratic range falloff to
    `range_m`. Works for a friendly base station and for a hostile jammer
    alike -- an emitter is an emitter; only the sign of "being reached" flips.
    Returns (field [n,n] float32, emitters [s,3]).
    """
    ys = (np.arange(n, dtype=np.float64) * cell) - half           # row -> y
    xs = (np.arange(n, dtype=np.float64) * cell) - half           # col -> x
    rx_z = ground.astype(np.float64) + rx_agl_m
    t = np.linspace(0.0, 1.0, samples + 2)[1:-1]                  # exclude endpoints
    field = np.zeros((n, n), np.float64)
    emitters = []
    for sx, sy in emitters_xy:
        sc = int(np.clip(round((sx + half) / cell), 0, n - 1))
        sr = int(np.clip(round((sy + half) / cell), 0, n - 1))
        sz = float(ground[sr, sc]) + mast_m
        emitters.append((float(sx), float(sy), sz))
        for r0 in range(0, n, 64):                                # chunk rows for memory
            r1 = min(n, r0 + 64)
            X, Y = np.meshgrid(xs, ys[r0:r1])                     # [R,n]
            Z = rx_z[r0:r1]
            # march emitter -> cell: [R,n,S] sample points
            px = sx + (X[..., None] - sx) * t
            py = sy + (Y[..., None] - sy) * t
            pz = sz + (Z[..., None] - sz) * t
            ci = np.clip(((px + half) / cell).round().astype(np.int32), 0, n - 1)
            ri = np.clip(((py + half) / cell).round().astype(np.int32), 0, n - 1)
            clearance = (pz - solid[ri, ci]).min(axis=2)          # worst point on the ray
            los = np.clip(clearance / max(graze_m, 1e-3), 0.0, 1.0)
            d = np.sqrt((X - sx) ** 2 + (Y - sy) ** 2 + (Z - sz) ** 2)
            falloff = np.clip(1.0 - (d / range_m) ** 2, 0.0, 1.0)
            np.maximum(field[r0:r1], los * falloff, out=field[r0:r1])
    return field.astype(np.float32), np.asarray(emitters, np.float32)


def bake_comms(solid, ground, n, half, cell, stations_xy, mast_m, rx_agl_m,
               range_m, graze_m, jammers_xy=(), jam_mast_m=40.0, jam_range_m=None):
    """Effective connectivity [0..1]: base-station signal degraded by jamming.

    Jamming is the same physics inverted: wherever the jammer has line of sight
    the link is suppressed, and the usable pockets are exactly the terrain
    shadows -- valleys, the far side of hills, behind buildings -- that block
    the jammer but still see a station. effective = signal x (1 - jam), so a
    pocket only counts when both conditions hold at once.
    Returns (comms, signal, jam, stations, jammers).
    """
    sig, stations = emitter_field(solid, ground, n, half, cell, stations_xy,
                                  mast_m, rx_agl_m, range_m, graze_m)
    if jammers_xy:
        jam, jammers = emitter_field(solid, ground, n, half, cell, jammers_xy,
                                     jam_mast_m, rx_agl_m, jam_range_m or range_m, graze_m)
    else:
        jam = np.zeros_like(sig)
        jammers = np.zeros((0, 3), np.float32)
    return (sig * (1.0 - jam)).astype(np.float32), sig, jam, stations, jammers


def site_roads(usd: Path, osm_path: Path | None):
    """Roads from the site's cached OSM dump, in the world's local frame."""
    osm_path = osm_path or usd.with_name("osm.json")
    if not osm_path.exists():
        print(f"no OSM dump at {osm_path}: road and parking layers will be empty")
        return []
    meta = stage_site(usd)
    if meta is None:
        print("USD carries no vesper_site metadata: cannot place roads")
        return []
    from vesper.worlds.geo import GeoSite, parse_osm
    site = GeoSite(meta["lat0"], meta["lon0"], meta["half_m"])
    return parse_osm(site, json.loads(osm_path.read_text()))["roads"]


def stage_site(usd: Path):
    st = Usd.Stage.Open(str(usd))
    data = st.GetRootLayer().customLayerData or {}
    return json.loads(data["vesper_site"]) if "vesper_site" in data else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("usd")
    ap.add_argument("--out", default=None, help="default <usd stem>_map.npz beside the world")
    ap.add_argument("--osm", default=None, help="OSM dump (default osm.json beside the world)")
    ap.add_argument("--max-slope-deg", type=float, default=14.0, help="drivable slope limit")
    ap.add_argument("--conceal-density", type=float, default=0.45,
                    help="canopy density above which a cell counts as concealment")
    ap.add_argument("--comms-station", action="append", default=None, metavar="X,Y",
                    help="base-station position in world metres; repeatable "
                         "(default one station at the world centre)")
    ap.add_argument("--comms-mast", type=float, default=25.0, help="antenna height above ground (m)")
    ap.add_argument("--comms-rx-agl", type=float, default=15.0,
                    help="nominal drone altitude the field is evaluated at (m AGL)")
    ap.add_argument("--comms-range", type=float, default=None,
                    help="radio range (m); default 1.6 x the world half-extent")
    ap.add_argument("--comms-graze", type=float, default=4.0,
                    help="rays clearing terrain by less than this are attenuated (m)")
    ap.add_argument("--jammer", action="append", default=None, metavar="X,Y",
                    help="hostile jammer position in world metres; repeatable. With a "
                         "jammer the picture inverts: the AO is denied wherever the "
                         "jammer has line of sight, and the usable pockets are the "
                         "terrain shadows (valleys, behind hills/buildings)")
    ap.add_argument("--jammer-mast", type=float, default=40.0, help="jammer antenna height (m)")
    ap.add_argument("--jammer-range", type=float, default=None,
                    help="jammer effective range (m); default 2.5 x the world half-extent "
                         "(aggressive: LOS alone decides inside the AO)")
    a = ap.parse_args()

    usd = Path(a.usd).resolve()
    stage = Usd.Stage.Open(str(usd))
    ground, half, cell = terrain_grid(stage)
    n = ground.shape[0]
    print(f"terrain {n}x{n} @ {cell} m, extent +-{half} m, z [{ground.min():.1f}, {ground.max():.1f}]")

    obstacle = ground.copy()
    nb = rasterize_mesh_tops(stage, "/World/buildings", n, half, cell, obstacle)
    bh = obstacle - ground
    building = (bh > 0.5).astype(np.uint8)
    print(f"buildings: {nb} faces, footprint {building.sum()} cells, tallest {bh.max():.1f} m")

    water = np.zeros((n, n), np.uint8)
    nw = rasterize_mesh_mask(stage, "/World/water", n, half, cell, water)
    print(f"water: {nw} cells")

    trees, species = read_trees(stage)
    canopy, dens = splat_canopy(trees, ground, n, half, cell)
    print(f"trees: {len(trees)}, canopy cells {(dens > 0.05).sum()}, "
          f"tallest {(canopy - ground).max():.1f} m")
    solid_trees = species_have_colliders(usd, species)
    tree_z, trunks = splat_tree_solids(trees, ground, n, half, cell)
    print(f"tree colliders: {'yes -- trees are solid in the map' if solid_trees else 'no -- trees are visual only'}")

    gy, gx = np.gradient(ground.astype(np.float64), cell)
    slope = np.degrees(np.arctan(np.hypot(gx, gy))).astype(np.float32)
    drivable = ((slope < a.max_slope_deg) & (bh < 0.5) & (water == 0)).astype(np.uint8)
    # keep vehicles off the very edge of the world
    m = max(2, int(20.0 / cell))
    drivable[:m] = drivable[-m:] = drivable[:, :m] = drivable[:, -m:] = 0
    concealed = (drivable & (dens > a.conceal_density)).astype(np.uint8)
    if solid_trees:
        # under canopy but not in the thick of it: a crawling tank needs a gap
        concealed &= (trunks <= 1).astype(np.uint8)
    print(f"drivable {drivable.sum()} cells ({100*drivable.mean():.0f}%), "
          f"concealed {concealed.sum()} cells ({100*concealed.mean():.1f}%)")

    # comms: what stops a ray is what stops the drone -- buildings always, hard trees too
    solid = np.maximum(obstacle, tree_z) if solid_trees else obstacle
    stations_xy = ([tuple(float(v) for v in s.split(",")) for s in a.comms_station]
                   if a.comms_station else [(0.0, 0.0)])
    jammers_xy = ([tuple(float(v) for v in s.split(",")) for s in a.jammer]
                  if a.jammer else [])
    comms_range = a.comms_range if a.comms_range else 1.6 * half
    jam_range = a.jammer_range if a.jammer_range else 2.5 * half
    comms, sig, jam, comms_stations, comms_jammers = bake_comms(
        solid, ground, n, half, cell, stations_xy, a.comms_mast, a.comms_rx_agl,
        comms_range, a.comms_graze, jammers_xy, a.jammer_mast, jam_range)
    # 0.35 mirrors vesper.worlds.heightmap.LINK_THRESHOLD (kept torch-free here)
    connected_frac = float((comms >= 0.35).mean())
    print(f"comms: {len(stations_xy)} station(s) range {comms_range:.0f} m, "
          f"{len(jammers_xy)} jammer(s) range {jam_range:.0f} m, "
          f"connected {100*connected_frac:.1f}% of cells"
          + (f" (signal alone would give {100*(sig >= 0.35).mean():.1f}%)" if jammers_xy else ""))

    roads = site_roads(usd, Path(a.osm) if a.osm else None)
    road, road_yaw = rasterize_roads(roads, n, half, cell)
    road &= drivable
    road_yaw = np.where(road > 0, road_yaw, 0.0).astype(np.float32)
    parking, park_yaw, bdist = parking_from_buildings(building, drivable, cell)
    print(f"roads: {len(roads)} ways, {road.sum()} cells ({100*road.mean():.1f}%); "
          f"parking {parking.sum()} cells ({100*parking.mean():.1f}%)")

    out = Path(a.out) if a.out else usd.with_name(usd.stem + "_map.npz")
    layers = dict(half_m=np.float32(half), cell=np.float32(cell),
                  ground_z=ground, obstacle_z=obstacle, canopy_z=canopy,
                  canopy_d=dens.astype(np.float32), drivable=drivable, concealed=concealed,
                  slope_deg=slope, trees=trees, road=road, road_yaw=road_yaw,
                  parking=parking, park_yaw=park_yaw, bdist=bdist, trunks=trunks,
                  comms=comms, comms_stations=comms_stations)
    if len(comms_jammers):
        layers["jam"] = jam                                 # raw interference, for the UI
        layers["comms_jammers"] = comms_jammers
    if solid_trees:
        layers["tree_z"] = tree_z
    np.savez_compressed(out, **layers)
    meta = {"usd": os.path.relpath(usd, out.parent), "n": int(n), "cell": float(cell),
            "half_m": float(half), "trees": int(len(trees)), "tree_solids": bool(solid_trees),
            "building_cells": int(building.sum()), "drivable_cells": int(drivable.sum()),
            "concealed_cells": int(concealed.sum()), "road_cells": int(road.sum()),
            "parking_cells": int(parking.sum()),
            "z_range": [float(ground.min()), float(ground.max())],
            "max_slope_deg": a.max_slope_deg, "conceal_density": a.conceal_density,
            "comms": {"stations": [[float(v) for v in s] for s in comms_stations],
                      "jammers": [[float(v) for v in s] for s in comms_jammers],
                      "range_m": float(comms_range), "jam_range_m": float(jam_range),
                      "mast_m": a.comms_mast, "jammer_mast_m": a.jammer_mast,
                      "rx_agl_m": a.comms_rx_agl, "graze_m": a.comms_graze,
                      "connected_frac": round(connected_frac, 4)}}
    out.with_suffix(".json").write_text(json.dumps(meta, indent=1))
    print(f"wrote {out} ({out.stat().st_size/1e6:.1f} MB) and {out.with_suffix('.json')}")


if __name__ == "__main__":
    main()
