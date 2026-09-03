"""Export a geo world USD to the torch-side map the search task needs.

The search policy has to reason about terrain, buildings and tree cover without
loading USD: where the ground is, what blocks line of sight, what attenuates it,
and where a forklift can actually drive. All of that is baked here, once, into
four rasters on the terrain's own grid.

    python3 scripts/export_world_map.py assets/cornell/cornell.usd

Writes <world>_map.npz (float32 rasters) + <world>_map.json (metadata):
  ground_z    (n,n)  terrain height, world z
  obstacle_z  (n,n)  top of solid obstacles (buildings), world z; = ground_z where none
  canopy_z    (n,n)  top of tree canopy, world z; = ground_z where none
  canopy_d    (n,n)  canopy density 0..1 (how much of the cell a crown covers)
  drivable    (n,n)  uint8: gentle slope, no building, no water -- where a vehicle spawns
  concealed   (n,n)  uint8: drivable AND under canopy -- where a *hidden* vehicle spawns

Only pxr + numpy, so it runs on the Mac; the rasters are what ship to the GPU.
"""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path

import numpy as np
from pxr import Usd, UsdGeom

# Crown geometry per species: (crown radius / tree height). Spruces are spires,
# beeches and hawthorns are round. Used to splat canopy discs -- the tree assets
# themselves are never loaded (they are instanceable references and slow to open).
SPECIES_H = {"Norway_Spruce": 18.0, "Lombardy_Poplar": 15.0, "Largetooth_Aspen": 9.0,
             "American_Beech": 7.0, "Hawthorn": 8.0, "Gray_Birch": 4.0}
SPECIES_CROWN = {"Norway_Spruce": 0.26, "Lombardy_Poplar": 0.16, "Largetooth_Aspen": 0.34,
                 "American_Beech": 0.46, "Hawthorn": 0.44, "Gray_Birch": 0.36}
DEFAULT_H, DEFAULT_CROWN = 10.0, 0.35


def terrain_grid(stage):
    """Terrain mesh -> (ground_z [n,n], half_m, cell). The mesh is a regular grid
    laid out row-major in y, so the points reshape straight into a raster."""
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
        return np.zeros((0, 4), np.float32)
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
        name = "?"
        refs = prim.GetMetadata("references")
        if refs and refs.prependedItems:
            name = Path(str(refs.prependedItems[0].assetPath)).stem
        rows.append((float(t[0]), float(t[1]), float(t[2]), float(s[0]) if s else 1.0, name))
    if not rows:
        return np.zeros((0, 4), np.float32)
    names = np.array([r[4] for r in rows])
    scales = np.array([r[3] for r in rows])
    out = np.zeros((len(rows), 4), np.float32)
    for i, r in enumerate(rows):
        med = float(np.median(scales[names == r[4]]))
        rel = r[3] / med if med > 0 else 1.0
        h = SPECIES_H.get(r[4], DEFAULT_H) * rel
        out[i] = (r[0], r[1], h, SPECIES_CROWN.get(r[4], DEFAULT_CROWN) * h)
    return out


def splat_canopy(trees, ground, n, half, cell):
    """Canopy top and density rasters from tree discs."""
    top = ground.copy()
    dens = np.zeros_like(ground)
    for x, y, h, r in trees:
        r = max(r, cell)
        c0 = max(0, int((x - r + half) / cell)); c1 = min(n - 1, int(math.ceil((x + r + half) / cell)))
        r0 = max(0, int((y - r + half) / cell)); r1 = min(n - 1, int(math.ceil((y + r + half) / cell)))
        if c1 < c0 or r1 < r0:
            continue
        cx = (np.arange(c0, c1 + 1) * cell) - half
        cy = (np.arange(r0, r1 + 1) * cell) - half
        d2 = (cx[None, :] - x) ** 2 + (cy[:, None] - y) ** 2
        inside = d2 <= r * r
        if not inside.any():
            continue
        sub_top = top[r0:r1 + 1, c0:c1 + 1]
        # crown top follows the local ground, so a tree on a slope leans with it
        cand = ground[r0:r1 + 1, c0:c1 + 1] + h
        np.copyto(sub_top, np.maximum(sub_top, cand), where=inside)
        sub_d = dens[r0:r1 + 1, c0:c1 + 1]
        # thicker toward the trunk: 1 at the centre falling to 0.35 at the rim
        prof = np.clip(1.0 - 0.65 * np.sqrt(np.maximum(d2, 0)) / r, 0.0, 1.0)
        np.copyto(sub_d, np.minimum(sub_d + prof, 1.0), where=inside)
    return top, dens


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("usd")
    ap.add_argument("--out", default=None, help="default <usd stem>_map.npz beside the world")
    ap.add_argument("--max-slope-deg", type=float, default=14.0, help="drivable slope limit")
    ap.add_argument("--conceal-density", type=float, default=0.45,
                    help="canopy density above which a cell counts as concealment")
    a = ap.parse_args()

    usd = Path(a.usd).resolve()
    stage = Usd.Stage.Open(str(usd))
    ground, half, cell = terrain_grid(stage)
    n = ground.shape[0]
    print(f"terrain {n}x{n} @ {cell} m, extent +-{half} m, z [{ground.min():.1f}, {ground.max():.1f}]")

    obstacle = ground.copy()
    nb = rasterize_mesh_tops(stage, "/World/buildings", n, half, cell, obstacle)
    bh = obstacle - ground
    print(f"buildings: {nb} faces, footprint {(bh > 0.5).sum()} cells, tallest {bh.max():.1f} m")

    water = np.zeros((n, n), np.uint8)
    nw = rasterize_mesh_mask(stage, "/World/water", n, half, cell, water)
    print(f"water: {nw} cells")

    trees = read_trees(stage)
    canopy, dens = splat_canopy(trees, ground, n, half, cell)
    print(f"trees: {len(trees)}, canopy cells {(dens > 0.05).sum()}, "
          f"tallest {(canopy - ground).max():.1f} m")

    gy, gx = np.gradient(ground.astype(np.float64), cell)
    slope = np.degrees(np.arctan(np.hypot(gx, gy))).astype(np.float32)
    drivable = ((slope < a.max_slope_deg) & (bh < 0.5) & (water == 0)).astype(np.uint8)
    # keep vehicles off the very edge of the world
    m = max(2, int(20.0 / cell))
    drivable[:m] = drivable[-m:] = drivable[:, :m] = drivable[:, -m:] = 0
    concealed = (drivable & (dens > a.conceal_density)).astype(np.uint8)
    print(f"drivable {drivable.sum()} cells ({100*drivable.mean():.0f}%), "
          f"concealed {concealed.sum()} cells ({100*concealed.mean():.1f}%)")

    out = Path(a.out) if a.out else usd.with_name(usd.stem + "_map.npz")
    np.savez_compressed(out, half_m=np.float32(half), cell=np.float32(cell),
                        ground_z=ground, obstacle_z=obstacle, canopy_z=canopy,
                        canopy_d=dens.astype(np.float32), drivable=drivable, concealed=concealed,
                        slope_deg=slope, trees=trees)
    meta = {"usd": os.path.relpath(usd, out.parent), "n": int(n), "cell": float(cell),
            "half_m": float(half), "trees": int(len(trees)), "building_cells": int((bh > 0.5).sum()),
            "drivable_cells": int(drivable.sum()), "concealed_cells": int(concealed.sum()),
            "z_range": [float(ground.min()), float(ground.max())],
            "max_slope_deg": a.max_slope_deg, "conceal_density": a.conceal_density}
    out.with_suffix(".json").write_text(json.dumps(meta, indent=1))
    print(f"wrote {out} ({out.stat().st_size/1e6:.1f} MB) and {out.with_suffix('.json')}")


if __name__ == "__main__":
    main()
