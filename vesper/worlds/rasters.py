"""Raster layers a search world needs beyond terrain and buildings.

scripts/export_world_map.py bakes a site USD (+ its cached OSM dump) into the
npz that vesper.worlds.heightmap.WorldMap loads on the GPU. The geometry-to-
raster steps live here, pure numpy + PIL, so they are unit-tested on a Mac
against synthetic inputs rather than only ever run against the real site.

All rasters share the terrain grid: (n, n), row indexes +y, column indexes +x,
both spanning [-half, half] in `cell` metre steps.
"""
from __future__ import annotations

import math

import numpy as np
from PIL import Image, ImageDraw

# Crown geometry per species: (crown radius / tree height). Spruces are spires,
# beeches and hawthorns are round. Used to splat canopy discs and crown solids
# -- the tree assets themselves are never loaded here.
SPECIES_H = {"Norway_Spruce": 18.0, "Lombardy_Poplar": 15.0, "Largetooth_Aspen": 9.0,
             "American_Beech": 7.0, "Hawthorn": 8.0, "Gray_Birch": 4.0}
SPECIES_CROWN = {"Norway_Spruce": 0.26, "Lombardy_Poplar": 0.16, "Largetooth_Aspen": 0.34,
                 "American_Beech": 0.46, "Hawthorn": 0.44, "Gray_Birch": 0.36}
DEFAULT_H, DEFAULT_CROWN = 10.0, 0.35

# Trunk and crown solids, as fractions of tree height. The same numbers author
# the colliders in vesper.worlds.geo, so PhysX and the map agree on what is hard.
TRUNK_R, TRUNK_TOP = 0.025, 0.60      # trunk radius, trunk collider top
CROWN_Z = 0.65                        # crown sphere centre

# Roads a forklift would use, with a paved width in metres. Footways, steps and
# paths are left out on purpose: a 1.2 m wide, 2.7 t forklift does not take the
# stairs, and on a campus the footway graph is denser than the road graph.
ROAD_WIDTH_M = {"motorway": 14.0, "trunk": 12.0, "primary": 11.0, "secondary": 10.0,
                "tertiary": 8.0, "unclassified": 7.0, "residential": 7.0, "service": 5.5,
                "living_street": 6.0, "track": 4.0, "tertiary_link": 7.0, "primary_link": 8.0,
                "secondary_link": 8.0, "trunk_link": 9.0}


def _to_px(xy, half, cell):
    return [((x + half) / cell, (y + half) / cell) for x, y in xy]


def rasterize_roads(roads, n, half, cell):
    """(mask uint8 [n,n], yaw float32 [n,n]) from [(LineString-like, tags)].

    The mask is every drivable road buffered to its class width; yaw is the
    road's direction in [0, pi) at each masked cell (a road has no forward, so
    the vehicle picks a sign). Anything not in ROAD_WIDTH_M is ignored.
    """
    mask = Image.new("L", (n, n), 0)
    yaw = Image.new("F", (n, n), 0.0)
    dm, dy = ImageDraw.Draw(mask), ImageDraw.Draw(yaw)
    for line, tags in roads:
        w = ROAD_WIDTH_M.get(tags.get("highway"))
        if w is None:
            continue
        pts = np.asarray(line.coords, dtype=float)
        wpx = max(1, int(round(w / cell)))
        for a, b in zip(pts[:-1], pts[1:]):
            d = b - a
            if np.hypot(*d) < 1e-6:
                continue
            ang = math.atan2(d[1], d[0]) % math.pi
            seg = _to_px([a, b], half, cell)
            dm.line(seg, fill=1, width=wpx)
            # +0.5: the "F" image stores 0 as "no road"; keep yaw strictly positive
            dy.line(seg, fill=float(ang + 0.5), width=wpx)
    m = np.asarray(mask, dtype=np.uint8).copy()
    y = np.asarray(yaw, dtype=np.float32).copy()
    y = np.where(m > 0, np.maximum(y - 0.5, 0.0), 0.0).astype(np.float32)
    return m, y


def chamfer_distance(mask: np.ndarray, cell: float) -> np.ndarray:
    """Approximate Euclidean distance (m) from every cell to the nearest set cell.

    3-4 chamfer metric via repeated vectorised passes: within ~8% of exact, no
    scipy, and fast enough for a 600x600 grid.
    """
    big = 1e9
    d = np.where(mask > 0, 0.0, big).astype(np.float64)
    if not (mask > 0).any():
        return np.full(mask.shape, big, np.float32)
    for _ in range(10000):
        prev = d
        c = d.copy()
        for axis, shift in ((0, 1), (0, -1), (1, 1), (1, -1)):
            r = np.roll(d, shift, axis=axis) + 3.0
            if shift == 1:
                (r[:1] if axis == 0 else r[:, :1])[...] = big
            else:
                (r[-1:] if axis == 0 else r[:, -1:])[...] = big
            c = np.minimum(c, r)
        for sy, sx in ((1, 1), (1, -1), (-1, 1), (-1, -1)):
            r = np.roll(np.roll(d, sy, axis=0), sx, axis=1) + 4.0
            (r[:1] if sy == 1 else r[-1:])[...] = big
            (r[:, :1] if sx == 1 else r[:, -1:])[...] = big
            c = np.minimum(c, r)
        d = c
        if np.array_equal(d, prev):
            break
    return (d / 3.0 * cell).astype(np.float32)


def parking_from_buildings(building_mask, drivable, cell, near_m=(1.0, 7.0)):
    """(parking uint8, park_yaw float32, bdist float32) -- the strip along a facade.

    A parked vehicle sits a few metres off a wall, nose along it. bdist is the
    chamfer distance to the nearest building cell; parking is the drivable band
    at near_m from one; park_yaw is the heading parallel to the wall, taken as
    perpendicular to the distance field's gradient.
    """
    bdist = chamfer_distance(building_mask, cell)
    band = (bdist >= near_m[0]) & (bdist <= near_m[1]) & (drivable > 0)
    # smooth the field a little before differentiating: chamfer is faceted
    sm = bdist.astype(np.float64)
    for _ in range(2):
        sm = (sm + np.roll(sm, 1, 0) + np.roll(sm, -1, 0) + np.roll(sm, 1, 1) + np.roll(sm, -1, 1)) / 5.0
    gy, gx = np.gradient(sm, cell)
    normal = np.arctan2(gy, gx)                       # away from the wall
    park_yaw = ((normal + math.pi / 2) % math.pi).astype(np.float32)
    return band.astype(np.uint8), np.where(band, park_yaw, 0.0).astype(np.float32), bdist


def splat_canopy(trees, ground, n, half, cell):
    """Canopy top and density rasters from tree discs [(x, y, h, crown_r)]."""
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


def splat_tree_solids(trees, ground, n, half, cell):
    """(tree_z float32, trunks float32) -- what a tree with colliders stops.

    tree_z is the top of the hard tree over each cell: the crown sphere's upper
    surface where the crown is, the trunk top over the trunk cell. trunks counts
    trunks per cell, for keeping vehicles out of the thick of a wood.
    """
    tree_z = ground.copy()
    trunks = np.zeros_like(ground)
    for x, y, h, crown_r in trees:
        ci = int((x + half) / cell + 0.5); ri = int((y + half) / cell + 0.5)
        if 0 <= ci < n and 0 <= ri < n:
            trunks[ri, ci] += 1.0
            tree_z[ri, ci] = max(tree_z[ri, ci], ground[ri, ci] + TRUNK_TOP * h)
        R = max(crown_r, cell * 0.5)
        zc = CROWN_Z * h
        c0 = max(0, int((x - R + half) / cell)); c1 = min(n - 1, int(math.ceil((x + R + half) / cell)))
        r0 = max(0, int((y - R + half) / cell)); r1 = min(n - 1, int(math.ceil((y + R + half) / cell)))
        if c1 < c0 or r1 < r0:
            continue
        cx = (np.arange(c0, c1 + 1) * cell) - half
        cy = (np.arange(r0, r1 + 1) * cell) - half
        d2 = (cx[None, :] - x) ** 2 + (cy[:, None] - y) ** 2
        inside = d2 <= R * R
        if not inside.any():
            continue
        cap = zc + np.sqrt(np.maximum(R * R - d2, 0.0))
        sub = tree_z[r0:r1 + 1, c0:c1 + 1]
        np.copyto(sub, np.maximum(sub, ground[r0:r1 + 1, c0:c1 + 1] + cap), where=inside)
    return tree_z.astype(np.float32), trunks.astype(np.float32)
