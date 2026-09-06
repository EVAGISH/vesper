"""Buildings for the geo world, in two shapes that cover a small post-Soviet town.

  house       small one-storey footprint: walls on the real footprint, a gable roof with
              an overhang on the footprint's minimum rotated rectangle, plaster/brick
              walls, slate or metal sheeting on the roof.
  block       everything taller: walls extruded from the footprint, one storey per
              3 m of wall so a panel-facade tile lines up window rows, flat bitumen roof.
  shed        garages, kiosks, sheds: low flat box, plain wall.
  industrial  factory/warehouse: tall flat box with corrugated cladding.

Classification is a hard-coded threshold on tags, then footprint area (Microsoft
footprints carry no tags): see classify().

Wall UVs are metres / tile span, with v = 0 at the wall base, so a texture tile that
depicts `span` metres of facade repeats correctly on every wall. Roof UVs run along
the ridge and down the slope for pitched roofs, and over x/y for flat ones.

Textures come from textures/<name>.jpg (generated once with scripts/gen_textures.py);
any that are missing fall back to the procedural tiles vesper.worlds.geo bakes.
"""
from __future__ import annotations

import math
from pathlib import Path

import mapbox_earcut as earcut
import numpy as np
from pxr import UsdGeom, UsdShade, Vt
from shapely.geometry import Polygon

from vesper.worlds.geo import _mesh, bake_facade_textures

TEX_DIR = Path(__file__).resolve().parents[2] / "textures"

# class -> (wall material names, roof material names)
STYLES = {
    "house":      (["house_wall", "house_wall_brick"], ["roof_slate", "roof_metal"]),
    "block":      (["soviet_panel", "soviet_panel_balcony"], ["roof_flat"]),
    "shed":       (["shed_brick"], ["roof_flat"]),
    "industrial": (["industrial_wall"], ["roof_flat"]),
}
PBR_DIR = TEX_DIR / "pbr"
# material name -> albedo tile (generated, with windows) or None to use the scan's own colour,
# metres one albedo tile covers, scan set for normal/roughness/AO, metres one scan tile covers
MATERIALS = {
    "house_wall":           ("house_wall", 3.5, "plaster", 1.5),
    "house_wall_brick":     ("house_wall_brick", 6.0, "brick", 3.0),
    "soviet_panel":         ("soviet_panel", 30.0, "concrete", 3.0),
    "soviet_panel_balcony": ("soviet_panel_balcony", 33.0, "concrete", 3.0),
    "shed_brick":           (None, 2.0, "brick_red", 2.0),
    "industrial_wall":      (None, 2.5, "industrial_wall", 2.5),
    "roof_flat":            (None, 4.0, "roof_flat", 4.0),
    "roof_slate":           (None, 1.5, "roof_slate", 1.5),
    "roof_metal":           (None, 1.5, "roof_metal", 1.5),
}
HOUSE_MAX_AREA = 320.0       # m² -- above this an untagged footprint is not a house
BLOCK_MIN_AREA = 600.0       # m² -- above this an untagged footprint is a 5-storey block
STOREY_M = 3.0
HOUSE_WALL_M = 3.0
ROOF_PITCH = math.tan(math.radians(30))
OVERHANG_M = 0.5

HOUSE_KINDS = {"house", "detached", "residential", "semidetached_house", "bungalow", "farm", "cabin", "hut"}
SHED_KINDS = {"garage", "garages", "shed", "kiosk", "roof", "carport", "greenhouse", "service", "ruins", "construction"}
IND_KINDS = {"industrial", "warehouse", "hangar", "factory", "manufacture"}
BLOCK_KINDS = {"apartments", "dormitory", "hotel", "hospital", "school", "university", "office", "public",
               "commercial", "retail", "supermarket", "civic", "kindergarten"}


def classify(tags: dict, poly: Polygon, rng) -> tuple[str, float]:
    """-> (class, height above base in metres)."""
    kind = tags.get("building", "yes")
    area = poly.area
    levels = None
    try:
        if "building:levels" in tags:
            levels = max(1, int(float(tags["building:levels"])))
    except ValueError:
        pass
    height = None
    try:
        if "height" in tags:
            height = float(str(tags["height"]).split()[0].replace(",", "."))
    except ValueError:
        pass
    if kind in SHED_KINDS or (kind == "yes" and area < 40):
        return "shed", height or float(rng.uniform(2.5, 3.2))
    if kind in IND_KINDS:
        return "industrial", height or float(rng.uniform(7.0, 11.0))
    if kind in HOUSE_KINDS and (levels or 1) <= 1 and area < 2 * HOUSE_MAX_AREA:
        return "house", HOUSE_WALL_M
    if kind in BLOCK_KINDS or (levels and levels >= 2) or (height and height > 5.0):
        if levels is None:
            levels = round(height / STOREY_M) if height else (5 if area > BLOCK_MIN_AREA else 2)
        return "block", levels * STOREY_M
    # untagged (Microsoft footprints, building=yes) or one-level: decide by size and shape
    rect = poly.minimum_rotated_rectangle
    short = min(math.dist(*rect.exterior.coords[0:2]), math.dist(*rect.exterior.coords[1:3]))
    if levels == 1:
        return ("house", HOUSE_WALL_M) if area < 2 * HOUSE_MAX_AREA else ("shed", 3.2)
    if area < HOUSE_MAX_AREA:
        return "house", HOUSE_WALL_M
    if short < 9.0:                                   # long thin untagged: garage rows, barns
        return "shed", float(rng.uniform(2.8, 3.4))
    if area < BLOCK_MIN_AREA:
        return "block", 2 * STOREY_M
    return "block", 5 * STOREY_M


def building_materials(out_dir: Path, rng) -> dict:
    """name -> {albedo, albedo_span, normal, rough, ao, detail_span}. Generated albedo tiles
    from textures/, scan maps from textures/pbr/<set>/ (scripts/fetch_pbr.py); anything
    missing falls back to the procedural tiles vesper.worlds.geo bakes."""
    fac, roofs = bake_facade_textures(out_dir, rng)          # always written: tests + fallback
    fallback = {"house_wall": fac[0], "house_wall_brick": fac[2], "soviet_panel": fac[3],
                "soviet_panel_balcony": fac[1], "shed_brick": fac[2], "industrial_wall": fac[3],
                "roof_slate": roofs[1], "roof_metal": roofs[0], "roof_flat": roofs[1]}
    mats = {}
    for name, (tile, span, scan, dspan) in MATERIALS.items():
        d = PBR_DIR / scan
        albedo = next((TEX_DIR / f"{tile}{e}" for e in (".jpg", ".png") if tile and (TEX_DIR / f"{tile}{e}").exists()), None)
        if albedo is None and (d / "color.jpg").exists():
            albedo, span = d / "color.jpg", dspan
        mats[name] = {"albedo": albedo or fallback[name], "albedo_span": span, "detail_span": dspan,
                      "normal": d / "normal.jpg" if (d / "normal.jpg").exists() else None,
                      "rough": d / "rough.jpg" if (d / "rough.jpg").exists() else None,
                      "ao": d / "ao.jpg" if (d / "ao.jpg").exists() else None}
    return mats


def pbr_material(stage, path: str, spec: dict, rel_dir: Path):
    """UsdPreviewSurface with albedo + normal + roughness + AO, each texture tiled at its own
    metre span via UsdTransform2d (mesh UVs are in metres)."""
    import os
    from pxr import Gf, Sdf
    mat = UsdShade.Material.Define(stage, path)
    st = UsdShade.Shader.Define(stage, path + "/st")
    st.CreateIdAttr("UsdPrimvarReader_float2")
    st.CreateInput("varname", Sdf.ValueTypeNames.Token).Set("st")
    pbr = UsdShade.Shader.Define(stage, path + "/pbr")
    pbr.CreateIdAttr("UsdPreviewSurface")
    pbr.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(0.0)
    pbr.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.85)

    def tex(name, file, span, srgb):
        xf = UsdShade.Shader.Define(stage, f"{path}/{name}_uv")
        xf.CreateIdAttr("UsdTransform2d")
        xf.CreateInput("in", Sdf.ValueTypeNames.Float2).ConnectToSource(st.ConnectableAPI(), "result")
        xf.CreateInput("scale", Sdf.ValueTypeNames.Float2).Set(Gf.Vec2f(1.0 / span, 1.0 / span))
        t = UsdShade.Shader.Define(stage, f"{path}/{name}")
        t.CreateIdAttr("UsdUVTexture")
        t.CreateInput("file", Sdf.ValueTypeNames.Asset).Set(os.path.relpath(file, rel_dir))
        t.CreateInput("sourceColorSpace", Sdf.ValueTypeNames.Token).Set("sRGB" if srgb else "raw")
        t.CreateInput("wrapS", Sdf.ValueTypeNames.Token).Set("repeat")
        t.CreateInput("wrapT", Sdf.ValueTypeNames.Token).Set("repeat")
        t.CreateInput("st", Sdf.ValueTypeNames.Float2).ConnectToSource(xf.ConnectableAPI(), "result")
        return t

    a = tex("albedo", spec["albedo"], spec["albedo_span"], True)
    pbr.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).ConnectToSource(a.ConnectableAPI(), "rgb")
    if spec.get("normal"):
        n = tex("normal", spec["normal"], spec["detail_span"], False)
        n.CreateInput("scale", Sdf.ValueTypeNames.Float4).Set(Gf.Vec4f(2, 2, 2, 1))
        n.CreateInput("bias", Sdf.ValueTypeNames.Float4).Set(Gf.Vec4f(-1, -1, -1, 0))
        pbr.CreateInput("normal", Sdf.ValueTypeNames.Normal3f).ConnectToSource(n.ConnectableAPI(), "rgb")
    if spec.get("rough"):
        r = tex("rough", spec["rough"], spec["detail_span"], False)
        pbr.GetInput("roughness").ConnectToSource(r.ConnectableAPI(), "r")
    if spec.get("ao"):
        o = tex("ao", spec["ao"], spec["detail_span"], False)
        pbr.CreateInput("occlusion", Sdf.ValueTypeNames.Float).ConnectToSource(o.ConnectableAPI(), "r")
    mat.CreateSurfaceOutput().ConnectToSource(pbr.ConnectableAPI(), "surface")
    return mat


def _gable_roof(poly: Polygon, wall_top: float):
    """Gable roof over the footprint's minimum rotated rectangle, with overhang.
    -> (slope quads [2][4 x (x,y,z)], gable triangles [2][3 x (x,y,z)], uv per slope quad)."""
    rect = np.asarray(poly.minimum_rotated_rectangle.exterior.coords)[:-1]
    e0 = rect[1] - rect[0]; e1 = rect[2] - rect[1]
    if np.linalg.norm(e0) < np.linalg.norm(e1):          # make e0 the long (ridge) axis
        rect = np.roll(rect, -1, axis=0); e0, e1 = e1, -e0
    L = float(np.linalg.norm(e0)); W = float(np.linalg.norm(e1))
    u = e0 / max(L, 1e-6); v = e1 / max(W, 1e-6)          # unit along ridge, across
    c = rect.mean(axis=0)
    hw = W / 2 + OVERHANG_M; hl = L / 2 + OVERHANG_M
    rise = (W / 2) * ROOF_PITCH
    ridge_z = wall_top + rise
    eave_z = wall_top - OVERHANG_M * ROOF_PITCH
    r0 = c - u * hl; r1 = c + u * hl                       # ridge ends (with overhang)
    quads = []
    for s in (+1, -1):
        a = r0 + v * s * hw; b = r1 + v * s * hw           # eave corners
        quads.append([(a[0], a[1], eave_z), (b[0], b[1], eave_z), (r1[0], r1[1], ridge_z), (r0[0], r0[1], ridge_z)])
    slope = math.hypot(hw, ridge_z - eave_z)
    uvs = [[(0, 0), (2 * hl, 0), (2 * hl, slope), (0, slope)]] * 2
    g0 = c - u * (L / 2); g1 = c + u * (L / 2)             # gable ends at the wall line
    gables = []
    for g in (g0, g1):
        p = g - v * (W / 2); q = g + v * (W / 2)
        gables.append([(p[0], p[1], wall_top), (q[0], q[1], wall_top), (g[0], g[1], ridge_z)])
    return quads, gables, uvs


def build_buildings(stage, terrain, osm, rng, mats: dict, rel_dir: Path):
    """All footprints into one mesh with a GeomSubset per material. -> (count, {index: height})."""
    pts, counts, idx, st = [], [], [], []
    faces = {name: [] for name in mats}
    heights = {}
    face_id = 0
    n_b = 0
    half = terrain.site.half_m

    def add_poly(verts, uvs, mat):
        nonlocal face_id
        b = len(pts); pts.extend(verts); st.extend(uvs)
        counts.append(len(verts)); idx.extend(range(b, b + len(verts)))
        faces[mat].append(face_id); face_id += 1

    def add_ring_cap(ring, z, mat, span):
        nonlocal face_id
        tris = earcut.triangulate_float32(ring.astype(np.float32), np.array([len(ring)], dtype=np.uint32))
        b = len(pts)
        pts.extend((x, y, z) for x, y in ring); st.extend((x / span, y / span) for x, y in ring)
        for t in range(0, len(tris), 3):
            counts.append(3); idx.extend(b + int(tris[k]) for k in (t, t + 1, t + 2))
            faces[mat].append(face_id); face_id += 1

    for bi, (poly, tags) in enumerate(osm["buildings"]):
        ring = np.asarray(poly.exterior.coords)[:-1]
        if len(ring) < 3 or not (np.abs(ring[:, 0]).max() < half and np.abs(ring[:, 1]).max() < half):
            continue
        if Polygon(ring).exterior.is_ccw is False:
            ring = ring[::-1]
        cls, h = classify(tags, poly, rng)
        walls, roofs = STYLES[cls]
        wall_mat = walls[bi % len(walls)]; roof_mat = roofs[bi % len(roofs)]
        wspan = rspan = 1.0                                  # UVs in metres; materials tile themselves
        gz = terrain.height(ring[:, 0], ring[:, 1])
        base = float(gz.min()) - 0.4
        top = base + 0.4 + h
        heights[bi] = h + (0.0 if cls != "house" else (poly.minimum_rotated_rectangle.length / 8) * ROOF_PITCH)
        along = 0.0
        for k in range(len(ring)):
            a, b = ring[k], ring[(k + 1) % len(ring)]
            L = float(np.linalg.norm(b - a))
            if L < 0.05:
                continue
            add_poly([(a[0], a[1], base), (b[0], b[1], base), (b[0], b[1], top), (a[0], a[1], top)],
                     [(along / wspan, 0.0), ((along + L) / wspan, 0.0),
                      ((along + L) / wspan, (top - base) / wspan), (along / wspan, (top - base) / wspan)], wall_mat)
            along += L
        if cls == "house":
            quads, gables, uvs = _gable_roof(poly, top)
            for q, uv in zip(quads, uvs):
                add_poly(q, [(x / rspan, y / rspan) for x, y in uv], roof_mat)
            for g in gables:
                gl = math.dist(g[0][:2], g[1][:2])
                add_poly(g, [(0, 0), (gl / wspan, 0), (gl / 2 / wspan, (g[2][2] - top) / wspan)], wall_mat)
            add_ring_cap(ring, top - 0.02, wall_mat, wspan)        # ceiling: hides the interior at overhang gaps
        else:
            add_ring_cap(ring, top, roof_mat, rspan)
        n_b += 1
    if not pts:
        return 0, heights
    mesh = _mesh(stage, "/World/buildings", pts, counts, idx, st=st, collide=True)
    for name, fids in faces.items():
        if not fids:
            continue
        sub = UsdGeom.Subset.CreateGeomSubset(mesh, name, UsdGeom.Tokens.face, Vt.IntArray(fids),
                                              UsdShade.Tokens.materialBind, UsdGeom.Tokens.partition)
        mat = pbr_material(stage, f"/World/Looks/{name}", mats[name], rel_dir)
        UsdShade.MaterialBindingAPI.Apply(sub.GetPrim()).Bind(mat)
    UsdGeom.Subset.SetFamilyType(mesh, "materialBind", UsdGeom.Tokens.partition)
    return n_b, heights
