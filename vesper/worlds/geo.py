"""Real-world site -> USD world (pure python: numpy, PIL, shapely, earcut, pxr).

Inputs (fetched by scripts/build_geo_world.py):
  dem.npy + dem_meta.json   elevation crop (EPSG:4326): USGS 3DEP 1 m in the US,
                            else Copernicus GLO-30 30 m
  naip.png                  (US only) 1 m true-colour orthophoto used as the ground albedo
  osm.json                  Overpass "out geom" dump: buildings, highways, landuse, natural, water
  assets/vegetation/Trees   NVIDIA tree USDs (cm units), plain-mesh species only

Output: <site>.usd with
  /World/terrain    grid mesh from the DEM, ground albedo baked from land cover + roads
  /World/buildings  OSM footprints extruded (height tags / levels / type defaults), facade + roof textures
  /World/water      flat water polygons
  /World/trees      PointInstancer: woodland polygons, tree rows, scrub, gardens
  /World/sun, /World/sky
  static triangle-mesh colliders on terrain and buildings.
Local frame: ENU meters, origin at (lat0, lon0), z = DEM height minus DEM height at the origin.
"""
from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass, field
from pathlib import Path

import mapbox_earcut as earcut
import numpy as np
from PIL import Image, ImageDraw
from pxr import Gf, Sdf, Usd, UsdGeom, UsdLux, UsdPhysics, UsdShade, Vt
from shapely.geometry import LineString, Point, Polygon
from shapely.strtree import STRtree

# ---------------------------------------------------------------- land cover palette
BASE_RGB = (118, 116, 74)          # dry steppe grass
LANDUSE_RGB = {
    "grass": (92, 124, 58), "grassland": (104, 122, 60), "meadow": (100, 124, 60), "park": (88, 120, 56),
    "recreation_ground": (96, 122, 60), "pitch": (90, 128, 60), "garden": (96, 118, 62), "flowerbed": (96, 110, 60),
    "farmland": (122, 104, 58), "farmyard": (120, 108, 70), "orchard": (96, 112, 56), "vineyard": (100, 110, 58),
    "residential": (110, 112, 72), "retail": (122, 120, 110), "commercial": (122, 120, 110),
    "industrial": (128, 126, 120), "railway": (110, 104, 96), "construction": (140, 130, 110),
    "cemetery": (98, 110, 68), "allotments": (108, 112, 66), "brownfield": (128, 118, 96), "greenfield": (110, 118, 66),
    "wood": (56, 70, 38), "forest": (56, 70, 38), "scrub": (84, 96, 52), "heath": (110, 112, 66), "wetland": (70, 92, 70),
    "sand": (176, 164, 130), "beach": (180, 170, 140), "water": (38, 58, 78), "reservoir": (38, 58, 78), "basin": (38, 58, 78),
}
ROAD_STYLE = {  # highway tag -> (width m, rgb)
    "motorway": (14, (58, 58, 60)), "trunk": (12, (58, 58, 60)), "primary": (10, (60, 60, 62)),
    "secondary": (8, (62, 62, 64)), "tertiary": (7, (66, 66, 68)), "residential": (5.5, (74, 74, 74)),
    "unclassified": (5, (76, 76, 76)), "living_street": (4.5, (80, 80, 80)), "service": (3.5, (92, 90, 86)),
    "track": (3, (112, 96, 70)), "footway": (1.6, (150, 140, 122)), "path": (1.2, (140, 128, 104)),
    "cycleway": (2, (110, 100, 90)), "pedestrian": (4, (150, 145, 135)), "steps": (1.5, (150, 140, 122)),
}
RAIL_STYLE = (4.0, (82, 76, 70))
WATER_KEYS = {"water", "reservoir", "basin"}
WOOD_KEYS = {"wood", "forest"}

# tree species: (file stem, target height m, share in woods, share in gardens/rows)
#
# Every entry MUST be a plain-mesh asset -- no PointInstancers of its own, or Isaac
# hangs a 100x-oversized branch cluster over the map (see _has_nested_instancer,
# which enforces this at build time). Of the 19 NVIDIA trees fetched only these 6
# qualify; Yellow_Pine / Black_Oak / Fraxinus / the maples / oaks / firs all nest.
# Target heights are kept near each asset's native height so the uniform scale
# stays ~1x and leaf detail doesn't stretch.
SPECIES = [
    ("Norway_Spruce", 18.0, 0.34, 0.05),      # native 17.7 m
    ("Lombardy_Poplar", 15.0, 0.20, 0.28),    # native 13.7 m
    ("Largetooth_Aspen", 9.0, 0.20, 0.14),    # native  7.9 m
    ("American_Beech", 7.0, 0.16, 0.15),      # native  6.0 m
    ("Hawthorn", 8.0, 0.06, 0.24),            # native  8.6 m
    ("Gray_Birch", 4.0, 0.04, 0.14),          # native  3.3 m -- scrub/understory
]


@dataclass
class GeoSite:
    lat0: float
    lon0: float
    half_m: float                      # world is a (2*half_m)^2 square
    res_m: float = 5.0                 # terrain grid spacing
    tex_px: int = 4096                 # ground albedo resolution
    seed: int = 0
    tree_spacing_m: float = 6.0        # woodland grid spacing (jittered)
    max_trees: int = 60000
    imagery_canopy: bool = True   # read unmapped tree cover out of the orthophoto
    leg_m: float = 250.0               # mission loop leg length (drives flight/video length)

    def to_local(self, lat, lon):
        x = (np.asarray(lon) - self.lon0) * 111320.0 * math.cos(math.radians(self.lat0))
        y = (np.asarray(lat) - self.lat0) * 110574.0
        return x, y


@dataclass
class BuildReport:
    terrain_verts: int = 0
    buildings: int = 0
    trees: int = 0
    water: int = 0
    z_range: list = field(default_factory=list)
    spawn_xy: list = field(default_factory=list)
    spawn_ground_z: float = 0.0
    waypoints: list = field(default_factory=list)
    takeoff_alt_m: float = 0.0
    usd: str = ""


# ---------------------------------------------------------------- terrain
class Terrain:
    def __init__(self, site: GeoSite, dem: np.ndarray, meta: dict):
        self.site = site
        a, b, c, d, e, f = meta["transform"]      # affine: lon = a*col + b*row + c ; lat = d*col + e*row + f
        n = int(round(2 * site.half_m / site.res_m)) + 1
        xs = np.linspace(-site.half_m, site.half_m, n)
        self.xs, self.n = xs, n
        X, Y = np.meshgrid(xs, xs)                 # row = y index, col = x index
        lon = site.lon0 + X / (111320.0 * math.cos(math.radians(site.lat0)))
        lat = site.lat0 + Y / 110574.0
        col = (lon - c) / a
        row = (lat - f) / e
        self.Z = self._bilinear(dem, col, row)
        z0 = float(self._bilinear(dem, np.array([[(site.lon0 - c) / a]]), np.array([[(site.lat0 - f) / e]]))[0, 0])
        self.Z = self.Z - z0
        self.z0_abs = z0

    @staticmethod
    def _bilinear(dem, col, row):
        H, W = dem.shape
        c0 = np.clip(np.floor(col).astype(int), 0, W - 2); r0 = np.clip(np.floor(row).astype(int), 0, H - 2)
        fc = np.clip(col - c0, 0, 1); fr = np.clip(row - r0, 0, 1)
        return ((1 - fr) * ((1 - fc) * dem[r0, c0] + fc * dem[r0, c0 + 1])
                + fr * ((1 - fc) * dem[r0 + 1, c0] + fc * dem[r0 + 1, c0 + 1]))

    def height(self, x, y):
        """Bilinear terrain height at local (x, y) arrays."""
        s = self.site
        gx = (np.asarray(x) + s.half_m) / s.res_m
        gy = (np.asarray(y) + s.half_m) / s.res_m
        return self._bilinear(self.Z, gx, gy)

    def mesh(self):
        n = self.n
        X, Y = np.meshgrid(self.xs, self.xs)
        pts = np.column_stack([X.ravel(), Y.ravel(), self.Z.ravel()])
        i = np.arange(n - 1); j = np.arange(n - 1)
        J, I = np.meshgrid(j, i)                   # I = row (y), J = col (x)
        v00 = (I * n + J).ravel(); v01 = v00 + 1; v10 = v00 + n; v11 = v10 + 1
        faces = np.column_stack([v00, v01, v11, v10]).ravel()   # CCW seen from +z
        st = np.column_stack([(X.ravel() + self.site.half_m) / (2 * self.site.half_m),
                              (Y.ravel() + self.site.half_m) / (2 * self.site.half_m)])
        return pts, faces, st


# ---------------------------------------------------------------- OSM parsing
def _ring_xy(site: GeoSite, geom):
    lat = np.array([p["lat"] for p in geom]); lon = np.array([p["lon"] for p in geom])
    x, y = site.to_local(lat, lon)
    return np.column_stack([x, y])


def parse_osm(site: GeoSite, osm: dict):
    """-> dict of lists: buildings [(Polygon, tags)], roads [(LineString, tags)], rails [LineString],
    areas [(Polygon, key)] for land cover, water [Polygon], tree_rows [LineString]."""
    out = {"buildings": [], "roads": [], "rails": [], "areas": [], "water": [], "tree_rows": []}
    for e in osm["elements"]:
        tags = e.get("tags", {})
        if e["type"] == "way" and "geometry" in e:
            xy = _ring_xy(site, e["geometry"])
            closed = len(xy) >= 4 and np.allclose(xy[0], xy[-1])
            if "building" in tags and closed:
                poly = Polygon(xy)
                if poly.is_valid and poly.area > 4:
                    out["buildings"].append((poly, tags))
            if "highway" in tags and len(xy) >= 2 and not (closed and "area" in tags):
                out["roads"].append((LineString(xy), tags))
            if tags.get("railway") in ("rail", "light_rail", "narrow_gauge") and len(xy) >= 2:
                out["rails"].append(LineString(xy))
            if tags.get("natural") == "tree_row" and len(xy) >= 2:
                out["tree_rows"].append(LineString(xy))
            if closed and "building" not in tags:
                key = tags.get("landuse") or tags.get("natural") or tags.get("leisure")
                if key in WATER_KEYS or tags.get("water"):
                    out["water"].append(Polygon(xy))
                elif key in LANDUSE_RGB:
                    p = Polygon(xy)
                    if p.is_valid:
                        out["areas"].append((p, key))
        elif e["type"] == "relation" and "members" in e:
            key = tags.get("landuse") or tags.get("natural") or tags.get("leisure")
            outers = [m for m in e["members"] if m.get("role") == "outer" and "geometry" in m]
            for m in outers:
                xy = _ring_xy(site, m["geometry"])
                if len(xy) < 4 or not np.allclose(xy[0], xy[-1]):
                    continue
                p = Polygon(xy)
                if not p.is_valid:
                    continue
                if "building" in tags:
                    out["buildings"].append((p, tags))
                elif key in WATER_KEYS or tags.get("water"):
                    out["water"].append(p)
                elif key in LANDUSE_RGB:
                    out["areas"].append((p, key))
    return out


def building_height(tags: dict, rng: np.random.Generator) -> float:
    try:
        if "height" in tags:
            return float(str(tags["height"]).split()[0].replace(",", "."))
    except ValueError:
        pass
    try:
        if "building:levels" in tags:
            return float(tags["building:levels"]) * 3.2 + 1.0
    except ValueError:
        pass
    kind = tags.get("building", "yes")
    if kind in ("house", "detached", "residential", "semidetached_house", "bungalow", "farm", "cabin"):
        return float(rng.uniform(4.5, 7.5))
    if kind in ("apartments", "dormitory", "hotel", "hospital", "school", "university", "office", "public"):
        return float(rng.uniform(12.0, 18.0))
    if kind in ("industrial", "warehouse", "hangar", "factory", "manufacture", "commercial", "retail", "supermarket"):
        return float(rng.uniform(7.0, 11.0))
    if kind in ("garage", "garages", "shed", "hut", "kiosk", "roof", "carport", "greenhouse", "service"):
        return float(rng.uniform(2.6, 3.4))
    return float(rng.uniform(4.5, 8.0))


# ---------------------------------------------------------------- textures
def _noise(size: int, cells: int, rng) -> np.ndarray:
    small = rng.random((cells, cells)).astype(np.float32)
    img = Image.fromarray((small * 255).astype(np.uint8)).resize((size, size), Image.BICUBIC)
    return np.asarray(img, dtype=np.float32) / 255.0


def bake_ground_texture(site: GeoSite, osm, out_png: Path, rng, ortho: Path | None = None) -> None:
    """Ground albedo. With `ortho` (a NAIP orthophoto covering exactly the site
    square) the real photograph is the albedo -- roads, field boundaries, mown
    grass, dirt tracks and tree shadows all come for free, which no painted
    land-cover palette can match. Without it, fall back to painting OSM polygons."""
    T = site.tex_px
    ppm = T / (2 * site.half_m)

    if ortho is not None and Path(ortho).exists():
        img = Image.open(ortho).convert("RGB").resize((T, T), Image.LANCZOS)
        arr = np.asarray(img, dtype=np.float32) / 255.0
        # NAIP is flown near noon and already carries baked-in sun; pull it back a
        # little so our own sun/sky lighting doesn't double up.
        arr = np.clip(arr * 0.92, 0, 1)
        Image.fromarray((arr * 255).astype(np.uint8)).save(out_png, optimize=False)
        return

    def w2p(xy):
        xy = np.asarray(xy)
        return [((x + site.half_m) * ppm, (site.half_m - y) * ppm) for x, y in xy]

    img = Image.new("RGB", (T, T), BASE_RGB)
    d = ImageDraw.Draw(img)
    # larger areas first so small ones win
    for poly, key in sorted(osm["areas"], key=lambda a: -a[0].area):
        d.polygon(w2p(np.asarray(poly.exterior.coords)), fill=LANDUSE_RGB[key])
        for hole in poly.interiors:
            d.polygon(w2p(np.asarray(hole.coords)), fill=BASE_RGB)
    for poly in osm["water"]:
        d.polygon(w2p(np.asarray(poly.exterior.coords)), fill=LANDUSE_RGB["water"])
    for line in osm["rails"]:
        d.line(w2p(np.asarray(line.coords)), fill=RAIL_STYLE[1], width=max(1, int(RAIL_STYLE[0] * ppm)), joint="curve")
    order = ["path", "footway", "steps", "cycleway", "track", "service", "living_street", "pedestrian",
             "unclassified", "residential", "tertiary", "secondary", "primary", "trunk", "motorway"]
    for cls in order:
        w, rgb = ROAD_STYLE[cls]
        for line, tags in osm["roads"]:
            if tags.get("highway") == cls:
                d.line(w2p(np.asarray(line.coords)), fill=rgb, width=max(1, int(w * ppm)), joint="curve")
    # building footprints: dirt apron
    for poly, _ in osm["buildings"]:
        d.polygon(w2p(np.asarray(poly.exterior.coords)), fill=(120, 110, 96))
    arr = np.asarray(img, dtype=np.float32) / 255.0
    n1 = _noise(T, 48, rng) * 0.35 + 0.825          # broad patches
    n2 = _noise(T, 512, rng) * 0.16 + 0.92          # fine grain
    arr = np.clip(arr * n1[..., None] * n2[..., None], 0, 1)
    Image.fromarray((arr * 255).astype(np.uint8)).save(out_png, optimize=False)


def bake_facade_textures(out_dir: Path, rng) -> tuple[list[Path], list[Path]]:
    """One storey per tile: plaster/brick with a window band. Returns (facades, roofs)."""
    facades, roofs = [], []
    tints = [(206, 196, 176), (188, 178, 160), (172, 150, 128), (200, 200, 196)]
    for i, tint in enumerate(tints):
        S = 512
        img = Image.new("RGB", (S, S), tint)
        d = ImageDraw.Draw(img)
        # two windows per 3 m tile, glass with frame
        for wx in (96, 320):
            d.rectangle([wx - 6, 150, wx + 102, 380], fill=(90, 90, 84))
            d.rectangle([wx, 156, wx + 96, 374], fill=(52, 66, 84))
            d.line([wx + 48, 156, wx + 48, 374], fill=(120, 120, 116), width=4)
            d.line([wx, 265, wx + 96, 265], fill=(120, 120, 116), width=4)
        d.rectangle([0, 470, S, S], fill=tuple(max(0, c - 40) for c in tint))      # plinth band
        arr = np.asarray(img, dtype=np.float32) / 255.0
        arr = np.clip(arr * (_noise(S, 64, rng) * 0.2 + 0.9)[..., None], 0, 1)
        p = out_dir / f"facade_{i}.png"; Image.fromarray((arr * 255).astype(np.uint8)).save(p); facades.append(p)
    for i, tint in enumerate([(112, 72, 58), (96, 96, 100), (120, 90, 60)]):
        S = 256
        img = Image.new("RGB", (S, S), tint); d = ImageDraw.Draw(img)
        for y in range(0, S, 32):
            d.line([0, y, S, y], fill=tuple(max(0, c - 30) for c in tint), width=3)
        arr = np.asarray(img, dtype=np.float32) / 255.0
        arr = np.clip(arr * (_noise(S, 32, rng) * 0.25 + 0.875)[..., None], 0, 1)
        p = out_dir / f"roof_{i}.png"; Image.fromarray((arr * 255).astype(np.uint8)).save(p); roofs.append(p)
    return facades, roofs


# ---------------------------------------------------------------- USD helpers
def _preview_material(stage, path: str, texture: Path | None, rel_dir: Path, rgb=(0.8, 0.8, 0.8),
                      roughness=0.9, metallic=0.0, uv_scale=1.0):
    mat = UsdShade.Material.Define(stage, path)
    sh = UsdShade.Shader.Define(stage, path + "/pbr")
    sh.CreateIdAttr("UsdPreviewSurface")
    sh.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(roughness)
    sh.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(metallic)
    if texture is None:
        sh.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*rgb))
    else:
        st = UsdShade.Shader.Define(stage, path + "/st")
        st.CreateIdAttr("UsdPrimvarReader_float2")
        st.CreateInput("varname", Sdf.ValueTypeNames.Token).Set("st")
        tex = UsdShade.Shader.Define(stage, path + "/tex")
        tex.CreateIdAttr("UsdUVTexture")
        tex.CreateInput("file", Sdf.ValueTypeNames.Asset).Set(str(Path(texture).relative_to(rel_dir)))
        tex.CreateInput("wrapS", Sdf.ValueTypeNames.Token).Set("repeat")
        tex.CreateInput("wrapT", Sdf.ValueTypeNames.Token).Set("repeat")
        if uv_scale != 1.0:
            xf = UsdShade.Shader.Define(stage, path + "/uvxf")
            xf.CreateIdAttr("UsdTransform2d")
            xf.CreateInput("in", Sdf.ValueTypeNames.Float2).ConnectToSource(st.ConnectableAPI(), "result")
            xf.CreateInput("scale", Sdf.ValueTypeNames.Float2).Set(Gf.Vec2f(uv_scale, uv_scale))
            tex.CreateInput("st", Sdf.ValueTypeNames.Float2).ConnectToSource(xf.ConnectableAPI(), "result")
        else:
            tex.CreateInput("st", Sdf.ValueTypeNames.Float2).ConnectToSource(st.ConnectableAPI(), "result")
        sh.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).ConnectToSource(tex.ConnectableAPI(), "rgb")
    mat.CreateSurfaceOutput().ConnectToSource(sh.ConnectableAPI(), "surface")
    return mat


def _mesh(stage, path, pts, counts, indices, st=None, collide=True):
    m = UsdGeom.Mesh.Define(stage, path)
    m.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(np.asarray(pts, dtype=np.float32)))
    m.CreateFaceVertexCountsAttr(Vt.IntArray.FromNumpy(np.asarray(counts, dtype=np.int32)))
    m.CreateFaceVertexIndicesAttr(Vt.IntArray.FromNumpy(np.asarray(indices, dtype=np.int32)))
    m.CreateSubdivisionSchemeAttr("none")
    m.CreateDoubleSidedAttr(True)
    ext = UsdGeom.PointBased(m).ComputeExtent(m.GetPointsAttr().Get())
    m.CreateExtentAttr(ext)
    if st is not None:
        pv = UsdGeom.PrimvarsAPI(m).CreatePrimvar("st", Sdf.ValueTypeNames.TexCoord2fArray,
                                                  UsdGeom.Tokens.vertex if len(st) == len(pts) else UsdGeom.Tokens.faceVarying)
        pv.Set(Vt.Vec2fArray.FromNumpy(np.asarray(st, dtype=np.float32)))
    if collide:
        UsdPhysics.CollisionAPI.Apply(m.GetPrim())
        UsdPhysics.MeshCollisionAPI.Apply(m.GetPrim()).CreateApproximationAttr().Set(UsdPhysics.Tokens.none)
    return m


# ---------------------------------------------------------------- buildings
def build_buildings(stage, terrain: Terrain, osm, rng, facades, roofs, rel_dir):
    """All footprints into one mesh with GeomSubsets per facade/roof material."""
    pts, counts, idx, st = [], [], [], []
    wall_faces = {i: [] for i in range(len(facades))}
    roof_faces = {i: [] for i in range(len(roofs))}
    face_id = 0
    n_b = 0
    half = terrain.site.half_m
    for bi, (poly, tags) in enumerate(osm["buildings"]):
        ring = np.asarray(poly.exterior.coords)[:-1]
        if len(ring) < 3:
            continue
        # Overpass returns everything in the padded query bbox, which reaches past
        # the terrain. A footprint outside the mesh sits on clamped edge height and
        # floats; drop anything not (mostly) inside the world square.
        if not (np.abs(ring[:, 0]).max() < half and np.abs(ring[:, 1]).max() < half):
            continue
        # CCW
        if Polygon(ring).exterior.is_ccw is False:
            ring = ring[::-1]
        h = building_height(tags, rng)
        gz = terrain.height(ring[:, 0], ring[:, 1])
        base = float(gz.min()) - 0.4
        top = float(gz.max()) + h if h < 4 else base + 0.4 + h
        top = max(top, base + 2.5)
        fi, ri = bi % len(facades), bi % len(roofs)
        # walls
        along = 0.0
        for k in range(len(ring)):
            a, b = ring[k], ring[(k + 1) % len(ring)]
            L = float(np.linalg.norm(b - a))
            if L < 0.05:
                continue
            base_i = len(pts)
            pts += [(a[0], a[1], base), (b[0], b[1], base), (b[0], b[1], top), (a[0], a[1], top)]
            st += [(along / 3.0, 0.0), ((along + L) / 3.0, 0.0), ((along + L) / 3.0, (top - base) / 3.0), (along / 3.0, (top - base) / 3.0)]
            counts.append(4); idx += [base_i, base_i + 1, base_i + 2, base_i + 3]
            wall_faces[fi].append(face_id); face_id += 1
            along += L
        # roof (earcut on the ring)
        verts = ring.astype(np.float32)
        tris = earcut.triangulate_float32(verts, np.array([len(verts)], dtype=np.uint32))
        base_i = len(pts)
        pts += [(x, y, top) for x, y in ring]
        st += [(x / 4.0, y / 4.0) for x, y in ring]
        for t in range(0, len(tris), 3):
            counts.append(3); idx += [base_i + int(tris[t]), base_i + int(tris[t + 1]), base_i + int(tris[t + 2])]
            roof_faces[ri].append(face_id); face_id += 1
        n_b += 1
    if not pts:
        return 0
    mesh = _mesh(stage, "/World/buildings", pts, counts, idx, st=st, collide=True)
    # per-vertex st here is faceVarying-sized? we appended one st per point -> vertex interpolation
    for i, p in enumerate(facades):
        if not wall_faces[i]:
            continue
        sub = UsdGeom.Subset.CreateGeomSubset(mesh, f"walls_{i}", UsdGeom.Tokens.face, Vt.IntArray(wall_faces[i]))
        mat = _preview_material(stage, f"/World/Looks/facade_{i}", p, rel_dir, roughness=0.85)
        UsdShade.MaterialBindingAPI.Apply(sub.GetPrim()).Bind(mat)
    for i, p in enumerate(roofs):
        if not roof_faces[i]:
            continue
        sub = UsdGeom.Subset.CreateGeomSubset(mesh, f"roofs_{i}", UsdGeom.Tokens.face, Vt.IntArray(roof_faces[i]))
        mat = _preview_material(stage, f"/World/Looks/roof_{i}", p, rel_dir, roughness=0.8)
        UsdShade.MaterialBindingAPI.Apply(sub.GetPrim()).Bind(mat)
    UsdGeom.Subset.SetFamilyType(mesh, "materialBind", UsdGeom.Tokens.partition)
    return n_b


# ---------------------------------------------------------------- water
def build_water(stage, terrain: Terrain, osm, rel_dir, max_drape_m: float = 12.0):
    """Flat water surfaces, clipped to the site square.

    Two things have to be enforced or a big lake wrecks the world. First, OSM
    water polygons routinely extend far outside the crop (Cayuga Lake ran 2.3 km
    past a 1 km world), so clip to the site. Second, water is level: sampling
    terrain height per vertex made the surface drape up the hillside as a giant
    sheet through the flight path. Each body gets one elevation, and a body whose
    terrain still varies more than `max_drape_m` under it is not a flat waterbody
    inside this crop (mis-tagged, or a shoreline the DEM disagrees with) and is
    dropped rather than drawn as a wall of water.
    """
    from shapely.geometry import box as _box
    site_box = _box(-terrain.site.half_m, -terrain.site.half_m, terrain.site.half_m, terrain.site.half_m)
    pts, counts, idx, kept = [], [], [], 0
    for poly in osm["water"]:
        clipped = poly.intersection(site_box)
        if clipped.is_empty:
            continue
        parts = list(getattr(clipped, "geoms", [clipped]))
        for part in parts:
            if part.geom_type != "Polygon" or part.area < 25.0:
                continue
            ring = np.asarray(part.exterior.coords)[:-1].astype(np.float32)
            if len(ring) < 3:
                continue
            zs = terrain.height(ring[:, 0], ring[:, 1])
            if float(zs.max() - zs.min()) > max_drape_m:
                continue                                   # would drape over terrain
            level = float(np.percentile(zs, 25)) + 0.12     # one flat level per body
            tris = earcut.triangulate_float32(ring, np.array([len(ring)], dtype=np.uint32))
            b = len(pts); pts += [(x, y, level) for x, y in ring]
            for t in range(0, len(tris), 3):
                counts.append(3); idx += [b + int(tris[t]), b + int(tris[t + 1]), b + int(tris[t + 2])]
            kept += 1
    if not pts:
        return 0
    m = _mesh(stage, "/World/water", pts, counts, idx, collide=False)
    mat = _preview_material(stage, "/World/Looks/water", None, rel_dir, rgb=(0.08, 0.16, 0.22), roughness=0.08)
    UsdShade.MaterialBindingAPI.Apply(m.GetPrim()).Bind(mat)
    return kept


# ---------------------------------------------------------------- trees
def _tree_native_height_m(usd_path: Path) -> float:
    st = Usd.Stage.Open(str(usd_path))
    cache = UsdGeom.XformCache()
    zmax, zmin = -1e9, 1e9
    for p in st.Traverse():
        if p.IsA(UsdGeom.Mesh):
            pts = UsdGeom.Mesh(p).GetPointsAttr().Get()
            if not pts:
                continue
            m = np.array(cache.GetLocalToWorldTransform(p)); w = np.asarray(pts) @ m[:3, :3] + m[3, :3]
            zmax, zmin = max(zmax, w[:, 2].max()), min(zmin, w[:, 2].min())
    return float((zmax - zmin) * UsdGeom.GetStageMetersPerUnit(st))


def _extent_from_descendants(prim) -> "Vt.Vec3fArray | None":
    """Bounding extent of every descendant mesh's points, expressed in `prim`'s own space."""
    cache = UsdGeom.XformCache()
    inv = np.array(cache.GetLocalToWorldTransform(prim)).T
    try:
        inv = np.linalg.inv(inv)
    except np.linalg.LinAlgError:
        return None
    lo = np.full(3, np.inf); hi = np.full(3, -np.inf)
    for d in Usd.PrimRange(prim):
        if d == prim or not d.IsA(UsdGeom.Mesh):
            continue
        pts = UsdGeom.Mesh(d).GetPointsAttr().Get()
        if not pts:
            continue
        m = np.array(cache.GetLocalToWorldTransform(d))
        w = np.asarray(pts) @ m[:3, :3] + m[3, :3]                  # -> world
        local = w @ inv[:3, :3].T + inv[:3, 3]                      # -> prim space
        lo = np.minimum(lo, local.min(axis=0)); hi = np.maximum(hi, local.max(axis=0))
    if not np.isfinite(lo).all():
        return None
    return Vt.Vec3fArray([Gf.Vec3f(*lo.astype(float)), Gf.Vec3f(*hi.astype(float))])


def _prepare_species_usd(src_usd: Path, out_dir: Path, name: str, target_h: float):
    """Prepare one species -> (usd_path, scale_to_metres).

    The layer holds the raw NVIDIA tree with its extents repaired; trees
    reference it instanceable, so that repair is authored once and shared by
    every copy (an instanceable prim cannot carry overs on its own descendants,
    which is why this needs its own layer).

    It deliberately does NOT author a scale op. A referencing prim that needs its
    own translate/rotate has to author an xformOpOrder, and that order replaces
    the referenced one wholesale rather than appending to it -- so a scale op
    here would simply be dropped, leaving the tree at native centimetre scale.
    The factor is returned instead and folded into each tree's own scale.
    """
    if _has_nested_instancer(src_usd):
        raise ValueError(
            f"tree species {name!r} contains nested PointInstancers; Isaac draws their branch "
            f"prototypes at the origin at native scale (the giant tree in the sky). "
            f"Use a plain-mesh species -- see _has_nested_instancer()."
        )
    out = Path(out_dir) / "species" / f"{name}.usd"
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists():
        out.unlink()
    stage = Usd.Stage.CreateNew(str(out))
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    root = UsdGeom.Xform.Define(stage, "/Tree")
    stage.SetDefaultPrim(root.GetPrim())
    native = _tree_native_height_m(src_usd)
    s = target_h / max(native, 0.1) * UsdGeom.GetStageMetersPerUnit(Usd.Stage.Open(str(src_usd)))
    root.GetPrim().GetReferences().AddReference(os.path.relpath(src_usd, out.parent))
    for prim in Usd.PrimRange(root.GetPrim()):
        if not prim.IsA(UsdGeom.Mesh):
            continue
        mesh = UsdGeom.Mesh(prim)
        pts = mesh.GetPointsAttr().Get()
        if pts:
            mesh.CreateExtentAttr(UsdGeom.PointBased(mesh).ComputeExtent(pts))
        else:
            # A Mesh with no points of its own but real geometry underneath (the spruce's
            # "sprucetrunk" parents 854k points). USD treats a Mesh as a leaf boundable, so
            # this extent bounds everything below it: zeroing it culls the whole tree and the
            # asset's shipped value is garbage. Author the union of the descendants.
            ext = _extent_from_descendants(prim)
            if ext is not None:
                mesh.CreateExtentAttr(ext)
    stage.GetRootLayer().Save()
    return out, s


def _has_nested_instancer(usd_path: Path) -> bool:
    """True if a tree asset contains PointInstancers of its own (branch clusters).

    Isaac's renderer draws PointInstancer prototype prims directly, in addition to
    instancing them. We hide our top-level prototypes by parking them under an
    Xform 10 km down, which works because instances inherit only the prototype
    ROOT's local ops. That trick does NOT reach instancers nested *inside* a
    prototype: Isaac draws those branch prototypes without the ancestor
    transforms, so they land at the origin at native (centimetre) scale -- a
    ~100x-oversized branch cluster hanging over the entire map, with a matching
    kilometre-wide shadow. Such assets are unusable as prototypes; pick a
    species whose geometry is plain meshes.
    """
    st = Usd.Stage.Open(str(usd_path))
    return any(p.GetTypeName() == "PointInstancer" for p in st.Traverse())


def _sample_polygon(poly: Polygon, spacing: float, rng) -> np.ndarray:
    minx, miny, maxx, maxy = poly.bounds
    xs = np.arange(minx, maxx, spacing); ys = np.arange(miny, maxy, spacing)
    if len(xs) == 0 or len(ys) == 0:
        return np.zeros((0, 2))
    X, Y = np.meshgrid(xs, ys)
    P = np.column_stack([X.ravel(), Y.ravel()]) + rng.uniform(-0.45, 0.45, (X.size, 2)) * spacing
    import shapely
    keep = shapely.contains_xy(poly, P[:, 0], P[:, 1])
    return P[keep]


def canopy_from_imagery(img_path, site: GeoSite, rng, spacing_m=7.0, tex_thr=3.2,
                        lum_thr=105.0, cluster_m=8.0, cover_thr=0.42):
    """Tree positions [M,2] read out of the site's own orthophoto.

    OSM tags only a fraction of a real campus's trees -- Cornell's 1.2 km square
    has 431 of them, all in one corner, which is not a place anything can hide.
    The orthophoto shows the rest. This crop is leaf-off, so greenness (ExG) is
    useless; what separates woodland from grass and asphalt there is texture:
    bare crowns over leaf litter are rough and dark, mown grass and pavement are
    smooth. Threshold on roughness + darkness, then require the hits to form
    clusters at crown scale so isolated speckles on parked cars and roof edges
    drop out, and scatter trees on a jittered grid inside what survives.

    Returned points are raw: the caller still applies the building/road/water
    exclusions every other tree source goes through.
    """
    from PIL import Image as _Im
    _Im.MAX_IMAGE_PIXELS = None

    def box_mean(a, k):
        """k x k running mean via an integral image (no scipy dependency)."""
        k = max(1, int(k) | 1)
        pad = k // 2
        b = np.pad(a, pad + 1, mode="edge")
        S = b.cumsum(0).cumsum(1)
        S = np.pad(S, ((1, 0), (1, 0)))
        h, w = a.shape
        r0 = np.arange(h)[:, None]; c0 = np.arange(w)[None, :]
        r1, c1 = r0 + k, c0 + k
        tot = S[r1, c1] - S[r0, c1] - S[r1, c0] + S[r0, c0]
        return tot / (k * k)
    img = _Im.open(img_path).convert("RGB")
    W = int(2 * site.half_m)                                   # work at 1 m/px
    lum = np.asarray(img.convert("L").resize((W * 4, W * 4), _Im.BILINEAR), dtype=np.float32)
    tex = lum.reshape(W, 4, W, 4).std(axis=(1, 3))             # roughness at ~0.5 m scale
    grey = np.asarray(img.resize((W, W), _Im.BILINEAR), dtype=np.float32).mean(-1)
    raw = ((tex > tex_thr) & (grey < lum_thr)).astype(np.float32)
    cover = box_mean(raw, max(3, int(cluster_m)))              # crown-scale support
    mask = cover > cover_thr

    step = max(1.0, spacing_m)
    gx = np.arange(step / 2, 2 * site.half_m, step)
    X, Y = np.meshgrid(gx, gx)
    P = np.stack([X.ravel(), Y.ravel()], 1)
    P = P + rng.uniform(-step * 0.4, step * 0.4, P.shape)
    col = np.clip(P[:, 0].astype(int), 0, W - 1)
    row = np.clip(P[:, 1].astype(int), 0, W - 1)
    keep = mask[row, col]
    # image row 0 is the north edge; world y grows north
    out = np.stack([P[keep, 0] - site.half_m, site.half_m - P[keep, 1]], 1)
    print(f"  imagery canopy: {100 * mask.mean():.0f}% of the site, {len(out)} candidate trees")
    return out


def build_trees(stage, site: GeoSite, terrain: Terrain, osm, rng, veg_dir: Path, rel_dir: Path,
                spawn_xy, clear_radius=15.0, canopy_xy=None):
    """Instanced trees over woodland polygons, scrub, tree rows, gardens, and
    (when given) canopy read from the site's orthophoto."""
    exclusion = [p.buffer(2.5) for p, _ in osm["buildings"]]
    exclusion += [l.buffer(ROAD_STYLE.get(t.get("highway"), (4, None))[0] / 2 + 1.0) for l, t in osm["roads"]]
    exclusion += [p for p in osm["water"]]
    tree_excl = STRtree(exclusion) if exclusion else None

    def allowed(P):
        if len(P) == 0:
            return P
        keep = np.ones(len(P), dtype=bool)
        if tree_excl is not None:
            hits = tree_excl.query(np.array([Point(x, y) for x, y in P]), predicate="intersects")
            keep[np.unique(hits[0])] = False
        d = np.hypot(P[:, 0] - spawn_xy[0], P[:, 1] - spawn_xy[1])
        keep &= d > clear_radius
        keep &= (np.abs(P[:, 0]) < site.half_m - 3) & (np.abs(P[:, 1]) < site.half_m - 3)
        return P[keep]

    woods = [p for p, k in osm["areas"] if k in WOOD_KEYS]
    scrub = [p for p, k in osm["areas"] if k == "scrub"]
    gardens = [p for p, k in osm["areas"] if k in ("residential", "allotments", "garden", "park", "cemetery", "orchard")]
    pos, kind = [], []           # kind: 0 wood, 1 row/garden, 2 scrub(small)
    for p in woods:
        P = allowed(_sample_polygon(p, site.tree_spacing_m, rng)); pos.append(P); kind.append(np.zeros(len(P), int))
    for p in scrub:
        P = allowed(_sample_polygon(p, site.tree_spacing_m * 2.2, rng)); pos.append(P); kind.append(np.full(len(P), 2))
    for p in gardens:
        P = allowed(_sample_polygon(p, 24.0, rng)); pos.append(P); kind.append(np.ones(len(P), int))
    for line in osm["tree_rows"]:
        n = max(1, int(line.length / 7.0))
        P = np.array([line.interpolate(i * line.length / n).coords[0] for i in range(n + 1)])
        P = allowed(P + rng.uniform(-0.8, 0.8, P.shape)); pos.append(P); kind.append(np.ones(len(P), int))
    if canopy_xy is not None and len(canopy_xy):
        P = allowed(np.asarray(canopy_xy, dtype=float)); pos.append(P); kind.append(np.zeros(len(P), int))
    if not pos:
        return 0, np.zeros((0, 2)), np.zeros(0)
    P = np.vstack(pos); K = np.concatenate(kind)
    if len(P) > site.max_trees:
        sel = rng.choice(len(P), site.max_trees, replace=False); P, K = P[sel], K[sel]
    n = len(P)
    if n == 0:
        return 0, np.zeros((0, 2)), np.zeros(0)
    # species per tree
    wood_w = np.array([s[2] for s in SPECIES]); wood_w /= wood_w.sum()
    row_w = np.array([s[3] for s in SPECIES]); row_w /= row_w.sum()
    proto_idx = np.where(K == 0, rng.choice(len(SPECIES), n, p=wood_w), rng.choice(len(SPECIES), n, p=row_w))
    scale = rng.uniform(0.85, 1.25, n) * np.where(K == 2, 0.45, 1.0)
    yaw = rng.uniform(0, 2 * np.pi, n)
    z = terrain.height(P[:, 0], P[:, 1]) - 0.05

    # One Xform per tree holding an *instanceable* reference to a prepared species
    # asset -- deliberately NOT a PointInstancer.
    #
    # Isaac draws PointInstancer prototype prims as ordinary geometry in addition to
    # instancing them, and it does so ignoring the prototype root's own xformOps AND
    # the transforms of its ancestors. So neither scaling the prototype nor parking it
    # under an Xform 10 km down removes it: the asset lands at the world origin at its
    # native (centimetre) scale, which is the ~1.7 km pine that hung over every frame.
    # Native USD scenegraph instancing has no such problem -- it is the same mechanism
    # Isaac Lab uses to clone environments -- and gives the same memory win, since all
    # trees of a species share one prototype.
    prepared = {name: _prepare_species_usd(veg_dir / "Trees" / f"{name}.usd", rel_dir, name, target_h)
                for name, target_h, _, _ in SPECIES}          # name -> (usd, scale_to_metres)
    UsdGeom.Scope.Define(stage, "/World/trees")
    for i in range(n):
        name = SPECIES[int(proto_idx[i])][0]
        species_usd, species_scale = prepared[name]
        xf = UsdGeom.Xform.Define(stage, f"/World/trees/t{i:05d}")
        xf.AddTranslateOp().Set(Gf.Vec3d(float(P[i, 0]), float(P[i, 1]), float(z[i])))
        xf.AddRotateZOp().Set(float(np.degrees(yaw[i])))
        s = float(scale[i]) * species_scale        # per-tree variation x cm->m and target height
        xf.AddScaleOp().Set(Gf.Vec3f(s, s, s))
        prim = xf.GetPrim()
        prim.GetReferences().AddReference(os.path.relpath(species_usd, rel_dir))
        prim.SetInstanceable(True)

    heights = np.array([SPECIES[i][1] for i in proto_idx]) * scale
    return n, P, heights


# ---------------------------------------------------------------- spawn + mission planning
def obstacle_height_map(site: GeoSite, terrain: Terrain, osm, cell=10.0, tree_xy=None, tree_h=None):
    """Max obstacle top (terrain + building/tree height) per cell, world-relative z.
    Trees come from the actual instances when given (gardens, rows), else from polygons."""
    n = int(2 * site.half_m / cell) + 1
    xs = np.linspace(-site.half_m, site.half_m, n)
    X, Y = np.meshgrid(xs, xs)
    H = terrain.height(X, Y).copy()
    ppm = 1.0 / cell
    img = Image.new("F", (n, n), 0.0); d = ImageDraw.Draw(img)

    def w2p(xy):
        return [((x + site.half_m) * ppm, (y + site.half_m) * ppm) for x, y in xy]
    for p, k in osm["areas"]:
        if k in WOOD_KEYS:
            d.polygon(w2p(np.asarray(p.exterior.coords)), fill=22.0)
        elif k == "scrub":
            d.polygon(w2p(np.asarray(p.exterior.coords)), fill=6.0)
    rng = np.random.default_rng(0)
    for p, t in osm["buildings"]:
        d.polygon(w2p(np.asarray(p.exterior.coords)), fill=float(building_height(t, rng)) + 1.0)
    for line in osm["tree_rows"]:
        d.line(w2p(np.asarray(line.coords)), fill=20.0, width=2)
    obst = np.asarray(img).copy()
    if tree_xy is not None and len(tree_xy):
        i = np.clip(((tree_xy[:, 1] + site.half_m) * ppm).astype(int), 0, n - 1)
        j = np.clip(((tree_xy[:, 0] + site.half_m) * ppm).astype(int), 0, n - 1)
        np.maximum.at(obst, (i, j), tree_h)
    return xs, H + obst


def _snap_to_open_ground(xy, osm, clear_m: float = 16.0, search_r: float = 220.0):
    """Nearest point to `xy` that is clear of buildings, roads and water.

    A hand-picked spawn is easy to place on a rooftop or inside a facade (which
    puts the cameras inside the building and every frame is wallpaper). The
    automatic picker already enforces this; an override has to be held to the
    same standard rather than trusted.
    """
    blockers = [p.buffer(clear_m) for p, _ in osm["buildings"]]
    blockers += [l.buffer(8.0) for l, _ in osm["roads"]]
    blockers += [p.buffer(12.0) for p in osm["water"]]
    if not blockers:
        return (round(xy[0], 1), round(xy[1], 1))
    tree = STRtree(blockers)
    x0, y0 = xy
    if not len(tree.query(Point(x0, y0), predicate="intersects")):
        return (round(x0, 1), round(y0, 1))
    for r in np.arange(5.0, search_r, 5.0):
        for ang in np.linspace(0, 2 * np.pi, max(12, int(r / 3)), endpoint=False):
            x, y = x0 + r * math.cos(ang), y0 + r * math.sin(ang)
            if not len(tree.query(Point(x, y), predicate="intersects")):
                print(f"  spawn ({x0:.0f},{y0:.0f}) was blocked (building/road/water); "
                      f"snapped {r:.0f} m to ({x:.0f},{y:.0f})")
                return (round(x, 1), round(y, 1))
    raise ValueError(f"no open ground within {search_r} m of {xy}")


def choose_spawn(site: GeoSite, terrain: Terrain, osm, rng, search_r=600.0):
    """Open ground, >=18 m from buildings/roads/woods/water, 25-90 m from the nearest wood, nearest to origin."""
    blockers = [p.buffer(18) for p, _ in osm["buildings"]] + [l.buffer(10) for l, _ in osm["roads"]]
    blockers += [p.buffer(15) for p, k in osm["areas"] if k in WOOD_KEYS or k == "scrub"] + [p.buffer(20) for p in osm["water"]]
    tree = STRtree(blockers)
    woods = [p for p, k in osm["areas"] if k in WOOD_KEYS]
    wood_tree = STRtree(woods) if woods else None
    best = None
    for r in np.arange(0, search_r, 10.0):
        for ang in np.linspace(0, 2 * np.pi, max(8, int(r / 6)), endpoint=False):
            x, y = r * math.cos(ang), r * math.sin(ang)
            pt = Point(x, y)
            if len(tree.query(pt, predicate="intersects")):
                continue
            dw = min(pt.distance(woods[i]) for i in wood_tree.query(pt.buffer(150))) if wood_tree and len(wood_tree.query(pt.buffer(150))) else 999.0
            if not (25.0 <= dw <= 90.0):
                continue
            score = abs(dw - 45.0) + 0.05 * r
            if best is None or score < best[0]:
                best = (score, x, y)
        if best is not None and r > 200:
            break
    if best is None:
        return (0.0, 0.0)
    return (round(best[1], 1), round(best[2], 1))


def plan_loop(site: GeoSite, terrain: Terrain, osm, spawn_xy, leg_m=250.0, clearance=20.0, min_alt=25.0,
              tree_xy=None, tree_h=None):
    xs, top = obstacle_height_map(site, terrain, osm, tree_xy=tree_xy, tree_h=tree_h)
    cell = xs[1] - xs[0]
    sx, sy = spawn_xy
    gz = float(terrain.height(np.array([sx]), np.array([sy]))[0])

    def top_at(x, y):
        i = int(np.clip((y + site.half_m) / cell, 0, len(xs) - 1)); j = int(np.clip((x + site.half_m) / cell, 0, len(xs) - 1))
        win = top[max(0, i - 2):i + 3, max(0, j - 2):j + 3]
        return float(win.max())
    s = leg_m / 2
    loop = [(s, 0), (s, s), (-s, s), (-s, -s), (s, -s), (0, 0)]        # (east, north) relative to spawn
    loop = [(e, n) for e, n in loop if abs(sx + e) < site.half_m - 30 and abs(sy + n) < site.half_m - 30] or [(0, 0)]
    pts = [(0, 0)] + loop
    need = []
    for (e0, n0), (e1, n1) in zip(pts, pts[1:]):
        k = int(np.hypot(e1 - e0, n1 - n0) / 5) + 1
        need.append(max(top_at(sx + e0 + (e1 - e0) * t, sy + n0 + (n1 - n0) * t) for t in np.linspace(0, 1, k)) - gz + clearance)
    alts = [round(max(min_alt, need[k], need[k + 1] if k + 1 < len(need) else 0), 1) for k in range(len(loop))]
    wps = [[float(n), float(e), a] for (e, n), a in zip(loop, alts)]
    return wps, round(max(min_alt, need[0]), 1), gz


# ---------------------------------------------------------------- top level
def build_site(site: GeoSite, data_dir: Path, veg_dir: Path, out_usd: Path,
               spawn_override=None) -> BuildReport:
    rng = np.random.default_rng(site.seed)
    data_dir, veg_dir, out_usd = Path(data_dir), Path(veg_dir).resolve(), Path(out_usd).resolve()
    out_dir = out_usd.parent; out_dir.mkdir(parents=True, exist_ok=True)
    dem = np.load(data_dir / "dem.npy"); meta = json.loads((data_dir / "dem_meta.json").read_text())
    osm = parse_osm(site, json.loads((data_dir / "osm.json").read_text()))
    terrain = Terrain(site, dem, meta)
    rep = BuildReport()

    stage = Usd.Stage.CreateNew(str(out_usd))
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z); UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    world = UsdGeom.Xform.Define(stage, "/World"); stage.SetDefaultPrim(world.GetPrim())
    UsdGeom.Scope.Define(stage, "/World/Looks")

    # terrain
    pts, faces, st = terrain.mesh()
    tex = out_dir / "ground.png"
    naip = data_dir / "naip.png"
    bake_ground_texture(site, osm, tex, rng, ortho=naip if naip.exists() else None)
    tmesh = _mesh(stage, "/World/terrain", pts, np.full(len(faces) // 4, 4), faces, st=st, collide=True)
    mat = _preview_material(stage, "/World/Looks/ground", tex, out_dir, roughness=0.95)
    UsdShade.MaterialBindingAPI.Apply(tmesh.GetPrim()).Bind(mat)
    rep.terrain_verts = len(pts); rep.z_range = [round(float(pts[:, 2].min()), 1), round(float(pts[:, 2].max()), 1)]

    facades, roofs = bake_facade_textures(out_dir, rng)
    rep.buildings = build_buildings(stage, terrain, osm, rng, facades, roofs, out_dir)
    rep.water = build_water(stage, terrain, osm, out_dir)

    if spawn_override is not None:
        spawn = _snap_to_open_ground(tuple(spawn_override), osm)
    else:
        spawn = choose_spawn(site, terrain, osm, rng)
    ortho = next((data_dir / f for f in ("naip.png", "imagery.png") if (data_dir / f).exists()), None)
    canopy_xy = canopy_from_imagery(ortho, site, rng) if (ortho and site.imagery_canopy) else None
    rep.trees, tree_xy, tree_h = build_trees(stage, site, terrain, osm, rng, veg_dir, out_dir, spawn,
                                             canopy_xy=canopy_xy)

    sun = UsdLux.DistantLight.Define(stage, "/World/sun")
    sun.CreateIntensityAttr(1700.0); sun.CreateAngleAttr(0.53); sun.CreateColorAttr(Gf.Vec3f(1.0, 0.97, 0.92))
    sxf = UsdGeom.Xformable(sun); sxf.AddRotateXOp().Set(-50.0); sxf.AddRotateZOp().Set(140.0)   # ~40 deg elevation, from the SE
    sky = UsdLux.DomeLight.Define(stage, "/World/sky")
    sky.CreateIntensityAttr(480.0); sky.CreateColorAttr(Gf.Vec3f(0.42, 0.60, 0.90))

    wps, tk, gz = plan_loop(site, terrain, osm, spawn, leg_m=site.leg_m, tree_xy=tree_xy, tree_h=tree_h)
    rep.spawn_xy = list(spawn); rep.spawn_ground_z = round(gz, 3); rep.waypoints = wps; rep.takeoff_alt_m = tk
    stage.GetRootLayer().customLayerData = {"vesper_site": json.dumps({"lat0": site.lat0, "lon0": site.lon0, "half_m": site.half_m})}
    stage.GetRootLayer().Save()
    rep.usd = str(out_usd)
    return rep
