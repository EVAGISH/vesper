"""Build a real-world site into a USD world + flight scenario (runs on the Mac, no Isaac).

    python3 scripts/build_geo_world.py cornell --lat 42.4475 --lon -76.4831 --half-km 1.0

Fetches (once, cached under assets/<site>/) an elevation crop -- USGS 3DEP 1 m
lidar plus NAIP 1 m aerial imagery inside the US, Copernicus 30 m elsewhere -- and an
Overpass dump of OSM buildings/roads/landuse/water, then writes
assets/<site>/<site>.usd (+ textures) and <site>0.json (ScenarioSpec) with an
automatically chosen open spawn next to the woods and a canopy-safe loop.
"""
import argparse
import json
import math
import time
from pathlib import Path

import numpy as np

from vesper.scenario.spec import imported_scenario
from vesper.worlds.geo import GeoSite, build_site

ap = argparse.ArgumentParser()
ap.add_argument("site")
ap.add_argument("--lat", type=float, required=True)
ap.add_argument("--lon", type=float, required=True)
ap.add_argument("--half-km", type=float, default=1.5)
ap.add_argument("--res", type=float, default=5.0, help="terrain grid spacing (m)")
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--variant", type=int, default=0,
                help="reseed only the scatter (trees, building heights): many worlds from one fetch")
ap.add_argument("--no-tree-colliders", action="store_true", help="visual-only trees (the old behaviour)")
ap.add_argument("--leg-m", type=float, default=90.0,
                help="mission loop leg length (m); ~90 gives a ~30 s flight video")
ap.add_argument("--max-sim-s", type=float, default=75.0)
ap.add_argument("--source", choices=["auto", "us", "global"], default="auto",
                help="auto picks USGS 3DEP+NAIP inside the US, Copernicus+painted elsewhere")
ap.add_argument("--dsm", default=None, metavar="GEOTIFF",
                help="ingest a provided elevation GeoTIFF (any CRS) instead of fetching. "
                     "A high-res Digital Surface Model (Maxar/Airbus, ~0.5 m) is the "
                     "high-fidelity path for places with no open lidar (e.g. Ukraine).")
ap.add_argument("--ortho", default=None, metavar="IMG",
                help="ingest a provided ortho image (GeoTIFF or PNG covering the site) as the ground albedo")
ap.add_argument("--surface-model", action="store_true",
                help="treat --dsm as a surface model (buildings+trees in the elevation): "
                     "build them into the terrain mesh, skip OSM prisms and stamped trees")
ap.add_argument("--smooth-dsm-m", type=float, default=3.0,
                help="median-filter window (m) for an ingested --dsm, to remove tree/edge "
                     "spikes; 0 disables")
ap.add_argument("--spawn", type=float, nargs=2, default=None, metavar=("X", "Y"),
                help="local ENU metres from the lat/lon origin; default picks open ground automatically")
ap.add_argument("--tex-px", type=int, default=4096,
                help="ground albedo resolution; 8192 halves texel size at the cost of memory")
ap.add_argument("--ms-buildings", action="store_true",
                help="supplement OSM with Microsoft Global ML Building Footprints "
                     "(free, covers Ukraine where OSM is thin); merged into osm.json")
ap.add_argument("--refetch", action="store_true")
a = ap.parse_args()


def _quadkey(lat, lon, z=9):
    import math as _m
    sin = _m.sin(_m.radians(lat))
    x = min((1 << z) - 1, max(0, int((lon + 180) / 360 * (1 << z))))
    y = min((1 << z) - 1, max(0, int((0.5 - _m.log((1 + sin) / (1 - sin)) / (4 * _m.pi)) * (1 << z))))
    qk = ""
    for i in range(z, 0, -1):
        d, mask = 0, 1 << (i - 1)
        if x & mask: d += 1
        if y & mask: d += 2
        qk += str(d)
    return qk


def fetch_ms_buildings(bbox, cache_dir):
    """MS Global ML Building Footprints intersecting bbox (S,W,N,E) as Overpass-
    style way elements. Free (ODbL), global; fills in where OSM buildings are sparse."""
    import csv as _csv, gzip as _gzip, io as _io, requests as _rq
    S, W, N, E = bbox
    qks = {_quadkey(la, lo) for la in (S, N, (S + N) / 2) for lo in (W, E, (W + E) / 2)}
    links = cache_dir / ".ms_dataset_links.csv"
    if not links.exists():
        r = _rq.get("https://minedbuildings.z5.web.core.windows.net/global-buildings/dataset-links.csv", timeout=120)
        r.raise_for_status(); links.write_bytes(r.content)
    urls = [row["Url"] for row in _csv.DictReader(links.read_text().splitlines())
            if row["QuadKey"] in qks]
    ways, wid = [], -10_000_000
    for url in urls:
        try:
            raw = _gzip.decompress(_rq.get(url, timeout=180).content).decode()
        except Exception:                                  # noqa: BLE001
            continue
        for line in raw.splitlines():
            if not line.strip():
                continue
            try:
                feat = json.loads(line)
                ring = feat["geometry"]["coordinates"][0]
            except (json.JSONDecodeError, KeyError, IndexError):
                continue
            if not any(W <= p[0] <= E and S <= p[1] <= N for p in ring):
                continue
            h = (feat.get("properties") or {}).get("height")
            tags = {"building": "yes"}
            if isinstance(h, (int, float)) and h > 0:
                tags["height"] = str(round(float(h), 1))
            ways.append({"type": "way", "id": wid, "tags": tags,
                         "geometry": [{"lat": p[1], "lon": p[0]} for p in ring]})
            wid -= 1
    return ways

root = Path(__file__).resolve().parents[1]
data = root / "assets" / a.site; data.mkdir(parents=True, exist_ok=True)
half_m = a.half_km * 1000.0
dlat = (half_m + 300) / 110574.0; dlon = (half_m + 300) / (111320.0 * math.cos(math.radians(a.lat)))
bbox = (a.lat - dlat, a.lon - dlon, a.lat + dlat, a.lon + dlon)          # S, W, N, E

US_LON = (-125.0, -66.0)
US_LAT = (24.0, 50.0)
in_us = US_LON[0] < a.lon < US_LON[1] and US_LAT[0] < a.lat < US_LAT[1]
use_us = {"us": True, "global": False, "auto": in_us}[a.source]
ingest = a.dsm is not None
if ingest:
    use_us = False
    print(f"source: provided DSM {a.dsm}" + (" (surface model)" if a.surface_model else "")
          + (f" + ortho {a.ortho}" if a.ortho else " + Esri ortho"))
elif use_us:
    print(f"source: USGS 3DEP (1 m lidar DEM) + NAIP (1 m aerial imagery)")
else:
    print("source: Copernicus GLO-30 (30 m DEM) + Esri World Imagery ortho + OSM")

if use_us and (a.refetch or not (data / "dem.npy").exists()):
    # 3DEP bare-earth elevation, ~1 m where lidar exists. We pick the request size
    # so one pixel is ~1 m, capped at the ImageServer's limit.
    import requests, rasterio, io
    S, W, N, E = bbox
    span_m = (half_m + 300) * 2
    px = int(min(4000, max(512, round(span_m / 1.0))))
    r = requests.get("https://elevation.nationalmap.gov/arcgis/rest/services/3DEPElevation/ImageServer/exportImage",
                     params={"bbox": f"{W},{S},{E},{N}", "bboxSR": 4326, "size": f"{px},{px}",
                             "format": "tiff", "pixelType": "F32", "noData": -9999,
                             "interpolation": "RSP_BilinearInterpolation", "f": "image"}, timeout=300)
    r.raise_for_status()
    with rasterio.open(io.BytesIO(r.content)) as ds:
        arr = ds.read(1).astype(np.float32)
    arr[arr < -1000] = np.nan
    if np.isnan(arr).all():
        raise SystemExit("3DEP returned no data here -- rerun with --source global")
    # fill any voids with the mean so the mesh stays continuous
    arr[np.isnan(arr)] = np.nanmean(arr)
    # transform for a north-up image spanning the bbox: lon = a*col + c ; lat = e*row + f
    tr = [(E - W) / px, 0.0, W, 0.0, -(N - S) / px, N]
    np.save(data / "dem.npy", arr)
    (data / "dem_meta.json").write_text(json.dumps({"transform": tr, "bbox": bbox, "source": "USGS 3DEP 1m"}))
    print(f"DEM (3DEP): {arr.shape} cells at ~{span_m/px:.1f} m/px, {arr.min():.0f}-{arr.max():.0f} m")

NY = (40.4 < a.lat < 45.1) and (-79.9 < a.lon < -71.8)

if use_us and NY and (a.refetch or not (data / "naip.png").exists()):
    # New York publishes statewide orthoimagery far sharper than NAIP's 0.6 m
    # (cars, crosswalks and footpaths resolve). Prefer it inside the state.
    import requests
    from PIL import Image as _Im
    import io as _io
    dlat_s = half_m / 110574.0
    dlon_s = half_m / (111320.0 * math.cos(math.radians(a.lat)))
    Wn, Sn, En, Nn = a.lon - dlon_s, a.lat - dlat_s, a.lon + dlon_s, a.lat + dlat_s
    TILES = max(1, a.tex_px // 1024)
    TPX = a.tex_px // TILES
    canvas = _Im.new("RGB", (TILES * TPX, TILES * TPX))
    ok = True
    for ty in range(TILES):
        for tx in range(TILES):
            w0 = Wn + (En - Wn) * tx / TILES
            w1 = Wn + (En - Wn) * (tx + 1) / TILES
            n1 = Nn - (Nn - Sn) * ty / TILES
            n0 = Nn - (Nn - Sn) * (ty + 1) / TILES
            rr = requests.get("https://orthos.its.ny.gov/arcgis/rest/services/wms/Latest/MapServer/export",
                              params={"bbox": f"{w0},{n0},{w1},{n1}", "bboxSR": 4326,
                                      "size": f"{TPX},{TPX}", "format": "jpg", "f": "image"}, timeout=300)
            if not rr.ok or rr.headers.get("content-type", "").startswith("application/json"):
                print(f"  NYS ortho tile failed ({rr.status_code}); falling back to NAIP")
                ok = False; break
            canvas.paste(_Im.open(_io.BytesIO(rr.content)).convert("RGB"), (tx * TPX, ty * TPX))
        if not ok:
            break
    if ok:
        canvas.save(data / "naip.png")
        print(f"NYS orthoimagery: {TILES*TPX}px mosaic at ~{2*half_m/(TILES*TPX):.2f} m/px")

if use_us and (a.refetch or not (data / "naip.png").exists()):
    # NAIP true-colour orthoimagery for EXACTLY the site square, so it maps 1:1
    # onto the terrain UVs (which run 0..1 across [-half_m, half_m]).
    import requests
    dlat_s = half_m / 110574.0
    dlon_s = half_m / (111320.0 * math.cos(math.radians(a.lat)))
    Wn, Sn, En, Nn = a.lon - dlon_s, a.lat - dlat_s, a.lon + dlon_s, a.lat + dlat_s
    # the ImageServer caps a single export, so mosaic a grid of tiles
    from PIL import Image as _Im
    import io as _io
    TILES, TPX = 3, 1400                      # 3x3 tiles -> 4200 px across the site
    canvas = _Im.new("RGB", (TILES * TPX, TILES * TPX))
    for ty in range(TILES):
        for tx in range(TILES):
            w0 = Wn + (En - Wn) * tx / TILES
            w1 = Wn + (En - Wn) * (tx + 1) / TILES
            n1 = Nn - (Nn - Sn) * ty / TILES          # row 0 is the north edge
            n0 = Nn - (Nn - Sn) * (ty + 1) / TILES
            rr = requests.get(
                "https://imagery.nationalmap.gov/arcgis/rest/services/USGSNAIPImagery/ImageServer/exportImage",
                params={"bbox": f"{w0},{n0},{w1},{n1}", "bboxSR": 4326,
                        "size": f"{TPX},{TPX}", "format": "jpgpng", "f": "image"}, timeout=300)
            rr.raise_for_status()
            if rr.headers.get("content-type", "").startswith("application/json"):
                raise SystemExit(f"NAIP error: {rr.text[:200]}")
            canvas.paste(_Im.open(_io.BytesIO(rr.content)).convert("RGB"), (tx * TPX, ty * TPX))
    canvas.save(data / "naip.png")
    print(f"NAIP: {TILES*TPX}px mosaic at ~{2*half_m/(TILES*TPX):.2f} m/px")

if ingest and (a.refetch or not (data / "dem.npy").exists()):
    # Ingest a provided elevation GeoTIFF (any CRS) -- the drop-in for Maxar/Airbus
    # or OpenDroneMap DSMs. The DSM's OWN extent defines the site: we center on it
    # and clamp half_m to fit inside, so the saved transform always matches the
    # array (a request bbox larger than the data would desync them and flatten it).
    import rasterio
    from rasterio.vrt import WarpedVRT
    with rasterio.open(a.dsm) as src, WarpedVRT(src, crs="EPSG:4326") as vrt:
        b = vrt.bounds
        a.lat = (b.bottom + b.top) / 2.0
        a.lon = (b.left + b.right) / 2.0
        span_lat_m = (b.top - b.bottom) * 110574.0
        span_lon_m = (b.right - b.left) * 111320.0 * math.cos(math.radians(a.lat))
        half_m = min(half_m, 0.45 * min(span_lat_m, span_lon_m))   # fit inside, 10% margin
        scale = min(1.0, 4000.0 / max(vrt.width, vrt.height))       # cap the raster size
        oh, ow = max(1, int(vrt.height * scale)), max(1, int(vrt.width * scale))
        arr = vrt.read(1, out_shape=(oh, ow)).astype(np.float32)
        tr = [(b.right - b.left) / ow, 0.0, b.left, 0.0, -(b.top - b.bottom) / oh, b.top]
    void = (arr <= -1000) | (arr == 0)                             # ODM fills outside-hull with 0/nodata
    arr[void] = np.nan
    if np.isnan(arr).all():
        raise SystemExit("DSM has no valid data")
    arr[np.isnan(arr)] = np.nanmin(arr)                            # voids drop to the low point, no spikes
    if a.smooth_dsm_m > 0:
        # Raw DSM meshed as terrain spikes at trees / reconstruction noise. A
        # median filter over ~smooth_dsm_m removes isolated spikes and turns tree
        # canopy into gentle mounds -- the honest form of DSM terrain.
        from scipy.ndimage import median_filter
        px_m = (b.right - b.left) * 111320.0 * math.cos(math.radians(a.lat)) / ow
        k = max(3, int(round(a.smooth_dsm_m / max(px_m, 1e-6))) | 1)   # odd window
        arr = median_filter(arr, size=min(k, 41)).astype(np.float32)
        print(f"  denoised DSM: {a.smooth_dsm_m:.0f} m median ({k}px)")
    dlat = (half_m + 300) / 110574.0
    dlon = (half_m + 300) / (111320.0 * math.cos(math.radians(a.lat)))
    bbox = (a.lat - dlat, a.lon - dlon, a.lat + dlat, a.lon + dlon)
    np.save(data / "dem.npy", arr)
    (data / "dem_meta.json").write_text(json.dumps(
        {"transform": tr, "bbox": list(bbox), "source": f"DSM {a.dsm}"}))
    print(f"DSM: {arr.shape} cells over a {2*half_m:.0f} m site, "
          f"{np.nanmin(arr):.0f}-{np.nanmax(arr):.0f} m; center {a.lat:.5f},{a.lon:.5f}")

if a.ortho and (a.refetch or not (data / "imagery.png").exists()):
    # Provided ortho -> ground albedo. GeoTIFF is warped+cropped to the site
    # square; a plain image is assumed to already cover it 1:1.
    from PIL import Image as _Im
    if a.ortho.lower().endswith((".tif", ".tiff")):
        import rasterio
        from rasterio.vrt import WarpedVRT
        from rasterio.windows import from_bounds
        dlat_s = half_m / 110574.0
        dlon_s = half_m / (111320.0 * math.cos(math.radians(a.lat)))
        with rasterio.open(a.ortho) as src, WarpedVRT(src, crs="EPSG:4326") as vrt:
            win = from_bounds(a.lon - dlon_s, a.lat - dlat_s, a.lon + dlon_s, a.lat + dlat_s,
                              transform=vrt.transform)
            rgb = vrt.read((1, 2, 3), window=win)
        _Im.fromarray(np.transpose(rgb, (1, 2, 0)).astype(np.uint8)).save(data / "imagery.png")
    else:
        _Im.open(a.ortho).convert("RGB").save(data / "imagery.png")
    print(f"ortho: ingested {a.ortho}")

if (not use_us) and (not ingest) and (a.refetch or not (data / "dem.npy").exists()):
    import rasterio
    from rasterio.windows import from_bounds
    # Copernicus GLO-30 tiles are 1x1 deg; a site can straddle a tile edge, so
    # sign the tile name from floor(lat/lon) of each corner and merge if needed.
    lat_t, lon_t = int(math.floor(a.lat)), int(math.floor(a.lon))
    ns, ew = ("N" if lat_t >= 0 else "S"), ("E" if lon_t >= 0 else "W")
    tile = f"Copernicus_DSM_COG_10_{ns}{abs(lat_t):02d}_00_{ew}{abs(lon_t):03d}_00_DEM"
    url = f"https://copernicus-dem-30m.s3.amazonaws.com/{tile}/{tile}.tif"
    with rasterio.Env(AWS_NO_SIGN_REQUEST="YES"), rasterio.open(url) as ds:
        win = from_bounds(bbox[1], bbox[0], bbox[3], bbox[2], transform=ds.transform)
        arr = ds.read(1, window=win).astype(np.float32); tr = ds.window_transform(win)
    np.save(data / "dem.npy", arr)
    (data / "dem_meta.json").write_text(json.dumps({"transform": list(tr)[:6], "bbox": bbox, "source": url}))
    print(f"DEM (Copernicus GLO-30): {arr.shape} cells, {arr.min():.0f}-{arr.max():.0f} m")

if (not use_us) and (not a.ortho) and (a.refetch or not (data / "imagery.png").exists()):
    # Global true-colour ortho. Esri World Imagery is sub-metre over most populated
    # areas (incl. Ukrainian cities) and covers the whole planet -- enough to drape
    # the ground and detect canopy. Same site-exact tiled mosaic as the NAIP path.
    # (Prototype source; a licensed feed -- Maxar/Airbus -- is the high-fidelity swap.)
    import requests
    from PIL import Image as _Im
    import io as _io
    dlat_s = half_m / 110574.0
    dlon_s = half_m / (111320.0 * math.cos(math.radians(a.lat)))
    Wn, Sn, En, Nn = a.lon - dlon_s, a.lat - dlat_s, a.lon + dlon_s, a.lat + dlat_s
    TILES = max(3, a.tex_px // 1024)
    TPX = a.tex_px // TILES
    canvas = _Im.new("RGB", (TILES * TPX, TILES * TPX))

    def _tile(w0, n0, w1, n1):
        # ArcGIS export is flaky under load; retry with backoff before giving up
        for attempt in range(4):
            try:
                rr = requests.get(
                    "https://services.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/export",
                    params={"bbox": f"{w0},{n0},{w1},{n1}", "bboxSR": 4326,
                            "size": f"{TPX},{TPX}", "format": "jpg", "f": "image"}, timeout=120)
                if rr.ok and not rr.headers.get("content-type", "").startswith(("application/json", "text/")):
                    return _Im.open(_io.BytesIO(rr.content)).convert("RGB")
            except requests.exceptions.RequestException:
                pass
            time.sleep(2 * (attempt + 1))
        return None

    missing = 0
    for ty in range(TILES):
        for tx in range(TILES):
            w0 = Wn + (En - Wn) * tx / TILES
            w1 = Wn + (En - Wn) * (tx + 1) / TILES
            n1 = Nn - (Nn - Sn) * ty / TILES
            n0 = Nn - (Nn - Sn) * (ty + 1) / TILES
            tile = _tile(w0, n0, w1, n1)
            if tile is None:
                missing += 1                          # tolerate a straggler; leave it black
                print(f"  Esri tile {ty},{tx} failed after retries; leaving a gap")
            else:
                canvas.paste(tile, (tx * TPX, ty * TPX))
    if missing < TILES * TILES:                       # got at least some imagery
        canvas.save(data / "imagery.png")
        print(f"imagery (Esri World Imagery): {TILES*TPX}px mosaic at ~{2*half_m/(TILES*TPX):.2f} m/px"
              + (f" ({missing} tile gaps)" if missing else ""))
    else:
        print("  all Esri tiles failed; ground will be painted from land cover")
if a.refetch or not (data / "osm.json").exists():
    import requests
    S, W, N, E = bbox
    q = (f'[out:json][timeout:180];(way["building"]({S},{W},{N},{E});relation["building"]({S},{W},{N},{E});'
         f'way["highway"]({S},{W},{N},{E});way["landuse"]({S},{W},{N},{E});relation["landuse"]({S},{W},{N},{E});'
         f'way["natural"]({S},{W},{N},{E});relation["natural"]({S},{W},{N},{E});way["waterway"]({S},{W},{N},{E});'
         f'way["railway"]({S},{W},{N},{E});way["leisure"]({S},{W},{N},{E});way["aeroway"]({S},{W},{N},{E}););out geom;')
    hdrs = {"User-Agent": "vesper-sim/0.1 (research; contact via repo)"}
    endpoints = ["https://overpass-api.de/api/interpreter",
                 "https://overpass.kumi.systems/api/interpreter",
                 "https://overpass.osm.ch/api/interpreter"]
    r = None
    for ep in endpoints:
        try:
            r = requests.post(ep, data={"data": q}, headers=hdrs, timeout=300)
            if r.ok:
                break
            print(f"  overpass {ep} -> {r.status_code}, trying next")
        except Exception as exc:
            print(f"  overpass {ep} failed: {exc}")
    if r is None or not r.ok:
        raise SystemExit("all Overpass endpoints failed")
    (data / "osm.json").write_bytes(r.content)
    print(f"OSM: {len(r.json()['elements'])} elements")

if a.ms_buildings and not a.surface_model:
    osm = json.loads((data / "osm.json").read_text())
    if not osm.get("vesper_ms_merged"):
        try:
            ms = fetch_ms_buildings(bbox, root)
            osm["elements"].extend(ms)
            osm["vesper_ms_merged"] = True
            (data / "osm.json").write_text(json.dumps(osm))
            print(f"MS footprints: +{len(ms)} buildings merged into OSM")
        except Exception as exc:                           # never let this fail the build
            print(f"MS footprints skipped: {exc}")

t0 = time.time()
site = GeoSite(a.lat, a.lon, half_m, res_m=a.res, seed=a.seed, variant=a.variant, leg_m=a.leg_m,
               tex_px=a.tex_px, tree_colliders=not a.no_tree_colliders)
rep = build_site(site, data, root / "assets" / "vegetation", data / f"{a.site}.usd",
                 spawn_override=a.spawn, surface_model=a.surface_model)
print(f"built in {time.time() - t0:.0f}s: terrain {rep.terrain_verts} verts z[{rep.z_range[0]},{rep.z_range[1]}], "
      f"{rep.buildings} buildings, {rep.water} water bodies, {rep.trees} trees")
print(f"spawn (x,y)={rep.spawn_xy} ground z={rep.spawn_ground_z}; takeoff {rep.takeoff_alt_m} m; waypoints {rep.waypoints}")
print(f"build manifest: {rep.manifest}")

spec = imported_scenario(str(Path(rep.usd).relative_to(root)), tuple(rep.spawn_xy), rep.spawn_ground_z, a.seed)
spec.world = a.site; spec.waypoints = rep.waypoints; spec.takeoff_alt_m = rep.takeoff_alt_m
spec.overview_cam = None
spec.max_sim_s = a.max_sim_s
out = spec.save(root / f"{a.site}0.json")
print("scenario:", out)
