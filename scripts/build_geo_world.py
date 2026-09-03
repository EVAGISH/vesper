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
ap.add_argument("--leg-m", type=float, default=90.0,
                help="mission loop leg length (m); ~90 gives a ~30 s flight video")
ap.add_argument("--max-sim-s", type=float, default=75.0)
ap.add_argument("--source", choices=["auto", "us", "global"], default="auto",
                help="auto picks USGS 3DEP+NAIP inside the US, Copernicus+painted elsewhere")
ap.add_argument("--refetch", action="store_true")
a = ap.parse_args()

root = Path(__file__).resolve().parents[1]
data = root / "assets" / a.site; data.mkdir(parents=True, exist_ok=True)
half_m = a.half_km * 1000.0
dlat = (half_m + 300) / 110574.0; dlon = (half_m + 300) / (111320.0 * math.cos(math.radians(a.lat)))
bbox = (a.lat - dlat, a.lon - dlon, a.lat + dlat, a.lon + dlon)          # S, W, N, E

US_LON = (-125.0, -66.0)
US_LAT = (24.0, 50.0)
in_us = US_LON[0] < a.lon < US_LON[1] and US_LAT[0] < a.lat < US_LAT[1]
use_us = {"us": True, "global": False, "auto": in_us}[a.source]
if use_us:
    print(f"source: USGS 3DEP (1 m lidar DEM) + NAIP (1 m aerial imagery)")
else:
    print("source: Copernicus GLO-30 (30 m) + painted land-cover ground")

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

if (not use_us) and (a.refetch or not (data / "dem.npy").exists()):
    import rasterio
    from rasterio.windows import from_bounds
    lat_t, lon_t = int(math.floor(a.lat)), int(math.floor(a.lon))
    tile = f"Copernicus_DSM_COG_10_N{lat_t:02d}_00_E{lon_t:03d}_00_DEM"
    url = f"https://copernicus-dem-30m.s3.amazonaws.com/{tile}/{tile}.tif"
    with rasterio.Env(AWS_NO_SIGN_REQUEST="YES"), rasterio.open(url) as ds:
        win = from_bounds(bbox[1], bbox[0], bbox[3], bbox[2], transform=ds.transform)
        arr = ds.read(1, window=win).astype(np.float32); tr = ds.window_transform(win)
    np.save(data / "dem.npy", arr)
    (data / "dem_meta.json").write_text(json.dumps({"transform": list(tr)[:6], "bbox": bbox, "source": url}))
    print(f"DEM: {arr.shape} cells, {arr.min():.0f}-{arr.max():.0f} m")
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

t0 = time.time()
site = GeoSite(a.lat, a.lon, half_m, res_m=a.res, seed=a.seed, leg_m=a.leg_m)
rep = build_site(site, data, root / "assets" / "vegetation", data / f"{a.site}.usd")
print(f"built in {time.time() - t0:.0f}s: terrain {rep.terrain_verts} verts z[{rep.z_range[0]},{rep.z_range[1]}], "
      f"{rep.buildings} buildings, {rep.water} water bodies, {rep.trees} trees")
print(f"spawn (x,y)={rep.spawn_xy} ground z={rep.spawn_ground_z}; takeoff {rep.takeoff_alt_m} m; waypoints {rep.waypoints}")

spec = imported_scenario(str(Path(rep.usd).relative_to(root)), tuple(rep.spawn_xy), rep.spawn_ground_z, a.seed)
spec.world = a.site; spec.waypoints = rep.waypoints; spec.takeoff_alt_m = rep.takeoff_alt_m
spec.overview_cam = None
spec.max_sim_s = a.max_sim_s
out = spec.save(root / f"{a.site}0.json")
print("scenario:", out)
