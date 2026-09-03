"""Build a real-world site into a USD world + flight scenario (runs on the Mac, no Isaac).

    python3 scripts/build_geo_world.py sloviansk --lat 48.845 --lon 37.635 --half-km 1.5

Fetches (once, cached under assets/<site>/) the Copernicus 30 m DEM crop and an
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
ap.add_argument("--refetch", action="store_true")
a = ap.parse_args()

root = Path(__file__).resolve().parents[1]
data = root / "assets" / a.site; data.mkdir(parents=True, exist_ok=True)
half_m = a.half_km * 1000.0
dlat = (half_m + 300) / 110574.0; dlon = (half_m + 300) / (111320.0 * math.cos(math.radians(a.lat)))
bbox = (a.lat - dlat, a.lon - dlon, a.lat + dlat, a.lon + dlon)          # S, W, N, E

if a.refetch or not (data / "dem.npy").exists():
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
    r = requests.get("https://overpass-api.de/api/interpreter", params={"data": q}, timeout=300); r.raise_for_status()
    (data / "osm.json").write_bytes(r.content)
    print(f"OSM: {len(r.json()['elements'])} elements")

t0 = time.time()
site = GeoSite(a.lat, a.lon, half_m, res_m=a.res, seed=a.seed)
rep = build_site(site, data, root / "assets" / "vegetation", data / f"{a.site}.usd")
print(f"built in {time.time() - t0:.0f}s: terrain {rep.terrain_verts} verts z[{rep.z_range[0]},{rep.z_range[1]}], "
      f"{rep.buildings} buildings, {rep.water} water bodies, {rep.trees} trees")
print(f"spawn (x,y)={rep.spawn_xy} ground z={rep.spawn_ground_z}; takeoff {rep.takeoff_alt_m} m; waypoints {rep.waypoints}")

spec = imported_scenario(str(Path(rep.usd).relative_to(root)), tuple(rep.spawn_xy), rep.spawn_ground_z, a.seed)
spec.world = a.site; spec.waypoints = rep.waypoints; spec.takeoff_alt_m = rep.takeoff_alt_m
spec.overview_cam = None
out = spec.save(root / f"{a.site}0.json")
print("scenario:", out)
