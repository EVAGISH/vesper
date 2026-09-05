"""Probe Google Photorealistic 3D Tiles coverage at a lat/lon.

    python3 scripts/probe_3dtiles.py --lat 50.4275 --lon 30.5367   # Kyiv
    python3 scripts/probe_3dtiles.py --lat 48.7233 --lon 37.5562   # Kramatorsk

Descends the global tileset toward the point and reports the finest geometric
error reached and whether leaf glb geometry exists there -- i.e. whether the
city has real photoreal 3D coverage. Reads GOOGLE_MAPS_API_KEY from env/.env.
"""
import argparse
import math
import os
import re
from pathlib import Path

import requests

ap = argparse.ArgumentParser()
ap.add_argument("--lat", type=float, required=True)
ap.add_argument("--lon", type=float, required=True)
ap.add_argument("--max-depth", type=int, default=28)
a = ap.parse_args()

KEY = os.environ.get("GOOGLE_MAPS_API_KEY")
if not KEY:
    env = Path(__file__).resolve().parents[1] / ".env"
    for line in env.read_text().splitlines() if env.exists() else []:
        m = re.match(r"^GOOGLE_MAPS_API_KEY=([^\s#]+)", line)
        if m:
            KEY = m.group(1); break
if not KEY:
    raise SystemExit("no GOOGLE_MAPS_API_KEY")

HOST = "https://tile.googleapis.com"
a_wgs, f = 6378137.0, 1 / 298.257223563
e2 = f * (2 - f)


def ecef(lat, lon, h=0.0):
    la, lo = math.radians(lat), math.radians(lon)
    N = a_wgs / math.sqrt(1 - e2 * math.sin(la) ** 2)
    return ((N + h) * math.cos(la) * math.cos(lo),
            (N + h) * math.cos(la) * math.sin(lo),
            (N * (1 - e2) + h) * math.sin(la))


P = ecef(a.lat, a.lon)


def in_box(box, p, slack=1.05):
    c = box[0:3]
    for i in (3, 6, 9):
        u = box[i:i + 3]
        uu = u[0] ** 2 + u[1] ** 2 + u[2] ** 2
        if uu == 0:
            continue
        t = ((p[0] - c[0]) * u[0] + (p[1] - c[1]) * u[1] + (p[2] - c[2]) * u[2]) / uu
        if abs(t) > slack:
            return False
    return True


def fetch(uri):
    url = uri if uri.startswith("http") else HOST + uri
    sep = "&" if "?" in url else "?"
    r = requests.get(f"{url}{sep}key={KEY}", timeout=30)
    r.raise_for_status()
    return r.json()


def descend(tile, depth, best):
    ge = tile.get("geometricError", best["ge"])
    bv = tile.get("boundingVolume", {})
    box = bv.get("box")
    if box and not in_box(box, P):
        return
    content = tile.get("content") or {}
    uri = content.get("uri") or content.get("url")
    if uri and uri.split("?")[0].endswith(".glb"):
        if ge < best["ge"]:
            best["ge"] = ge; best["glb"] = True
        return
    if ge < best["ge"] and (tile.get("children") or uri):
        best["ge"] = min(best["ge"], ge)
    if depth >= a.max_depth:
        return
    if uri and uri.split("?")[0].endswith(".json"):
        try:
            sub = fetch(uri)
            root = sub.get("root")
            if root:
                descend(root, depth + 1, best)
        except requests.exceptions.RequestException:
            pass
    for ch in tile.get("children", []) or []:
        descend(ch, depth + 1, best)


best = {"ge": float("inf"), "glb": False}
descend(fetch("/v1/3dtiles/root.json")["root"], 0, best)
print(f"lat={a.lat} lon={a.lon}")
if best["glb"] and best["ge"] < 100:
    print(f"COVERED: photoreal leaf geometry present, finest geometricError ~{best['ge']:.1f} m")
elif best["ge"] < 1e9:
    print(f"partial: finest geometricError ~{best['ge']:.1f} m, leaf glb={best['glb']}")
else:
    print("NO detailed coverage reached at this point")
