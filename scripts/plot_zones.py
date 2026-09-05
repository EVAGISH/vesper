"""Draw a site's zones over its orthophoto: what the sim actually uses, as a picture.

    python3 scripts/plot_zones.py                      # Cornell -> docs/cornell_zones.png
    python3 scripts/plot_zones.py --map assets/<site>/<site>_map.npz --out docs/<site>_zones.png

The friendly/hunting split is painted from the rasterised masks, not the raw
polygons, so what you see is what the reward reads. Spawn dots are drawn with
the environment's own sampler. Pure numpy + PIL: runs on the Mac, no Isaac.
"""
import argparse
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw

from vesper.worlds.heightmap import WorldMap
from vesper.worlds.rasters import chamfer_distance
from vesper.worlds.zones import Zones, find_zones

ROOT = Path(__file__).resolve().parents[1]
ap = argparse.ArgumentParser()
ap.add_argument("--map", default=str(ROOT / "assets" / "cornell" / "cornell_map.npz"))
ap.add_argument("--zones", default=None, help="default: beside the map, else <site>_zones.json")
ap.add_argument("--ground", default=None, help="default: ground.png beside the map")
ap.add_argument("--out", default=str(ROOT / "docs" / "cornell_zones.png"))
ap.add_argument("--arena", type=float, default=590.0)
ap.add_argument("--px", type=int, default=1400)
ap.add_argument("--spawns", type=int, default=200)
a = ap.parse_args()

Image.MAX_IMAGE_PIXELS = None
mp = Path(a.map)
zpath = a.zones or find_zones(mp, ROOT)
if not zpath:
    raise SystemExit(f"no zones file for {mp}")
z = Zones.load(zpath)
d = np.load(mp)
half, cell, n = float(d["half_m"]), float(d["cell"]), d["ground_z"].shape[0]
lm, sm = z.masks(n, half, cell)
W = a.px
ground = Path(a.ground) if a.ground else mp.with_name("ground.png")
base = (Image.open(ground).convert("RGB").resize((W, W), Image.LANCZOS)
        if ground.exists() else Image.new("RGB", (W, W), (40, 44, 40)))


def px(x, y):
    return ((x + half) / (2 * half) * W, (half - y) / (2 * half) * W)


# the split, painted from the masks the reward reads
friendly = Image.fromarray((sm[::-1] * 255).astype(np.uint8)).resize((W, W), Image.NEAREST).convert("L")
arr = np.asarray(base).astype(np.float32)
mask = (np.asarray(friendly) > 127)[..., None]
cool = np.clip(arr * np.array([0.55, 0.95, 1.45]) + np.array([10, 30, 60]), 0, 255)
warm = np.clip(arr * np.array([0.85, 0.68, 0.55]), 0, 255)
base = Image.fromarray(np.where(mask, cool, warm).astype(np.uint8))

ov = Image.new("RGBA", (W, W), (0, 0, 0, 0))
dr = ImageDraw.Draw(ov)


def label(at, text, fill, anchor="mm", pad=5):
    x, y = at
    b = dr.textbbox((x, y), text, anchor=anchor)
    dr.rectangle([b[0] - pad, b[1] - pad, b[2] + pad, b[3] + pad], fill=(0, 0, 0, 170))
    dr.text((x, y), text, fill=fill, anchor=anchor)


for poly in z.safe:
    pts = [px(*p) for p in poly]
    dr.line(pts + [pts[0]], fill=(90, 230, 255, 255), width=7)
    dr.line(pts + [pts[0]], fill=(255, 255, 255, 255), width=2)
A = a.arena
dr.rectangle([*px(-A, A), *px(A, -A)], outline=(255, 255, 255, 230), width=3)
label((px(-A, A)[0] + 8, px(-A, A)[1] + 8), f"ARENA  {2*A:.0f} x {2*A:.0f} m", (255, 255, 255, 245), "lt")

out = chamfer_distance(1 - (sm > 0).astype(np.uint8), cell) if sm.any() else None
if z.launch:
    dr.polygon([px(*p) for p in z.launch], fill=(40, 220, 80, 120), outline=(140, 255, 150, 255))
    cx = sum(p[0] for p in z.launch) / len(z.launch)
    cy = sum(p[1] for p in z.launch) / len(z.launch)
    w = WorldMap(mp); w.attach_zones(z)
    g = torch.Generator().manual_seed(0)
    xy, _ = w.sample_cells_xy(w.launch, a.spawns, g, half=A * 0.95)
    for p in xy.tolist():
        x, y = px(*p)
        dr.ellipse([x - 2.5, y - 2.5, x + 2.5, y + 2.5], fill=(255, 255, 255, 235))
    label(px(cx, cy - 52), f"LAUNCH  ·  {a.spawns} random spawns", (150, 255, 160, 255))
    if out is not None and (lm > 0).any():
        ex, ey = px(cx + 120, cy + 40)
        dr.line([px(cx, cy), (ex, ey)], fill=(255, 210, 60, 255), width=7)
        dr.polygon([(ex, ey), (ex - 20, ey - 8), (ex - 16, ey + 13)], fill=(255, 210, 60, 255))
        label((ex + 14, ey - 4),
              f"get clear:  {out[lm > 0].min():.0f}-{out[lm > 0].max():.0f} m", (255, 215, 90, 255), "lm")

m100 = 100.0 / (2 * half) * W
bx, by = 45, W - 45
dr.line([bx, by, bx + m100, by], fill=(255, 255, 255, 255), width=5)
for e in (bx, bx + m100):
    dr.line([e, by - 8, e, by + 8], fill=(255, 255, 255, 255), width=4)
label((bx + m100 / 2, by - 22), "100 m", (255, 255, 255, 255), "ms")
label((45, 34), f"{mp.stem.replace('_map', '')} — friendly ground (blue) vs hunting ground",
      (255, 255, 255, 255), "lt")

Path(a.out).parent.mkdir(parents=True, exist_ok=True)
Image.alpha_composite(base.convert("RGBA"), ov).convert("RGB").save(a.out)
print(f"wrote {a.out}")
if out is not None and (lm > 0).any():
    print(f"friendly {100*float((sm>0).mean()):.0f}% of the raster; "
          f"exit from the pad {out[lm>0].min():.0f}-{out[lm>0].max():.0f} m")
