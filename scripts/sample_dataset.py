"""Draw a random sample of a COCO dataset with its boxes, for eyeballing.

    python3 scripts/sample_dataset.py scratch/datasets/btr80 --n 200 \
        --out scratch/review/btr80

Writes the annotated frames plus contact sheets, and prints the distribution
that decides whether the set is worth training on: boxes per image, box size in
pixels, and how much of the set is empty. Plain PIL, so it runs on the droplet
next to the data or on the Mac after an rsync -- no Isaac, no torch.
"""
import argparse
import json
import random
from pathlib import Path

from PIL import Image, ImageDraw

ap = argparse.ArgumentParser()
ap.add_argument("dataset", help="dir holding train/ valid/ test/")
ap.add_argument("--n", type=int, default=200)
ap.add_argument("--out", default=None, help="default: <dataset>/../review/<name>")
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--split", nargs="*", default=["train", "valid", "test"])
ap.add_argument("--sheet-cols", type=int, default=8)
ap.add_argument("--sheet-tile", type=int, default=320)
args = ap.parse_args()

root = Path(args.dataset).resolve()
out = Path(args.out).resolve() if args.out else root.parent.parent / "review" / root.name
out.mkdir(parents=True, exist_ok=True)

frames, stats = [], []
for split in args.split:
    ann_path = root / split / "_annotations.coco.json"
    if not ann_path.exists():
        continue
    d = json.loads(ann_path.read_text())
    by_image = {}
    for a in d["annotations"]:
        by_image.setdefault(a["image_id"], []).append(a["bbox"])
    for im in d["images"]:
        frames.append((split, im["file_name"], by_image.get(im["id"], [])))

if not frames:
    raise SystemExit(f"no COCO splits found under {root}")

rng = random.Random(args.seed)
sample = rng.sample(frames, min(args.n, len(frames)))

tiles = []
for i, (split, name, boxes) in enumerate(sample):
    img = Image.open(root / split / name).convert("RGB")
    dr = ImageDraw.Draw(img)
    for x, y, w, h in boxes:
        # two strokes: a dark backing line keeps the box readable on pale sky
        dr.rectangle([x, y, x + w, y + h], outline=(0, 0, 0), width=4)
        dr.rectangle([x, y, x + w, y + h], outline=(60, 255, 90), width=2)
        stats.append((w, h))
    dr.text((6, 6), f"{split}/{name}  {len(boxes)} box(es)", fill=(255, 220, 60))
    img.save(out / f"{i:03d}_{split}_{name}")
    tiles.append(img)

cols = args.sheet_cols
T = args.sheet_tile
per_sheet = cols * 5
for s in range(0, len(tiles), per_sheet):
    chunk = tiles[s:s + per_sheet]
    rows = (len(chunk) + cols - 1) // cols
    sheet = Image.new("RGB", (cols * T, rows * T), (18, 18, 20))
    for k, im in enumerate(chunk):
        t = im.copy(); t.thumbnail((T, T))
        sheet.paste(t, ((k % cols) * T, (k // cols) * T))
    sheet.save(out / f"sheet_{s // per_sheet:02d}.jpg", quality=88)

empty = sum(1 for _, _, b in frames if not b)
areas = sorted(w * h for w, h in stats)
sides = sorted(min(w, h) for w, h in stats)


def pct(v, p):
    return v[int(p * (len(v) - 1))] if v else 0


print(f"dataset {root}: {len(frames)} images, {sum(len(b) for _, _, b in frames)} boxes")
print(f"  empty frames: {empty} ({empty / len(frames):.1%})")
print(f"  boxes/image:  {sum(len(b) for _, _, b in frames) / len(frames):.2f}")
if sides:
    print(f"  short side px: p5={pct(sides, .05):.0f} p50={pct(sides, .5):.0f} "
          f"p95={pct(sides, .95):.0f}")
    print(f"  box area px^2: p5={pct(areas, .05):.0f} p50={pct(areas, .5):.0f} "
          f"p95={pct(areas, .95):.0f}")
print(f"wrote {len(sample)} annotated frames + "
      f"{len(list(out.glob('sheet_*.jpg')))} contact sheets -> {out}")
