"""Run a trained RF-DETR over a split and draw what it actually predicts.

    scratch/venv-rfdetr/bin/python scripts/predict_review.py \
        --checkpoint scratch/runs/rfdetr-nano/checkpoint_best_total.pth \
        --dataset scratch/datasets/btr80 --split test --out scratch/review/preds

Ground truth is drawn in green, predictions in magenta with their confidence, so
misses and false positives are visible as unpaired boxes rather than hidden
inside an mAP number. Each frame is captioned with its own TP/FP/FN at IoU 0.5.

This is the counterpart to sample_dataset.py: that one shows what the model was
told, this one shows what it learned.
"""
import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

REPO = Path(__file__).resolve().parents[1]
SIZES = {"nano": "RFDETRNano", "small": "RFDETRSmall",
         "medium": "RFDETRMedium", "base": "RFDETRBase", "large": "RFDETRLarge"}

ap = argparse.ArgumentParser()
ap.add_argument("--checkpoint", required=True)
ap.add_argument("--dataset", default="scratch/datasets/btr80")
ap.add_argument("--split", default="test")
ap.add_argument("--model", default="nano", choices=sorted(SIZES))
ap.add_argument("--out", default="scratch/review/preds")
ap.add_argument("--threshold", type=float, default=0.5)
ap.add_argument("--limit", type=int, default=0, help="0 = the whole split")
ap.add_argument("--sheet-cols", type=int, default=6)
ap.add_argument("--sheet-tile", type=int, default=420)
args = ap.parse_args()

root = (REPO / args.dataset).resolve() / args.split
out = (REPO / args.out).resolve()
out.mkdir(parents=True, exist_ok=True)

coco = json.loads((root / "_annotations.coco.json").read_text())
gt = {}
for a in coco["annotations"]:
    x, y, w, h = a["bbox"]
    gt.setdefault(a["image_id"], []).append([x, y, x + w, y + h])
images = coco["images"]
if args.limit:
    images = images[:args.limit]

from rfdetr import __dict__ as rfdetr_ns  # noqa: E402

model = rfdetr_ns[SIZES[args.model]](pretrain_weights=str(Path(args.checkpoint).resolve()))
model.optimize_for_inference()


def iou(a, b):
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


tot_tp = tot_fp = tot_fn = 0
tiles, rows = [], []
for i, im_rec in enumerate(images):
    path = root / im_rec["file_name"]
    img = Image.open(path).convert("RGB")
    det = model.predict(np.asarray(img), threshold=args.threshold)
    pred = [list(map(float, b)) for b in np.asarray(det.xyxy).reshape(-1, 4)]
    conf = [float(c) for c in np.asarray(det.confidence).reshape(-1)] if det.confidence is not None \
        else [1.0] * len(pred)
    truth = gt.get(im_rec["id"], [])

    # greedy match, highest IoU first, one ground-truth box per prediction
    matched, used = [], set()
    for pi, p in enumerate(pred):
        best, bj = 0.0, -1
        for j, t in enumerate(truth):
            if j in used:
                continue
            v = iou(p, t)
            if v > best:
                best, bj = v, j
        if best >= 0.5:
            used.add(bj)
            matched.append(pi)
    tp, fp, fn = len(matched), len(pred) - len(matched), len(truth) - len(used)
    tot_tp += tp; tot_fp += fp; tot_fn += fn

    dr = ImageDraw.Draw(img)
    for t in truth:                                     # ground truth: green
        dr.rectangle(t, outline=(0, 0, 0), width=4)
        dr.rectangle(t, outline=(60, 255, 90), width=2)
    for pi, p in enumerate(pred):                       # prediction: magenta
        colour = (255, 60, 200) if pi in matched else (255, 90, 40)
        dr.rectangle(p, outline=(0, 0, 0), width=4)
        dr.rectangle(p, outline=colour, width=2)
        dr.text((p[0] + 3, max(0, p[1] - 12)), f"{conf[pi]:.2f}", fill=colour)
    dr.text((6, 6), f"{im_rec['file_name']}  TP {tp}  FP {fp}  FN {fn}", fill=(255, 220, 60))
    img.save(out / f"{i:03d}_{im_rec['file_name']}")
    tiles.append(img)
    rows.append({"file": im_rec["file_name"], "tp": tp, "fp": fp, "fn": fn,
                 "n_pred": len(pred), "n_gt": len(truth)})

cols, T = args.sheet_cols, args.sheet_tile
per = cols * 5
for s in range(0, len(tiles), per):
    chunk = tiles[s:s + per]
    n_rows = (len(chunk) + cols - 1) // cols
    sheet = Image.new("RGB", (cols * T, n_rows * T), (18, 18, 20))
    for k, im in enumerate(chunk):
        t = im.copy(); t.thumbnail((T, T))
        sheet.paste(t, ((k % cols) * T, (k // cols) * T))
    sheet.save(out / f"pred_sheet_{s // per:02d}.jpg", quality=88)

prec = tot_tp / max(tot_tp + tot_fp, 1)
rec = tot_tp / max(tot_tp + tot_fn, 1)
summary = {"split": args.split, "images": len(images), "threshold": args.threshold,
           "tp": tot_tp, "fp": tot_fp, "fn": tot_fn,
           "precision": round(prec, 4), "recall": round(rec, 4),
           "f1": round(2 * prec * rec / max(prec + rec, 1e-9), 4), "per_image": rows}
(out / "prediction_summary.json").write_text(json.dumps(summary, indent=1))
print(f"{len(images)} images @ threshold {args.threshold}, IoU 0.5")
print(f"  TP {tot_tp}  FP {tot_fp}  FN {tot_fn}")
print(f"  precision {prec:.3f}  recall {rec:.3f}  f1 {summary['f1']:.3f}")
print(f"wrote {len(tiles)} frames + sheets -> {out}")
