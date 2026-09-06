"""Fine-tune RF-DETR on the synthetic BTR-80A set, on the GPU droplet.

    scratch/venv-rfdetr/bin/python scripts/train_rfdetr.py \
        --dataset scratch/datasets/btr80 --model nano --epochs 40

Runs in the venv scripts/rfdetr_setup.sh builds, not in the Isaac container:
Isaac's image is rebuilt from a Dockerfile and anything pip-installed into a
running container dies with it. Everything it writes lands under scratch/, which
infra/do/snapshot.sh clears before imaging the disk.

Model size is a deployment decision, not an accuracy one. Nano is the default
because the airframe carries a Jetson-class part and a detector that cannot keep
up with the camera is not a detector; --model medium is there for measuring what
accuracy the small one is giving away.
"""
import argparse
import json
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]

SIZES = {"nano": "RFDETRNano", "small": "RFDETRSmall",
         "medium": "RFDETRMedium", "base": "RFDETRBase", "large": "RFDETRLarge"}

ap = argparse.ArgumentParser()
ap.add_argument("--dataset", default="scratch/datasets/btr80",
                help="repo-relative dir holding train/ valid/ test/")
ap.add_argument("--model", default="nano", choices=sorted(SIZES))
ap.add_argument("--out", default=None, help="default: scratch/runs/rfdetr-<model>")
ap.add_argument("--epochs", type=int, default=40)
ap.add_argument("--batch-size", type=int, default=8)
ap.add_argument("--grad-accum-steps", type=int, default=2)
ap.add_argument("--lr", type=float, default=1e-4)
ap.add_argument("--resolution", type=int, default=None,
                help="input square, multiple of patch_size*num_windows: 32 for "
                     "nano/small/medium/large, 56 for base; default is the model's own")
ap.add_argument("--num-workers", type=int, default=4)
ap.add_argument("--early-stopping", action="store_true")
ap.add_argument("--resume", default=None, help="checkpoint to continue from")
ap.add_argument("--seed", type=int, default=0)
args = ap.parse_args()

dataset = (REPO / args.dataset).resolve()
out = Path(args.out).resolve() if args.out else REPO / "scratch" / "runs" / f"rfdetr-{args.model}"

for split in ("train", "valid"):
    ann = dataset / split / "_annotations.coco.json"
    if not ann.exists():
        sys.exit(f"missing {ann}\nGenerate the dataset first:\n"
                 "  /isaac-sim/python.sh scripts/gen_detect_dataset.py --images 3000")
counts = {}
for split in ("train", "valid", "test"):
    ann = dataset / split / "_annotations.coco.json"
    if ann.exists():
        d = json.loads(ann.read_text())
        counts[split] = (len(d["images"]), len(d["annotations"]))
print(f"dataset {dataset}")
for split, (n_img, n_box) in counts.items():
    print(f"  {split}: {n_img} images, {n_box} boxes")

from rfdetr import __dict__ as rfdetr_ns  # noqa: E402

out.mkdir(parents=True, exist_ok=True)
model = rfdetr_ns[SIZES[args.model]]()

train_kwargs = dict(
    dataset_dir=str(dataset),
    output_dir=str(out),
    epochs=args.epochs,
    batch_size=args.batch_size,
    grad_accum_steps=args.grad_accum_steps,
    lr=args.lr,
    num_workers=args.num_workers,
    seed=args.seed,
    early_stopping=args.early_stopping,
    tensorboard=False,
    wandb=False,
    run_test="test" in counts,
)
if args.resolution:
    # train() is the only path that checks the value divides patch_size *
    # num_windows; the constructor accepts any integer and breaks later.
    train_kwargs["resolution"] = args.resolution
if args.resume:
    train_kwargs["resume"] = str(Path(args.resume).resolve())

print(f"\ntraining {SIZES[args.model]} -> {out}")
print(json.dumps({k: v for k, v in train_kwargs.items()}, indent=1))
t0 = time.time()
model.train(**train_kwargs)
mins = (time.time() - t0) / 60

summary = {"model": SIZES[args.model], "dataset": str(dataset), "counts": counts,
           "epochs": args.epochs, "minutes": round(mins, 1),
           "output_dir": str(out), "args": vars(args)}
(out / "vesper_train_summary.json").write_text(json.dumps(summary, indent=1))
print(f"\ndone in {mins:.1f} min -> {out}")
print("checkpoints:", sorted(p.name for p in out.glob('*.pth'))[:6])
print(f"\nnext:\n  scratch/venv-rfdetr/bin/python scripts/export_rfdetr.py "
      f"--checkpoint {out}/checkpoint_best_total.pth --model {args.model}")
