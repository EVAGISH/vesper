"""Export a trained RF-DETR to ONNX and to a TensorRT engine, and time both.

    scratch/venv-rfdetr/bin/python scripts/export_rfdetr.py \
        --checkpoint scratch/runs/rfdetr-nano/checkpoint_best_total.pth --model nano

Two artefacts, on purpose:

  rfdetr-<size>.onnx      portable. Survives a driver upgrade, a different GPU
                          and the trip to the airframe's Jetson, and is the
                          thing worth pulling home.
  rfdetr-<size>_fp16.trt  fast. Target-specific compilation: the engine is tied
                          to this exact GPU and TensorRT version, so it is
                          rebuilt on whatever hardware actually runs it and
                          never treated as a deliverable.

rfdetr names both files after the model variant, not after us -- the engine
carries the precision it was actually built at (_fp32 under --fp32).

The latency number printed at the end is the one that decides whether the
detector can sit in the control loop at all, so it is measured rather than
assumed -- on this GPU, at the resolution the model was trained at.
"""
import argparse
import json
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SIZES = {"nano": "RFDETRNano", "small": "RFDETRSmall",
         "medium": "RFDETRMedium", "base": "RFDETRBase", "large": "RFDETRLarge"}

ap = argparse.ArgumentParser()
ap.add_argument("--checkpoint", required=True)
ap.add_argument("--model", default="nano", choices=sorted(SIZES))
ap.add_argument("--out", default=None, help="default: alongside the checkpoint, in export/")
ap.add_argument("--resolution", type=int, default=None)
ap.add_argument("--batch-size", type=int, default=1)
ap.add_argument("--no-trt", action="store_true", help="ONNX only")
ap.add_argument("--fp32", action="store_true", help="build the engine at FP32")
ap.add_argument("--bench-iters", type=int, default=200)
args = ap.parse_args()

ckpt = Path(args.checkpoint).resolve()
if not ckpt.exists():
    raise SystemExit(f"no checkpoint at {ckpt}")
out = Path(args.out).resolve() if args.out else ckpt.parent / "export"
out.mkdir(parents=True, exist_ok=True)

from rfdetr import __dict__ as rfdetr_ns  # noqa: E402

kwargs = {"pretrain_weights": str(ckpt)}
if args.resolution:
    kwargs["resolution"] = args.resolution
model = rfdetr_ns[SIZES[args.model]](**kwargs)

report = {"checkpoint": str(ckpt), "model": SIZES[args.model], "artifacts": {}}

print(f"exporting ONNX -> {out}")
t0 = time.time()
onnx_path = model.export(output_dir=str(out), format="onnx", batch_size=args.batch_size)
report["artifacts"]["onnx"] = {"path": str(onnx_path), "seconds": round(time.time() - t0, 1)}
print(f"  {onnx_path}")

if not args.no_trt:
    print(f"\nbuilding TensorRT engine (fp16={not args.fp32}) -- this compiles for THIS GPU")
    t0 = time.time()
    try:
        trt_path = model.export(output_dir=str(out), format="tensorrt",
                                batch_size=args.batch_size, fp16=not args.fp32)
        # Record the precision that was actually built, not the one asked for.
        # TensorRT 11.2 does not expose the FP16 builder flag and silently falls
        # back to FP32 with only a warning -- rfdetr encodes the truth in the
        # filename suffix, so trust that over our own request.
        built_fp16 = "_fp16" in Path(trt_path).name
        report["artifacts"]["tensorrt"] = {"path": str(trt_path),
                                           "fp16_requested": not args.fp32,
                                           "fp16_built": built_fp16,
                                           "seconds": round(time.time() - t0, 1)}
        if (not args.fp32) and not built_fp16:
            print("  NOTE: FP16 was requested but an FP32 engine was built "
                  "(this TensorRT has no FP16 builder flag)")
        print(f"  {trt_path}")
    except Exception as e:
        report["artifacts"]["tensorrt"] = {"error": f"{type(e).__name__}: {e}"}
        print(f"  TensorRT export failed: {e}\n"
              "  (needs rfdetr[tensorrt]; rerun scripts/rfdetr_setup.sh)")

# --- latency, measured on this GPU
try:
    import numpy as np
    import torch

    model.optimize_for_inference()
    res = getattr(model.model, "resolution", None) or args.resolution or 560
    dummy = (np.random.rand(res, res, 3) * 255).astype("uint8")
    for _ in range(10):
        model.predict(dummy, threshold=0.5)
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(args.bench_iters):
        model.predict(dummy, threshold=0.5)
    torch.cuda.synchronize()
    ms = (time.time() - t0) / args.bench_iters * 1000
    report["torch_latency_ms"] = round(ms, 2)
    report["torch_fps"] = round(1000.0 / ms, 1)
    report["resolution"] = int(res)
    print(f"\npytorch (optimised) @ {res}px: {ms:.2f} ms/frame, {1000 / ms:.0f} fps")
except Exception as e:
    report["torch_latency_ms"] = f"{type(e).__name__}: {e}"
    print(f"\nlatency benchmark skipped: {e}")

(out / "export_report.json").write_text(json.dumps(report, indent=1))
print(f"\nreport -> {out / 'export_report.json'}")
print("pull the small artefacts home with scripts/detect_pull.sh")
