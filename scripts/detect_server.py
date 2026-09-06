"""Serve a trained RF-DETR to the simulator, on the GPU box.

    scratch/venv-rfdetr/bin/python scripts/detect_server.py \
        --checkpoint scratch/runs/rfdetr-nano/checkpoint_best_total.pth --model nano

Why a server and not an import. rfdetr lives in the venv scripts/rfdetr_setup.sh
builds under scratch/; the simulator lives in the Isaac container, which is
rebuilt from a Dockerfile and forgets anything pip-installed into it. The two
cannot share a Python. They can share a machine: the sim container runs with
host networking, so this process on the droplet's loopback is one hop from the
training run and no frame ever leaves the box.

Protocol, deliberately small:

  GET  /health   {"ready": bool, "model": ..., "checkpoint": ..., "resolution": n,
                  "device": ..., "images": n, "seconds": s}
  POST /detect   body = JSON header line + "\n" + raw uint8 N*H*W*3
                 header {"n","h","w","threshold"}
                 -> {"boxes": [[[x0,y0,x1,y1], ...], ...], "scores": [[...], ...]}
                 boxes are xyxy in the submitted image's own pixel coordinates.

The model loads on a background thread so the socket is answering (and /health
is honest about `ready`) while torch and the weights come up, which takes tens
of seconds and would otherwise look like a dead port to whoever launched it.
"""
import argparse
import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SIZES = {"nano": "RFDETRNano", "small": "RFDETRSmall",
         "medium": "RFDETRMedium", "base": "RFDETRBase", "large": "RFDETRLarge"}

ap = argparse.ArgumentParser()
ap.add_argument("--checkpoint", required=True, help="repo-relative or absolute .pth")
ap.add_argument("--model", default="nano", choices=sorted(SIZES))
ap.add_argument("--resolution", type=int, default=None,
                help="input square the model runs at; default is the checkpoint's own")
ap.add_argument("--host", default="127.0.0.1")
ap.add_argument("--port", type=int, default=8181)
ap.add_argument("--threshold", type=float, default=0.5, help="default score threshold")
ap.add_argument("--batch", type=int, default=32,
                help="images per forward pass; the whole request is chunked to this")
args = ap.parse_args()

ckpt = Path(args.checkpoint)
if not ckpt.is_absolute():
    ckpt = REPO / ckpt
if not ckpt.exists():
    raise SystemExit(f"no checkpoint at {ckpt}")

STATE = {"ready": False, "status": "loading", "model": SIZES[args.model],
         "checkpoint": str(ckpt), "resolution": args.resolution, "device": "?",
         "batch": None, "images": 0, "seconds": 0.0, "error": None}
MODEL = {"m": None}
LOCK = threading.Lock()          # one GPU, one forward pass at a time


def _load():
    t0 = time.time()
    try:
        import numpy as np  # noqa: F401  (imported here so the failure is reported, not fatal)
        import torch
        from rfdetr import __dict__ as rfdetr_ns

        kwargs = {"pretrain_weights": str(ckpt)}
        if args.resolution:
            kwargs["resolution"] = args.resolution
        m = rfdetr_ns[SIZES[args.model]](**kwargs)
        try:
            # The optimized model is compiled for ONE batch size and refuses any
            # other, so the compile size and the chunk size below are the same
            # number and short chunks are padded up to it.
            m.optimize_for_inference(batch_size=args.batch)
            STATE["batch"] = args.batch
        except Exception as e:                      # older rfdetr, or no compile path
            STATE["batch"] = None                   # uncompiled: any batch is fine
            print(f"optimize_for_inference skipped: {e}", flush=True)
        MODEL["m"] = m
        STATE["device"] = "cuda" if torch.cuda.is_available() else "cpu"
        STATE["resolution"] = getattr(m, "resolution", args.resolution)
        STATE["ready"] = True
        STATE["status"] = "ready"
        print(f"detector ready in {time.time() - t0:.1f}s: {SIZES[args.model]} "
              f"{ckpt.name} on {STATE['device']}", flush=True)
    except Exception as e:                          # a dead server must say why
        STATE["status"] = "failed"
        STATE["error"] = f"{type(e).__name__}: {e}"
        print(f"detector failed to load: {STATE['error']}", flush=True)


def _predict(frames, threshold):
    """frames: list of HxWx3 uint8 arrays -> (boxes, scores) lists."""
    m = MODEL["m"]
    fixed = STATE["batch"]
    boxes, scores = [], []
    for i in range(0, len(frames), args.batch):
        chunk = frames[i:i + args.batch]
        want = len(chunk)
        if fixed and want < fixed:
            # a compiled model only accepts its own batch size: pad the tail with
            # copies of the last frame and throw the extra results away
            chunk = chunk + [chunk[-1]] * (fixed - want)
        with LOCK:
            dets = m.predict(chunk, threshold=threshold)
        # rfdetr returns one Detections for a single image, a list for a batch
        if not isinstance(dets, (list, tuple)):
            dets = [dets]
        for d in dets[:want]:
            xyxy = getattr(d, "xyxy", None)
            conf = getattr(d, "confidence", None)
            boxes.append([] if xyxy is None else [[round(float(v), 2) for v in b] for b in xyxy])
            scores.append([] if conf is None else [round(float(c), 4) for c in conf])
    return boxes, scores


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *a):                      # one line per request is noise here
        pass

    def _json(self, code, payload):
        body = json.dumps(payload).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path.startswith("/health"):
            self._json(200, STATE)
        else:
            self._json(404, {"error": "no such path"})

    def do_POST(self):
        if not self.path.startswith("/detect"):
            return self._json(404, {"error": "no such path"})
        if not STATE["ready"]:
            return self._json(503, {"error": STATE["error"] or STATE["status"]})
        try:
            length = int(self.headers.get("Content-Length") or 0)
            raw = self.rfile.read(length)
            head, _, payload = raw.partition(b"\n")
            hdr = json.loads(head)
            n, h, w = int(hdr["n"]), int(hdr["h"]), int(hdr["w"])
            thr = float(hdr.get("threshold", args.threshold))
            want = n * h * w * 3
            if len(payload) != want:
                return self._json(400, {"error": f"expected {want} bytes, got {len(payload)}"})
            import numpy as np
            # frombuffer is read-only and torch refuses to own it: copy once here
            frames = np.frombuffer(payload, dtype=np.uint8).reshape(n, h, w, 3).copy()
            t0 = time.time()
            boxes, scores = _predict([frames[i] for i in range(n)], thr)
            STATE["images"] += n
            STATE["seconds"] = round(STATE["seconds"] + (time.time() - t0), 2)
            self._json(200, {"boxes": boxes, "scores": scores})
        except Exception as e:
            self._json(500, {"error": f"{type(e).__name__}: {e}"})


threading.Thread(target=_load, daemon=True).start()
srv = ThreadingHTTPServer((args.host, args.port), Handler)
print(f"detect server on http://{args.host}:{args.port} "
      f"({SIZES[args.model]}, {ckpt.name}) -- loading", flush=True)
srv.serve_forever()
