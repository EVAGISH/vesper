"""replay.json -> three.mp4 + three_fpv.mp4: the on-device three.js render lane.

Re-films a run's after-action log (vesper.native.replay schema) in the SAME
three.js scene the web client's live 3D view draws — real terrain, buildings,
the Kenney low-poly trees, drone/tank models — entirely on this machine:
headless Chrome drives the app's /render page frame-by-frame (deterministic
seek, no wall clock) and pipes the captures into ffmpeg. A ~60 s sortie
exports in about a minute; the photoreal Isaac lane on the GPU box stays the
separate, slower track for the hero shot (scripts/render_isaac_replay.py).

    .venv/bin/python scripts/render_three_replay.py runs/<id>
        [--cam both|world|fpv] [--fps 24] [--width 1280] [--height 720] [--full]

Writes three.mp4 (chase of the hero drone — the one that strikes — lingering
on the impact) and three_fpv.mp4 (the hero's own forward-down lens, held on
the last thing it saw after the airframe is expended) into the run dir, where
the Runs tab picks up any *.mp4 automatically.
"""
import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CLIENT = ROOT / "web" / "client"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir", help="run directory containing replay.json")
    ap.add_argument("--cam", choices=["world", "fpv", "both"], default="both")
    ap.add_argument("--fps", type=int, default=24)
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=720)
    ap.add_argument("--full", action="store_true",
                    help="render the whole log (default trims to ~4 s past the last strike)")
    args = ap.parse_args()

    run = Path(args.run_dir).resolve()      # the driver runs from web/client
    replay = run / "replay.json"
    if not replay.is_file():
        sys.exit(f"no replay.json in {run}")
    world = json.loads(replay.read_text())["world"]

    # world geometry + ground ortho, same artifacts the live view is served
    sys.path.insert(0, str(ROOT))
    from vesper.worlds.webgeo import ensure_ground_jpg, ensure_world3d
    world_json = ensure_world3d(world, ROOT / "assets")
    ground = ensure_ground_jpg(world, ROOT / "assets")

    cams = {"world": "world", "fpv": "fpv", "both": "world,fpv"}[args.cam]
    cmd = ["node", str(CLIENT / "scripts" / "export_replay.mjs"),
           "--replay", str(replay), "--world-json", str(world_json),
           "--out-world", str(run / "three.mp4"), "--out-fpv", str(run / "three_fpv.mp4"),
           "--cams", cams, "--fps", str(args.fps),
           "--width", str(args.width), "--height", str(args.height)]
    if ground:
        cmd += ["--ground", str(ground)]
    if args.full:
        cmd += ["--full"]

    t0 = time.time()
    r = subprocess.run(cmd, cwd=CLIENT)
    dt = time.time() - t0
    made = [n for n in ("three.mp4", "three_fpv.mp4") if (run / n).is_file()]
    print(f"[three] {', '.join(made) or 'NOTHING'} in {dt:.0f}s -> {run}", flush=True)
    sys.exit(r.returncode)


if __name__ == "__main__":
    main()
