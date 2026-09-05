"""replay.json -> drone's-eye (FPV) mp4: what the drone sees as it strikes.

The top-down tactical view is render_replay.py; this is the first-person lens --
the body-fixed, forward-down camera (the same cone the policy flies), raymarched
from the world rasters (vesper.native.camera). Targets and other drones are drawn
in-frame, so a tank grows in the sensor as the drone dives on it.

    .venv/bin/python scripts/render_fpv_replay.py runs/<id> [--device mps] [--stride 2]

Writes <run>/fpv.mp4. Photoreal (Isaac RTX) fpv is the separate render_isaac_replay track.
"""
import argparse
import json
import math
import subprocess
from pathlib import Path

import numpy as np

from vesper.lab.search_task import SearchCfg
from vesper.native.camera import RasterCamera
from vesper.worlds.heightmap import WorldMap

ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / "assets"
FPS = 24


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("--world", default=None)
    ap.add_argument("--stride", type=int, default=2)
    ap.add_argument("--device", default="mps")
    ap.add_argument("--res", type=int, default=720)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    run = Path(args.run_dir)
    rep = json.loads((run / "replay.json").read_text())
    world = args.world or rep["world"]
    frames = rep["frames"][:: max(1, args.stride)]
    K = rep["targets"]
    tcfg = SearchCfg()

    wmap = WorldMap(str(ASSETS / world / f"{world}_map.npz"), device=args.device)
    ortho = ASSETS / world / "ground.png"
    cam = RasterCamera(wmap, ortho_path=str(ortho) if ortho.exists() else None,
                       res=(args.res, args.res), fov_half_deg=tcfg.fov_half_deg,
                       device=args.device)

    out = Path(args.out) if args.out else run / "fpv.mp4"
    ff = subprocess.Popen(
        ["ffmpeg", "-y", "-f", "rawvideo", "-pixel_format", "rgb24",
         "-video_size", f"{args.res}x{args.res}", "-framerate", str(FPS), "-i", "-",
         "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20",
         "-preset", "medium", "-movflags", "+faststart", str(out)],
        stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    import torch
    for fr in frames:
        d0 = fr["d"][0]
        yaw = fr["hdg"][0]
        quat = [math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2)]   # level, yaw-only
        # targets on the ground; only draw those already detected (what the
        # operator would have on the map). z from the terrain under them.
        tgt = []
        for i in range(K):
            tx, ty, known, reached = fr["tg"][i]
            gz = float(wmap.ground_at(torch.tensor([tx], device=args.device),
                                      torch.tensor([ty], device=args.device)))
            tgt.append([tx, ty, gz + 1.0])
        img = cam.render(d0, quat, tcfg.cam_pitch_deg, targets=tgt, drones=fr["d"][1:])
        ff.stdin.write(np.asarray(img, np.uint8).tobytes())
    ff.stdin.close()
    ff.wait()
    print(f"wrote {out} ({len(frames)} frames, {len(frames)/FPS:.0f}s)")


if __name__ == "__main__":
    main()
