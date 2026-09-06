"""progression/<tag>/replay.json -> progression.mp4: the three.js training reel.

A GPU-box training run with --defer_progression leaves one short replay per
policy snapshot under runs/<id>/progression/ (plus index.json with captions),
instead of rendering anything on the box — the box has no web client. This
script re-films every snapshot in the app's three.js scene on this machine
(the same headless-Chrome lane as scripts/render_three_replay.py), stamps the
iteration caption on each clip, and concatenates them oldest-first into
runs/<id>/progression.mp4 — the policy going from aimless to lethal.

    .venv/bin/python scripts/render_progression.py runs/<id>
        [--fps 24] [--width 1280] [--height 720]

The next server on :3199 is booted once (production `next start`; a dev server
is never reused — see export_replay.mjs) and shared by every clip.
"""
import argparse
import json
import os
import signal
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CLIENT = ROOT / "web" / "client"
FFMPEG = "/opt/homebrew/bin/ffmpeg" if Path("/opt/homebrew/bin/ffmpeg").exists() else "ffmpeg"
FONTS = ["/System/Library/Fonts/Monaco.ttf",
         "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
         "/System/Library/Fonts/Supplemental/Arial.ttf"]


def _probe(url: str) -> bool:
    try:
        with urllib.request.urlopen(f"{url}/render", timeout=2) as r:
            return r.status < 500
    except OSError:
        return False


def ensure_server():
    """A production next server for the /render page: reuse :3199 when one
    answers, else build (first use only) and boot our own."""
    url = "http://127.0.0.1:3199"
    if _probe(url):
        return url, None
    if not (CLIENT / ".next" / "BUILD_ID").exists():
        print("[progression] no production build — running next build (one-time)…", flush=True)
        subprocess.run(["npx", "next", "build"], cwd=CLIENT, check=True)
    print("[progression] booting next start on :3199", flush=True)
    child = subprocess.Popen(["npx", "next", "start", "-p", "3199"], cwd=CLIENT,
                             stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                             start_new_session=True)
    for _ in range(60):
        time.sleep(1)
        if _probe(url):
            return url, child
    raise RuntimeError("next start did not come up on :3199")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir", help="run directory containing progression/*/replay.json")
    ap.add_argument("--fps", type=int, default=24)
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=720)
    args = ap.parse_args()

    run = Path(args.run_dir).resolve()
    prog = run / "progression"
    index_f = prog / "index.json"
    if index_f.is_file():
        entries = json.loads(index_f.read_text())
    else:                                       # fall back to whatever replays are there
        entries = [{"tag": d.name, "caption": d.name.upper(),
                    "replay": f"progression/{d.name}/replay.json"}
                   for d in sorted(prog.iterdir()) if (d / "replay.json").is_file()]
    entries = [e for e in entries if (run / e["replay"]).is_file()]
    if not entries:
        sys.exit(f"no progression replays in {prog}")
    # oldest policy first, final last — index.json is written in that order, but
    # sort defensively: numeric tags ascending, "final" to the back
    entries.sort(key=lambda e: (e["tag"] == "final", e["tag"]))

    world = json.loads((run / entries[0]["replay"]).read_text())["world"]
    sys.path.insert(0, str(ROOT))
    from vesper.worlds.webgeo import ensure_ground_jpg, ensure_world3d
    world_json = ensure_world3d(world, ROOT / "assets")
    ground = ensure_ground_jpg(world, ROOT / "assets")

    font = next((f for f in FONTS if Path(f).exists()), None)
    t0 = time.time()
    url, child = ensure_server()
    clips = []
    try:
        for e in entries:
            rp = run / e["replay"]
            raw = rp.parent / "clip.mp4"
            cmd = ["node", str(CLIENT / "scripts" / "export_replay.mjs"),
                   "--replay", str(rp), "--world-json", str(world_json),
                   "--out-world", str(raw), "--cams", "world", "--full",
                   "--fps", str(args.fps), "--width", str(args.width),
                   "--height", str(args.height), "--url", url]
            if ground:
                cmd += ["--ground", str(ground)]
            r = subprocess.run(cmd, cwd=CLIENT)
            if r.returncode != 0 or not raw.is_file():
                print(f"[progression] clip {e['tag']} FAILED — skipping", flush=True)
                continue
            out = rp.parent / "captioned.mp4"
            if font:
                text = e["caption"].replace(":", r"\:").replace("'", "")
                vf = (f"drawtext=fontfile={font}:text='{text}':x=28:y=24:fontsize=34:"
                      f"fontcolor=white:box=1:boxcolor=black@0.55:boxborderw=10")
                cap = subprocess.run([FFMPEG, "-y", "-nostdin", "-loglevel", "error",
                                      "-i", str(raw), "-vf", vf, "-c:v", "libx264",
                                      "-preset", "veryfast", "-crf", "21",
                                      "-pix_fmt", "yuv420p", str(out)])
                if cap.returncode != 0:
                    out = raw                                  # caption is best-effort
            else:
                out = raw
            clips.append(out)
            print(f"[progression] clip {e['tag']} ready ({e['caption']})", flush=True)
        if not clips:
            sys.exit("[progression] every clip failed")
        lst = prog / "concat.txt"
        lst.write_text("".join(f"file '{c}'\n" for c in clips))
        final = run / "progression.mp4"
        # captioned clips share identical encode settings, so stream-copy concat;
        # a mixed list (a caption fell back to the raw clip) re-encodes instead.
        mixed = any(c.name == "clip.mp4" for c in clips) and any(
            c.name == "captioned.mp4" for c in clips)
        codec = ["-c:v", "libx264", "-preset", "veryfast", "-crf", "18",
                 "-pix_fmt", "yuv420p"] if mixed else ["-c", "copy"]
        subprocess.run([FFMPEG, "-y", "-nostdin", "-loglevel", "error", "-f", "concat",
                        "-safe", "0", "-i", str(lst), *codec, "-movflags", "+faststart",
                        str(final)], check=True)
        print(f"[progression] {len(clips)} clips -> {final} "
              f"in {time.time() - t0:.0f}s", flush=True)
    finally:
        if child:
            try:
                os.killpg(child.pid, signal.SIGTERM)
            except OSError:
                pass
            subprocess.run(["pkill", "-f", "next start -p 3199"],
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


if __name__ == "__main__":
    main()
