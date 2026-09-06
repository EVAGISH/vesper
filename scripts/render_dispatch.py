"""One entry point for every replay renderer — the render FLAG lives here.

    .venv/bin/python scripts/render_dispatch.py runs/<id> --renderers three,tactical

Three lanes, all reading the same runs/<id>/replay.json, all coexisting:

    tactical  scripts/render_replay.py       2D top-down mp4, Mac, seconds
    three     scripts/render_three_replay.py three.js chase+fpv mp4s, Mac, ~a minute
    isaac     scripts/render_isaac_replay.py photoreal RTX on the GPU box, slow —
              pushed over SSH, results rsync'd back (the hero-shot lane)

Called by the warm session's RECORD SORTIE, by train_search_native.py after
training, and by the web UI's POST /api/runs/<id>/render. Lanes run
sequentially, each best-effort: one failing never blocks the others.
"""
import argparse
import json
import os
import re
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RENDERERS = ("tactical", "three", "isaac")

# droplet access for the photoreal lane (same key/layout web/server/app.py uses)
_SSH_KEY = os.path.expanduser(os.environ.get("KEY_FILE", "~/.ssh/vesper.pem"))
_SSH_OPTS = f"ssh -i {_SSH_KEY} -o BatchMode=yes -o StrictHostKeyChecking=accept-new"


def _droplet_ip():
    """Public IP of the GPU droplet, or None (box down / no token in .env)."""
    token = os.environ.get("DIGITALOCEAN_TOKEN")
    if not token:
        envf = ROOT / ".env"
        if envf.exists():
            for line in envf.read_text().splitlines():
                m = re.match(r"^(?:export\s+)?DIGITALOCEAN_TOKEN=[\"']?([^\"'#\s]+)", line)
                if m:
                    token = m.group(1)
                    break
    if not token:
        return None
    name = os.environ.get("DROPLET_NAME", "vesper-dev")
    req = urllib.request.Request(
        "https://api.digitalocean.com/v2/droplets?tag_name=vesper",
        headers={"Authorization": f"Bearer {token}"})
    try:
        with urllib.request.urlopen(req, timeout=6) as r:
            for drop in json.load(r).get("droplets", []):
                if drop.get("name") != name:
                    continue
                for net in drop.get("networks", {}).get("v4", []):
                    if net.get("type") == "public":
                        return net.get("ip_address")
    except OSError:
        pass
    return None


def _set_isaac_status(run_dir: Path, status: str):
    """Stamp the photoreal render's state into the run's manifest, so the Runs
    tab can show 'rendering photoreal…' / 'pending' beside the fast lanes."""
    mf = run_dir / "manifest.json"
    try:
        m = json.loads(mf.read_text())
    except (OSError, json.JSONDecodeError):
        m = {}
    m["isaac"] = status
    mf.write_text(json.dumps(m))


def render_isaac(run_dir: Path):
    """Photoreal Isaac render on the droplet: push replay.json, run
    render_isaac_replay.py in the Isaac container, pull isaac*.mp4 back.
    Best-effort: droplet down leaves the run marked isaac:"pending"."""
    run_id = run_dir.name
    _set_isaac_status(run_dir, "rendering")
    ip = _droplet_ip()
    if not ip:
        _set_isaac_status(run_dir, "pending")
        print(f"[dispatch] photoreal deferred: gpu box is offline (runs/{run_id})", flush=True)
        return
    try:
        subprocess.run(["rsync", "-az", "-e", _SSH_OPTS, str(run_dir),
                        f"root@{ip}:vesper/runs/"], timeout=300, check=True)
        r = subprocess.run(
            ["ssh", "-i", _SSH_KEY, "-o", "BatchMode=yes",
             "-o", "StrictHostKeyChecking=accept-new", f"root@{ip}",
             f"cd vesper && docker compose -f docker/compose.yml run --rm "
             f"--name vsp_replay_{run_id} sim "
             f"/isaac-sim/python.sh scripts/render_isaac_replay.py runs/{run_id}"],
            capture_output=True, text=True, timeout=2400)
        subprocess.run(["rsync", "-az", "-e", _SSH_OPTS,
                        f"root@{ip}:vesper/runs/{run_id}/isaac*.mp4",
                        str(run_dir) + "/"], timeout=300)
        ok = (run_dir / "isaac_fpv.mp4").exists()
        _set_isaac_status(run_dir, "done" if ok else "failed")
        print(f"[dispatch] photoreal {'done' if ok else 'FAILED'} -> runs/{run_id}"
              + ("" if ok else f" ({(r.stderr or r.stdout)[-200:]})"), flush=True)
    except (OSError, subprocess.SubprocessError) as e:
        _set_isaac_status(run_dir, "pending")
        print(f"[dispatch] photoreal deferred ({e}); re-run render_isaac_replay "
              f"against runs/{run_id} later", flush=True)


def dispatch(run_dir: Path, renderers: list[str]) -> int:
    """Run each requested lane in turn; returns the count that failed."""
    failed = 0
    for r in renderers:
        t0 = time.time()
        try:
            if r == "tactical":
                subprocess.run([sys.executable, "scripts/render_replay.py", str(run_dir)],
                               cwd=ROOT, check=True)
            elif r == "three":
                subprocess.run([sys.executable, "scripts/render_three_replay.py",
                                str(run_dir)], cwd=ROOT, check=True)
            elif r == "isaac":
                render_isaac(run_dir)
            print(f"[dispatch] {r} finished in {time.time() - t0:.0f}s", flush=True)
        except subprocess.SubprocessError as e:
            failed += 1
            print(f"[dispatch] {r} FAILED: {e}", flush=True)
    return failed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("--renderers", default="three,tactical",
                    help=f"comma list from {RENDERERS}")
    args = ap.parse_args()
    run_dir = Path(args.run_dir)
    if not (run_dir / "replay.json").is_file():
        sys.exit(f"no replay.json in {run_dir}")
    rends = [r.strip() for r in args.renderers.split(",") if r.strip()]
    bad = set(rends) - set(RENDERERS)
    if bad:
        sys.exit(f"unknown renderer(s) {sorted(bad)}; pick from {RENDERERS}")
    sys.exit(1 if dispatch(run_dir, rends) else 0)


if __name__ == "__main__":
    main()
