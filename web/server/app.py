"""Vesper runs browser -- local web UI over runs/ and the scenario specs.

    .venv/bin/python -m uvicorn web.server.app:app --port 8777
    open http://localhost:8777

Reads the same artifacts every run already writes (manifest.json, *.mp4,
track.png, trajectory.parquet, scenario.json); no state of its own. Videos are
served with HTTP Range support so <video> can stream and scrub.
"""
import json
import re
from pathlib import Path

import pyarrow.parquet as pq
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, Response

ROOT = Path(__file__).resolve().parents[2]
RUNS = ROOT / "runs"
RUN_ID = re.compile(r"^[\w.-]+$")

app = FastAPI(title="vesper")

MEDIA_TYPES = {".mp4": "video/mp4", ".png": "image/png", ".json": "application/json",
               ".jsonl": "text/plain", ".parquet": "application/octet-stream"}


def _run_dir(run_id: str) -> Path:
    if not RUN_ID.match(run_id):
        raise HTTPException(400, "bad run id")
    d = RUNS / run_id
    if not d.is_dir():
        raise HTTPException(404, "no such run")
    return d


@app.get("/api/runs")
def list_runs():
    out = []
    for d in sorted(RUNS.iterdir(), reverse=True) if RUNS.is_dir() else []:
        mf = d / "manifest.json"
        if not d.is_dir() or not mf.exists():
            continue
        try:
            manifest = json.loads(mf.read_text())
        except json.JSONDecodeError:
            manifest = {}
        files = sorted(p.name for p in d.iterdir() if p.suffix in MEDIA_TYPES)
        out.append({"id": d.name, "manifest": manifest, "files": files})
    return out


@app.get("/api/runs/{run_id}/trajectory")
def trajectory(run_id: str, max_points: int = 2000):
    f = _run_dir(run_id) / "trajectory.parquet"
    if not f.exists():
        raise HTTPException(404, "no trajectory")
    t = pq.read_table(f, columns=["t", "px", "py", "pz"])
    stride = max(1, t.num_rows // max_points)
    cols = {c: t.column(c).to_pylist()[::stride] for c in ("t", "px", "py", "pz")}
    return cols


@app.get("/api/scenarios")
def scenarios():
    out = []
    for f in sorted(ROOT.glob("*.json")):
        try:
            spec = json.loads(f.read_text())
        except (json.JSONDecodeError, UnicodeDecodeError):
            continue
        if not isinstance(spec, dict) or "waypoints" not in spec:
            continue
        out.append({
            "file": f.name,
            "world": spec.get("world"),
            "terrain_usd": (spec.get("terrain") or {}).get("usd"),
            "waypoints": len(spec.get("waypoints") or []),
            "wind_ms": spec.get("wind_speed_ms"),
            "visibility_m": spec.get("visibility_m"),
            "cruise_ms": spec.get("cruise_ms"),
            "max_sim_s": spec.get("max_sim_s"),
            "command": f"docker compose run --rm sim /isaac-sim/python.sh scripts/fly_mission.py {f.name}",
        })
    return out


@app.get("/media/{run_id}/{name}")
def media(run_id: str, name: str, request: Request):
    d = _run_dir(run_id)
    f = d / name
    if not f.is_file() or f.parent != d or f.suffix not in MEDIA_TYPES:
        raise HTTPException(404, "no such file")
    mtype = MEDIA_TYPES[f.suffix]
    size = f.stat().st_size
    range_header = request.headers.get("range")
    if not range_header:
        return FileResponse(f, media_type=mtype)
    m = re.match(r"bytes=(\d*)-(\d*)", range_header)
    start = int(m.group(1) or 0)
    end = min(int(m.group(2) or size - 1), size - 1)
    with open(f, "rb") as fh:
        fh.seek(start)
        chunk = fh.read(end - start + 1)
    return Response(chunk, status_code=206, media_type=mtype, headers={
        "Content-Range": f"bytes {start}-{end}/{size}",
        "Accept-Ranges": "bytes",
        "Content-Length": str(len(chunk)),
    })


@app.get("/")
def index():
    return FileResponse(Path(__file__).parents[1] / "index.html", media_type="text/html")
