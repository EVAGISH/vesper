"""Vesper runs browser -- local web UI over runs/ and the scenario specs.

    .venv/bin/python -m uvicorn web.server.app:app --port 8777
    open http://localhost:8777

Reads the same artifacts every run already writes (manifest.json, *.mp4,
track.png, trajectory.parquet, scenario.json); no state of its own. Videos are
served with HTTP Range support so <video> can stream and scrub.
"""
import json
import os
import re
import subprocess
import tempfile
import time
import urllib.request
from pathlib import Path

import pyarrow.parquet as pq
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, Response
from pydantic import BaseModel

ROOT = Path(__file__).resolve().parents[2]
RUNS = ROOT / "runs"
ASSETS = ROOT / "assets"
RUN_ID = re.compile(r"^[\w.-]+$")
POLICY_PATH = re.compile(r"^runs/[\w.-]+/[\w.-]+\.pt$")

# GPU box access — same key and layout capture_pull.sh already uses.
KEY_FILE = os.path.expanduser(os.environ.get("KEY_FILE", "~/.ssh/vesper.pem"))
REMOTE_DIR = "vesper"
JOBS_FILE = ROOT / ".vesper_jobs.json"

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


@app.get("/api/models")
def models():
    """Every policy checkpoint under runs/ (train_* writes runs/<id>/*.pt),
    with the final training metrics from the sibling curve.jsonl when present."""
    out = []
    for d in sorted(RUNS.iterdir(), reverse=True) if RUNS.is_dir() else []:
        if not d.is_dir():
            continue
        metrics = {}
        curve = d / "curve.jsonl"
        if curve.exists():
            try:
                lines = [ln for ln in curve.read_text().splitlines() if ln.strip()]
                if lines:
                    last = json.loads(lines[-1])
                    metrics = {k: v for k, v in last.items() if isinstance(v, (int, float))}
            except (json.JSONDecodeError, UnicodeDecodeError, OSError):
                pass
        for f in sorted(d.glob("*.pt")):
            st = f.stat()
            out.append({
                "run": d.name,
                "file": f.name,
                "path": f"runs/{d.name}/{f.name}",
                "bytes": st.st_size,
                "mtime": st.st_mtime,
                "metrics": metrics,
            })
    return out


_live_cache = {"t": 0.0, "ip": None}


def _droplet_ip():
    """Public IP of the GPU droplet (tag vesper, name from DROPLET_NAME).
    Cached 30 s; None when the box is down or no DIGITALOCEAN_TOKEN in .env."""
    now = time.time()
    if now - _live_cache["t"] < 30:
        return _live_cache["ip"]
    token = os.environ.get("DIGITALOCEAN_TOKEN")
    if not token:
        env = ROOT / ".env"
        if env.exists():
            for line in env.read_text().splitlines():
                m = re.match(r"^(?:export\s+)?DIGITALOCEAN_TOKEN=[\"']?([^\"'#\s]+)", line)
                if m:
                    token = m.group(1)
                    break
    ip = None
    if token:
        name = os.environ.get("DROPLET_NAME", "vesper-dev")
        req = urllib.request.Request(
            "https://api.digitalocean.com/v2/droplets?tag_name=vesper",
            headers={"Authorization": f"Bearer {token}"},
        )
        try:
            with urllib.request.urlopen(req, timeout=4) as r:
                for drop in json.load(r).get("droplets", []):
                    if drop.get("name") != name:
                        continue
                    for net in drop.get("networks", {}).get("v4", []):
                        if net.get("type") == "public":
                            ip = net.get("ip_address")
        except OSError:
            pass
    _live_cache.update(t=now, ip=ip)
    return ip


@app.get("/api/live")
def live():
    return {"ip": _droplet_ip()}


# ---------------------------------------------------------------- jobs
# The one write path: launch whitelisted sim jobs on the GPU box over SSH,
# each in a named docker container so status and stop are just `docker ps`
# and `docker rm -f`. The registry is a JSON file next to runs/.

JOB_KINDS = {
    "train": "scripts/train_search.py --num_envs 1024 --iters 1500 --headless",
    "fly": "scripts/fly_search.py --policy {policy} --seconds 90 --headless --enable_cameras",
    "eval": "scripts/eval_search.py --policy {policy} --num_envs 256 --episodes 400 --headless",
    "mission": "scripts/fly_mission.py {scenario}",
    "live": "scripts/live_world.py assets/cornell/cornell.usd",
}
SCENARIO_FILE = re.compile(r"^[\w.-]+\.json$")


def _ssh(cmd: str, timeout: int = 25):
    if not os.path.exists(KEY_FILE):
        raise HTTPException(503, f"ssh key not found: {KEY_FILE}")
    ip = _droplet_ip()
    if not ip:
        raise HTTPException(503, "gpu box is offline")
    r = subprocess.run(
        ["ssh", "-i", KEY_FILE, "-o", "BatchMode=yes", "-o", "ConnectTimeout=8",
         "-o", "StrictHostKeyChecking=accept-new", f"root@{ip}", cmd],
        capture_output=True, text=True, timeout=timeout,
    )
    return r.returncode, r.stdout, r.stderr


def _load_jobs():
    try:
        return json.loads(JOBS_FILE.read_text())
    except (OSError, json.JSONDecodeError):
        return []


def _save_jobs(jobs):
    JOBS_FILE.write_text(json.dumps(jobs[-40:], indent=1))


class JobReq(BaseModel):
    kind: str
    policy: str | None = None
    scenario: str | None = None


@app.post("/api/jobs")
def start_job(req: JobReq):
    if req.kind not in JOB_KINDS:
        raise HTTPException(400, "unknown job kind")
    tmpl = JOB_KINDS[req.kind]
    if "{policy}" in tmpl:
        if not req.policy or not POLICY_PATH.match(req.policy):
            raise HTTPException(400, "policy must look like runs/<id>/<name>.pt")
        tmpl = tmpl.format(policy=req.policy)
    if "{scenario}" in tmpl:
        if not req.scenario or not SCENARIO_FILE.match(req.scenario) \
                or not (ROOT / req.scenario).is_file():
            raise HTTPException(400, "unknown scenario file")
        tmpl = tmpl.format(scenario=req.scenario)
    if req.kind == "live":
        # one live session at a time -- reuse it instead of stacking GPU copies
        for j in _load_jobs():
            if j["kind"] == "live" and j["status"] == "running":
                return {"id": j["id"]}
    jid = time.strftime(f"job-%Y%m%d-%H%M%S-{req.kind}")
    # sim uses host networking, so the WebRTC ports pass through without flags.
    # `run -d` detaches on the box; attached-run + nohup keeps the ssh session
    # open until the sim exits, which times out the API call.
    remote = (
        f"cd {REMOTE_DIR} && "
        f"docker compose -f docker/compose.yml run -d --rm --name vsp_{jid} sim "
        f"/isaac-sim/python.sh {tmpl} && echo launched"
    )
    rc, out, err = _ssh(remote)
    if rc != 0 or "launched" not in out:
        raise HTTPException(502, f"launch failed: {err.strip() or out.strip()}")
    jobs = _load_jobs()
    jobs.append({"id": jid, "kind": req.kind, "policy": req.policy,
                 "started": time.time(), "status": "running", "finished": None,
                 "log": ""})
    _save_jobs(jobs)
    return {"id": jid}


_jobs_cache = {"t": 0.0, "data": None}


@app.get("/api/jobs")
def list_jobs():
    now = time.time()
    if _jobs_cache["data"] is not None and now - _jobs_cache["t"] < 5:
        return _jobs_cache["data"]
    jobs = _load_jobs()
    open_jobs = [j for j in jobs if j["status"] == "running"]
    if open_jobs and _droplet_ip():
        tails = "; ".join(
            f"echo __JOB__{j['id']}; docker logs --tail 12 vsp_{j['id']} 2>&1 | tail -c 600"
            for j in open_jobs
        )
        try:
            rc, out, _ = _ssh(f"docker ps --format '{{{{.Names}}}}'; echo __SEP__; {tails}")
        except HTTPException:
            rc, out = 1, ""
        if rc == 0:
            names, _, logpart = out.partition("__SEP__")
            alive = set(names.split())
            sections = logpart.split("__JOB__")
            logs = {}
            for s in sections:
                s = s.strip("\n")
                if s:
                    head, _, body = s.partition("\n")
                    logs[head.strip()] = body[-600:]
            for j in jobs:
                if j["status"] != "running":
                    continue
                j["log"] = logs.get(j["id"], j.get("log", ""))
                if f"vsp_{j['id']}" not in alive:
                    died_fast = now - j["started"] < 60
                    looks_broken = re.search(
                        r"error|not found|traceback|no such|failed",
                        j.get("log") or "", re.IGNORECASE)
                    j["status"] = "failed" if (died_fast and looks_broken) else "done"
                    j["finished"] = now
            _save_jobs(jobs)
    elif open_jobs:
        for j in open_jobs:
            j["log"] = (j.get("log") or "") or "(box unreachable)"
    data = list(reversed(jobs[-12:]))
    _jobs_cache.update(t=now, data=data)
    return data


@app.post("/api/jobs/{jid}/stop")
def stop_job(jid: str):
    if not RUN_ID.match(jid):
        raise HTTPException(400, "bad job id")
    _ssh(f"docker rm -f vsp_{jid} >/dev/null 2>&1; true")
    jobs = _load_jobs()
    for j in jobs:
        if j["id"] == jid and j["status"] == "running":
            j["status"] = "stopped"
            j["finished"] = time.time()
    _save_jobs(jobs)
    _jobs_cache["t"] = 0.0
    return {"ok": True}


@app.post("/api/sync")
def sync_runs():
    """Pull run artifacts from the box into the local runs/ mirror."""
    ip = _droplet_ip()
    if not ip:
        raise HTTPException(503, "gpu box is offline")
    RUNS.mkdir(exist_ok=True)
    r = subprocess.run(
        ["rsync", "-az", "-e",
         f"ssh -i {KEY_FILE} -o BatchMode=yes -o StrictHostKeyChecking=accept-new",
         f"root@{ip}:{REMOTE_DIR}/runs/", str(RUNS) + "/"],
        capture_output=True, text=True, timeout=300,
    )
    if r.returncode != 0:
        raise HTTPException(502, f"rsync failed: {r.stderr.strip()[-300:]}")
    n = sum(1 for d in RUNS.iterdir() if d.is_dir())
    return {"ok": True, "runs": n}


# ---------------------------------------------------------------- site map
@app.get("/api/site")
def site():
    """Worlds with a baked map: extent metadata + the ground ortho for map views."""
    out = []
    for mj in sorted(ASSETS.glob("*/*_map.json")) if ASSETS.is_dir() else []:
        try:
            meta = json.loads(mj.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        world = mj.parent.name
        if (mj.parent / "ground.png").exists():
            out.append({"world": world, "half_m": meta.get("half_m"),
                        "ground": f"/site/{world}/ground"})
    return out


@app.get("/site/{world}/ground")
def site_ground(world: str):
    """Ground ortho, downscaled once to a web-friendly jpg (the source is 30 MB)."""
    if not RUN_ID.match(world):
        raise HTTPException(400, "bad world")
    src = ASSETS / world / "ground.png"
    if not src.is_file():
        raise HTTPException(404, "no ground texture")
    cache = Path(tempfile.gettempdir()) / f"vesper_{world}_ground.jpg"
    if not cache.exists() or cache.stat().st_mtime < src.stat().st_mtime:
        try:
            from PIL import Image
            Image.MAX_IMAGE_PIXELS = None
            img = Image.open(src).convert("RGB")
            img.thumbnail((2048, 2048), Image.BILINEAR)
            img.save(cache, "JPEG", quality=82)
        except ImportError:
            return FileResponse(src, media_type="image/png")
    return FileResponse(cache, media_type="image/jpeg")


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
