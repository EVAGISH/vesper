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
import sys
import tempfile
import threading
import time
import urllib.parse
import urllib.request
from pathlib import Path

import pyarrow.parquet as pq
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
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


# ---------------------------------------------------------------- environments
# Add any place as a world: build_geo_world runs locally (Copernicus + Esri +
# OSM, no keys), then the world rsyncs to the GPU box so Isaac can use it. The
# build is a local subprocess tracked in memory; its log tails into /api/environments.

SITE_NAME = re.compile(r"^[a-z][a-z0-9_]{1,31}$")
ENV_BUILDS: dict[str, dict] = {}


class EnvBuildReq(BaseModel):
    name: str
    lat: float
    lon: float
    half_km: float = 1.0


def _env_log(name: str) -> Path:
    return ROOT / f".envbuild_{name}.log"


def _run_env_build(name: str, lat: float, lon: float, half_km: float) -> None:
    logf = _env_log(name)
    try:
        with open(logf, "w") as f:
            p = subprocess.run(
                [sys.executable, "scripts/build_geo_world.py", name,
                 "--lat", str(lat), "--lon", str(lon), "--half-km", str(half_km),
                 "--source", "global"],
                cwd=ROOT, stdout=f, stderr=subprocess.STDOUT, timeout=1800)
        ok = p.returncode == 0 and (ROOT / "assets" / name / f"{name}.usd").exists()
        if ok:                                              # push the new world to the box
            ip = _droplet_ip()
            if ip:
                ssh = f"ssh -i {KEY_FILE} -o StrictHostKeyChecking=accept-new"
                subprocess.run(["rsync", "-az", "-e", ssh, str(ROOT / "assets" / name),
                                f"root@{ip}:{REMOTE_DIR}/assets/"], timeout=900)
                subprocess.run(["rsync", "-az", "-e", ssh, str(ROOT / f"{name}0.json"),
                                f"root@{ip}:{REMOTE_DIR}/"], timeout=120)
        ENV_BUILDS[name].update(status="done" if ok else "failed", finished=time.time())
    except Exception as e:                                  # noqa: BLE001
        ENV_BUILDS[name].update(status="failed", finished=time.time(), error=str(e))


@app.get("/api/geocode")
def geocode(q: str):
    """Place name -> coordinates, so users search a city/area instead of typing
    lat/lon. OpenStreetMap Nominatim (free, no key)."""
    q = q.strip()
    if len(q) < 2:
        return []
    try:
        url = "https://nominatim.openstreetmap.org/search?" + urllib.parse.urlencode(
            {"q": q, "format": "jsonv2", "limit": 5})
        req = urllib.request.Request(url, headers={"User-Agent": "vesper-sim/0.1 (research)"})
        with urllib.request.urlopen(req, timeout=8) as r:
            hits = json.load(r)
    except (OSError, json.JSONDecodeError):
        raise HTTPException(502, "geocoding service unavailable")
    out = []
    for h in hits:
        # suggest a radius from the place's bounding box, clamped to a sane range
        half_km = 1.0
        bb = h.get("boundingbox")
        if bb and len(bb) == 4:
            import math as _m
            s, n, w, e = (float(bb[0]), float(bb[1]), float(bb[2]), float(bb[3]))
            lat_c = (s + n) / 2
            span_km = max((n - s) * 110.574,
                          (e - w) * 111.320 * _m.cos(_m.radians(lat_c))) / 2
            half_km = round(min(3.0, max(0.3, span_km)), 1)
        out.append({"name": h.get("display_name", q), "lat": float(h["lat"]),
                    "lon": float(h["lon"]), "half_km": half_km,
                    "type": h.get("type", "")})
    return out


@app.post("/api/environments/build")
def build_environment(req: EnvBuildReq):
    if not SITE_NAME.match(req.name):
        raise HTTPException(400, "name: lowercase letter then letters/digits/underscore")
    if not (-90 < req.lat < 90 and -180 < req.lon < 180):
        raise HTTPException(400, "bad lat/lon")
    if not (0.1 <= req.half_km <= 3.0):
        raise HTTPException(400, "half_km must be 0.1-3.0")
    cur = ENV_BUILDS.get(req.name)
    if cur and cur["status"] == "running":
        raise HTTPException(409, "already building")
    ENV_BUILDS[req.name] = {"status": "running", "started": time.time(), "finished": None}
    threading.Thread(target=_run_env_build,
                     args=(req.name, req.lat, req.lon, req.half_km), daemon=True).start()
    return {"name": req.name, "status": "running"}


def _run_reconstruct(name: str, half_km: float) -> None:
    """photos -> OpenDroneMap (DSM+ortho) on the box -> --dsm surface-model world.
    Needs the GPU box up (ODM is heavy); the world syncs back when done."""
    logf = _env_log(name)
    src = ROOT / "assets" / name / "src_images"

    def log(msg):
        with open(logf, "a") as f:
            f.write(msg + "\n")

    try:
        ip = _droplet_ip()
        if not ip:
            log("GPU box is down -- relaunch it (infra/do/launch.sh), then re-run.")
            ENV_BUILDS[name].update(status="needs_box", finished=time.time())
            return
        ssh = f"ssh -i {KEY_FILE} -o StrictHostKeyChecking=accept-new"
        remote = f"/root/photogrammetry/{name}"
        log(f"uploading images to the box ({ip})...")
        subprocess.run(["ssh", "-i", KEY_FILE, f"root@{ip}",
                        f"mkdir -p {remote}/images"], timeout=60)
        subprocess.run(["rsync", "-az", "-e", ssh, f"{src}/",
                        f"root@{ip}:{remote}/images/"], timeout=1800, check=True)
        log("running OpenDroneMap (photogrammetry -- 20-60 min)...")
        odm = (f"docker run --rm -v /root/photogrammetry:/datasets opendronemap/odm "
               f"--project-path /datasets {name} --dsm --fast-orthophoto --skip-report")
        r = subprocess.run(["ssh", "-i", KEY_FILE, f"root@{ip}", odm],
                           capture_output=True, text=True, timeout=14400)
        dsm_remote = f"{remote}/odm_dem/dsm.tif"
        rc = subprocess.run(["ssh", "-i", KEY_FILE, f"root@{ip}",
                             f"test -f {dsm_remote} && echo ok"], capture_output=True, text=True)
        if "ok" not in rc.stdout:
            log("ODM produced no DSM -- images likely lack overlap/coverage.")
            ENV_BUILDS[name].update(status="failed", finished=time.time())
            return
        log("reconstruction done; pulling DSM + ortho...")
        dst = ROOT / "assets" / name
        subprocess.run(["rsync", "-az", "-e", ssh,
                        f"root@{ip}:{dsm_remote}",
                        f"root@{ip}:{remote}/odm_orthophoto/odm_orthophoto.tif",
                        f"{dst}/"], timeout=1800)
        log("building the USD world from the reconstruction...")
        p = subprocess.run(
            [sys.executable, "scripts/build_geo_world.py", name,
             "--lat", "0", "--lon", "0", "--half-km", str(half_km),
             "--dsm", f"assets/{name}/dsm.tif",
             "--ortho", f"assets/{name}/odm_orthophoto.tif", "--surface-model"],
            cwd=ROOT, capture_output=True, text=True, timeout=1800)
        with open(logf, "a") as f:
            f.write(p.stdout[-2000:] + "\n" + p.stderr[-1000:] + "\n")
        ok = (ROOT / "assets" / name / f"{name}.usd").exists()
        if ok:
            subprocess.run(["rsync", "-az", "-e", ssh, str(ROOT / "assets" / name),
                            f"root@{ip}:{REMOTE_DIR}/assets/"], timeout=900)
            subprocess.run(["rsync", "-az", "-e", ssh, str(ROOT / f"{name}0.json"),
                            f"root@{ip}:{REMOTE_DIR}/"], timeout=120)
        ENV_BUILDS[name].update(status="done" if ok else "failed", finished=time.time())
    except Exception as e:                                  # noqa: BLE001
        log(f"error: {e}")
        ENV_BUILDS[name].update(status="failed", finished=time.time())


@app.post("/api/environments/reconstruct")
async def reconstruct_environment(name: str = Form(...), half_km: float = Form(0.3),
                                  files: list[UploadFile] = File(...)):
    if not SITE_NAME.match(name):
        raise HTTPException(400, "name: lowercase letter then letters/digits/underscore")
    imgs = [f for f in files if (f.filename or "").lower().endswith((".jpg", ".jpeg", ".png", ".tif", ".tiff"))]
    if len(imgs) < 8:
        raise HTTPException(400, "need at least ~8 overlapping photos to reconstruct")
    if ENV_BUILDS.get(name, {}).get("status") == "running":
        raise HTTPException(409, "already processing")
    src = ROOT / "assets" / name / "src_images"
    src.mkdir(parents=True, exist_ok=True)
    for i, f in enumerate(imgs):
        ext = Path(f.filename or f"img{i}.jpg").suffix or ".jpg"
        (src / f"{i:04d}{ext}").write_bytes(await f.read())
    _env_log(name).write_text(f"received {len(imgs)} photos\n")
    ENV_BUILDS[name] = {"status": "running", "started": time.time(), "finished": None,
                        "mode": "photogrammetry"}
    threading.Thread(target=_run_reconstruct, args=(name, half_km), daemon=True).start()
    return {"name": name, "status": "running", "photos": len(imgs)}


@app.get("/api/environments")
def environments():
    """Built worlds (assets/<name>/<name>.usd) plus any in-flight builds."""
    out = []
    adir = ROOT / "assets"
    for d in sorted(adir.iterdir()) if adir.is_dir() else []:
        usd = d / f"{d.name}.usd"
        if not d.is_dir() or not usd.exists():
            continue
        b = ENV_BUILDS.get(d.name, {})
        out.append({
            "name": d.name,
            "usd": f"assets/{d.name}/{d.name}.usd",
            "scenario": f"{d.name}0.json" if (ROOT / f"{d.name}0.json").exists() else None,
            "map": f"assets/{d.name}/{d.name}_map.npz" if (d / f"{d.name}_map.npz").exists() else None,
            "mb": round(usd.stat().st_size / 1e6, 1),
            "build_status": b.get("status", "done"),
        })
    seen = {o["name"] for o in out}
    for name, b in ENV_BUILDS.items():                      # builds not yet on disk
        if name not in seen:
            log = _env_log(name)
            tail = log.read_text()[-400:] if log.exists() else ""
            out.append({"name": name, "usd": None, "scenario": None, "map": None,
                        "build_status": b["status"], "log": tail})
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
    # persistent sim: world stays loaded, feeds + /state stay up, deploy/reset
    # are instant commands instead of a fresh Isaac boot
    "warm": "scripts/warm_session.py --num_envs 8 --cameras --policy runs/friend-checkpoints/search.pt",
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


WORLD_USD = re.compile(r"^assets/[\w.-]+/[\w.-]+\.usd$")
WORLD_MAP = re.compile(r"^assets/[\w.-]+/[\w.-]+_map\.npz$")


class JobReq(BaseModel):
    kind: str
    policy: str | None = None
    scenario: str | None = None
    world: str | None = None
    map: str | None = None


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
    # train/eval in a chosen world: append --world/--map (search task needs the map)
    if req.kind in ("train", "eval") and req.world:
        if not WORLD_USD.match(req.world):
            raise HTTPException(400, "world must be assets/<name>/<name>.usd")
        tmpl += f" --world {req.world}"
        if req.map and WORLD_MAP.match(req.map):
            tmpl += f" --map {req.map}"
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
ACTIVE_FILE = ROOT / ".vesper_active.json"


class ActiveReq(BaseModel):
    name: str


def _site_entry(world: str) -> dict:
    d = ASSETS / world
    half = None
    mj = d / f"{world}_map.json"
    if mj.exists():
        try:
            half = json.loads(mj.read_text()).get("half_m")
        except (OSError, json.JSONDecodeError):
            pass
    return {
        "name": world, "world": world, "half_m": half,
        "ground": f"/site/{world}/ground" if (d / "ground.png").exists() else None,
        "scenario": f"{world}0.json" if (ROOT / f"{world}0.json").exists() else None,
        "usd": f"assets/{world}/{world}.usd" if (d / f"{world}.usd").exists() else None,
        "map": f"assets/{world}/{world}_map.npz" if (d / f"{world}_map.npz").exists() else None,
    }


@app.get("/api/active")
def get_active():
    """The environment the app is working with (map, live overlay, job defaults).
    Falls back to the first built world so the map is never empty."""
    name = None
    if ACTIVE_FILE.exists():
        try:
            name = json.loads(ACTIVE_FILE.read_text()).get("name")
        except (OSError, json.JSONDecodeError):
            pass
    if not name or not (ASSETS / name / f"{name}.usd").exists():
        cands = [d.name for d in sorted(ASSETS.iterdir())
                 if d.is_dir() and (d / f"{d.name}.usd").exists()] if ASSETS.is_dir() else []
        name = cands[0] if cands else None
    return _site_entry(name) if name else None


@app.post("/api/active")
def set_active(req: ActiveReq):
    if not SITE_NAME.match(req.name) or not (ASSETS / req.name / f"{req.name}.usd").exists():
        raise HTTPException(404, "no such world")
    ACTIVE_FILE.write_text(json.dumps({"name": req.name}))
    return _site_entry(req.name)


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
