"""Vesper runs browser -- local web UI over runs/ and the scenario specs.

    .venv/bin/python -m uvicorn web.server.app:app --port 8777
    open http://localhost:8777

Reads the same artifacts every run already writes (manifest.json, *.mp4,
track.png, trajectory.parquet, scenario.json); no state of its own. Videos are
served with HTTP Range support so <video> can stream and scrub.
"""
import json
import math
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


_RUN_CONTENT = {"curve.jsonl", "results.jsonl", "report.json", "events.json",
                "trajectory.parquet"}


def _has_media(d: Path, names: set[str]) -> bool:
    """True when a run carries renderable media -- at the top level OR one dir
    down (e.g. frames/*.png), so a run whose media lives in a subdir still
    surfaces. One level only; stops at the first hit to stay cheap."""
    if any(n.endswith((".mp4", ".png")) for n in names):
        return True
    for sub in d.iterdir():
        if sub.is_dir():
            try:
                if any(p.suffix in (".png", ".mp4") for p in sub.iterdir()):
                    return True
            except OSError:
                continue
    return False


@app.get("/api/runs")
def list_runs():
    out = []
    for d in sorted(RUNS.iterdir()) if RUNS.is_dir() else []:
        if not d.is_dir():
            continue
        try:
            names = {p.name for p in d.iterdir()}
            # a run is anything with a manifest OR real content (training curves,
            # sweeps, media). A dir holding only checkpoints (*.pt/*.onnx) is a
            # model pool (e.g. friend-checkpoints), surfaced by /api/models -- not
            # a sortie, so it must not become the default "latest".
            has_content = ("manifest.json" in names or names & _RUN_CONTENT
                           or _has_media(d, names))
            if not has_content:
                continue
            manifest = {}
            if "manifest.json" in names:
                try:
                    manifest = json.loads((d / "manifest.json").read_text())
                except (json.JSONDecodeError, OSError):     # half-written manifest
                    pass
            if "name" not in manifest:                      # synthesize for capture-less runs
                manifest["name"] = d.name.split("-", 2)[-1] if "-" in d.name else d.name
            manifest.setdefault("started", d.stat().st_mtime)   # always dated, even with a name
            files = sorted(p.name for p in d.iterdir() if p.suffix in MEDIA_TYPES)
            out.append({"id": d.name, "manifest": manifest, "files": files})
        except OSError:                                     # dir vanished / unreadable mid-scan
            continue
    # newest first by real timestamp (manifest.started, else dir mtime), so dated
    # sorties lead and a lexicographic name (friend-*) can't hijack "latest".
    out.sort(key=lambda r: r["manifest"].get("started") or 0, reverse=True)
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


RENDERERS = {"tactical", "three", "isaac"}


class RenderReq(BaseModel):
    renderers: str = "three"        # comma list of {tactical, three, isaac}


@app.post("/api/runs/{run_id}/render")
def render_run(run_id: str, req: RenderReq):
    """Kick a replay render for an existing run — the renderer FLAG's endpoint.

    tactical = 2D top-down (Mac, seconds); three = three.js chase+fpv (Mac,
    ~a minute, scripts/render_three_replay.py); isaac = photoreal on the GPU
    box (slow; the hero shot). Detached: poll /api/runs for the mp4s."""
    d = _run_dir(run_id)
    if not (d / "replay.json").is_file():
        raise HTTPException(400, "run has no replay.json")
    rends = [r.strip() for r in req.renderers.split(",") if r.strip()]
    if not rends or set(rends) - RENDERERS:
        raise HTTPException(400, f"renderers must be from {sorted(RENDERERS)}")
    py = ROOT / ".venv" / "bin" / "python"
    subprocess.Popen(
        [str(py) if py.exists() else sys.executable, "scripts/render_dispatch.py",
         str(d), "--renderers", ",".join(rends)],
        cwd=ROOT, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        start_new_session=True)
    return {"ok": True, "renderers": rends}


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
ENV_BUILDS_FILE = ROOT / ".vesper_envbuilds.json"   # survives server restarts (uvicorn --reload)


def _save_env_builds() -> None:
    try:
        ENV_BUILDS_FILE.write_text(json.dumps(ENV_BUILDS, indent=1))
    except OSError:
        pass


def _env_update(name: str, **kw) -> None:
    ENV_BUILDS.setdefault(name, {}).update(kw)
    _save_env_builds()


class EnvBuildReq(BaseModel):
    name: str
    lat: float
    lon: float
    half_km: float = 1.0
    launch: list[float] | None = None            # [lat, lon] of the launch pin (5 m pad)
    safe: list[list[list[float]]] = []           # friendly zones: [[[lat, lon], ...], ...]


# ---- site frame <-> geo. Site metres are ENU about the world centre (lat0, lon0),
# the same frame the builder, the map rasters and the zones file use.
LAUNCH_R_M = 5.0                                 # launch pad radius
LAUNCH_TREE_CLEAR_M = 5.0                        # no trunk closer than this to the pin


def _site_xy(lat: float, lon: float, lat0: float, lon0: float) -> list[float]:
    return [round((lon - lon0) * 111320.0 * math.cos(math.radians(lat0)), 2),
            round((lat - lat0) * 110574.0, 2)]


def _geo(x: float, y: float, lat0: float, lon0: float) -> list[float]:
    return [round(lat0 + y / 110574.0, 7),
            round(lon0 + x / (111320.0 * math.cos(math.radians(lat0))), 7)]


def _launch_polygon(x: float, y: float, r: float = LAUNCH_R_M, n: int = 24) -> list[list[float]]:
    return [[round(x + r * math.cos(2 * math.pi * i / n), 2),
             round(y + r * math.sin(2 * math.pi * i / n), 2)] for i in range(n)]


def _zones_doc(lat0: float, lon0: float, launch_geo: list[float] | None,
               safe_geo: list[list[list[float]]], launch_xy: list[float] | None = None) -> dict:
    """The zones.json body: `launch` / `safe` polygons in site metres (what
    vesper.worlds.zones.Zones loads), plus the raw pin and the lat/lon originals
    so the map UI can redraw them without the site frame."""
    if launch_xy is None and launch_geo is not None:
        launch_xy = _site_xy(launch_geo[0], launch_geo[1], lat0, lon0)
    if launch_xy is not None and launch_geo is None:
        launch_geo = _geo(launch_xy[0], launch_xy[1], lat0, lon0)
    safe_xy = [[_site_xy(la, lo, lat0, lon0) for la, lo in poly] for poly in safe_geo]
    return {
        "launch": _launch_polygon(*launch_xy) if launch_xy else None,
        "safe": safe_xy,
        "launch_point": ({"x": launch_xy[0], "y": launch_xy[1], "r_m": LAUNCH_R_M,
                          "lat": launch_geo[0], "lon": launch_geo[1]} if launch_xy else None),
        "safe_geo": safe_geo,
        "center": [lat0, lon0],
    }


def _tree_clearance(world: str, x: float, y: float) -> float | None:
    """Distance (m) from (x, y) to the nearest trunk in the world's map, or None
    when the world has no map yet (pre-build: the builder clears trees itself)."""
    npz = ASSETS / world / f"{world}_map.npz"
    if not npz.exists():
        return None
    try:
        import numpy as np
        trees = np.load(npz)["trees"]                 # [(x, y, height, crown_r)]
    except Exception:                                 # noqa: BLE001
        return None
    if len(trees) == 0:
        return float("inf")
    return float(np.hypot(trees[:, 0] - x, trees[:, 1] - y).min())


def _write_zones(world: str, doc: dict) -> Path:
    d = ASSETS / world
    d.mkdir(parents=True, exist_ok=True)
    f = d / "zones.json"
    f.write_text(json.dumps(doc, indent=1))
    ip = _droplet_ip()
    if ip:                                            # the box copy is what training reads
        subprocess.run(["rsync", "-az", "-e", _ssh_opts(), str(f), f"root@{ip}:{REMOTE_DIR}/assets/{world}/"],
                       timeout=60, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return f


def _env_log(name: str) -> Path:
    return ROOT / f".envbuild_{name}.log"


def _export_map(name: str, logf: Path) -> None:
    """Bake the world-map (ground/obstacle/canopy/drivable/concealed rasters) so
    the search task can train here. Without it a new world is not trainable."""
    with open(logf, "a") as f:
        f.write("exporting world-map (needed for training)...\n")
        subprocess.run([sys.executable, "scripts/export_world_map.py",
                        f"assets/{name}/{name}.usd"], cwd=ROOT,
                       stdout=f, stderr=subprocess.STDOUT, timeout=900)


GEO_PY = "/root/geo-venv/bin/python"           # infra/do/geo_env.sh puts the build env here
# what the Mac keeps of a world built on the box: everything except the big rasters and
# the USD, which only Isaac (on the box) reads. ground.jpg is the web preview.
MIRROR_EXCLUDES = ["imagery.png", "naip.png", "ground.png", "dem.npy", "*.usd", "*.glb", "src_images"]


def _ssh_opts():
    return f"ssh -i {KEY_FILE} -o BatchMode=yes -o StrictHostKeyChecking=accept-new"


def _tex_px(size_km: float) -> int:
    """Ground albedo resolution: ~0.5 m/px up to 4 km, then 8192 (memory + Esri's own limit)."""
    return 4096 if size_km <= 2.5 else 8192


def _build_cmd(name: str, lat: float, lon: float, size_km: float, py: str,
               spawn_xy: list[float] | None = None) -> list[str]:
    cmd = [py, "-u", "scripts/build_geo_world.py", name,
           "--lat", str(lat), "--lon", str(lon), "--size-km", str(size_km),
           "--tex-px", str(_tex_px(size_km)), "--source", "global", "--ms-buildings"]
    if spawn_xy is not None:                        # the operator's launch pin (builder clears trees around it)
        cmd += ["--spawn", str(spawn_xy[0]), str(spawn_xy[1])]
    return cmd


def _reconcile_launch(name: str, lat0: float, lon0: float, logf: Path) -> None:
    """The builder may snap a pin off a rooftop/road; keep zones.json on the spawn
    it actually used so the pad, the scenario and the map agree."""
    zf = ASSETS / name / "zones.json"
    mf = ASSETS / name / f"{name}_build.json"
    if not zf.exists() or not mf.exists():
        return
    try:
        doc = json.loads(zf.read_text()); rep = json.loads(mf.read_text()).get("report", {})
        sx, sy = rep.get("spawn_xy") or (None, None)
        lp = doc.get("launch_point")
        if sx is None or not lp:
            return
        moved = math.hypot(sx - lp["x"], sy - lp["y"])
        if moved > 0.5:
            with open(logf, "a") as f:
                f.write(f"launch pin moved {moved:.0f} m to open ground by the builder\n")
            _write_zones(name, _zones_doc(lat0, lon0, None, doc.get("safe_geo") or [], launch_xy=[sx, sy]))
        else:
            _write_zones(name, doc)                   # push the local copy to the box
    except (OSError, json.JSONDecodeError, KeyError, TypeError):
        return


def _run_env_build_local(name: str, lat: float, lon: float, size_km: float, logf: Path,
                         spawn_xy: list[float] | None = None) -> bool:
    """Fallback when the box is down: build here (the geo pipeline needs no GPU)."""
    with open(logf, "a") as f:
        f.write("box is down -- building locally (slower: home bandwidth)\n")
        p = subprocess.run(_build_cmd(name, lat, lon, size_km, sys.executable, spawn_xy),
                           cwd=ROOT, stdout=f, stderr=subprocess.STDOUT, timeout=3600)
    ok = p.returncode == 0 and (ROOT / "assets" / name / f"{name}.usd").exists()
    if ok:
        _export_map(name, logf)                                 # make it trainable
        subprocess.run([sys.executable, "scripts/ground_preview.py", f"assets/{name}"], cwd=ROOT,
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=300)
        ip = _droplet_ip()
        if ip:                                                  # push the new world to the box
            subprocess.run(["rsync", "-az", "-e", _ssh_opts(), str(ROOT / "assets" / name),
                            f"root@{ip}:{REMOTE_DIR}/assets/"], timeout=1800)
            subprocess.run(["rsync", "-az", "-e", _ssh_opts(), str(ROOT / f"{name}0.json"),
                            f"root@{ip}:{REMOTE_DIR}/"], timeout=120)
    return ok


def _launch_box_build(name: str, lat: float, lon: float, size_km: float,
                      spawn_xy: list[float] | None, logf: Path, ip: str) -> str | None:
    """Start the geo build + map export + preview on the box as a detached job
    with a unique log/exit marker; returns the job id (None if it did not start)."""
    job = f"envbuild_{name}_{int(time.time())}"
    rlog, rexit = f"/root/{job}.log", f"/root/{job}.exit"
    cmd = " ".join(_build_cmd(name, lat, lon, size_km, GEO_PY, spawn_xy))
    chain = (f"cd {REMOTE_DIR} && {cmd}"
             f" && echo '== exporting world-map (needed for training)'"
             f" && {GEO_PY} -u scripts/export_world_map.py assets/{name}/{name}.usd"
             f" && {GEO_PY} scripts/ground_preview.py assets/{name}")
    # the job script goes over stdin (no inline quoting games), then runs detached
    script = f"#!/usr/bin/env bash\n( {chain} ) > {rlog} 2>&1\necho $? > {rexit}\n"   # subshell: redirect covers the whole chain
    rc, out, err = _ssh_script(script, f"/root/{job}.sh", timeout=30)
    with open(logf, "a") as f:
        f.write(f"building on the box ({ip}) as {job}\n")
        if rc != 0 or "launched" not in out:
            f.write(f"could not launch: {(err or out).strip()[-300:]}\n")
            return None
    return job


def _poll_box_job(name: str, job: str, ip: str, logf: Path, started: float) -> int | None:
    """Copy the remote log into the local one every few seconds until the exit
    marker appears. Returns the exit code, or None on timeout / never started."""
    rlog, rexit = f"/root/{job}.log", f"/root/{job}.exit"
    deadline = started + 3600
    first_output_by = started + 90                              # the job prints within seconds
    while time.time() < deadline:
        time.sleep(3)
        try:
            rc, out, _ = _ssh(f"cat {rexit} 2>/dev/null; echo __SEP__; cat {rlog} 2>/dev/null", timeout=20)
        except (HTTPException, subprocess.TimeoutExpired):
            continue                                            # transient: box busy / ssh hiccup
        head, _, tail = out.partition("__SEP__")
        if not tail.strip() and time.time() > first_output_by:
            with open(logf, "a") as f:
                f.write("the job never started on the box (no log after 90 s)\n")
            return None
        body = "\n".join(ln for ln in tail.splitlines()
                         if ln.strip() and not ln.startswith("#") and "Warning" not in ln
                         and "oriented_envelope" not in ln and not ln.startswith("  return"))
        logf.write_text(f"building on the box ({ip}) as {job}\n{body}\n")
        if head.strip().lstrip("-").isdigit():
            return int(head.strip())
    with open(logf, "a") as f:
        f.write("timed out waiting for the box\n")
    return None


def _mirror_box_world(name: str, ip: str, logf: Path) -> bool:
    """Bring the light files back (metadata, map rasters, preview) -- not the big
    rasters or the USD, which only Isaac on the box reads."""
    with open(logf, "a") as f:
        f.write("mirroring world metadata + map back to this machine...\n")
    ex = sum((["--exclude", e] for e in MIRROR_EXCLUDES), [])
    r1 = subprocess.run(["rsync", "-az", "-e", _ssh_opts(), *ex,
                         f"root@{ip}:{REMOTE_DIR}/assets/{name}/", str(ROOT / "assets" / name) + "/"],
                        capture_output=True, text=True, timeout=1800)
    r2 = subprocess.run(["rsync", "-az", "-e", _ssh_opts(),
                         f"root@{ip}:{REMOTE_DIR}/{name}0.json", str(ROOT / f"{name}0.json")],
                        capture_output=True, text=True, timeout=120)
    ok = r1.returncode == 0 and r2.returncode == 0 and (ROOT / "assets" / name / f"{name}_build.json").exists()
    with open(logf, "a") as f:
        f.write("done\n" if ok else f"mirror failed: {(r1.stderr or r2.stderr).strip()[-200:]}\n")
    return ok


def _run_env_build_box(name: str, lat: float, lon: float, size_km: float, ip: str, logf: Path,
                       spawn_xy: list[float] | None = None) -> bool:
    """Build on the GPU box: it has datacenter bandwidth to Esri/Copernicus/Overpass,
    and the finished world lands where Isaac and training read it (no upload over the
    home link, which measures ~1 MB/s). The Mac then mirrors the light files back.
    The job id is persisted so a restarted server re-attaches instead of losing it."""
    job = _launch_box_build(name, lat, lon, size_km, spawn_xy, logf, ip)
    if not job:
        return False
    _env_update(name, job=job, ip=ip, lat=lat, lon=lon, size_km=size_km)
    code = _poll_box_job(name, job, ip, logf, time.time())
    if code != 0:
        if code is not None:
            with open(logf, "a") as f:
                f.write("build failed on the box\n")
        return False
    return _mirror_box_world(name, ip, logf)


def _resume_box_builds() -> None:
    """After a server restart: re-attach to box jobs that were still running."""
    try:
        saved = json.loads(ENV_BUILDS_FILE.read_text()) if ENV_BUILDS_FILE.exists() else {}
    except (OSError, json.JSONDecodeError):
        saved = {}
    ENV_BUILDS.update(saved)
    for name, b in list(saved.items()):
        if b.get("status") != "running":
            continue
        if not b.get("job"):                                     # local build: the process died with us
            _env_update(name, status="failed", finished=time.time(), error="server restarted mid-build")
            continue

        def _resume(name=name, b=b):
            logf = _env_log(name)
            with open(logf, "a") as f:
                f.write("server restarted; re-attached to the running box job\n")
            ok = False
            try:
                code = _poll_box_job(name, b["job"], b["ip"], logf, b.get("started", time.time()))
                ok = code == 0 and _mirror_box_world(name, b["ip"], logf)
                if ok:
                    _reconcile_launch(name, b["lat"], b["lon"], logf)
            except Exception as e:                              # noqa: BLE001
                with open(logf, "a") as f:
                    f.write(f"error: {e}\n")
            _env_update(name, status="done" if ok else "failed", finished=time.time())

        threading.Thread(target=_resume, daemon=True).start()


def _run_env_build(name: str, lat: float, lon: float, size_km: float,
                   launch: list[float] | None = None, safe: list | None = None) -> None:
    logf = _env_log(name)
    logf.write_text("")
    try:
        spawn_xy = None
        if launch or safe:                          # operator zones, saved before the build starts
            doc = _zones_doc(lat, lon, launch, safe or [])
            (ASSETS / name).mkdir(parents=True, exist_ok=True)
            (ASSETS / name / "zones.json").write_text(json.dumps(doc, indent=1))
            if doc["launch_point"]:
                spawn_xy = [doc["launch_point"]["x"], doc["launch_point"]["y"]]
        ip = _droplet_ip()
        ok = (_run_env_build_box(name, lat, lon, size_km, ip, logf, spawn_xy) if ip
              else _run_env_build_local(name, lat, lon, size_km, logf, spawn_xy))
        if ok:
            _reconcile_launch(name, lat, lon, logf)
        _env_update(name, status="done" if ok else "failed", finished=time.time())
    except Exception as e:                                  # noqa: BLE001
        with open(logf, "a") as f:
            f.write(f"error: {e}\n")
        _env_update(name, status="failed", finished=time.time(), error=str(e))


_resume_box_builds()


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
    if not (0.5 <= req.half_km <= 4.0):
        raise HTTPException(400, "half_km must be 0.5-4.0 (1-8 km wide)")
    cur = ENV_BUILDS.get(req.name)
    if cur and cur["status"] == "running":
        raise HTTPException(409, "already building")
    # a scenario file with this name that is NOT a built world (e.g. the repo's site0.json)
    # would be overwritten by the build's own scenario
    if (ROOT / f"{req.name}0.json").exists() and not (ASSETS / req.name).is_dir():
        raise HTTPException(409, f"'{req.name}' is taken by an existing scenario file; pick another name")
    ENV_BUILDS[req.name] = {"status": "running", "started": time.time(), "finished": None,
                            "where": "box" if _droplet_ip() else "local"}
    _save_env_builds()
    if req.launch is not None:
        lx, ly = _site_xy(req.launch[0], req.launch[1], req.lat, req.lon)
        if max(abs(lx), abs(ly)) > req.half_km * 1000 - LAUNCH_R_M:
            raise HTTPException(400, "launch pin must sit inside the area")
    threading.Thread(target=_run_env_build,
                     args=(req.name, req.lat, req.lon, 2 * req.half_km, req.launch, req.safe),
                     daemon=True).start()
    return {"name": req.name, "status": "running", "where": ENV_BUILDS[req.name]["where"]}


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
            _export_map(name, logf)                         # make it trainable
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
    _save_env_builds()
    threading.Thread(target=_run_reconstruct, args=(name, half_km), daemon=True).start()
    return {"name": name, "status": "running", "photos": len(imgs)}


def _world_built(name: str) -> bool:
    """A world exists when its USD is here or its build manifest is (a box build
    mirrors the manifest + map back and leaves the USD on the box for Isaac)."""
    d = ASSETS / name
    return d.is_dir() and ((d / f"{name}.usd").exists() or (d / f"{name}_build.json").exists())


@app.get("/api/environments")
def environments():
    """Built worlds (assets/<name>/, here or mirrored from the box) plus in-flight builds."""
    out = []
    adir = ROOT / "assets"
    for d in sorted(adir.iterdir()) if adir.is_dir() else []:
        if not d.is_dir() or not _world_built(d.name):
            continue
        usd = d / f"{d.name}.usd"
        b = ENV_BUILDS.get(d.name, {})
        log = _env_log(d.name)
        out.append({
            "name": d.name,
            "center": _world_center(d.name),                # [lat, lon] or None
            "half_m": _world_half_m(d.name),
            "usd": f"assets/{d.name}/{d.name}.usd",         # valid on the box even if not mirrored
            "scenario": f"{d.name}0.json" if (ROOT / f"{d.name}0.json").exists() else None,
            "map": f"assets/{d.name}/{d.name}_map.npz" if (d / f"{d.name}_map.npz").exists() else None,
            "mb": round(usd.stat().st_size / 1e6, 1) if usd.exists() else None,
            "build_status": b.get("status", "done"),
            "log": log.read_text()[-600:] if b and log.exists() else None,
            "demo": _demo_media(d.name),
        })
    seen = {o["name"] for o in out}
    for name, b in ENV_BUILDS.items():                      # builds not yet on disk
        if name not in seen:
            log = _env_log(name)
            tail = log.read_text()[-600:] if log.exists() else ""
            out.append({"name": name, "usd": None, "scenario": None, "map": None,
                        "build_status": b["status"], "where": b.get("where"), "log": tail})
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
                    # drop NaN/Inf -- not JSON-compliant, and they'd 500 the whole response
                    metrics = {k: v for k, v in last.items()
                               if isinstance(v, (int, float)) and math.isfinite(v)}
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
                "onnx": f"runs/{d.name}/{f.stem}.onnx" if (d / f"{f.stem}.onnx").exists() else None,
            })
    return out


class ExportReq(BaseModel):
    policy: str


@app.post("/api/models/export")
def export_model(req: ExportReq):
    """Export a checkpoint to ONNX for onboard hardware (the deploy step)."""
    if not POLICY_PATH.match(req.policy) or not (ROOT / req.policy).is_file():
        raise HTTPException(400, "policy must be runs/<id>/<name>.pt")
    onnx = Path(req.policy).with_suffix(".onnx")
    r = subprocess.run([sys.executable, "scripts/export_policy.py", req.policy],
                       cwd=ROOT, capture_output=True, text=True, timeout=180)
    if r.returncode != 0 or not (ROOT / onnx).exists():
        raise HTTPException(500, f"export failed: {(r.stderr or r.stdout)[-300:]}")
    return {"onnx": str(onnx), "bytes": (ROOT / onnx).stat().st_size,
            "url": f"/download/{onnx.parts[1]}/{onnx.name}"}


@app.get("/download/{run_id}/{name}")
def download(run_id: str, name: str):
    if not RUN_ID.match(run_id) or not RUN_ID.match(name):
        raise HTTPException(400, "bad path")
    f = RUNS / run_id / name
    if not f.is_file() or f.suffix not in (".onnx", ".pt"):
        raise HTTPException(404, "no such artifact")
    return FileResponse(f, media_type="application/octet-stream", filename=name)


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


_local_live_cache = {"t": 0.0, "up": False}


def _local_live():
    """True when a warm session (native or otherwise) is serving on this
    machine's 8180. Cached 5 s; the probe is a fast localhost connect."""
    now = time.time()
    if now - _local_live_cache["t"] < 5:
        return _local_live_cache["up"]
    up = False
    try:
        with urllib.request.urlopen("http://127.0.0.1:8180/state", timeout=0.5) as r:
            up = r.status == 200
    except OSError:
        pass
    _local_live_cache.update(t=now, up=up)
    return up


@app.get("/api/live")
def live(request: Request = None):
    # the demo is the native session on this machine. When it's up, it wins;
    # when it's down, the live view is STANDBY (null) -- do NOT fall back to the
    # droplet, or stopping the mission flips the UI to a stale Isaac session on
    # the box ("random" old feeds). The Isaac/droplet path is opt-in via ?box=1.
    if _local_live():
        return {"ip": "localhost"}
    if request is not None and request.query_params.get("box"):
        return {"ip": _droplet_ip()}
    return {"ip": None}


# ---------------------------------------------------------------- native session
# The demo path: launch/kill the NATIVE warm session (scripts/warm_session_native.py)
# as a local subprocess. No Isaac, no droplet — the session serves /state on 8180
# and /api/live flips to "localhost" while it is up. The Isaac/droplet lane below
# (/api/jobs) stays for work that genuinely needs the GPU box.

SESSION_FILE = ROOT / ".vesper_session.json"
SESSION_LOG = ROOT / ".vesper_session.log"
# demo defaults: the kramatorsk AO with its best converged policy; reach_radius 40
# registers the policy's close passes as strikes, episode_s 240 gives a full
# search→detect→neutralize mission before any rollover. The policy is the
# frontier/loitering-munition retrain (search-lm-full): searches without the
# scan-orbit, expends the striking drone on impact, and pairs with the live
# session's shared-fleet-coverage so the swarm self-spreads. The prior
# demo-kram-lean checkpoint was the circler this replaced.
SESSION_DEFAULTS = {
    "map": "assets/kramatorsk/kramatorsk_map.npz",
    "policy": "runs/20260905-230222-search-lm-full/search.pt",
    "reach_radius": 40.0,
    "episode_s": 240.0,
}


class SessionReq(BaseModel):
    map: str | None = None
    policy: str | None = None
    reach_radius: float | None = None
    episode_s: float | None = None


def _session_read() -> dict | None:
    try:
        return json.loads(SESSION_FILE.read_text())
    except (OSError, json.JSONDecodeError):
        return None


def _session_alive(info: dict | None) -> bool:
    if not info or not info.get("pid"):
        return False
    try:
        os.kill(int(info["pid"]), 0)
        return True
    except (OSError, ValueError):
        return False


@app.get("/api/session")
def session_status():
    info = _session_read()
    alive = _session_alive(info)
    return {"running": alive, **({k: info.get(k) for k in ("pid", "started", "args")} if alive and info else {})}


@app.post("/api/session/start")
def session_start(req: SessionReq):
    info = _session_read()
    if _session_alive(info) or _local_live():
        raise HTTPException(409, "a live session is already running")
    p = dict(SESSION_DEFAULTS)
    if req.map is not None:
        if not WORLD_MAP.match(req.map):
            raise HTTPException(400, "map must be assets/<world>/<world>_map.npz")
        p["map"] = req.map
    if req.policy is not None:
        if not POLICY_PATH.match(req.policy):
            raise HTTPException(400, "policy must be runs/<id>/<name>.pt")
        p["policy"] = req.policy
    if req.reach_radius is not None:
        p["reach_radius"] = req.reach_radius
    if req.episode_s is not None:
        p["episode_s"] = req.episode_s
    if not (ROOT / p["map"]).is_file():
        raise HTTPException(400, f"no such map: {p['map']}")
    if not (ROOT / p["policy"]).is_file():
        raise HTTPException(400, f"no such policy: {p['policy']}")
    py = ROOT / ".venv" / "bin" / "python"
    cmd = [str(py) if py.exists() else sys.executable, "scripts/warm_session_native.py",
           "--map", p["map"], "--policy", p["policy"],
           "--reach_radius", str(p["reach_radius"]), "--episode_s", str(p["episode_s"])]
    log = open(SESSION_LOG, "w")
    # detached (its own session) so uvicorn --reload restarts never take it down
    proc = subprocess.Popen(cmd, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                            stdin=subprocess.DEVNULL, start_new_session=True)
    SESSION_FILE.write_text(json.dumps({"pid": proc.pid, "started": time.time(), "args": p}))
    _local_live_cache["t"] = 0.0                    # next /api/live probes for real
    return {"pid": proc.pid, "args": p}


@app.post("/api/session/stop")
def session_stop():
    import signal
    info = _session_read()
    if not _session_alive(info):
        # no tracked child — but a session started by hand may still own 8180;
        # stop means stop, so take that one down too
        try:
            r = subprocess.run(["lsof", "-ti", "tcp:8180", "-sTCP:LISTEN"],
                               capture_output=True, text=True, timeout=5)
            for pid_s in r.stdout.split():
                os.kill(int(pid_s), signal.SIGTERM)
        except (OSError, ValueError, subprocess.TimeoutExpired):
            pass
        SESSION_FILE.unlink(missing_ok=True)
        _local_live_cache.update(t=0.0, up=False)
        return {"ok": True, "was_running": False}
    pid = int(info["pid"])
    try:
        os.killpg(pid, signal.SIGTERM)              # its own session → pid == pgid
    except OSError:
        try:
            os.kill(pid, signal.SIGTERM)
        except OSError:
            pass
    for _ in range(20):                             # up to ~2 s for a clean exit
        if not _session_alive(info):
            break
        time.sleep(0.1)
    if _session_alive(info):
        try:
            os.killpg(pid, signal.SIGKILL)
        except OSError:
            pass
    SESSION_FILE.unlink(missing_ok=True)
    _local_live_cache.update(t=0.0, up=False)
    return {"ok": True, "was_running": True}


# ---------------------------------------------------------------- jobs (Isaac lane)
# Everything below launches on the GPU box — Isaac only, chosen explicitly from
# the Models/Environments pages. The demo's live session is /api/session above.
# The one write path: launch whitelisted sim jobs on the GPU box over SSH,
# each in a named docker container so status and stop are just `docker ps`
# and `docker rm -f`. The registry is a JSON file next to runs/.

JOB_KINDS = {
    "train": "scripts/train_search.py --num_envs 1024 --iters 1500 --headless",
    "fly": "scripts/fly_search.py --policy {policy} --seconds 90 --headless --enable_cameras",
    "eval": "scripts/eval_search.py --policy {policy} --num_envs 256 --episodes 400 --headless",
    "mission": "scripts/fly_mission.py {scenario}",
    "live": "scripts/live_world.py {world}",
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


def _ssh_script(script: str, remote_path: str, timeout: int = 30):
    """Write `script` to remote_path over stdin and start it detached (nohup); returns
    (rc, stdout, stderr). Avoids quoting a whole command line through ssh + bash -c."""
    if not os.path.exists(KEY_FILE):
        raise HTTPException(503, f"ssh key not found: {KEY_FILE}")
    ip = _droplet_ip()
    if not ip:
        raise HTTPException(503, "gpu box is offline")
    r = subprocess.run(
        ["ssh", "-i", KEY_FILE, "-o", "BatchMode=yes", "-o", "ConnectTimeout=8",
         "-o", "StrictHostKeyChecking=accept-new", f"root@{ip}",
         # only the job goes to the background; `cat` must keep ssh's stdin
         f"cat > {remote_path} && chmod +x {remote_path} && "
         f"{{ nohup {remote_path} >/dev/null 2>&1 </dev/null & }} && echo launched"],
        input=script, capture_output=True, text=True, timeout=timeout,
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
    # jobs default to the active environment when world/scenario aren't given,
    # so selecting a world in the UI makes everything run there.
    active = get_active() or {}
    if "{scenario}" in tmpl:
        scen = req.scenario or active.get("scenario")
        if not scen or not SCENARIO_FILE.match(scen) or not (ROOT / scen).is_file():
            raise HTTPException(400, "no scenario (and no active world scenario)")
        tmpl = tmpl.format(scenario=scen)
    if "{world}" in tmpl:                                   # live viewport
        world = req.world or active.get("usd")
        if not world or not WORLD_USD.match(world):
            raise HTTPException(400, "no world — select an active environment")
        tmpl = tmpl.format(world=world)
    if req.kind in ("train", "eval", "warm"):              # search task wants world + map
        world = req.world or active.get("usd")
        mp = req.map or active.get("map")
        if world and WORLD_USD.match(world):
            tmpl += f" --world {world}"
            if mp and WORLD_MAP.match(mp):
                tmpl += f" --map {mp}"
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


def _world_center(world: str):
    """[lat, lon] of a built world, from the DEM bbox the builder wrote
    (dem_meta.json bbox = [S, W, N, E] around the requested centre)."""
    mj = ASSETS / world / "dem_meta.json"
    try:
        s, w, n, e = json.loads(mj.read_text())["bbox"]
        return [round((s + n) / 2, 6), round((w + e) / 2, 6)]
    except (OSError, ValueError, KeyError, TypeError):
        return None


def _world_half_m(world: str):
    """Exact site half-extent (m). The world builder stamps it into the USD's
    customLayerData; the map needs it to scale the ground and place live drones."""
    d = ASSETS / world
    mj = d / f"{world}_map.json"
    if mj.exists():
        try:
            h = json.loads(mj.read_text()).get("half_m")
            if h:
                return h
        except (OSError, json.JSONDecodeError):
            pass
    usd = d / f"{world}.usd"
    if usd.exists():
        try:
            from pxr import Usd
            data = Usd.Stage.Open(str(usd)).GetRootLayer().customLayerData
            site = json.loads(data.get("vesper_site", "{}"))
            if site.get("half_m"):
                return site["half_m"]
        except Exception:                                  # noqa: BLE001
            pass
    return None


def _demo_media(world: str) -> list[dict]:
    """Cached demo pack for a world: pre-rendered stills/flyovers under
    assets/<world>/demo/. Instant + reliable — no box needed to show them."""
    dd = ASSETS / world / "demo"
    if not dd.is_dir():
        return []
    return [{"file": p.name, "url": f"/demo/{world}/{p.name}",
             "kind": "video" if p.suffix == ".mp4" else "image"}
            for p in sorted(dd.iterdir())
            if p.suffix in (".png", ".jpg", ".mp4")]


def _site_entry(world: str) -> dict:
    d = ASSETS / world
    half = _world_half_m(world)
    return {
        "name": world, "world": world, "half_m": half,
        "ground": f"/site/{world}/ground" if ((d / "ground.png").exists() or (d / "ground.jpg").exists()) else None,
        "scenario": f"{world}0.json" if (ROOT / f"{world}0.json").exists() else None,
        "usd": f"assets/{world}/{world}.usd" if _world_built(world) else None,
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
    if not name or not _world_built(name):
        cands = [d.name for d in sorted(ASSETS.iterdir())
                 if d.is_dir() and _world_built(d.name)] if ASSETS.is_dir() else []
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
        if (mj.parent / "ground.png").exists() or (mj.parent / "ground.jpg").exists():
            out.append({"world": world, "half_m": meta.get("half_m"),
                        "ground": f"/site/{world}/ground"})
    return out


@app.get("/api/zones/{world}")
def zones(world: str):
    """Operator zones for a world: the launch pad and any no-track safe areas.

    assets/<world>/zones.json wins; a tracked <world>_zones.json at the repo
    root is the fallback default (vesper.worlds.zones.find_zones)."""
    if not RUN_ID.match(world):
        raise HTTPException(400, "bad world")
    center = _world_center(world)
    for cand in (ASSETS / world / "zones.json", ROOT / f"{world}_zones.json"):
        if cand.is_file():
            try:
                d = json.loads(cand.read_text())
            except (OSError, json.JSONDecodeError):
                raise HTTPException(500, f"{cand.name} is not valid JSON")
            out = {"world": world, "launch": d.get("launch"), "safe": list(d.get("safe") or []),
                   "source": cand.name, "launch_point": d.get("launch_point"),
                   "safe_geo": d.get("safe_geo"), "center": d.get("center") or center}
            # older files (site metres only): derive the geo versions from the world centre
            if center and out["launch_point"] is None and out["launch"]:
                xs = [p[0] for p in out["launch"]]; ys = [p[1] for p in out["launch"]]
                cx, cy = sum(xs) / len(xs), sum(ys) / len(ys)
                la, lo = _geo(cx, cy, *center)
                out["launch_point"] = {"x": round(cx, 2), "y": round(cy, 2), "r_m": LAUNCH_R_M, "lat": la, "lon": lo}
            if center and out["safe_geo"] is None:
                out["safe_geo"] = [[_geo(x, y, *center) for x, y in poly] for poly in out["safe"]]
            return out
    return {"world": world, "launch": None, "safe": [], "source": None,
            "launch_point": None, "safe_geo": [], "center": center}


class ZonesReq(BaseModel):
    launch: list[float] | None = None            # [lat, lon]
    safe: list[list[list[float]]] = []           # [[[lat, lon], ...], ...]


@app.put("/api/zones/{world}")
def put_zones(world: str, req: ZonesReq):
    """Set/replace a built world's launch pad + friendly zones. The pad must be
    inside the world and at least LAUNCH_TREE_CLEAR_M from every trunk in its map."""
    if not RUN_ID.match(world) or not _world_built(world):
        raise HTTPException(404, "no such world")
    center = _world_center(world)
    half = _world_half_m(world)
    if not center:
        raise HTTPException(409, "world has no centre metadata (dem_meta.json)")
    if req.launch is not None:
        x, y = _site_xy(req.launch[0], req.launch[1], *center)
        if half and max(abs(x), abs(y)) > half - LAUNCH_R_M:
            raise HTTPException(400, "launch pin must sit inside the world")
        clear = _tree_clearance(world, x, y)
        if clear is not None and clear < LAUNCH_TREE_CLEAR_M:
            raise HTTPException(400, f"launch pin is {clear:.1f} m from a tree; needs > {LAUNCH_TREE_CLEAR_M:.0f} m")
    for poly in req.safe:
        if len(poly) < 3:
            raise HTTPException(400, "a friendly zone needs at least 3 points")
    _write_zones(world, _zones_doc(center[0], center[1], req.launch, req.safe))
    return zones(world)


@app.get("/demo/{world}/{name}")
def demo_media(world: str, name: str):
    """Serve a cached demo-pack file (assets/<world>/demo/<name>)."""
    if not RUN_ID.match(world) or not RUN_ID.match(name):
        raise HTTPException(400, "bad path")
    f = ASSETS / world / "demo" / name
    if not f.is_file() or f.suffix not in MEDIA_TYPES:
        raise HTTPException(404, "no such demo file")
    return FileResponse(f, media_type=MEDIA_TYPES[f.suffix])


@app.get("/api/world3d/{world}")
def world3d(world: str):
    """Geometry for the in-browser 3D view (components/world-view.tsx).

    Everything is derived from data the sim itself flies against, so the view
    and the task agree: terrain + tree placement from <world>_map.npz, crisp
    building footprints from osm.json with heights read back off the obstacle
    raster (the truth the sensor raymarches). Built once, cached beside the
    assets, rebuilt when the map is re-exported.
    """
    if not RUN_ID.match(world):
        raise HTTPException(400, "bad world")
    from vesper.worlds.webgeo import ensure_world3d
    try:
        cache = ensure_world3d(world, ASSETS)
    except FileNotFoundError:
        raise HTTPException(404, "world has no baked map")
    return FileResponse(cache, media_type="application/json")


@app.get("/site/{world}/ground")
def site_ground(world: str):
    """Ground ortho, downscaled once to a web-friendly jpg (the source is 30 MB)."""
    if not RUN_ID.match(world):
        raise HTTPException(400, "bad world")
    src = ASSETS / world / "ground.png"
    if not src.is_file():
        pre = ASSETS / world / "ground.jpg"                 # mirrored preview of a box build
        if pre.is_file():
            return FileResponse(pre, media_type="image/jpeg")
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
