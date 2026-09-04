# Vesper Client UI — Agent Brief

You are building the web client for **Vesper**, a drone-simulation gym that turns
real-world places into training and evaluation environments. Read `STACK.md` and
`STEPS.md` at the repo root first — they define the system you are the window into.

## What exists (do not rebuild)

- **Backend**: `web/server/app.py` — FastAPI, stateless, filesystem-is-the-database.
  Run it with `python -m uvicorn web.server.app:app --port 8777`.
  Endpoints, all working:
  - `GET /api/runs` — every run under `runs/`, each `{id, manifest, files}`.
    `manifest.json` fields: `name, scene, started, finished, streams {name: kind},
    frames, fps, resolution`.
  - `GET /api/runs/{id}/trajectory?max_points=N` — flight path from
    `trajectory.parquet` as `{t[], px[], py[], pz[]}` (meters, world frame,
    spawn at origin).
  - `GET /api/scenarios` — every scenario spec JSON at the repo root:
    `{file, world, terrain_usd, waypoints, wind_ms, visibility_m, cruise_ms,
    max_sim_s, command}`.
  - `GET /media/{run_id}/{file}` — run artifacts with HTTP Range support
    (videos scrub properly in `<video>`).
- **Reference frontend**: `web/index.html` — a working single-file prototype
  (runs sidebar, video cards, canvas trajectory plot with hover, environments
  tab with copyable launch commands). It proves the data contract. Replace it,
  but keep feature parity as the V1 floor.
- Run kinds you will encounter (check `files` per run, render what exists):
  - flight runs: `overview.mp4` (chase), `fpv.mp4`, `trajectory.parquet`, `scenario.json`
  - search-policy runs: add `chase.mp4`, `track.png`, `events.json`
    (timeline: first-sighting / reached per vehicle)
  - training runs: `curve.jsonl` (one JSON object per iteration; plot reward/success vs iter)
  - sweep runs: `report.json`, `results.jsonl` (per-variant outcome; each failure
    should link to its replayable run)
  - view runs: `view_*.png` stills

## Architecture (decided — follow it)

- `web/server/` — the FastAPI app already lives here; extend it as needed, keep
  it stateless and read-only over `runs/` (exception: a later run-trigger endpoint).
- `web/client/` — **Next.js (App Router) + Tailwind + shadcn/ui**, dark theme
  default. Proxy `/api` and `/media` to the FastAPI server in dev
  (`next.config` rewrites → `http://localhost:8777`).
- No database. New runs appear when files appear; poll or revalidate cheaply.
- Charts: single accent hue for single-series (time-gradient allowed);
  categorical series get fixed hue assignments, never cycled; every chart gets
  hover tooltips; text in text colors, never series colors.

## Phase 0 — DESIGN FIRST (stop for approval)

The dashboard's look is **not decided**. Before writing app code, produce 2–3
distinct design directions as static mockups (HTML or images the user can open):
e.g. (a) mission-control dense grid, (b) media-first gallery with big video,
(c) timeline/feed of runs. Each mockup must show: the runs list, one run's
detail (2 videos + trajectory + metadata), and the environments view. Present
them and **wait for the user to pick one** before Phase 1. Note in your
presentation which direction you recommend and why.

## Phase 1 — V1: the runs browser

Feature parity with `web/index.html`, in the chosen design:
run list with artifact badges → run detail (all videos, stills, top-down
trajectory plot with hover, metadata, scenario params) → environments page
(scenario cards + copy launch command). Plus what the prototype lacks:
- `curve.jsonl` training-curve chart on training runs
- `events.json` timeline on search runs (sighting/reach markers, jump video to t)
- `report.json` sweep table where each failure row links to its run

## Phase 2 — V2: live (design the slot in V1, implement after V1 ships)

- "Live" panel on the home page: latest frame / MJPEG stream from an
  in-progress run on the GPU box (backend endpoint to be added — coordinate).
- Embedded interactive viewport via `@nvidia/omniverse-webrtc-streaming-library`
  pointed at the box's IP (ports 49100 TCP/UDP already open) — this is a
  stretch goal; do not block V1/V2 on it.

## Phase 3 — V3: 3D trajectory over the site ground texture (three.js or
embedded rerun web viewer). Defer until asked.

## Constraints

- Videos can be 30+ MB — always stream via `/media` (Range), never import/copy.
- Trajectory plots: equal-aspect axes (it's a map), meters labeled, start/end
  marked.
- The site world frame: spawn at origin, +x east, +y north, z up (meters).
- Everything must work offline/local — no external CDNs at runtime.
- Run `npm run build` clean and keep the FastAPI server importable
  (`python -m uvicorn web.server.app:app` or equivalent) before calling any
  phase done.

## Working agreement

Commit at each phase boundary on a branch (`web-client`), never to `main`.
After Phase 0 approval, don't re-litigate the design. Ask the user only for
decisions this brief doesn't cover.
