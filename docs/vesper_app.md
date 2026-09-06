# Vesper — the app

Vesper is a **flight lab and operations console for drone autonomy in real
places**. One system, two modes over a shared substrate. This doc supersedes the
earlier `ui_agent_brief.md` / `ui_layout_brief.md` framing (the Live page as an
ad-hoc panel grid); keep those only for component-level detail.

## The one-sentence product

Turn a real place into a simulator, **fly** policies in it live (Operations),
and **make them better** by training and sweeping for failures (Lab) — where
every live flight becomes training data and every trained policy is deployable
live.

## Two modes

### Operations — "fly it now" (map-first)
- **AO map is the hero**: all drones' live positions, tracks, and coverage/belief
  over the real site. In the throughput lane this is a whole swarm sweeping at
  once — the demo showpiece. (`components/site-map.tsx`, extended to plot live
  drone state.)
- **Live camera feeds** as a supporting panel — the drone's own view
  (`components/drone-feeds.tsx`, MJPEG from the box, port 8180).
- **Deploy / task** controls: pick a policy + scenario, deploy into the warm
  session (instant — see below). Task verbs later (search / hold / goto / RTL).
- **Event feed**: detections, reaches, low battery, lost link, geofence.

### Lab — "make it better" (the R&D loop)
- **Environments** — the real sites (build/select). (`app/environments`)
- **Runs & Trajectories** — every run logged; view path + sensors + video;
  **compare** runs side by side. (`app/runs`, `trajectory-plot`, `run-detail`)
- **Cohorts** — *the missing spine*: select a set of trajectories/scenarios, then
  **train or evaluate** a policy against exactly that set, and see the result.
  This is the connective tissue the app currently lacks.
- **Sweeps** — N seeded scenarios → success by condition (visibility/wind/…) →
  click a failure row → replay it. (`sweep-table`, `eval_search.py`)
- **Models & Training** — launch training, watch the curve, get a checkpoint,
  score it across a cohort. (`app/models`, `curve-charts`)

## What makes it ONE app (not two glued together)

Shared substrate + artifacts that flow between modes:
- **One run/trajectory store** (parquet + manifests under `runs/`).
- A mission flown in **Operations lands as a run in the Lab**.
- A policy trained in the **Lab is deployable in Operations**.
- **One warm sim session** on the box serves both (below).

The demo rides this flow: fly live → it becomes data → train on that data →
redeploy, measurably better.

## Shared infra: the warm session (build this FIRST)

Today every action spawns a fresh process that reloads Isaac + the world +
shaders — 1–2 min of dead time before anything happens. Unacceptable for a demo
and for interactive use.

**Fix:** one long-running sim process on the box that loads the world **once** and
then accepts commands over HTTP (a sibling of the existing frame server in
`vesper/capture/live.py`):
- `POST /reset {scenario}` → new episode/mission in the already-loaded world ≈ **~1 s**.
- `POST /deploy {policy, scenario}` → load a policy and fly it, in place.
- feeds broadcast **continuously**; the Live panel is never dark.

`live_world.py` proves the persistent session; the frame server proves streaming.
What's new: the command endpoint inside that session, and routing missions/deploys
through it instead of `docker compose run` per action. The jobs API
(`web/server/app.py`) shifts from "launch a process" to "send a command to the
warm session" for the interactive kinds (mission/fly/deploy); training/sweeps stay
as their own processes (they're batch, not interactive).

## Navigation

Top-level split, not a flat tab row:
- **OPERATIONS**: Live map + feeds + deploy.
- **LAB**: Environments · Runs · Cohorts · Sweeps · Models.

## Non-goals / guardrails
- Don't conflate the drone feed (operations) with the Isaac editor viewport
  (a world-inspection tool — keep it demoted, one click away).
- One visual system (existing dark HUD). No new deps without reason.
- Backend stays stateless over `runs/` except the warm-session command path.
- Commit to `web-client`; never `main`.
