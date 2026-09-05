# VESPER — Build Steps

Big steps, in order. Every step ends with something you can look at from the Mac — a frame, a video, a livestream, or a rerun view. No step is "done" on logs alone. Stack details and rationale live in `STACK.md`.

---

## Step 0 — A box that renders

Stand up the GPU box and prove the render path end-to-end before anything else.

- Launch a DigitalOcean GPU droplet (RTX 4000 Ada, NVIDIA AI/ML Ready image — driver and Docker preinstalled): `infra/do/launch.sh`, then `provision.sh` to copy the repo, log into NGC, and build the container (`isaac-lab:2.3.x`; Pegasus + PX4 layers come in Step 1).
- Headless Isaac Sim script: flat ground + a quadrotor asset, an RTX camera, save PNG frames and stitch an MP4.
- `scripts/capture_pull.sh` syncs `runs/<id>/` down to the Mac.
- Livestream mode working: WebRTC client on the Mac shows the same scene interactively.

**Inspect:** an MP4 on your Mac of a drone sitting (or tumbling — physics on, nobody flying) on basic terrain, and the same scene live in the streaming client. *This is the "receive a render from the cloud box" milestone — nothing is built until this exists.*

---

## Step 1 — A drone that flies

Fidelity lane first, because it needs no code of ours to fly: Pegasus + PX4 SITL.

- Pegasus vehicle + PX4 SITL boot in the container; scripted takeoff → waypoint square → land (via MAVSDK offboard or PX4 mission).
- Chase camera follows; every run auto-captures frames/MP4.
- Keyboard/joystick teleop over the livestream as a second control path.

**Inspect:** MP4 of a stabilized quadrotor flying a square on basic terrain under a real autopilot; you fly it yourself over the livestream.

---

## Step 2 — Runs become data

The recording contract, before anything is trained or swept.

- `scenario/` spec (seeded JSON) and `record/` parquet schema (full state + actions + sensor obs, every step).
- Step 1 flights write logs; `viz/` renders any log in rerun on the Mac (3D path, attitude, time series).
- Replay: pick any past run, regenerate its video from the log.

**Inspect:** rerun on the Mac showing a flight's 3D trajectory scrubbed back and forth; a video regenerated from a log rather than a live sim.

---

## Step 3 — Real worlds

The scenario spec starts producing terrain worth flying in.

- `worlds/`: spec → USD — extruded prisms (Pipeline A geometry), simple terrain, restricted zones; wind/visibility fields in the spec even if not yet physical.
- Fly Step 1's mission through a generated world; collisions are real (clip a building, PhysX says so).

**Inspect:** MP4 + livestream of the drone flying between extruded buildings from a real aerial footprint; one deliberate crash into a wall.

---

## Step 4 — The throughput lane exists

Isaac Lab, stock parts, measured — before any porting work.

- `Isaac-Quadcopter-Direct-v0` headless at 2–4k envs; record env-steps/s.
- Re-measure with `RayCasterCamera` attached, then a small `TiledCamera`. These three numbers decide the sensor story at scale.
- Render gate for scale: capture a grid-view video of ~16 envs flying simultaneously.

**Inspect:** the benchmark table (steps/s: bare / raycast / tiled camera) and a tiled video of many drones at once.

---

## Step 5 — Our drone at scale

Port the models so the throughput lane flies *our* vehicle, not the stock Crazyflie.

- `dynamics/`: Pegasus rotor/motor/drag model in batched torch. Verify against Pegasus native: same inputs, single vehicle, overlaid trajectories in rerun.
- `control/`: PX4-style cascade (position → attitude → rate + allocator) in torch. Same verification against real PX4 from Step 1.
- `VesperQuad` env in `lab/`: global-prim world (no per-env cloning), collision filtering, hover + waypoint under the torch controller at thousands of envs, writing `record/` logs.

**Inspect:** rerun overlay of torch-port vs. Pegasus/PX4 trajectories on the same scenario (the fidelity-gap picture), and a video of the vectorized swarm flying waypoints in a generated world.

---

## Step 6 — Eyes only

Sensors become the policy's entire world.

- `sensors/`: noise, bias drift, latency, dropout, visibility clipping in torch; ray-cast range sensors and (env-count permitting, per Step 4) depth cameras wired into `VesperQuad` observations.
- The observation builder reads sensor buffers only — ground truth never leaks into the actor's obs. Privileged state allowed for a critic, nothing else.
- Visibility affects the sensor: fog clips rays; the failure mechanism from the original plan is now real in Isaac.

**Inspect:** rerun view of one drone's sensor rays / depth image alongside its trajectory; the same scene at full and degraded visibility, visibly different observations.

---

## Step 7 — Sweeps

The evaluation loop, minus learning.

- `scenario/` randomizer over spawn, wind, visibility, noise, zones; `eval/` runs N seeded variants through the throughput lane, bins success by condition dimension, emits findings.
- Every failing run is one click from its log replay and a regenerated video.

**Inspect:** a sweep report (success by visibility/wind/etc.) where each failure row links to a watchable replay.

---

## Step 8 — Realistic terrain

Step 3's prisms prove collisions; this step makes the worlds look and behave like the real place, so that what the sensors see in sim is what they would see on site. Kept after Sweeps so the eval loop exists before the worlds get expensive.

**Inspect:** MP4 + livestream of the drone flying the Step 3 mission through a photoreal version of the same site, side by side with the prism version; a short table of sweep outcomes per world type.

---

## Step 9 — Policy

Only now. Residual-over-guidance (or setpoint policy — decided by what Steps 5–6 showed), trained in the throughput lane in the world types Step 8 justified, validated in the fidelity lane, demos recorded via Step 1 teleop. Details deliberately deferred.

**Inspect:** side-by-side videos — human demo vs. policy on the same scenario; policy in the fidelity lane under real PX4.

---

## Step 10 — Search, then reach

Step 9's pursuit policy is handed a vector to its target every step, so it never
has to find anything. This step removes that.

- Several forklifts scattered at random over the Cornell world every reset, by
  concealment class: driving in the open, painted down, crawling under canopy,
  parked against the buildings. Which slot holds which is reshuffled per episode.
- The policy sees a *belief*, not the truth: what a downward camera cone reports,
  denied by terrain and buildings, attenuated by foliage, scaled by contrast —
  plus a coverage grid of what it has already swept. Under a random policy the
  sensor denies ~97% of target-steps; `check_search.py` asserts that it stays a
  search and does not decay into a chase.
- One world, many drones: the site is a global prim, `env_spacing` is 0, and the
  environments are separated by collision filtering (STACK.md §3, rule 1).
- Tree cover comes from the site's orthophoto, not OSM tags — a campus's canopy is
  almost entirely unmapped, and without it there is nowhere to hide.

**Inspect:** `runs/<id>/chase.mp4` with the policy's own belief burned into the
frame, `overview.mp4` of the sweep, and `track.png` — drone path, vehicle paths,
and where each vehicle was first seen and reached, over the site's ground texture.

---

## Step 11 — Eyes on the airframe

Step 10's drone found forklifts with a geometric detector fed the truth, and
navigated on its own world position. This step makes the environment honest
before any policy is trained on it: the actor gets a camera and its own
instruments, nothing else.

- The camera is body-fixed and pitched 40° forward-down with a 110° lens, so the
  drone sees where it is flying and the ground ahead. The SE3 inner loop yaws
  the nose onto the velocity, and the action is a body-frame velocity command.
- No GPS. The actor's observation is the rendered frame (`pixels`) plus what the
  airframe measures (`policy`): body velocity, gravity direction, rates,
  rangefinder height, clock. World position, the belief and the coverage grid
  live in a separate `privileged` vector for the critic and a state-based
  teacher only.
- A first sighting is decided by the camera's segmentation mask, not a cone.
  The geometric cone stays, pitched to the same axis, for training a teacher
  at thousands of environments without rendering.
- One shared world means every camera sees every vehicle, so vehicles come in
  `--groups`: G sets on the site, environment i hunts set i % G, a group's
  episodes start and end together. Forklifts follow the roads, park along
  facades, ramp their speed and cannot spin on the spot.
- Trees are physical: a trunk capsule and a crown sphere per species, authored
  once and shared by every instance; the map's solid layer carries the same
  geometry so a crash in the task is a contact in PhysX.
- Worlds are repeatable: `<site>_build.json` records seed, variant and input
  hashes; `--variant` reshuffles trees and building heights from one fetch.

**Inspect:** `check_search.py --camera` passes on the box and reports env-steps/s
and VRAM with 64 tiled cameras; `fpv.mp4` from `fly_search.py --camera` shows
the forward view with the policy's 128 px input inset, rolling with the hull.

---

## Step 12 — Close the loop: touch it, with nothing but the camera

Step 11 made the environment honest. This step makes the task closeable end to
end by one small network that could run on the airframe.

**The task** (`vesper.lab.chase_task`, `chase_env.py`). Forklifts drive around
the site as global prims — no assignment, every drone can hit every one. A
drone launches from the **launch zone**, has to *see* a forklift with its own
camera, fly to it through the trees and buildings, and **touch** it. The touch
is a PhysX contact, it ends the episode, and the bonus scales with how much of
the episode is left, so the only thing worth optimising is time. A contact with
anything else is a crash.

**Zones** (`vesper.worlds.zones`, `<site>_zones.json`). Safe polygons are
**friendly ground**, and the launch pad sits inside one. Every drone spawns at a
random point on the pad with a random heading, and every step it spends over
friendly ground costs — so the first thing worth learning is to leave. The
penalty is monotone in the distance to the boundary (worst deep inside, half at
the line, zero 25 m clear of it), so it points the way out the whole time, and
nothing about it ends the episode. A forklift on friendly ground is protected:
no sighting bonus, no touch reward, touching it ends nothing. Both polygons are
drawn on the AO map.

On Cornell the slope bisects the site. Everything below the break is friendly
(42% of the 1200 m square, traced along the −12 m contour, so the line follows
Libe Slope and runs east along both gorge floors); the campus above it is the
hunting ground where the forklifts drive. The whole raster is in play — the
arena is the terrain's own edge, not a box inside it — and the pad is a 50 m
square at (−170, −250), 65 to 132 m of flying to clear friendly ground.

The actor has no map, so on a general site it could not perceive that boundary.
On one fixed site it can learn it from landmarks, which is what this is; if that
proves too slow, `ChaseEnvCfg.geofence` appends the signed distance to the zone
to the proprio vector — the geofence receiver a real airframe carries.

**The policy** (`vesper.lab.vision`, `recurrent_ppo.py`). RGB + depth at 96 px
and the 11 proprio values, four stride-2 convolutions, a 256-unit GRU, actor
and belief heads; an asymmetric critic that sees the privileged vector.
**1.44M parameters on the airframe, 17.9M multiply-adds per frame** — an order
of magnitude under a Jetson Orin Nano at 25 Hz. The GRU is what holds a
forklift through the seconds it is out of frame, turns frame-to-frame growth
into range, and remembers which way the drone has already looked. The belief
head regresses the true relative vector whenever a forklift is in frame, which
is what teaches the encoder to see forklifts long before the sparse touch
reward could.

**Depth** is a stereo-class sensor, not the renderer's truth: clipped at 20 m,
error growing with range squared, holes on far and thin returns
(`vesper.sensors.depth`). Trees are solid and shaped like trees — every species
mesh carries a convex-decomposition collider, so a gap the depth camera shows
is a gap the drone can fly through.

**Inspect:** `check_chase.py --camera` passes on the box and reports
env-steps/s and VRAM; `runs/<id>/fpv.mp4` from `fly_chase.py` shows the forward
view with the policy's own RGB and depth tensors inset and its belief head's
cross on the forklift; `track.png` shows the launch zone, the safe zones, the
forklift paths and where each touch happened.

---

## Standing rules

- Every step ends with a commit on completion, and messy in-between states get committed too — the branch history should let any step be revisited.
- Every run, every step, auto-captures at minimum an overview MP4 and a parquet log. Inspection is a default, not a favor.
- Pure modules (`scenario`, `dynamics`, `control`, `sensors`, `record`, `viz`) get CPU tests in CI from the step that creates them.
- A step's inspect artifact gets glanced at by a second person before the step is called done.
