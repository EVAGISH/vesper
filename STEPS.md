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

- Pipeline A, textured: drape the aerial orthophoto over the terrain mesh and project facade imagery onto the extruded prisms; real elevation (DEM) replaces flat ground.
- Pipeline B (photo → generative mesh) and Pipeline C (video → splat) evaluated as world sources. Test Isaac Sim 5.x's neural-reconstruction (3DGS) rendering on our splats; if it renders and collides acceptably, splats become a world type in `worlds/`, otherwise B/C are cut and A-textured is the ceiling.
- Every world type ships with a collision proxy — splats and generative meshes get a simplified mesh so PhysX behaves the same regardless of how the world is rendered.
- Realism is measured, not eyeballed: fly the same scenario over prism, textured, and splat versions of one site; diff the depth/range observations and the sweep outcomes from Step 7. Divergence tells you which fidelity the policy actually needs.
- Ray-cast sensors read the collision proxy and RTX cameras read the visuals, so the two can disagree; the log records which world type produced each observation.

**Inspect:** MP4 + livestream of the drone flying the Step 3 mission through a photoreal version of the same site, side by side with the prism version; a short table of sweep outcomes per world type.

---

## Step 9 — Policy

Only now. Residual-over-guidance (or setpoint policy — decided by what Steps 5–6 showed), trained in the throughput lane in the world types Step 8 justified, validated in the fidelity lane, demos recorded via Step 1 teleop. Details deliberately deferred.

**Inspect:** side-by-side videos — human demo vs. policy on the same scenario; policy in the fidelity lane under real PX4.

---

## Standing rules

- Every step ends with a commit on completion, and messy in-between states get committed too — the branch history should let any step be revisited.
- Every run, every step, auto-captures at minimum an overview MP4 and a parquet log. Inspection is a default, not a favor.
- Pure modules (`scenario`, `dynamics`, `control`, `sensors`, `record`, `viz`) get CPU tests in CI from the step that creates them.
- A step's inspect artifact gets glanced at by a second person before the step is called done.
