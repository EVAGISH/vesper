# VESPER — Simulation Stack & Environment Plan

Replaces the original hackathon plan. The product thesis is unchanged (real place → gym → demos → policy → sweep → failure discovery → harder worlds). What changes: the simulator is no longer a hand-rolled TypeScript engine — it is NVIDIA Isaac running on cloud GPUs, inspected from the Mac via livestream and rendered artifacts.

Sections are marked **[confident]** or **[open]**. Open items are deliberately undecided until the environment is up and measured. The build order lives in `STEPS.md`.

---

## 1. Goal

A faithful, GPU-accelerated drone simulation environment that:

- runs headless at scale on cloud GPUs for sweeps and training,
- can be flown manually with a real autopilot in the loop for demonstrations and validation,
- gives the policy access **only** to simulated sensor outputs, never ground truth,
- can produce renders (frames, videos, live stream) from the cloud box on demand, so the state of things is manually inspectable at every stage,
- is fed by one scenario spec and emits one trajectory format, so every downstream tool is indifferent to which lane produced the data.

Get this working end-to-end first. Policy, dashboard, and world-generation pipelines build on top of it.

---

## 2. Stack **[confident]**

| Component | Version | Role |
|---|---|---|
| Isaac Sim | **5.1** | Physics (PhysX), RTX rendering, USD scene, sensors |
| Isaac Lab | **2.3.x** | Vectorized environments, batched sensors, headless stepping, RL/eval plumbing |
| Pegasus Simulator | **v5.1.0** | Multirotor dynamics + PX4/ArduPilot/MAVLink bridge, single-vehicle fidelity |
| PX4 | 1.16 SITL | Real autopilot: cascaded control, EKF2, failsafes, flight modes |
| Rerun | current | Mac-native viewer for trajectory logs, sensor images, time series |
| Container | `nvcr.io/nvidia/isaac-lab:2.3.x` (NGC) | Base image; everything above installs on top |

**Why this pin.** Isaac Lab 2.3.x is the stable line built on Isaac Sim 5.1. Pegasus's latest release targets Isaac 5.x and has no Isaac Sim 6.0 release. Isaac Lab 3.0 (beta, Isaac Sim 6.0) drops 5.1 support — adopting it means losing Pegasus. Revisit only if Pegasus ships a 6.0 release. **[open]** timing of that migration.

**Platform.** Isaac runs on Linux only (Ubuntu 22.04, NVIDIA driver 570+) and needs an NVIDIA GPU — it does not run on macOS, and a Linux container on a Mac gets no GPU. Everything Isaac happens on the cloud box; the Mac sees it through the livestream, pulled render artifacts, and rerun.

---

## 3. Architecture: two lanes, one contract **[confident]**

```
              scenario spec  (seeded JSON: world, spawn, wind, visibility, zones, sensor noise)
                     │
      ┌──────────────┴───────────────┐
Throughput lane                  Fidelity lane
Isaac Lab 2.3, one GPU            Isaac Sim + Pegasus + PX4 SITL
thousands of envs                 1–4 vehicles, real autopilot
torch dynamics + torch controller  manual teleop, RTX cameras, validation
      └──────────────┬───────────────┘
              trajectory log  (parquet, full state + sensor obs every step)
                     │
              rerun viewer / rendered frames & videos
```

Both lanes implement the same `Env` interface (`reset(spec) → obs`, `step(action) → obs, done, info`) and write the same log. The lanes differ in physics backend, sensors, and scale.

### Why Pegasus is not the scale path

Pegasus evaluates each vehicle's dynamics in a per-vehicle Python object and pushes the resulting force/torque into PhysX one prim at a time. It is a Python loop over vehicles and was never designed to vectorize. PX4 is a separate process per vehicle and cannot vectorize at all.

### What scales instead

Pegasus's dynamics *model* is small and public: quadratic rotor thrust/torque from rotor speed, first-order motor lag, linear + quadratic body drag, gravity. Ported to batched torch ops over `[num_envs, 4]` rotor states, it runs inside an Isaac Lab `DirectRLEnv` with a single `set_external_force_and_torque` call for all envs; PhysX integrates rigid bodies and resolves collisions against the world on the GPU. This is how Isaac Lab's own quadcopter env, Aerial Gym, and OmniDrones work. Same equations, different evaluator.

### The inner-loop controller: what can and cannot be GPU-accelerated

Three different things get called "the controller":

| Inner loop at scale | Cost to port | Fidelity vs. real PX4 |
|---|---|---|
| Pegasus's built-in geometric (SE(3)) controller → torch | hours | different controller, competent |
| PX4's `PositionControl` / `AttitudeControl` / `RateControl` + control allocation → torch | days | same control laws; no estimator, modes, or failsafes |
| Real PX4, one process per vehicle | none | exact, but ~tens of vehicles per box |

PX4 itself cannot be vectorized: it is a C++ flight stack (scheduler, uORB, EKF2, mode state machine) with internal state, not a function of state. Its control laws, however, are isolated classes of a few hundred lines with public gains — those port cleanly. The plan is the middle row. Nobody should expect PX4 to run at thousands of envs.

The torch controller is the one honest fidelity gap. It is measurable: fly the same seeded scenario in both lanes and diff the trajectories. Because the learned policy sits *above* the controller (setpoints or residual corrections), the gap should stay small; when it doesn't, the fidelity lane is what catches it.

### Two rules for the throughput lane

1. **World is a global prim, not cloned per env.** Isaac Lab's default replicates each env's scene at spatial offsets; thousands of copies of a city block will not fit. The world lives once (like Isaac Lab's shared-terrain locomotion envs), drones are the per-env asset, env spacing ≈ 0, inter-env collision filtering on. Thousands of drones, one world.
2. **Replay from logs, never by re-simulation.** Isaac Lab is deterministic across runs on identical hardware and version, but the GPU pipeline is fragile (runtime parameter changes perturb low-order bits; runtime randomization of physics materials is discouraged). Log full state every step; "click the failing run" replays the log.

---

## 4. Sensors **[confident on mechanism, open on scale]**

**The rule:** the policy observes sensor outputs only. Enforced in one place — the env's observation builder reads sensor buffers, never ground-truth state. (Standard exception: a critic may see privileged state during training; the actor never does.)

| Sensor | Throughput lane (Isaac Lab) | Fidelity lane |
|---|---|---|
| Range / lidar / depth-by-raycast | `RayCaster`, `RayCasterCamera` (Warp, GPU, no RTX) | same, or RTX lidar |
| RGB / depth camera (rendered) | `TiledCamera` (RTX, batched) | RTX camera |
| IMU | `Imu` sensor or torch model | Pegasus / PX4 |
| GPS, barometer, magnetometer | torch model | Pegasus / PX4 |
| Contact | `ContactSensor` | PhysX |

Noise, bias drift, latency, dropout, and visibility clipping are ours, written once in torch and shared by both lanes. Fog on a rendered camera is a render setting; on a ray-cast sensor it is range clipping — the two are not the same effect, and the log records which sensor produced the observation.

**[open]** RTX cameras are the one thing that sharply caps env count. Likely outcome: range/depth sensors at scale, RGB in the fidelity lane and at small env counts. Measured early (see `STEPS.md`).

**The search task's camera (Step 11).** A body-fixed `TiledCamera` per drone, pitched 40° forward-down, 110° square lens, RGB + instance segmentation at 128 px. Its pose and intrinsics come from one config (`SearchCfg.cam_pitch_deg`, `fov_half_deg`) shared with the geometric cone, the render camera in the flight scripts, and the coverage footprint, so every consumer looks at the same patch of ground. Because the world is one global prim, every camera sees every vehicle: vehicle sets are shared by groups of environments (`SearchEnvCfg.n_groups`), which is the honest way to keep hundreds of cameras in one scene. The actor is GPS-denied -- pixels plus proprio only -- and the privileged vector (world pose, belief, coverage) is reserved for the critic and for a state-based teacher.

---

## 5. Cloud & inspection **[confident on shape, open on sizing]**

- **Compute: DigitalOcean GPU Droplets.** Dev box: **RTX 4000 Ada 20 GB** (~$0.76/hr); scale tier when VRAM demands it: **L40S 48 GB** (~$1.57/hr). Real VMs, per-second billing, UDP allowed — livestream works.
- No image baking needed: DO's "NVIDIA AI/ML Ready" base image ships driver + Docker. The daily cycle is `launch.sh` → work → `snapshot.sh` (power off, snapshot, destroy — DO bills powered-off droplets, so there is no cheap "stopped" state; the snapshot preserves the docker image and shader cache).
- **Storage:** run artifacts live on the droplet and rsync to the Mac; the snapshot carries the docker image and shader cache between days. No object store yet.
- **Fan-out:** one box at a few thousand envs covers the stress-test story. Multi-box only when sweeping many worlds × many policies.
- Plan on ~2 boxes; one shared dev box gets contended fast.
- `infra/do/`: `launch.sh`, `ssh.sh`, `provision.sh` (rsync repo + NGC login + build), `snapshot.sh`, `destroy.sh`. Plain curl + jq, no doctl dependency.

**Manual inspection — three channels, cheapest first:**

1. **Rendered artifacts.** Headless runs attach a chase/overview camera and write PNG frames or MP4 to `runs/<id>/`; a one-liner rsyncs them to the Mac. Every milestone below has one of these as its exit criterion — "it works" always comes with a picture.
2. **Trajectory logs in rerun.** Every run writes the parquet log; rerun on the Mac shows the 3D path over the world, sensor streams, and time series — no GPU needed to look.
3. **Live: WebRTC livestream.** Isaac Sim's livestream mode + the WebRTC Streaming Client on the Mac, for interactive poking and teleop. Heaviest channel; needed for flying, not for checking.

**[open]** RunPod for headless sweep fan-out later (cheapest per-FLOP; container-shaped, no UDP — fine for batch, wrong for the dev box).

---

## 6. Repository **[confident]**

Heavy software (Isaac Sim ~10 GB, Isaac Lab, Pegasus, PX4 build) lives in the Docker image, never in git. USD worlds, splats, and run artifacts live on the droplet, gitignored locally. The repo holds Python, the Dockerfile, infra scripts, and config.

```
vesper/
├── STACK.md / STEPS.md
├── pyproject.toml            # `vesper` package; heavy imports kept out of __init__
├── docker/
│   ├── Dockerfile            # FROM isaac-lab:2.3.x → + Pegasus v5.1.0 + PX4 v1.16 build + pip -e .
│   ├── compose.yml           # gpu, livestream ports, mounts repo + assets/ + runs/
│   └── entrypoint.sh
├── infra/
│   ├── do/                   # DigitalOcean: launch, ssh, provision, snapshot, destroy
│   └── batch/                # later
├── vesper/
│   ├── env.py                # the `Env` interface + obs/action specs shared by both lanes
│   ├── scenario/             # spec dataclasses + seeded randomizer               pure python
│   ├── dynamics/             # torch multirotor model (Pegasus port), Dryden wind   pure torch
│   ├── control/              # torch PX4-style cascade + allocator                 pure torch
│   ├── sensors/              # noise/latency/dropout models                        pure torch
│   ├── record/               # trajectory schema (parquet), writer, replay         pure python
│   ├── capture/              # chase/overview camera, frame/MP4 writer            container
│   ├── worlds/               # spec → USD stage builder, asset registry            container
│   ├── lab/                  # throughput lane: Isaac Lab DirectRLEnv, sensor adapters → Env
│   ├── fidelity/             # fidelity lane: Pegasus wrapper, PX4 launcher, MAVSDK teleop → Env
│   ├── viz/                  # rerun logging of any trajectory log                 pure python
│   └── eval/                 # sweep runner, per-dimension binning, findings
├── scripts/                  # view.sh, capture_pull.sh, bench.py, fly.py, sweep.py, train.py
├── tests/                    # pure modules only — CPU, no Isaac; run in CI
├── assets/                   # gitignored; synced from the droplet
└── web/                      # later: browser trajectory viewer
```

**Organizing rule:** modules marked *pure* import only torch/numpy (plus rerun for `viz`) and are unit-testable on a Mac CPU. Only `capture`, `worlds`, `lab`, and `fidelity` import `isaaclab`/`omni`, and only inside the container. This keeps the dynamics/controller ports developable and testable without a GPU box, and lets the torch dynamics be verified against Pegasus's native model before trusting it at thousands of envs.

Start `vesper/lab` from Isaac Lab's project generator (`isaaclab.sh --new`) rather than hand-rolling the extension layout and Hydra configs.

---

## 7. Open questions

- **Throughput: MEASURED** (2026-09-02, RTX 6000 Ada, `Isaac-Quadcopter-Direct-v0` headless, resets included): 118k env-steps/s @1024 envs, 233k @2048, **456k @4096** — near-linear scaling, GPU not yet saturated. At 50 Hz control that is roughly 9,000 sim-seconds per wall-second aggregate: a 40 s mission × 200 variants in ~1 s of stepping. Tiled RTX cameras (`Isaac-Cartpole-RGB-Camera-Direct-v0` @256 envs): 13.4k env-steps/s — vision is ~2× per-env cost and caps env count, as predicted. VesperQuad (our Iris dynamics + SE(3) controller in the loop): **321k env-steps/s @4096** — a 512-variant, 45-sim-second city sweep with per-env wind/visibility/noise and torch rays runs in **51 s wall**.
- **Cameras at scale.** First datapoint above: tiled RGB at 256 envs runs 13.4k env-steps/s (~50× realtime aggregate) — viable for vision training at hundreds of envs, not thousands. Depth-by-raycast at high env counts still unmeasured.
- **Controller port scope.** How much of PX4's cascade to reproduce (rate loop only? full position cascade? feed-forward terms?). Decide after the two-lane diff shows where the gap is.
- **Splat worlds.** Isaac Sim 5.x added neural-reconstruction (3DGS) rendering. If it handles our splats, the photoreal-twin beat lives inside Isaac; otherwise it stays a separate viewer or is cut.
- **World-generation pipelines** (A: aerial → prisms, B: photo → generative mesh, C: video → splat). Unchanged as inputs; A is the only one the environment depends on. B and C are evaluated after the environment works.
- **Browser.** Rerun + pulled renders cover inspection for now. A browser viewer (`web/`) only if people need to watch runs without installing anything. Whether any of the original TypeScript sim survives — probably not, but not decided.
- **Autopilot.** PX4 by default; Pegasus also supports ArduPilot if a reason appears.
- **Policy architecture.** Residual-over-guidance (a learned correction on top of a conventional guidance law) is still the leading candidate. Where it acts (setpoints vs. corrections) depends on which controller layers get ported.
- **Isaac Lab 3.0 / Isaac Sim 6.0.** Multi-backend physics and kit-less install are attractive. Blocked on Pegasus.
