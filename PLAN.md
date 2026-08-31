# VESPER

**Turn any real-world image into a training and evaluation gym for autonomous systems.**

> "Autonomy teams ship policies the way software teams shipped code in 1995 — test it by hand, deploy it, hope. Vesper turns real places into simulation gyms where you demonstrate a mission, train a policy, and automatically generate the edge cases that break it. Find the failure before the field does."

---

## Thesis

The four-mode structure and the closed loop are the product:

```
Real-world image / footage
        ↓
     3D world
        ↓
  Human demonstrations
        ↓
  Autonomous policy
        ↓
  Scenario sweep (hundreds of variants)
        ↓
  Failure discovery          ← the moat
        ↓
  Harder training worlds  ⟲  ← the wow
```

Three deliberate upgrades over the first draft:

1. **Positioning** — sell the evaluation loop ("CI for autonomy"), not the simulator. Simulators put you next to Unreal and Isaac; "your policy gets a failing test, then a fix" is a story every judge understands and nobody else in the room is telling.
2. **Policy architecture** — a learned residual over a conventional guidance law, which makes the wow moment *genuinely real* instead of staged (details below).
3. **Image-first world generation** — aerial images, single photos, and drone footage all become flyable worlds through parallel pipelines feeding one sim. The biggest version of the pitch: *any image of the real world becomes a gym.*

For dnhacks the live mission can be recon/ISR — that audience expects it. Search-and-rescue, inspection, agriculture, and delivery become the market-breadth slide.

---

## World Generation — image in, world out

The front door of the product: **drop in a real-world image, get a flyable, physics-backed gym.** Three input modalities, in order of reliability:

### Pipeline A — Overhead image (satellite tile or drone still) → world  *[must work]*

```
aerial image
  → building footprint segmentation (SAM2 via segment-geospatial)
  → height inference (shadow length / footprint-area heuristic, or a monocular height model)
  → extruded prisms + road ribbons → three.js scene
```

Fast, reliable, and produces **clean collision geometry you fully control** — which everything downstream depends on. Demo beat: *"drop in any aerial photo, get a gym in seconds."*

### Pipeline B — Ground-level photo → generative 3D world  *[the frontier beat]*

Single photograph → explorable 3D world via **HunyuanWorld-1.0** (open-source, runs on the GPU VM; image → panorama → layered 3D mesh, exportable) or World Labs **Marble** (API route). Voxelize the output mesh/splats into an occupancy grid for collision. One photo becomes a flyable world — the most frontier-feeling 30 seconds available to any team at the hackathon.

### Pipeline C — Drone video → photoreal splat twin

```
1–3 min orbit footage → ffmpeg frames → COLMAP poses
  → splat training (splatfacto / OpenSplat) on GPU VM (~20–40 min, ~$0.50/hr Runpod)
  → .splat rendered in-browser with gsplat.js
```

Collision from the footage itself: voxelize the Gaussian centers into an occupancy grid and raycast against it. **The move that wins the room:** fly a drone over the venue the morning of, and train a policy inside a photoreal twin of the building the judges are standing in. (Shoot your own footage or use clearly-licensed video; online rips are for testing the pipeline only.)

### Bonus — or just type an address

OSM Overpass footprints + AWS Terrain Tiles → the same extruded world from a location search. A few hours of work, kept as filler-content generator and fallback, no longer the headline.

**The architectural payoff:** the simulator only ever queries a collision structure (prisms or voxel grid), so all pipelines feed the *identical* flight stack, policy, and evaluation loop. The GPU VM is an offline reconstruction worker; the app stays a URL judges open themselves.

---

## Simulation — a PX4-style flight stack in miniature

No Unity, no Unreal, no off-the-shelf engine — but this is not a toy. ~800–1,000 lines of TypeScript implementing the same architecture as a real autopilot:

- **6-DOF rigid-body quadrotor dynamics** — quaternion attitude, per-rotor thrust and torque, aerodynamic drag, gravity. A real airframe model, not a point mass.
- **Cascaded PID control** — position → velocity → attitude → rate loops, mirroring the PX4 flight stack. The "low-level stabilization stays conventional" claim is now literally true and recognizable to anyone who has flown real hardware.
- **Dryden turbulence model** (MIL-HDBK-1797) for wind gusts — the standard used in actual aerospace simulation. Continuous gusts, not a constant wind vector.
- **Sensor suite** — GPS noise + dropout, IMU bias drift, perception rays clipped to visibility with Gaussian range noise.
- **Fixed timestep, seeded RNG** — bit-exact determinism: same seed, same outcome, every failing run replayable by clicking its row.

Rendering-free by construction → the same core runs at 60 fps for teleop and **hundreds-to-thousands× realtime headless in web workers**. Hundreds of stress-test variants finish in seconds — real, not faked scale.

**Why not Isaac Sim / Gazebo / a physics engine:** those buy contact resolution and photoreal sensors at the cost of the two properties the entire evaluation loop stands on — bit-exact deterministic replay (real engines' solvers aren't reproducible) and 1000× headless throughput (Isaac runs ~realtime per GPU instance). The rehearsed answer for judges: *"Isaac runs one scenario at realtime fidelity; Vesper runs a thousand at a thousand-x to find which scenario is worth Isaac's time."* Isaac's place is the roadmap slide — the graduation step between Vesper and hardware.

## Policy — learn a residual, not flight from scratch

Behavior cloning from five demos onto a raw waypoint network looks drunk on stage. Instead, mirror what real autonomy companies do:

- **Base layer (conventional, never trained):** classical guidance — goal attraction, obstacle/zone repulsion via potential fields. Deterministic, always competent.
- **Learned layer (from your demos):** a small MLP (2×64, tfjs) that outputs *corrections* to the base layer's heading and speed, trained by behavior cloning. It learns the human's judgment — how wide to swing around buildings, when to slow, how to sequence checkpoints — not how to fly.

State vector (egocentric): ~16 raycast distances **clipped to sensor range**, goal bearing/distance, velocity, altitude, wind, restricted-zone proximity. Action at 5 Hz: heading delta, speed, climb.

~5 demos × 40 s × 5 Hz ≈ 1,000 samples, augmented (mirroring, ray noise) to ~10k → trains in **under 30 seconds, live, in the browser, on stage**, loss curve animating.

### Why the wow moment is real, not scripted

Visibility *is* the policy's sensor range. Demos flown in clear weather have long rays; in fog, rays clip early, the input distribution shifts off the training data, and the learned correction degrades near obstacles — the policy **genuinely fails more in fog, for a mechanistic reason you can explain in one sentence**. The recovery is equally real: generate hard variants concentrated in the failing region, augment demos with clipped-ray noise, fly one or two fresh demos in fog, retrain — the number actually goes up. When a judge asks "is the improvement real?", that question becomes your best moment.

---

## The Four Modes

### Build
- Address search or footage upload → world generation with a build animation.
- Asset palette: vehicles, checkpoints, search target, obstacles, drawn restricted-zone polygons; environment sliders (time of day, visibility, wind).
- **Natural-language missions**: "Search the north lots, avoid the school grounds, return within 3 minutes" → Claude via Vercel AI Gateway → `{checkpoints, restricted_zones, constraints}` rendered on the map for the user to confirm. NL proposes, the human approves — also the right product answer.

### Demonstrate
- WASD + QE teleop, chase cam, recording full state + action at 5 Hz.
- The personality of this mode is the toast: `Demo 07 saved · 42 s · 0 collisions · mission complete`. A demo shelf shows every run as a card with a trajectory thumbnail. It must feel like **data collection**, not gameplay. A crash is saved too — "that's data."

### Train
- Select demos → **Train** → live loss curve → "Policy v1 ready" in under a minute.
- Immediately: **Watch autonomous run** — human trajectory vs. policy trajectory side by side on the same map. The emotional core of the product.

### Stress Test
- Randomize over spawn, obstacle placement/density, visibility, wind, sensor noise, target location, zone geometry. Seeded → every run replayable.
- Dashboard: overall success rate; **success binned by each condition dimension** (where the 88% → 57% chart lives); failure-mode breakdown (collision / zone violation / timeout / target missed); failure heatmap on the actual map.
- Vesper auto-writes the finding: *"Success drops 31 points when visibility < 200 m."* One button beside it: **Generate Hard Cases** → new scenarios concentrated in the failing region → back to Train. The loop closes on screen.

---

## Demo Script — three minutes, one story

| Time | Beat | The line |
|------|------|----------|
| 0:00 | Drop a real aerial image (or the venue splat twin) → the world rises out of the dark | "This is a real place, built from a single image." |
| 0:30 | Type the mission in English; checkpoints + restricted zone render | "Missions in natural language, grounded to real geometry." |
| 1:00 | Fly one clean demo (practiced, ~35 s); shelf already holds demos 01–06 | "Every flight is training data." |
| 1:40 | **Train** — loss converges live; policy flies the mission autonomously | "Stabilization stays conventional. The judgment is learned." |
| 2:10 | **Stress Test** — 200 variants in seconds: 88% overall, **57% in low visibility**; click a failing run, watch it clip a building in fog | "In the real world you find this failure over a real neighborhood." |
| 2:40 | **Generate Hard Cases** → retrain → re-sweep: low-visibility at **84%** | "Vesper found the weakness, built the worlds that target it, and closed the loop. That's the product." |

Numbers on stage are live; report the real ones whatever they are — a real 79% beats an 84% you have to defend. Insurance: cached weights for both policy versions and a full screen-recording, used only if the venue network dies.

---

## Build Plan — 36 hours, four people

| Owner | Workstream | Hours 0–12 | Hours 12–24 | Hours 24–36 |
|-------|-----------|------------|-------------|-------------|
| **A** World | Image → 3D | Pipeline A: SAM2 footprint segmentation, height inference, extrusion, r3f scene | Fog/wind/night visuals, build animation, asset placement; Pipelines B/C running on GPU VM | Generative/splat worlds integrated; venue flyover; pre-cache 2 demo worlds |
| **B** Sim | Physics → recording | 6-DOF quadrotor + cascaded PID, teleop, collisions, zones | Dryden wind, sensor models, recording, replay, headless worker batch runner | Seeded scenario randomizer, replay-from-row |
| **C** Learning | Policy → dashboard | State featurizer, base guidance law, BC residual in tfjs | Stress-test dashboard, binned analysis, auto-finding | Hard-case generation, retrain flow, cached weights |
| **D** Product | Shell → pitch | App shell, four-mode nav, dark UI system | NL mission → JSON via AI Gateway, demo shelf, toasts | Deck, roadmap slide, rehearse the 3-minute script ×5 |

**Integration checkpoints:** hour 12 — teleop a drone through a generated world. Hour 24 — full loop end-to-end, however ugly. The loop working ugly at 24 beats any mode being beautiful.

**Tonight's de-risk (1 hour):** run any licensed aerial orbit clip through splatfacto on a rented GPU. Proves the tier-2 pipeline end-to-end before demo minutes get committed to it.

## Stretch Tier — if ahead at hour 24

Ordered by wow-per-hour. These are real, not vapor:

1. **Autonomous curriculum loop** — chain Generate Hard Cases → retrain → re-sweep with no human in the loop; leave it running an hour and show policy v1 → v5 success climbing on one chart. The strongest possible closer: *Vesper improves policies while you sleep.*
2. **Multi-drone missions** — the sim is a pure function; N drones is a loop. Coordinated area search with deconflicted sectors demos beautifully in ISR framing.
3. **Fleet-scale evaluation** — the deterministic sim runs anywhere JS runs; fan 1,000 scenarios across serverless functions and show a live counter. Turns "massive parallel training: fake it" into "massive parallel evaluation: real."
4. **Live venue reconstruction** — kick off the splat job during judging of other teams; reveal the venue twin in your slot.

## Cut Lines — in order, without hesitation

1. Stretch tier (any of it)
2. Failure clustering → binned marginals only (80 lines that read as insight)
3. Weather variety → fog + wind only; fog is the one the story needs
4. Gamepad, day/night cycle, multiple drone types
5. Sim-to-real → one roadmap slide. Never build.

## Risks

| Risk | Likelihood | Response |
|------|-----------|----------|
| Height inference from imagery is rough | High | Footprints matter, heights don't: heuristic heights look fine. Nobody can tell. |
| Generative world (Pipeline B) is janky | Medium | Demo runs on Pipeline A worlds; B becomes a "same loop, generated world" beat or slide. |
| Splat reconstruction fails/ugly | Medium | Same fallback: Pipeline A carries the demo; splat becomes a slide. |
| Live training misbehaves on stage | Medium | Cached weights v1 + v2; loss curve replays from recorded history. |
| Teleop crash during live demo | Medium | Crashing is fine — save it ("that's data too"), load a good demo from the shelf. |
| Policy looks incompetent | Low — *because residual* | Increase base-law weight; worst case it's 90% guidance law and still honest ("learned corrections over conventional guidance"). |
| Overpass down at demo time | Low | Demo locations pre-cached as static JSON at build time. |
| "Is the improvement real?" | Certain | Yes. Rays clipped by visibility, one sentence. Your best moment. |

---

## Stack

Next.js on Vercel · three.js / react-three-fiber · SAM2 / segment-geospatial · HunyuanWorld-1.0 or Marble · gsplat.js · pure-TS deterministic 6-DOF sim in web workers · tfjs · Claude via Vercel AI Gateway · GPU VM (Runpod) for segmentation + reconstruction jobs · localStorage + static JSON, no database, no auth.

*The loop is the product: find the failure before the field does.*
