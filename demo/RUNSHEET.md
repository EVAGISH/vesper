# Vesper demo — run of show (~90s)

**One coherent arc:** a real place you can't safely operate in → reconstruct it →
train autonomy in it → validate with a real autopilot → **deploy the policy to the
drone's onboard hardware.** Each beat = one artifact. The arc is the point; the
capabilities hang off it.

## The 5 beats
1. **Real place** — Kramatorsk (can't fly/test there). Environments page → Kramatorsk (ACTIVE), its cached demo gallery loads instantly.
2. **Reconstruct** — from free satellite+map data (`demo/01_kramatorsk_reconstructed.png`) **or** an operator's own drone footage (`demo/05_photogrammetry_from_photos.png`).
3. **Train** — a search policy learns it: curve climbs (`demo/06_training_curve.jsonl`) + the hunt, drone's-eye with belief HUD (`demo/03_policy_hunting_fpv.mp4`) + from above (`demo/04_policy_track.png`).
4. **Validate** — a real PX4 autopilot flies it in sim (`demo/02_kramatorsk_flight.mp4`).
5. **Deploy** — export the policy to the aircraft's hardware: `demo/07_deployable_policy.onnx` (493 KB, → TensorRT on Jetson). *This is the payoff — it doesn't just stay in sim.*


## Before you present (de-risk)
- Start the **warm session** 5–10 min ahead so the live beat is already up (never start it live).
- Have this `demo/` folder open as the video/image fallback.
- Backend: `.venv/bin/python -m uvicorn web.server.app:app --port 8777 --reload`
- Frontend: `cd web/client && npm run dev` → http://localhost:3000
- **One GPU = one sim.** While the warm session runs, do NOT click Fly/Train/Warm again.

All of beats 1–5 above are **pre-rendered local files browsed in the UI → cannot fail to load.**

## The live beat (optional, only if pre-warmed)
- Live page: AO map with drones over Kramatorsk + live camera feed (warm session on the box).
- If it's not up, skip it — the videos carry the story.

## Honest framing
- Show reconstruction **from altitude/oblique** (how a drone sees it) — credible there.
- Lead with the **capability** (reconstruct + train + fly anywhere), not facade detail.
