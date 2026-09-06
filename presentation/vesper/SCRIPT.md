# Vesper — 3-minute pitch script

One block per slide. Advance on the **bold cue**. Target times add to ~3:00 with a ~50s live demo.

---

## Slide 1 — Hook (the fiber photo) · ~0:25

We built **Vesper** — an autonomous-drone training platform.

We were inspired by the war in Ukraine, where drones have become the king of the battlefield — but every drone is chained to its ability to talk to an operator. The jamming is so intense that both sides have turned to primitive solutions: **this image** — they literally fly drones over spools of fiber-optic cable, and those cables now litter the battlefield.

Jam the link and an FPV is a falling brick. So they fly on twenty kilometers of fiber — a leash. **You can't scale a leash.**

> **Advance** on "you can't scale a leash."

---

## Slide 2 — The thesis · ~0:25

To solve this, we build purpose-driven models — small and performant enough to fly the drone itself, autonomously, on the edge.

That by itself is not a new idea. Edge autonomy isn't new. **It fails** — because those models are trained in generic simulators that look nothing like where the drone will actually fly.

**Ours train in the fight.** Training on the real battlefield is the new part.

> **Advance** on "train in the fight."

---

## Slide 3 — The pipeline · ~0:30

Three steps got this done. *(walk the strip left to right)*

**First, we reconstruct the battlefield.** Point Vesper at any coordinates on Earth and it builds a digital twin from satellite imagery and elevation data — real terrain, real buildings. This background isn't a photo; it's our sim. The operator picks a launch point and can mark friendly zones the drones will never attack.

Then we train in that twin, model the RF environment, and deploy — steps two and three, coming up.

> **Advance** after touching the last node.

---

## Slide 4 — Training · ~0:25

**Second, we run millions of simulations in the twin** — search, detect, strike — far more flight data than any real fleet could ever log.

Reinforcement learning defines the behavior an operator wants, and the model improves with every trajectory. Iteration zero wanders. By the final policy it's killing inside twelve seconds. Nobody scripted any of that — it's learned.

> **Advance** on "it's learned."

---

## Slide 5 —