# Vesper UI — Live page layout reorganization

Scope: **layout and information hierarchy of the Live page only** (`web/client/app/page.tsx`).
Do not change the data sources, the backend, or the other routes (`/runs`,
`/models`, `/environments`). Reuse the existing components; move and resize them,
don't rewrite them. Keep the dark mission-control look already in `globals.css`.

## Read first

- `web/client/app/page.tsx` — the page you are reorganizing.
- These components it composes (leave their internals alone unless a prop is
  genuinely missing):
  - `components/drone-feeds.tsx` — **the live drone camera feed** (MJPEG from the
    box, port 8180). Big primary camera + thumbnail strip; self-serve "FLY
    MISSION" standby when nothing is in the air. This is the star of the page.
  - `components/live-viewport.tsx` — the Isaac **editor** viewport (WebRTC, free-fly
    camera for inspecting the world). Secondary/opt-in — NOT the drone. Today it's
    a header toggle on the same panel as the feed.
  - `components/site-map.tsx` — top-down AO map (world frame, N up).
  - `components/job-controls.tsx` — `JobsPanel` (jobs running on the box, with
    STOP + PULL ARTIFACTS) and `JobButton`.
- `components/topbar.tsx` — global nav (LIVE / RUNS / MODELS / ENVIRONMENTS).
  Don't duplicate nav on the page.

## The problem

The current Live page crams the camera feed, the Isaac viewport toggle, the AO
map, "Latest sortie," and the Jobs panel into an ad-hoc grid. Hierarchy is flat —
nothing signals what the operator looks at first, second, glance-only. The
feed/viewport share one panel, which reads as one confusing thing.

## What the operator actually needs (hierarchy to encode)

1. **Watch** — the live drone feed. Primary, largest, top-left, always the
   biggest element when a drone is flying.
2. **Situate** — the AO map: where the drone(s) are over the site. Second focus,
   beside the feed.
3. **Control / status** — Jobs on the box (what's running, launch, stop). A rail,
   not a hero.
4. **Reference** — Latest sortie / recent runs. Small, a jump-off to `/runs`.

The Isaac editor viewport is a *tool*, not part of the operational loop — keep it
one click away (a toggle or a small "inspect world" affordance), never competing
with the feed for space.

## Target layout (a starting point — you may improve it, justify if you deviate)

Two-column operational layout on desktop, single column stacked on mobile:

```
┌──────────────────────────────┬─────────────────────┐
│  LIVE DOWNLINK (drone feed)   │  AO MAP             │
│  big primary camera           │  (tall)             │
│  + thumbnail strip            │                     │
│                               ├─────────────────────┤
│  [inspect world ⧉ toggle]     │  JOBS ON THE BOX    │
│                               │  (status + launch)  │
├──────────────────────────────┤                     │
│  LATEST SORTIE (compact)      │                     │
└──────────────────────────────┴─────────────────────┘
```

Left column ≈ 60% width and led by the feed; right column ≈ 40%, map over jobs.
Feed keeps 16:9. Everything above the fold on a 1440×900 screen without inner
scroll except the Jobs list.

## Rules

- Consistent `Panel` chrome (the existing header style) on every box; one visual
  system, aligned gaps (the current `gap-3` grid is fine).
- Empty/standby and offline states must look intentional, not broken — reuse the
  patterns already in `drone-feeds.tsx`.
- No new dependencies. No backend changes. `npm run build` and
  `npx eslint app/page.tsx` must pass clean (no unused imports, no
  setState-in-effect).
- Responsive: the two columns collapse to one on `< lg`, feed first.

## Done when

`npm run build` is clean, and a screenshot of the Live page at 1440×900 shows the
drone feed as the clear primary, the AO map beside it, jobs as a side rail, and
the Isaac viewport reachable but not occupying primary space. Commit to the
`web-client` branch; do not touch `main`.
