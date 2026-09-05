"use client";

import { useCallback, useEffect, useMemo, useRef, useState } from "react";

// Cinematic search-mission visualization for the native warm session.
//
// Tells the drone search-and-find story on a 2D canvas: a swarm sweeps a real
// satellite-imaged AO, a "fog of war" veil clears behind the sensors, the lead
// drone's camera footprint paints the ground it is looking at RIGHT NOW, and
// enemy vehicles stay hidden until a sensor catches them — at which point they
// flash, get LABELED as DETECTED, and later marked NEUTRALIZED. A cinematic
// auto-camera keeps the lead framed with lookahead; the operator can grab to
// pan/zoom and the camera eases back after they let go.
//
// Data path: polls http://<ip>:8180/state every 150 ms (CORS-open on the box);
// geometry + ortho come once from /api/world3d/<world>. World frame is metres,
// +x east / +y north / +z up, origin centre, extent ±half_m — same transform
// math as site-map.tsx. Positions are interpolated between polls on rAF for
// buttery motion. Never crashes: with no telemetry it holds AWAITING TELEMETRY.

const POLL_MS = 150;
const MASK = 1024; // offscreen coverage resolution (world-space, ±half)
const ACCENT = "#0ca30c";
const LINK_BLUE = "#3b8dff"; // confirmed-connectivity layer (distinct from search green)

type SDrone = { x: number; y: number; z: number; q?: number[]; linked?: boolean };
type SVehicle = {
  x: number; y: number; z?: number; hdg?: number; found: boolean; reached: boolean;
  // sighted by the drone but not yet relayed (it was jammed at the time); the
  // operator's map intentionally does NOT show these until the report lands
  pending?: boolean;
};
// mission-accumulated connectivity map over the AO (±half m): one digit per
// cell — 0 unknown, 1 confirmed link, 2 confirmed dead zone; row 0 = south
type SComms = { n: number; half: number; grid: string; denied_frac?: number };
type State = {
  t: number;
  world?: string;
  policy?: string;
  drone0?: { speed: number; vz: number; agl: number };
  found: number;
  reached: number;
  targets: number;
  pending?: number;
  comms?: SComms;
  comms_denied?: number;
  drones: SDrone[];
  vehicles: SVehicle[];
};
type World3D = {
  world: string;
  half_m: number;
  ground: string;
  buildings?: { p: [number, number][]; h: number; z: number }[];
};

// smoothed, render-time drone with a heading we can point the glyph at
type RDrone = { x: number; y: number; hdg: number; agl: number; init: boolean };
// per-target detection bookkeeping so the animation fires exactly once
type Det = { foundAt: number; reachedAt: number };
// lead-drone path point; null marks a gap (sortie rollover / respawn) so the
// trail can persist across episodes without streaking to the new spawn
type TrailPt = [number, number] | null;

// ── module-level persistence ─────────────────────────────────────────────────
// Live is its own route: switching top-nav tabs unmounts this component and
// returning remounts it. Everything the operator perceives as "the mission so
// far" — coverage fog, eased camera, trail, detection bookkeeping, last state —
// lives HERE, keyed by world, so a remount restores instead of resetting.
// Module scope also makes StrictMode's dev double-mount harmless: both mounts
// read/write the same store.
type Persisted = {
  cov: HTMLCanvasElement;
  view: { cx: number; cy: number; zoom: number };
  trail: TrailPt[];
  rDrones: RDrone[];
  det: Map<number, Det>;
  state: State | null;
};
const persistStore = new Map<string, Persisted>();
let lastWorldName: string | null = null;

// one /api/world3d fetch (+ ortho decode) per world for the app's lifetime —
// returning to Live must not re-fetch (that re-fetch was the remount flash).
// The shared promise also dedupes StrictMode's double-mounted first poll.
const worldLoads = new Map<
  string,
  Promise<{ wd: World3D; img: HTMLImageElement | null } | null>
>();
function loadWorld(name: string) {
  let p = worldLoads.get(name);
  if (!p) {
    p = fetch(`/api/world3d/${name}`, { cache: "no-store" })
      .then((r) => (r.ok ? r.json() : null))
      .then((wd: World3D | null) =>
        wd
          ? new Promise<{ wd: World3D; img: HTMLImageElement | null }>((res) => {
              const img = new window.Image();
              img.onload = () => res({ wd, img });
              img.onerror = () => res({ wd, img: null });
              img.src = wd.ground;
            })
          : null,
      )
      .catch(() => {
        worldLoads.delete(name); // transient failure: allow a retry next poll
        return null;
      });
    worldLoads.set(name, p);
  }
  return p;
}

const clamp = (v: number, lo: number, hi: number) => Math.max(lo, Math.min(hi, v));
const lerp = (a: number, b: number, t: number) => a + (b - a) * t;
// shortest-path angle interpolation (radians)
function alerp(a: number, b: number, t: number) {
  let d = ((b - a + Math.PI) % (Math.PI * 2)) - Math.PI;
  if (d < -Math.PI) d += Math.PI * 2;
  return a + d * t;
}

export function TacticalView({ ip }: { ip: string }) {
  const wrapRef = useRef<HTMLDivElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);

  // ── live data (refs so the rAF loop reads the freshest without re-render) ──
  const stateRef = useRef<State | null>(null);
  const worldRef = useRef<World3D | null>(null);
  const imgRef = useRef<HTMLImageElement | null>(null);
  const [world, setWorld] = useState<World3D | null>(null);
  const [imgReady, setImgReady] = useState(false);
  const [hud, setHud] = useState<State | null>(null);
  const [link, setLink] = useState<"wait" | "live" | "lost">("wait");
  const [sortieAt, setSortieAt] = useState(0); // "NEW SORTIE" HUD note

  // ── render state ──
  const rDronesRef = useRef<RDrone[]>([]);
  const trailRef = useRef<TrailPt[]>([]); // lead drone path (null = gap)
  const detRef = useRef<Map<number, Det>>(new Map());
  const covRef = useRef<HTMLCanvasElement | null>(null); // coverage mask (world space)
  const fogRef = useRef<HTMLCanvasElement | null>(null); // per-frame viewport scratch
  // comms layer: rebuilt from the server's revealed grid whenever it changes
  // (server-truth, not client-synthesized like the coverage fog — the field
  // comes from the baked raster, so only the sim knows where the dead zones are)
  const commsRef = useRef<HTMLCanvasElement | null>(null);
  const commsGridRef = useRef<string>("");
  const linkPctRef = useRef(0); // % of AO cells confirmed connected (HUD)
  const sizeRef = useRef({ w: 0, h: 0, dpr: 1 });
  const focusRef = useRef({ x: 0, y: 0, at: -1e9 }); // detection focus pull

  // camera: zoom 1 = whole AO fits; auto follows the lead with lookahead
  const view = useRef({ cx: 0, cy: 0, zoom: 3.3 });
  const manualUntil = useRef(0);
  const drag = useRef({ on: false, x: 0, y: 0, moved: 0 });

  // ── poll /state ──────────────────────────────────────────────────────────
  useEffect(() => {
    if (!ip) return;
    let alive = true;
    let misses = 0;
    const poll = () =>
      fetch(`http://${ip}:8180/state`, { cache: "no-store" })
        .then((r) => (r.ok ? r.json() : null))
        .then((d: State | null) => {
          if (!alive) return;
          if (d && Array.isArray(d.drones) && d.world) {
            misses = 0;
            ingest(d);
            setHud(d);
            setLink("live");
          } else {
            misses++;
            if (misses > 4) setLink((l) => (l === "live" ? "lost" : "wait"));
          }
        })
        .catch(() => {
          if (!alive) return;
          misses++;
          if (misses > 4) setLink((l) => (l === "live" ? "lost" : "wait"));
        });
    poll();
    const id = setInterval(poll, POLL_MS);
    return () => {
      alive = false;
      clearInterval(id);
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [ip]);

  // restore the previous visit's picture immediately on remount (no flash,
  // no AWAITING TELEMETRY blip), and write everything back on unmount.
  // StrictMode's dev double mount/unmount just round-trips the same store.
  useEffect(() => {
    let alive = true;
    if (lastWorldName) {
      const saved = persistStore.get(lastWorldName);
      if (saved?.state) {
        stateRef.current = saved.state;
        setHud(saved.state);
      }
      loadWorld(lastWorldName).then((res) => {
        if (alive && res && !worldRef.current) applyWorld(res);
      });
    }
    return () => {
      alive = false;
      const w = worldRef.current;
      if (w && covRef.current) {
        persistStore.set(w.world, {
          cov: covRef.current,
          view: { ...view.current },
          trail: trailRef.current,
          rDrones: rDronesRef.current,
          det: detRef.current,
          state: stateRef.current,
        });
      }
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // the NEW SORTIE note fades itself out shortly after a rollover
  useEffect(() => {
    if (!sortieAt) return;
    const id = setTimeout(() => setSortieAt(0), 2600);
    return () => clearTimeout(id);
  }, [sortieAt]);

  // adopt a loaded world: restore this world's persisted picture if we have
  // one (route remount), otherwise start the accumulators fresh (world change)
  const applyWorld = (res: { wd: World3D; img: HTMLImageElement | null }) => {
    const wd = res.wd;
    worldRef.current = wd;
    lastWorldName = wd.world;
    setWorld(wd);
    imgRef.current = res.img;
    setImgReady(!!res.img);
    const saved = persistStore.get(wd.world);
    if (saved) {
      covRef.current = saved.cov;
      view.current = { ...saved.view };
      trailRef.current = saved.trail;
      rDronesRef.current = saved.rDrones;
      detRef.current = saved.det;
    } else {
      detRef.current.clear();
      trailRef.current = [];
      rDronesRef.current = [];
      const cov = covRef.current;
      if (cov) cov.getContext("2d")!.clearRect(0, 0, MASK, MASK);
      commsGridRef.current = ""; // next poll rebuilds the comms layer fresh
      commsRef.current = null;
      linkPctRef.current = 0;
    }
  };

  // rebuild the comms overlay canvas from the server's revealed grid. One pixel
  // per cell, alpha baked in; render scales it up with smoothing so the coarse
  // grid reads as soft radio coverage, not blocks.
  const rebuildComms = (cm: SComms) => {
    if (!cm.grid || cm.grid === commsGridRef.current) return;
    commsGridRef.current = cm.grid;
    let c = commsRef.current;
    if (!c || c.width !== cm.n) {
      c = document.createElement("canvas");
      c.width = cm.n;
      c.height = cm.n;
      commsRef.current = c;
    }
    const ctx = c.getContext("2d")!;
    const img = ctx.createImageData(cm.n, cm.n);
    let connected = 0;
    for (let r = 0; r < cm.n; r++) {
      const ir = cm.n - 1 - r; // grid row 0 = south; image row 0 = top (north)
      for (let col = 0; col < cm.n; col++) {
        const v = cm.grid.charCodeAt(r * cm.n + col) - 48;
        const o = (ir * cm.n + col) * 4;
        if (v === 1) {
          // confirmed link: cool blue wash
          img.data[o] = 59; img.data[o + 1] = 141; img.data[o + 2] = 255;
          img.data[o + 3] = 84;
          connected++;
        } else if (v === 2) {
          // confirmed dead zone: faint warm dark — visible, never shouting
          img.data[o] = 224; img.data[o + 1] = 72; img.data[o + 2] = 59;
          img.data[o + 3] = 46;
        }
      }
    }
    ctx.putImageData(img, 0, 0);
    linkPctRef.current = Math.round((100 * connected) / (cm.n * cm.n));
  };

  // an episode ended and the warm session auto-reset the lead (all targets
  // reached, crash, out-of-bounds, flip, or timeout). Keep the accumulated
  // picture — just fade it a notch, re-arm per-target animation state, and
  // note the new sortie instead of hard-resetting the view.
  // NOTE: rollover frequency is a warm-session concern (episode_length_s) —
  // tune it there when dialing in the demo, not here.
  const onRollover = () => {
    detRef.current.clear(); // fresh episode: found/reached edges must re-fire
    const t = trailRef.current;
    if (t.length && t[t.length - 1] !== null) t.push(null); // gap, don't clear
    const cov = covRef.current;
    if (cov) {
      // fade (don't wipe) coverage so the searched AO persists across sorties
      const cg = cov.getContext("2d")!;
      cg.globalCompositeOperation = "destination-out";
      cg.fillStyle = "rgba(0,0,0,0.35)";
      cg.fillRect(0, 0, MASK, MASK);
      cg.globalCompositeOperation = "source-over";
    }
    setSortieAt(performance.now());
  };

  // fold a fresh poll into the render state: fetch geometry on world change,
  // detect episode rollovers, and catch target found/reached rising edges
  const ingest = (d: State) => {
    const prev = stateRef.current;
    stateRef.current = d;
    const w = worldRef.current;
    if (!w || w.world !== d.world) {
      // new (or first) world — geometry + ortho come from the module cache
      const wn = d.world!;
      loadWorld(wn).then((res) => {
        if (res && worldRef.current?.world !== wn && stateRef.current?.world === wn)
          applyWorld(res);
      });
    } else if (prev && prev.world === d.world && d.t < prev.t - 1.0) {
      // mission clock snapped backwards on the same world → episode rollover
      onRollover();
    }
    if (d.comms) rebuildComms(d.comms);
    const now = performance.now();
    d.vehicles.forEach((v, i) => {
      let rec = detRef.current.get(i);
      if (!rec) {
        rec = { foundAt: v.found ? now : 0, reachedAt: v.reached ? now : 0 };
        // if it's already found at first sight, don't animate a stale one hard
        if (v.found) rec.foundAt = now - 3000;
        if (v.reached) rec.reachedAt = now - 3000;
        detRef.current.set(i, rec);
      } else {
        if (v.found && rec.foundAt === 0) {
          rec.foundAt = now;
          focusRef.current = { x: v.x, y: v.y, at: now }; // pull the eye over
        }
        if (v.reached && rec.reachedAt === 0) rec.reachedAt = now;
      }
    });
  };

  // ── world↔screen mapping for the current viewport (mirrors site-map) ──────
  const mapping = useCallback(() => {
    const { w: W, h: H } = sizeRef.current;
    const half = worldRef.current?.half_m ?? 1000;
    const fit = Math.min(W, H) / (2 * half);
    const ppm = fit * view.current.zoom;
    const spanX = W / (2 * ppm), spanY = H / (2 * ppm);
    view.current.cx = spanX >= half ? 0 : clamp(view.current.cx, -half + spanX, half - spanX);
    view.current.cy = spanY >= half ? 0 : clamp(view.current.cy, -half + spanY, half - spanY);
    const { cx, cy } = view.current;
    return {
      W, H, half, ppm,
      toX: (wx: number) => W / 2 + (wx - cx) * ppm,
      toY: (wy: number) => H / 2 + (cy - wy) * ppm,
      toWorld: (sx: number, sy: number): [number, number] => [
        cx + (sx - W / 2) / ppm,
        cy - (sy - H / 2) / ppm,
      ],
    };
  }, []);

  // world (metres) → coverage-mask pixel (north up), shared by stamp + composite
  const toMask = (x: number, y: number, half: number): [number, number] => [
    ((x + half) / (2 * half)) * MASK,
    ((half - y) / (2 * half)) * MASK,
  ];

  // ── main animation loop ────────────────────────────────────────────────────
  useEffect(() => {
    if (!covRef.current) {
      const c = document.createElement("canvas");
      c.width = MASK; c.height = MASK;
      covRef.current = c;
    }
    if (!fogRef.current) fogRef.current = document.createElement("canvas");

    let raf = 0;
    let prev = performance.now();

    const frame = (now: number) => {
      raf = requestAnimationFrame(frame);
      const dt = clamp((now - prev) / 1000, 0, 0.1);
      prev = now;
      step(now, dt);
      render(now);
    };
    raf = requestAnimationFrame(frame);
    return () => cancelAnimationFrame(raf);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // advance interpolated positions, headings, coverage, and the camera
  const step = (now: number, dt: number) => {
    const st = stateRef.current;
    const w = worldRef.current;
    if (!st || !w) return;
    const half = w.half_m;
    const kPos = 1 - Math.exp(-dt * 6); // position smoothing toward latest poll

    const rs = rDronesRef.current;
    st.drones.forEach((d, i) => {
      let r = rs[i];
      if (!r || !r.init) {
        r = { x: d.x, y: d.y, hdg: 0, agl: d.z, init: true };
        rs[i] = r;
      }
      const px = r.x, py = r.y;
      const agl = i === 0 && st.drone0 ? st.drone0.agl : d.z;
      // an episode reset teleports every drone to a fresh spawn. Snap to it
      // rather than gliding the smoother across the whole map (that glide was
      // the "teleporting"), and break the lead's trail so it doesn't streak —
      // a gap marker keeps the previous sortie's path on the map.
      if (Math.hypot(d.x - r.x, d.y - r.y) > 40) {
        r.x = d.x; r.y = d.y; r.agl = agl;
        if (i === 0) {
          const t = trailRef.current;
          if (t.length && t[t.length - 1] !== null) t.push(null);
        }
      } else {
        r.x = lerp(r.x, d.x, kPos);
        r.y = lerp(r.y, d.y, kPos);
        r.agl = lerp(r.agl, agl, kPos);
        const vx = r.x - px, vy = r.y - py;
        if (Math.hypot(vx, vy) > 0.02) {
          // screen angle: north is up (−screen y), east is +screen x
          const target = Math.atan2(-vy, vx);
          r.hdg = alerp(r.hdg, target, 1 - Math.exp(-dt * 5));
        }
      }
    });
    if (rs.length > st.drones.length) rs.length = st.drones.length;

    // lead trail (world coords), decimated
    const lead = rs[0];
    if (lead) {
      const t = trailRef.current;
      const last = t[t.length - 1];
      if (!last || Math.hypot(last[0] - lead.x, last[1] - lead.y) > 1.5) t.push([lead.x, lead.y]);
      if (t.length > 700) t.shift();
    }

    // stamp coverage: soft disc under every sensor, ahead-and-below the camera
    const cov = covRef.current!;
    const cg = cov.getContext("2d")!;
    rs.forEach((r) => {
      const agl = r.agl;
      const radM = clamp(agl * 0.8, 18, 120);
      // camera pitches forward-down: footprint centre sits ahead of the drone
      const dirX = Math.cos(r.hdg), dirY = -Math.sin(r.hdg); // back to world dir
      const fx = r.x + dirX * agl * 0.6;
      const fy = r.y + dirY * agl * 0.6;
      const [mx, my] = toMask(fx, fy, half);
      const rp = (radM / (2 * half)) * MASK;
      const grad = cg.createRadialGradient(mx, my, 0, mx, my, rp);
      grad.addColorStop(0, "rgba(255,255,255,0.55)");
      grad.addColorStop(0.7, "rgba(255,255,255,0.28)");
      grad.addColorStop(1, "rgba(255,255,255,0)");
      cg.fillStyle = grad;
      cg.beginPath();
      cg.arc(mx, my, rp, 0, 7);
      cg.fill();
    });

    // ── cinematic camera ─────────────────────────────────────────────────
    if (now > manualUntil.current && lead) {
      const dirX = Math.cos(lead.hdg), dirY = -Math.sin(lead.hdg);
      let tx = lead.x + dirX * 45; // lookahead ahead of travel
      let ty = lead.y + dirY * 45;
      let tz = 3.3;
      // a fresh detection pulls the eye toward the target for ~3 s
      const f = focusRef.current;
      const age = (now - f.at) / 1000;
      if (age >= 0 && age < 3.2) {
        const pull = Math.sin(clamp(age / 3.2, 0, 1) * Math.PI) * 0.6;
        tx = lerp(tx, (lead.x + f.x) / 2, pull);
        ty = lerp(ty, (lead.y + f.y) / 2, pull);
        tz = lerp(3.3, 4.3, pull);
      }
      const k = 1 - Math.exp(-dt * 1.6); // slow, filmic ease
      view.current.cx = lerp(view.current.cx, tx, k);
      view.current.cy = lerp(view.current.cy, ty, k);
      view.current.zoom = lerp(view.current.zoom, tz, k);
    }
  };

  // paint one frame
  const render = (now: number) => {
    const cv = canvasRef.current;
    if (!cv) return;
    const { w: W, h: H, dpr } = sizeRef.current;
    if (W === 0 || H === 0) return;
    const g = cv.getContext("2d")!;
    g.setTransform(dpr, 0, 0, dpr, 0, 0);
    // base clear (dark tactical ground) — never a black void
    g.fillStyle = "#070b0e";
    g.fillRect(0, 0, W, H);

    const w = worldRef.current;
    const img = imgRef.current;
    if (!w) return;
    const m = mapping();
    const size = 2 * m.half * m.ppm;
    const ox = m.toX(-m.half), oy = m.toY(m.half);

    // 1 ── satellite ortho, desaturated + darkened, blue-grey tactical tint
    if (img) {
      g.save();
      g.beginPath();
      g.rect(0, 0, W, H);
      g.clip();
      try {
        g.filter = "saturate(0.55) brightness(0.62) contrast(1.05)";
      } catch {}
      g.drawImage(img, ox, oy, size, size);
      g.filter = "none";
      g.restore();
      g.fillStyle = "rgba(18,32,44,0.34)"; // cool tint so bright UI reads on top
      g.fillRect(0, 0, W, H);
    }

    // building footprints for depth/context (crisp thin outlines)
    if (w.buildings && m.ppm > 0.02) {
      g.strokeStyle = "rgba(150,180,200,0.16)";
      g.lineWidth = 1;
      for (const b of w.buildings) {
        if (!b.p || b.p.length < 3) continue;
        g.beginPath();
        g.moveTo(m.toX(b.p[0][0]), m.toY(b.p[0][1]));
        for (let i = 1; i < b.p.length; i++) g.lineTo(m.toX(b.p[i][0]), m.toY(b.p[i][1]));
        g.closePath();
        g.stroke();
      }
    }

    // 2 ── fog of war: veil everywhere, cleared where the swarm has looked
    const cov = covRef.current;
    const fog = fogRef.current;
    if (cov && fog) {
      if (fog.width !== Math.round(W) || fog.height !== Math.round(H)) {
        fog.width = Math.max(1, Math.round(W));
        fog.height = Math.max(1, Math.round(H));
      }
      const fg = fog.getContext("2d")!;
      fg.setTransform(1, 0, 0, 1, 0, 0);
      fg.clearRect(0, 0, fog.width, fog.height);
      // dark tactical veil over the whole frame
      fg.fillStyle = "rgba(4,9,12,0.62)";
      fg.fillRect(0, 0, fog.width, fog.height);
      // punch the searched area out of the veil
      fg.globalCompositeOperation = "destination-out";
      fg.drawImage(cov, ox, oy, size, size);
      fg.globalCompositeOperation = "source-over";
      g.drawImage(fog, 0, 0, W, H);

      // faint green "searched" wash + scanlines, clipped to the cleared area
      fg.globalCompositeOperation = "source-over";
      fg.clearRect(0, 0, fog.width, fog.height);
      fg.fillStyle = "rgba(12,163,12,0.10)";
      fg.fillRect(0, 0, fog.width, fog.height);
      fg.strokeStyle = "rgba(12,163,12,0.13)";
      fg.lineWidth = 1;
      for (let y = 0; y < fog.height; y += 4) {
        fg.beginPath();
        fg.moveTo(0, y + 0.5);
        fg.lineTo(fog.width, y + 0.5);
        fg.stroke();
      }
      fg.globalCompositeOperation = "destination-in";
      fg.drawImage(cov, ox, oy, size, size);
      fg.globalCompositeOperation = "source-over";
      g.drawImage(fog, 0, 0, W, H);
    }

    const st = stateRef.current;
    const rs = rDronesRef.current;

    // 2.5 ── RF connectivity: blue where the swarm has CONFIRMED a link to the
    // base station, faint red where it has confirmed a dead zone. Server truth
    // (the baked comms raster), revealed as the drones fly — composites over
    // the fog so confirmed radio coverage reads even in unsearched veil.
    const cc = commsRef.current;
    const cm = st?.comms;
    if (cc && cm) {
      const ch = cm.half;
      g.save();
      g.imageSmoothingEnabled = true;
      g.drawImage(cc, m.toX(-ch), m.toY(ch), 2 * ch * m.ppm, 2 * ch * m.ppm);
      g.restore();
    }

    // 3 ── lead sensor beam + footprint (where it's looking RIGHT NOW)
    const lead = rs[0];
    if (lead) {
      const radM = clamp(lead.agl * 0.8, 18, 120);
      const dirX = Math.cos(lead.hdg), dirY = -Math.sin(lead.hdg);
      const fcx = lead.x + dirX * lead.agl * 0.6;
      const fcy = lead.y + dirY * lead.agl * 0.6;
      const ds = { x: m.toX(lead.x), y: m.toY(lead.y) };
      const fs = { x: m.toX(fcx), y: m.toY(fcy) };
      const rpx = radM * m.ppm;
      // screen-space heading + perpendicular for the wedge edges
      const sa = Math.atan2(fs.y - ds.y, fs.x - ds.x);
      const px = Math.cos(sa + Math.PI / 2), py = Math.sin(sa + Math.PI / 2);
      const lft = { x: fs.x + px * rpx, y: fs.y + py * rpx };
      const rgt = { x: fs.x - px * rpx, y: fs.y - py * rpx };
      // beam
      const beam = g.createLinearGradient(ds.x, ds.y, fs.x, fs.y);
      beam.addColorStop(0, "rgba(12,163,12,0)");
      beam.addColorStop(1, "rgba(12,163,12,0.20)");
      g.fillStyle = beam;
      g.beginPath();
      g.moveTo(ds.x, ds.y);
      g.lineTo(lft.x, lft.y);
      g.lineTo(rgt.x, rgt.y);
      g.closePath();
      g.fill();
      // footprint disc + sweeping leading edge
      g.save();
      g.strokeStyle = "rgba(12,163,12,0.55)";
      g.lineWidth = 1.4;
      g.beginPath();
      g.arc(fs.x, fs.y, rpx, 0, 7);
      g.stroke();
      const pulse = 0.5 + 0.5 * Math.sin(now / 260);
      g.strokeStyle = `rgba(120,255,120,${0.35 + 0.4 * pulse})`;
      g.lineWidth = 2;
      g.beginPath();
      g.arc(fs.x, fs.y, rpx, sa - 0.7, sa + 0.7);
      g.stroke();
      g.restore();
    }

    // 3.5 ── lead trail (dark halo + accent), split at sortie gaps; segments
    // from earlier sorties stay on the map but render faded
    const tr = trailRef.current;
    if (tr.length > 1) {
      const segs: [number, number][][] = [[]];
      for (const p of tr) {
        if (p === null) segs.push([]);
        else segs[segs.length - 1].push(p);
      }
      segs.forEach((seg, si) => {
        if (seg.length < 2) return;
        const old = si < segs.length - 1;
        for (const pass of [
          { style: old ? "rgba(0,0,0,0.25)" : "rgba(0,0,0,0.5)", width: 4 },
          { style: old ? "rgba(12,163,12,0.28)" : "rgba(12,163,12,0.7)", width: 1.6 },
        ]) {
          g.strokeStyle = pass.style;
          g.lineWidth = pass.width;
          g.lineJoin = "round";
          g.beginPath();
          g.moveTo(m.toX(seg[0][0]), m.toY(seg[0][1]));
          for (let i = 1; i < seg.length; i++) g.lineTo(m.toX(seg[i][0]), m.toY(seg[i][1]));
          g.stroke();
        }
      });
    }

    // 4 ── drones: lead as a glowing aircraft glyph, swarm as dim markers
    rs.forEach((r, i) => {
      const sx = m.toX(r.x), sy = m.toY(r.y);
      // r.hdg is already the screen-space travel angle (atan2(-vy, vx), north
      // up); the glyph's local +x points along it, so rotate by +hdg, not -hdg.
      const sa = r.hdg;
      g.save();
      g.translate(sx, sy);
      g.rotate(sa);
      if (i === 0) {
        g.shadowColor = "rgba(12,163,12,0.9)";
        g.shadowBlur = 14;
        g.fillStyle = ACCENT;
        g.strokeStyle = "#dfffdf";
        g.lineWidth = 1.2;
        g.beginPath(); // forward chevron
        g.moveTo(10, 0);
        g.lineTo(-6, -6);
        g.lineTo(-3, 0);
        g.lineTo(-6, 6);
        g.closePath();
        g.fill();
        g.shadowBlur = 0;
        g.strokeStyle = "rgba(255,255,255,0.85)";
        g.stroke();
      } else {
        g.fillStyle = "rgba(150,210,235,0.7)";
        g.beginPath();
        g.moveTo(5, 0);
        g.lineTo(-3.5, -3);
        g.lineTo(-3.5, 3);
        g.closePath();
        g.fill();
      }
      g.restore();
      // link lost: amber ring + slash so a drone in a dead zone reads at a glance
      if (st?.drones[i]?.linked === false) {
        g.strokeStyle = "rgba(255,158,68,0.9)";
        g.lineWidth = 1.4;
        g.beginPath();
        g.arc(sx, sy, 10, 0, 7);
        g.stroke();
        g.beginPath();
        g.moveTo(sx - 7, sy + 7);
        g.lineTo(sx + 7, sy - 7);
        g.stroke();
      }
    });

    // 5 ── targets + detection animation
    if (st) {
      st.vehicles.forEach((v, i) => {
        const rec = detRef.current.get(i);
        const sx = m.toX(v.x), sy = m.toY(v.y);
        if (!v.found && !v.reached) {
          // undiscovered: barely-there ghost only (the drone hasn't found it)
          g.fillStyle = "rgba(255,255,255,0.06)";
          g.beginPath();
          g.arc(sx, sy, 3, 0, 7);
          g.fill();
          return;
        }
        const reached = v.reached;
        const col = reached ? "#e0483b" : ACCENT;
        const foundAge = rec ? (now - rec.foundAt) / 1000 : 99;

        // expanding detection ring on the found rising edge
        if (foundAge >= 0 && foundAge < 1.2) {
          const p = foundAge / 1.2;
          g.strokeStyle = `rgba(12,163,12,${0.9 * (1 - p)})`;
          g.lineWidth = 2;
          g.beginPath();
          g.arc(sx, sy, 6 + p * 40, 0, 7);
          g.stroke();
        }

        // marker — diamond, struck when neutralized
        g.save();
        g.translate(sx, sy);
        g.rotate(Math.PI / 4);
        g.fillStyle = "rgba(0,0,0,0.55)";
        g.fillRect(-5, -5, 10, 10);
        g.strokeStyle = col;
        g.lineWidth = 1.8;
        g.strokeRect(-5, -5, 10, 10);
        g.restore();
        if (reached) {
          g.strokeStyle = col;
          g.lineWidth = 1.6;
          g.beginPath();
          g.moveTo(sx - 7, sy - 7); g.lineTo(sx + 7, sy + 7);
          g.moveTo(sx + 7, sy - 7); g.lineTo(sx - 7, sy + 7);
          g.stroke();
        }

        // label box with leader line
        const id = `TARGET-${String(i + 1).padStart(2, "0")}`;
        const status = reached ? "NEUTRALIZED" : "DETECTED";
        const l1 = `◈ ${id} · TANK`;
        const appear = clamp(foundAge / 0.35, 0, 1); // fade/slide in
        g.save();
        g.globalAlpha = appear;
        const lx = sx + 18, ly = sy - 34 - (1 - appear) * 6;
        g.strokeStyle = reached ? "rgba(224,72,59,0.6)" : "rgba(12,163,12,0.6)";
        g.lineWidth = 1;
        g.beginPath();
        g.moveTo(sx, sy);
        g.lineTo(lx - 6, ly + 22);
        g.lineTo(lx, ly + 22);
        g.stroke();
        g.font = "600 11px ui-monospace, SFMono-Regular, monospace";
        const w1 = g.measureText(l1).width;
        g.font = "10px ui-monospace, monospace";
        const w2 = g.measureText(status).width;
        const bw = Math.max(w1, w2) + 14;
        g.fillStyle = "rgba(6,12,15,0.82)";
        g.fillRect(lx, ly - 2, bw, 26);
        g.fillStyle = col;
        g.fillRect(lx, ly - 2, 2, 26); // accent spine
        g.fillStyle = "#eaf5ea";
        g.font = "600 11px ui-monospace, SFMono-Regular, monospace";
        g.fillText(l1, lx + 7, ly + 8);
        g.fillStyle = col;
        g.font = "10px ui-monospace, monospace";
        g.fillText(status, lx + 7, ly + 20);
        g.restore();
      });
    }

    // 6 ── cinematic frame: corner ticks + centre reticle
    drawFrame(g, W, H, now > manualUntil.current);
  };

  const drawFrame = (g: CanvasRenderingContext2D, W: number, H: number, auto: boolean) => {
    const pad = 16, len = 22;
    g.strokeStyle = "rgba(180,210,225,0.5)";
    g.lineWidth = 1.4;
    const corner = (x: number, y: number, dx: number, dy: number) => {
      g.beginPath();
      g.moveTo(x, y + dy * len); g.lineTo(x, y); g.lineTo(x + dx * len, y);
      g.stroke();
    };
    corner(pad, pad, 1, 1);
    corner(W - pad, pad, -1, 1);
    corner(pad, H - pad, 1, -1);
    corner(W - pad, H - pad, -1, -1);
    // centre reticle only while the camera is driving (auto)
    if (auto) {
      const cx = W / 2, cy = H / 2, r = 26;
      g.strokeStyle = "rgba(12,163,12,0.35)";
      g.lineWidth = 1;
      g.beginPath();
      g.arc(cx, cy, r, 0, 7);
      g.stroke();
      g.beginPath();
      for (const a of [0, Math.PI / 2, Math.PI, (3 * Math.PI) / 2]) {
        g.moveTo(cx + Math.cos(a) * (r - 6), cy + Math.sin(a) * (r - 6));
        g.lineTo(cx + Math.cos(a) * (r + 6), cy + Math.sin(a) * (r + 6));
      }
      g.stroke();
    }
  };

  // ── sizing (dpr-aware) ─────────────────────────────────────────────────────
  useEffect(() => {
    const wrap = wrapRef.current, cv = canvasRef.current;
    if (!wrap || !cv) return;
    const resize = () => {
      const r = wrap.getBoundingClientRect();
      const dpr = window.devicePixelRatio || 1;
      sizeRef.current = { w: r.width, h: r.height, dpr };
      cv.width = Math.max(1, Math.round(r.width * dpr));
      cv.height = Math.max(1, Math.round(r.height * dpr));
    };
    resize();
    const ro = new ResizeObserver(resize);
    ro.observe(wrap);
    return () => ro.disconnect();
  }, []);

  // ── operator override: drag to pan, wheel to zoom, camera eases back ───────
  useEffect(() => {
    const cv = canvasRef.current;
    if (!cv) return;
    const onWheel = (e: WheelEvent) => {
      e.preventDefault();
      const r = cv.getBoundingClientRect();
      const m = mapping();
      const [wx, wy] = m.toWorld(e.clientX - r.left, e.clientY - r.top);
      const v = view.current;
      const nz = clamp(v.zoom * Math.exp(-e.deltaY * 0.0015), 1, 14);
      const s = v.zoom / nz;
      v.cx = wx - (wx - v.cx) * s;
      v.cy = wy - (wy - v.cy) * s;
      v.zoom = nz;
      manualUntil.current = performance.now() + 4500;
    };
    cv.addEventListener("wheel", onWheel, { passive: false });
    return () => cv.removeEventListener("wheel", onWheel);
  }, [mapping]);

  const resetView = () => {
    manualUntil.current = 0; // hand control back to the auto-camera immediately
  };

  const world3dReady = !!world;

  return (
    <div className="relative aspect-video w-full overflow-hidden bg-[#070b0e]">
      <div ref={wrapRef} className="absolute inset-0">
        <canvas
          ref={canvasRef}
          className="absolute inset-0 h-full w-full touch-none"
          style={{ cursor: drag.current.on ? "grabbing" : "grab" }}
          onPointerDown={(e) => {
            drag.current = { on: true, x: e.clientX, y: e.clientY, moved: 0 };
            (e.target as HTMLElement).setPointerCapture(e.pointerId);
          }}
          onPointerMove={(e) => {
            if (!drag.current.on) return;
            const m = mapping();
            const dx = e.clientX - drag.current.x, dy = e.clientY - drag.current.y;
            drag.current.moved += Math.abs(dx) + Math.abs(dy);
            view.current.cx -= dx / m.ppm;
            view.current.cy += dy / m.ppm;
            drag.current.x = e.clientX;
            drag.current.y = e.clientY;
            manualUntil.current = performance.now() + 4500;
          }}
          onPointerUp={() => {
            drag.current.on = false;
          }}
          onPointerLeave={() => {
            drag.current.on = false;
          }}
        />
      </div>

      {/* ── HUD overlay: premium, minimal, monospace ── */}
      <div className="pointer-events-none absolute inset-0 select-none font-mono">
        {/* top-left: mission identity */}
        <div className="absolute left-6 top-6 flex flex-col gap-1">
          <div className="flex items-center gap-2 text-[11px] font-semibold uppercase tracking-[0.24em] text-white/90">
            <span
              className="inline-block h-1.5 w-1.5 rounded-full"
              style={{
                background: link === "live" ? ACCENT : link === "lost" ? "#e0483b" : "#8a949c",
                boxShadow: link === "live" ? `0 0 8px ${ACCENT}` : undefined,
              }}
            />
            VESPER · TACTICAL
          </div>
          {hud && (
            <div className="text-[9.5px] uppercase tracking-[0.18em] text-white/45">
              {hud.world} · AO {(worldRef.current?.half_m ?? 1000) * 2} M · N↑
            </div>
          )}
        </div>

        {/* top-centre: brief note when the session rolls into a new episode */}
        {sortieAt > 0 && (
          <div className="absolute left-1/2 top-6 -translate-x-1/2 border border-white/15 bg-black/60 px-3 py-1 text-[10px] font-semibold uppercase tracking-[0.3em] text-white/80">
            ▶ new sortie
          </div>
        )}

        {/* top-right: mission clock + policy */}
        {hud && (
          <div className="absolute right-6 top-6 flex flex-col items-end gap-1">
            <div className="text-[22px] font-semibold tabular-nums leading-none text-white/90">
              T+{hud.t.toFixed(1)}
              <span className="ml-1 text-[10px] font-normal text-white/40">S</span>
            </div>
            <div className="text-[9.5px] uppercase tracking-[0.18em] text-white/45">
              policy · {hud.policy ?? "—"}
            </div>
          </div>
        )}

        {/* bottom-left: search status + tallies */}
        {hud ? (
          <div className="absolute bottom-6 left-6 flex flex-col gap-2">
            <div
              className="flex items-center gap-2 text-[11px] uppercase tracking-[0.22em]"
              style={{ color: hud.reached >= hud.targets && hud.targets > 0 ? ACCENT : "#e8f2ea" }}
            >
              <span className="relative flex h-2 w-2">
                <span
                  className="absolute inline-flex h-full w-full animate-ping rounded-full opacity-70"
                  style={{ background: ACCENT }}
                />
                <span className="relative inline-flex h-2 w-2 rounded-full" style={{ background: ACCENT }} />
              </span>
              {hud.reached >= hud.targets && hud.targets > 0 ? "AO CLEARED" : "SEARCHING"}
            </div>
            {hud.drones[0]?.linked === false && (
              <div className="text-[10px] uppercase tracking-[0.2em] text-amber-400/90">
                ⚠ lead jammed · reports held
              </div>
            )}
            <div className="flex gap-5">
              <Stat label="FOUND" value={`${hud.found}/${hud.targets}`} accent />
              <Stat label="NEUTRALIZED" value={`${hud.reached}/${hud.targets}`} />
              <Stat label="ASSETS" value={`${hud.drones.length}`} />
              {hud.comms && (
                <Stat label="LINK" value={`${linkPctRef.current}% AO`} color={LINK_BLUE} />
              )}
            </div>
          </div>
        ) : (
          <div className="absolute inset-0 flex flex-col items-center justify-center gap-3">
            <div className="text-sm uppercase tracking-[0.4em] text-white/60">
              AWAITING TELEMETRY
            </div>
            <div className="text-[10px] uppercase tracking-[0.2em] text-white/30">
              link · {ip}:8180
            </div>
          </div>
        )}

        {/* bottom-right: lead telemetry */}
        {hud?.drone0 && (
          <div className="absolute bottom-6 right-6 flex gap-5">
            <Stat label="LEAD SPD" value={`${hud.drone0.speed.toFixed(1)} m/s`} />
            <Stat label="AGL" value={`${hud.drone0.agl.toFixed(0)} m`} />
            <Stat label="VZ" value={`${hud.drone0.vz >= 0 ? "+" : ""}${hud.drone0.vz.toFixed(1)}`} />
          </div>
        )}

        {/* recenter affordance (only meaningful once framed) */}
        {world3dReady && (
          <button
            onClick={resetView}
            className="pointer-events-auto absolute right-6 top-1/2 -translate-y-1/2 border border-white/15 bg-black/50 px-2 py-1 text-[9px] uppercase tracking-[0.16em] text-white/55 transition-colors hover:border-white/40 hover:text-white/90"
            title="resume auto-camera"
          >
            ⟳ recenter
          </button>
        )}
      </div>

      {(!imgReady || !world3dReady) && link !== "wait" && (
        <div className="pointer-events-none absolute bottom-6 left-1/2 -translate-x-1/2 font-mono text-[9px] uppercase tracking-[0.2em] text-white/30">
          loading ortho…
        </div>
      )}
    </div>
  );
}

function Stat({
  label,
  value,
  accent,
  color,
}: {
  label: string;
  value: string;
  accent?: boolean;
  color?: string;
}) {
  return (
    <div className="flex flex-col gap-0.5">
      <span className="text-[8.5px] uppercase tracking-[0.18em] text-white/35">{label}</span>
      <span
        className="text-[14px] font-semibold tabular-nums leading-none"
        style={{ color: color ?? (accent ? ACCENT : "rgba(234,245,234,0.92)") }}
      >
        {value}
      </span>
    </div>
  );
}
