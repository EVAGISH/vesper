"use client";

// Offline replay renderer — the headless-capture side of the three.js render
// lane. Builds the SAME scene as the live view (lib/scene3d.ts) from a run's
// replay.json, then steps through it deterministically: no RAF, no wall clock.
// The export driver (scripts/export_replay.mjs) navigates here headless, calls
// window.__vesper.seek(i) frame by frame and pipes the returned JPEG data URLs
// into ffmpeg. Also usable in a normal browser for spot checks:
//
//   /render?run=<id>&cam=world|fpv&fps=24&w=1280&h=720[&full=1]
//
// Data comes from /__render/replay.json + /__render/world.json when the driver
// intercepts those routes, else falls back to the live API (/media, /api).

import { useEffect, useRef, useState } from "react";
import * as THREE from "three";
import {
  buildWorldScene, CAM_PITCH_RAD, droneModel, tankModel, w2t, type World3D,
} from "@/lib/scene3d";

type Replay = {
  world: string;
  half_m: number;
  dt: number;
  targets: number;
  frames: {
    t: number;
    d: number[][];          // [drone][x,y,z]
    hdg: number[];          // [drone] world yaw
    tg: number[][];         // [target][x,y,found,reached]
    agl: number;
    dead?: number[];        // drone indices expended by their strike
  }[];
};

type Kill = { k: number; t: number; striker: number };

declare global {
  interface Window {
    __vesper?: {
      ready: boolean;
      error: string | null;
      meta: { frames: number; fps: number; hero: number; kills: Kill[]; duration: number };
      seek: (i: number) => string;
    };
  }
}

const LERP_ANGLE = (a: number, b: number, f: number) => {
  let d = b - a;
  while (d > Math.PI) d -= 2 * Math.PI;
  while (d < -Math.PI) d += 2 * Math.PI;
  return a + d * f;
};

async function fetchFirst<T>(urls: string[]): Promise<T> {
  for (const u of urls) {
    try {
      const r = await fetch(u, { cache: "no-store" });
      if (r.ok) return (await r.json()) as T;
    } catch { /* next */ }
  }
  throw new Error(`no source answered: ${urls.join(", ")}`);
}

/** Kill events: each `reached` 0->1 flip, striker = the drone that died with it. */
function findKills(rep: Replay): Kill[] {
  const kills: Kill[] = [];
  const t0 = rep.frames[0].t;
  let prevDead = new Set<number>(rep.frames[0].dead ?? []);
  const prevReached = rep.frames[0].tg.map((t) => t[3]);
  for (let j = 1; j < rep.frames.length; j++) {
    const fr = rep.frames[j];
    const nowDead = new Set<number>(fr.dead ?? []);
    for (let k = 0; k < fr.tg.length; k++) {
      if (fr.tg[k][3] && !prevReached[k]) {
        const fresh = [...nowDead].filter((d) => !prevDead.has(d));
        let striker = fresh.length ? fresh[0] : -1;
        if (striker < 0) {
          // no expend logged (e.g. non-munition eval): nearest drone at impact
          let best = Infinity;
          fr.d.forEach((p, di) => {
            const dd = (p[0] - fr.tg[k][0]) ** 2 + (p[1] - fr.tg[k][1]) ** 2;
            if (dd < best) { best = dd; striker = di; }
          });
        }
        kills.push({ k, t: fr.t - t0, striker });
      }
      prevReached[k] = fr.tg[k][3];
    }
    prevDead = nowDead;
  }
  return kills;
}

export default function RenderPage() {
  const holder = useRef<HTMLDivElement>(null);
  const [status, setStatus] = useState("loading replay…");

  useEffect(() => {
    const el = holder.current;
    if (!el) return;
    let disposed = false;
    let renderer: THREE.WebGLRenderer | null = null;

    (async () => {
      const qs = new URLSearchParams(window.location.search);
      const run = qs.get("run") ?? "";
      const cam = (qs.get("cam") ?? "world") as "world" | "fpv";
      const fps = Number(qs.get("fps") ?? 24);
      const W = Number(qs.get("w") ?? 1280);
      const H = Number(qs.get("h") ?? 720);
      const full = qs.get("full") === "1";

      const rep = await fetchFirst<Replay>([
        "/__render/replay.json",
        ...(run ? [`/media/${run}/replay.json`] : []),
      ]);
      const wd = await fetchFirst<World3D>([
        "/__render/world.json",
        `/api/world3d/${rep.world}`,
      ]);

      renderer = new THREE.WebGLRenderer({ antialias: true, preserveDrawingBuffer: true });
      renderer.shadowMap.enabled = true;
      renderer.shadowMap.type = THREE.PCFSoftShadowMap;
      renderer.setPixelRatio(1);
      renderer.setSize(W, H);
      el.appendChild(renderer.domElement);
      const scene = new THREE.Scene();
      const camera = new THREE.PerspectiveCamera(62, W / H, 0.5, 4000);

      setStatus("building world…");
      const groundAt = await buildWorldScene(scene, wd);
      if (disposed) return;

      // wait for the terrain ortho to finish decoding so frame 0 is textured
      // (onLoad if the manager still has fetches in flight, else the timeout)
      await new Promise<void>((res) => {
        let done = false;
        const finish = () => { if (!done) { done = true; res(); } };
        THREE.DefaultLoadingManager.onLoad = finish;
        setTimeout(finish, 3500);
      });

      // ── actors ──
      const F = rep.frames;
      const t0 = F[0].t;
      const nDrones = F[0].d.length;
      const drones = Array.from({ length: nDrones }, () => droneModel());
      drones.forEach((d) => scene.add(d));
      const nT = rep.targets;
      const tanks = Array.from({ length: nT }, () => tankModel());
      const tankMats: THREE.MeshStandardMaterial[][] = tanks.map((tk) => {
        const ms: THREE.MeshStandardMaterial[] = [];
        tk.traverse((o) => {
          const m = (o as THREE.Mesh).material as THREE.MeshStandardMaterial | undefined;
          if (m && !ms.includes(m)) ms.push(m);
        });
        return ms;
      });
      const tankBase = tankMats.map((ms) => ms.map((m) => m.color.clone()));
      tanks.forEach((tk) => scene.add(tk));
      const tankHdg = new Array(nT).fill(0);

      // strike pyro: a fireball + rising smoke ball per kill, driven by clip time
      const boomM = new THREE.MeshBasicMaterial({
        color: 0xff8a2a, transparent: true, opacity: 0.9, fog: false,
      });
      const smokeM = new THREE.MeshBasicMaterial({
        color: 0x22201d, transparent: true, opacity: 0.65,
      });
      const boomG = new THREE.IcosahedronGeometry(1, 1);
      const booms = Array.from({ length: nT }, () => {
        const fire = new THREE.Mesh(boomG, boomM.clone());
        const smoke = new THREE.Mesh(boomG, smokeM.clone());
        fire.visible = smoke.visible = false;
        scene.add(fire, smoke);
        return { fire, smoke };
      });

      // ── clip framing ──
      const kills = findKills(rep);
      const tLast = F[F.length - 1].t - t0;
      const lastKill = kills.length ? kills[kills.length - 1].t : null;
      const tEnd = full || lastKill === null ? tLast : Math.min(tLast, lastKill + 4);
      const hero = kills.length ? kills[kills.length - 1].striker : 0;
      const heroDeathT = kills.length ? kills[kills.length - 1].t : Infinity;
      const total = Math.max(1, Math.ceil(tEnd * fps));

      const frameDt = F.length > 1 ? (F[F.length - 1].t - t0) / (F.length - 1) : 1;
      const sample = (t: number) => {
        const f = Math.min(Math.max(t / frameDt, 0), F.length - 1.001);
        const j = Math.floor(f);
        return { a: F[j], b: F[Math.min(j + 1, F.length - 1)], f: f - j, j };
      };

      // sequential camera state (the driver always seeks 0..N in order)
      const chaseDir = new THREE.Vector2(1, 0);
      let camInit = false;
      const heldPos = new THREE.Vector3();
      const heldQuat = new THREE.Quaternion();
      let held = false;

      const fwd = new THREE.Vector3();
      const up = new THREE.Vector3();

      function seek(i: number): string {
        const t = i / fps;
        const { a, b, f } = sample(t);
        const deadSet = new Set(a.dead ?? []);

        for (let d = 0; d < nDrones; d++) {
          const pa = a.d[d], pb = b.d[d];
          drones[d].position.copy(w2t(
            pa[0] + (pb[0] - pa[0]) * f,
            pa[1] + (pb[1] - pa[1]) * f,
            pa[2] + (pb[2] - pa[2]) * f));
          const yaw = LERP_ANGLE(a.hdg[d], b.hdg[d], f);
          // forward lean from ground speed reads as flight, not hovering boxes
          const sp = Math.hypot(pb[0] - pa[0], pb[1] - pa[1]) / Math.max(frameDt, 1e-3);
          drones[d].rotation.set(0, yaw, -Math.min(0.4, sp * 0.022), "YXZ");
          drones[d].visible = !deadSet.has(d);
        }

        for (let k = 0; k < nT; k++) {
          const ta = a.tg[k], tb = b.tg[k];
          const x = ta[0] + (tb[0] - ta[0]) * f;
          const y = ta[1] + (tb[1] - ta[1]) * f;
          tanks[k].position.copy(w2t(x, y, groundAt(x, y)));
          const mv = Math.hypot(tb[0] - ta[0], tb[1] - ta[1]);
          if (mv > 0.05) tankHdg[k] = Math.atan2(tb[1] - ta[1], tb[0] - ta[0]);
          tanks[k].rotation.y = tankHdg[k];
          const kill = kills.find((kk) => kk.k === k);
          const age = kill ? t - kill.t : -1;
          const wrecked = kill !== undefined && age >= 0;
          tankMats[k].forEach((m, mi) => {
            m.color.copy(tankBase[k][mi]);
            if (wrecked) m.color.multiplyScalar(0.25);
          });
          const { fire, smoke } = booms[k];
          if (kill && age >= 0 && age < 1.2) {
            const e = age / 1.2;
            fire.visible = true;
            fire.position.copy(tanks[k].position).add(new THREE.Vector3(0, 2, 0));
            fire.scale.setScalar(2.5 + 13 * Math.sqrt(e));
            (fire.material as THREE.MeshBasicMaterial).opacity = 0.95 * (1 - e);
          } else fire.visible = false;
          if (kill && age >= 0.15 && age < 4.5) {
            const e = (age - 0.15) / 4.35;
            smoke.visible = true;
            smoke.position.copy(tanks[k].position).add(new THREE.Vector3(0, 3 + 16 * e, 0));
            smoke.scale.setScalar(3 + 9 * e);
            (smoke.material as THREE.MeshBasicMaterial).opacity = 0.6 * (1 - e);
          } else smoke.visible = false;
        }

        // ── camera rig ──
        const hp = drones[hero].position;
        const impact = kills.length
          ? tanks[kills[kills.length - 1].k].position
          : hp;
        if (cam === "fpv") {
          if (t <= heroDeathT + 0.02) {
            const yaw = LERP_ANGLE(a.hdg[hero], b.hdg[hero], f);
            // through the task's own lens: forward along heading, pitched down
            const cy = Math.cos(yaw), sy = Math.sin(yaw);
            const cp = Math.cos(CAM_PITCH_RAD), sp2 = Math.sin(CAM_PITCH_RAD);
            fwd.copy(w2t(cy * cp, sy * cp, -sp2));
            up.copy(w2t(cy * sp2, sy * sp2, cp));
            camera.position.copy(hp);
            camera.up.copy(up);
            const look = hp.clone().add(fwd);
            // terminal guidance: through the hero's last 2.5 s the seeker
            // settles on the target, so the impact happens ON camera
            const dive = kills.find(
              (kk) => kk.striker === hero && t > kk.t - 2.5 && t <= kk.t + 0.02);
            if (dive) {
              const wgt = Math.min(1, (t - (dive.t - 2.5)) / 1.8);
              look.lerp(tanks[dive.k].position, wgt);
            }
            camera.lookAt(look);
            heldPos.copy(camera.position);
            heldQuat.copy(camera.quaternion);
            held = true;
          } else if (held) {
            // the airframe is expended: the last thing its camera saw, held
            camera.position.copy(heldPos);
            camera.quaternion.copy(heldQuat);
          }
        } else {
          const heroAlive = t <= heroDeathT + 0.02;
          if (heroAlive) {
            const va = a.d[hero], vb = b.d[hero];
            const vx = vb[0] - va[0], vy = vb[1] - va[1];
            const sp = Math.hypot(vx, vy);
            if (sp > 0.02) {
              chaseDir.x = chaseDir.x * 0.9 + (vx / sp) * 0.1;
              chaseDir.y = chaseDir.y * 0.9 + (vy / sp) * 0.1;
              chaseDir.normalize();
            }
            const back = w2t(-chaseDir.x * 22, -chaseDir.y * 22, 9);
            const goal = hp.clone().add(back);
            if (!camInit) { camera.position.copy(goal); camInit = true; }
            else camera.position.lerp(goal, 0.14);
            camera.up.set(0, 1, 0);
            // a strike inside its linger window pulls the gaze to the impact
            const lk = kills.find((kk) => t >= kk.t - 1.5 && t < kk.t + 3.5);
            if (lk) camera.lookAt(tanks[lk.k].position.clone().lerp(hp, 0.15));
            else camera.lookAt(hp);
          } else {
            // the hero died striking: slow orbit around the wreck
            const w = t - heroDeathT;
            const ang = 0.25 * w;
            camera.position.set(
              impact.x + Math.cos(ang) * 42,
              impact.y + 20,
              impact.z + Math.sin(ang) * 42);
            camera.up.set(0, 1, 0);
            camera.lookAt(impact);
          }
        }

        renderer!.render(scene, camera);
        return renderer!.domElement.toDataURL("image/jpeg", 0.92);
      }

      window.__vesper = {
        ready: true,
        error: null,
        meta: { frames: total, fps, hero, kills, duration: tEnd },
        seek,
      };
      seek(0);
      setStatus("");
    })().catch((e) => {
      const msg = e instanceof Error ? e.message : String(e);
      setStatus(`render setup failed: ${msg}`);
      window.__vesper = {
        ready: false, error: msg,
        meta: { frames: 0, fps: 0, hero: 0, kills: [], duration: 0 },
        seek: () => "",
      };
    });

    return () => {
      disposed = true;
      renderer?.dispose();
      if (renderer?.domElement.parentElement === el) el.removeChild(renderer.domElement);
    };
  }, []);

  return (
    <div className="min-h-screen bg-black p-2">
      <div ref={holder} />
      {status && (
        <div className="p-4 font-mono text-xs tracking-[0.25em] text-neutral-400">{status}</div>
      )}
    </div>
  );
}
