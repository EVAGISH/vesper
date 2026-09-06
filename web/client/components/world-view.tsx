"use client";

import { useEffect, useRef, useState } from "react";
import * as THREE from "three";
import { OrbitControls } from "three/examples/jsm/controls/OrbitControls.js";
import {
  buildWorldScene, CAM_PITCH_RAD, type CamMode, droneModel, q2t, tankModel, w2t,
  type World3D,
} from "@/lib/scene3d";

// The live downlink, rendered in the browser: real terrain, buildings, trees
// and vehicles from /api/world3d/<world> (geometry derived from the same map
// the sim flies against), animated from the warm session's /state at 25 Hz sim
// time. The GPU doing the drawing is the viewer's own — the sim never renders.
// Scene construction (terrain/buildings/trees/models, frame math) lives in
// lib/scene3d.ts, shared verbatim with the offline replay exporter.

type StateDrone = { x: number; y: number; z: number; q?: number[]; expended?: boolean };
type StateVehicle = { x: number; y: number; z?: number; hdg?: number; found: boolean; reached: boolean };
type LiveState = {
  t: number; world?: string; drones: StateDrone[]; vehicles: StateVehicle[];
  found: number; reached: number; targets: number;
};

const POLL_MS = 150;

export function WorldView({ ip }: { ip: string }) {
  const holder = useRef<HTMLDivElement>(null);
  const [mode, setMode] = useState<CamMode>("chase");
  const modeRef = useRef<CamMode>("chase");
  modeRef.current = mode;
  const [status, setStatus] = useState<string>("connecting…");

  useEffect(() => {
    const el = holder.current;
    if (!el) return;
    let dead = false;

    const renderer = new THREE.WebGLRenderer({ antialias: true });
    renderer.shadowMap.enabled = true;
    renderer.shadowMap.type = THREE.PCFSoftShadowMap;
    renderer.toneMapping = THREE.ACESFilmicToneMapping;   // same grade as the export
    renderer.toneMappingExposure = 1.15;
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    el.appendChild(renderer.domElement);

    const scene = new THREE.Scene();
    scene.background = new THREE.Color(0x9db8d2);
    const camera = new THREE.PerspectiveCamera(62, 16 / 9, 0.5, 4000);
    camera.position.set(0, 300, 300);

    const controls = new OrbitControls(camera, renderer.domElement);
    controls.enabled = false;
    controls.maxPolarAngle = Math.PI / 2 - 0.05;

    // live actors, populated once geometry + first state arrive
    const drones: THREE.Group[] = [];
    const tanks: THREE.Group[] = [];
    let half = 500;
    let fogSet = false;

    // interpolation buffer: render slightly in the past between /state samples
    let prev: LiveState | null = null;
    let next: LiveState | null = null;
    let nextAt = 0, span = POLL_MS;
    const chaseDir = new THREE.Vector2(1, 0);

    async function buildWorld(world: string) {
      const d: World3D = await fetch(`/api/world3d/${world}`).then((r) => r.json());
      if (dead) return;
      half = d.half_m;
      await buildWorldScene(scene, d);      // terrain + buildings + trees + lights
      fogSet = true;
      setStatus("");
    }

    function ensureActors(s: LiveState) {
      while (drones.length < s.drones.length) {
        const g = droneModel();
        drones.push(g);
        scene.add(g);
      }
      while (tanks.length < s.vehicles.length) {
        const g = tankModel();
        tanks.push(g);
        scene.add(g);
      }
    }

    let built: string | null = null;
    const poll = setInterval(async () => {
      try {
        const s: LiveState = await fetch(`http://${ip}:8180/state`, { cache: "no-store" })
          .then((r) => r.json());
        if (dead) return;
        if (s.world && built !== s.world) {
          built = s.world;
          buildWorld(s.world).catch(() => setStatus("world geometry unavailable"));
        } else if (!s.world && built === null) {
          setStatus("session publishes no 3D state (Isaac session? use the MJPEG feeds)");
        }
        prev = next ?? s;
        next = s;
        span = Math.max(60, performance.now() - nextAt);
        nextAt = performance.now();
        ensureActors(s);
      } catch {
        if (!dead) setStatus("no live session");
      }
    }, POLL_MS);

    const lerp3 = (a: StateDrone, b: StateDrone, f: number) =>
      w2t(a.x + (b.x - a.x) * f, a.y + (b.y - a.y) * f, a.z + (b.z - a.z) * f);

    let raf = 0;
    const fwdW = new THREE.Vector3();
    const upW = new THREE.Vector3();
    const animate = () => {
      raf = requestAnimationFrame(animate);
      if (prev && next) {
        const f = Math.min(1, (performance.now() - nextAt) / span);
        for (let i = 0; i < drones.length && i < next.drones.length; i++) {
          const a = prev.drones[i] ?? next.drones[i];
          const b = next.drones[i];
          // a loitering munition that struck is expended: its glyph is gone
          drones[i].visible = !b.expended;
          drones[i].position.copy(lerp3(a, b, f));
          if (a.q && b.q) drones[i].quaternion.copy(q2t(a.q).slerp(q2t(b.q), f));
        }
        for (let i = 0; i < tanks.length && i < next.vehicles.length; i++) {
          const v = next.vehicles[i];
          tanks[i].position.copy(w2t(v.x, v.y, (v.z ?? 0) - 0.9));
          if (v.hdg !== undefined) tanks[i].rotation.y = v.hdg;
        }
        const d0 = drones[0];
        if (d0) {
          const m = modeRef.current;
          if (m === "fpv" && next.drones[0].q) {
            // through the task's own lens: forward-down, horizon banks with the airframe
            const q = d0.quaternion;
            fwdW.set(Math.cos(CAM_PITCH_RAD), -Math.sin(CAM_PITCH_RAD), 0);
            // body->three: model +x is body forward, +y is body up
            fwdW.set(fwdW.x, fwdW.y, 0).applyQuaternion(q);
            upW.set(Math.sin(CAM_PITCH_RAD), Math.cos(CAM_PITCH_RAD), 0).applyQuaternion(q);
            camera.position.copy(d0.position);
            camera.up.copy(upW);
            camera.lookAt(d0.position.clone().add(fwdW));
          } else if (m === "chase") {
            const va = prev.drones[0], vb = next.drones[0];
            const vx = vb.x - va.x, vy = vb.y - va.y;
            const sp = Math.hypot(vx, vy);
            if (sp > 0.05) {
              chaseDir.x = chaseDir.x * 0.92 + (vx / sp) * 0.08;
              chaseDir.y = chaseDir.y * 0.92 + (vy / sp) * 0.08;
              chaseDir.normalize();
            }
            const back = w2t(-chaseDir.x * 22, -chaseDir.y * 22, 9);
            camera.position.lerp(d0.position.clone().add(back), 0.12);
            camera.up.set(0, 1, 0);
            camera.lookAt(d0.position);
          } else {
            controls.target.lerp(d0.position, 0.05);
          }
        }
      }
      controls.enabled = modeRef.current === "orbit";
      if (controls.enabled) controls.update();
      if (!fogSet) scene.fog = null;
      renderer.render(scene, camera);
    };
    animate();

    const ro = new ResizeObserver(() => {
      const w = el.clientWidth, h = el.clientHeight;
      renderer.setSize(w, h);
      camera.aspect = w / h;
      camera.updateProjectionMatrix();
    });
    ro.observe(el);

    return () => {
      dead = true;
      clearInterval(poll);
      cancelAnimationFrame(raf);
      ro.disconnect();
      controls.dispose();
      renderer.dispose();
      el.removeChild(renderer.domElement);
    };
  }, [ip]);

  return (
    <div className="relative aspect-video w-full bg-black">
      <div ref={holder} className="absolute inset-0" />
      <div className="absolute left-2 top-2 flex gap-1">
        {(["chase", "fpv", "orbit"] as CamMode[]).map((m) => (
          <button
            key={m}
            onClick={() => setMode(m)}
            className={`cursor-pointer border border-border/60 px-2 py-0.5 font-mono text-[10px] uppercase tracking-widest ${
              mode === m ? "bg-foreground text-background" : "bg-background/60 text-muted-foreground hover:text-foreground"
            }`}
          >
            {m}
          </button>
        ))}
      </div>
      {status && (
        <div className="absolute inset-0 flex items-center justify-center">
          <span className="font-mono text-xs tracking-[0.25em] text-muted-foreground">{status}</span>
        </div>
      )}
    </div>
  );
}
