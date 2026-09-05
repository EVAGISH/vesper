"use client";

import { useEffect, useRef, useState } from "react";
import * as THREE from "three";
import { OrbitControls } from "three/examples/jsm/controls/OrbitControls.js";
import { mergeGeometries } from "three/examples/jsm/utils/BufferGeometryUtils.js";

// The live downlink, rendered in the browser: real terrain, buildings, trees
// and vehicles from /api/world3d/<world> (geometry derived from the same map
// the sim flies against), animated from the warm session's /state at 25 Hz sim
// time. The GPU doing the drawing is the viewer's own — the sim never renders.
//
// Frames: sim world is x east / y north / z up; three.js is y up. The mapping
// used throughout is (x, y, z)_world -> (x, z, -y)_three.

type CamMode = "chase" | "fpv" | "orbit";

type StateDrone = { x: number; y: number; z: number; q?: number[] };
type StateVehicle = { x: number; y: number; z?: number; hdg?: number; found: boolean; reached: boolean };
type LiveState = {
  t: number; world?: string; drones: StateDrone[]; vehicles: StateVehicle[];
  found: number; reached: number; targets: number;
};

const POLL_MS = 150;
const CAM_PITCH_RAD = (40 * Math.PI) / 180; // the task's forward-down lens

const w2t = (x: number, y: number, z: number) => new THREE.Vector3(x, z, -y);

// world-frame wxyz quaternion -> three.js quaternion in the y-up scene
const BASIS = new THREE.Quaternion().setFromAxisAngle(new THREE.Vector3(1, 0, 0), -Math.PI / 2);
const BASIS_INV = BASIS.clone().invert();
function q2t(q: number[]): THREE.Quaternion {
  const qw = new THREE.Quaternion(q[1], q[2], q[3], q[0]);
  return BASIS.clone().multiply(qw).multiply(BASIS_INV);
}

function droneModel(): THREE.Group {
  const g = new THREE.Group();
  const dark = new THREE.MeshStandardMaterial({ color: 0x2a2d31, roughness: 0.7 });
  const accent = new THREE.MeshStandardMaterial({ color: 0xd8dade, roughness: 0.5 });
  const body = new THREE.Mesh(new THREE.BoxGeometry(0.7, 0.16, 0.32), dark);
  g.add(body);
  const rotor = new THREE.CylinderGeometry(0.24, 0.24, 0.02, 12);
  for (const [ax, az] of [[0.32, 0.32], [0.32, -0.32], [-0.32, 0.32], [-0.32, -0.32]]) {
    const arm = new THREE.Mesh(new THREE.BoxGeometry(0.5, 0.05, 0.05), dark);
    arm.position.set(ax * 0.6, 0.02, az * 0.6);
    arm.rotation.y = Math.atan2(-az, ax);
    g.add(arm);
    const r = new THREE.Mesh(rotor, accent);
    r.position.set(ax, 0.1, az);
    g.add(r);
  }
  g.traverse((o) => { o.castShadow = true; });
  return g;
}

function tankModel(): THREE.Group {
  const g = new THREE.Group();
  const hullM = new THREE.MeshStandardMaterial({ color: 0x4c5340, roughness: 0.9 });
  const trackM = new THREE.MeshStandardMaterial({ color: 0x24261f, roughness: 1.0 });
  const hull = new THREE.Mesh(new THREE.BoxGeometry(6.4, 1.4, 2.8), hullM);
  hull.position.y = 1.15;
  const turret = new THREE.Mesh(new THREE.BoxGeometry(2.6, 0.9, 1.9), hullM);
  turret.position.set(-0.3, 2.2, 0);
  const barrel = new THREE.Mesh(new THREE.CylinderGeometry(0.12, 0.15, 4.4, 8), trackM);
  barrel.rotation.z = Math.PI / 2;
  barrel.position.set(3.0, 2.25, 0);
  for (const s of [-1, 1]) {
    const track = new THREE.Mesh(new THREE.BoxGeometry(6.8, 0.9, 0.7), trackM);
    track.position.set(0, 0.45, s * 1.45);
    g.add(track);
  }
  g.add(hull, turret, barrel);
  g.traverse((o) => { o.castShadow = true; });
  return g;
}

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
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    el.appendChild(renderer.domElement);

    const scene = new THREE.Scene();
    scene.background = new THREE.Color(0x9db8d2);
    const camera = new THREE.PerspectiveCamera(62, 16 / 9, 0.5, 4000);
    camera.position.set(0, 300, 300);

    const hemi = new THREE.HemisphereLight(0xcfe4ff, 0x63705d, 0.85);
    scene.add(hemi);
    const sun = new THREE.DirectionalLight(0xfff2dd, 2.2);
    sun.position.set(-420, 560, 300);
    sun.castShadow = true;
    sun.shadow.mapSize.set(2048, 2048);
    const sc = sun.shadow.camera as THREE.OrthographicCamera;
    sc.left = -700; sc.right = 700; sc.top = 700; sc.bottom = -700; sc.far = 2400;
    scene.add(sun);

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
      const d = await fetch(`/api/world3d/${world}`).then((r) => r.json());
      if (dead) return;
      half = d.half_m;
      scene.fog = new THREE.Fog(0x9db8d2, half * 0.9, half * 2.6);
      fogSet = true;

      // terrain: displaced grid, draped with the ground ortho
      const { n, step, z } = d.terrain as { n: number; step: number; z: number[] };
      const geo = new THREE.PlaneGeometry(2 * half, 2 * half, n - 1, n - 1);
      geo.rotateX(-Math.PI / 2);                    // plane XY -> XZ, +y up
      const pos = geo.attributes.position as THREE.BufferAttribute;
      for (let r = 0; r < n; r++) {
        for (let c = 0; c < n; c++) {
          const i = r * n + c;
          // plane rows run +z (south); map row r is y = r*step - half (north up)
          const zr = n - 1 - r;
          pos.setY(i, z[zr * n + c]);
        }
      }
      geo.computeVertexNormals();
      const mat = new THREE.MeshStandardMaterial({ roughness: 1.0, metalness: 0.0 });
      if (d.ground) {
        const tex = new THREE.TextureLoader().load(d.ground);
        tex.colorSpace = THREE.SRGBColorSpace;
        tex.anisotropy = 8;
        mat.map = tex;
      } else {
        mat.color = new THREE.Color(0x6b7a5e);
      }
      const terrain = new THREE.Mesh(geo, mat);
      terrain.receiveShadow = true;
      scene.add(terrain);

      // buildings: one merged extrusion mesh, concrete with per-part variation
      const parts: THREE.BufferGeometry[] = [];
      for (const b of d.buildings as { p: number[][]; h: number; z: number }[]) {
        const shape = new THREE.Shape(b.p.map(([x, y]) => new THREE.Vector2(x, y)));
        const g = new THREE.ExtrudeGeometry(shape, { depth: b.h, bevelEnabled: false });
        g.rotateX(-Math.PI / 2);                   // (x,y,ext) -> (x, ext up, -y)
        g.translate(0, b.z, 0);
        parts.push(g);
      }
      if (parts.length) {
        const merged = mergeGeometries(parts, false)!;
        merged.computeVertexNormals();
        const bm = new THREE.Mesh(merged, new THREE.MeshStandardMaterial({
          color: 0xb8b2a8, roughness: 0.95, flatShading: true,
        }));
        bm.castShadow = true;
        bm.receiveShadow = true;
        scene.add(bm);
        parts.forEach((p) => p.dispose());
      }

      // trees: two instanced meshes (trunks + crowns), scaled per tree
      const trees = d.trees as number[][];
      if (trees.length) {
        const trunkG = new THREE.CylinderGeometry(0.14, 0.22, 1, 5);
        trunkG.translate(0, 0.5, 0);
        const crownG = new THREE.IcosahedronGeometry(1, 1);
        const trunkM = new THREE.MeshStandardMaterial({ color: 0x4a3a28, roughness: 1 });
        const crownM = new THREE.MeshStandardMaterial({ color: 0x3d5a34, roughness: 1 });
        const trunks = new THREE.InstancedMesh(trunkG, trunkM, trees.length);
        const crowns = new THREE.InstancedMesh(crownG, crownM, trees.length);
        const m = new THREE.Matrix4();
        const q = new THREE.Quaternion();
        const col = new THREE.Color();
        for (let i = 0; i < trees.length; i++) {
          const [x, y, gz, h] = trees[i];
          const s = h * 0.42;
          m.compose(w2t(x, y, gz), q, new THREE.Vector3(1, h * 0.45, 1));
          trunks.setMatrixAt(i, m);
          m.compose(w2t(x, y, gz + h * 0.62), q,
                    new THREE.Vector3(s * 0.8, s, s * 0.8));
          crowns.setMatrixAt(i, m);
          col.setHSL(0.28 + ((i * 37) % 13) / 90, 0.42, 0.24 + ((i * 53) % 11) / 70);
          crowns.setColorAt(i, col);
        }
        crowns.castShadow = true;
        scene.add(trunks, crowns);
      }
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
