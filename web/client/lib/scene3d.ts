// Shared three.js scene kit for the sim world — the ONE place world geometry,
// tree/vehicle models, lights and frame math are defined, so the live view
// (components/world-view.tsx) and the offline replay exporter (app/render)
// produce pixel-identical worlds from the same /api/world3d payload.
//
// Frames: sim world is x east / y north / z up; three.js is y up. The mapping
// used throughout is (x, y, z)_world -> (x, z, -y)_three.

import * as THREE from "three";
import { GLTFLoader } from "three/examples/jsm/loaders/GLTFLoader.js";
import { mergeGeometries } from "three/examples/jsm/utils/BufferGeometryUtils.js";

export type CamMode = "chase" | "fpv" | "orbit";

export const CAM_PITCH_RAD = (40 * Math.PI) / 180; // the task's forward-down lens

export const w2t = (x: number, y: number, z: number) => new THREE.Vector3(x, z, -y);

// world-frame wxyz quaternion -> three.js quaternion in the y-up scene
export const BASIS = new THREE.Quaternion().setFromAxisAngle(
  new THREE.Vector3(1, 0, 0), -Math.PI / 2);
const BASIS_INV = BASIS.clone().invert();
export function q2t(q: number[]): THREE.Quaternion {
  const qw = new THREE.Quaternion(q[1], q[2], q[3], q[0]);
  return BASIS.clone().multiply(qw).multiply(BASIS_INV);
}

/** /api/world3d/<world> payload (web/server world3d endpoint). */
export type World3D = {
  world: string;
  half_m: number;
  ground: string | null;
  terrain: { n: number; step: number; z: number[] };
  buildings: { p: number[][]; h: number; z: number }[];
  trees: number[][]; // [x, y, ground_z, height_m]
};

// ── actor models ──────────────────────────────────────────────────────────

export function droneModel(): THREE.Group {
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

export function tankModel(): THREE.Group {
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

// ── sky + lights ──────────────────────────────────────────────────────────

export const SKY = 0x9db8d2;

export function addSkyAndLights(scene: THREE.Scene) {
  scene.background = new THREE.Color(SKY);
  const hemi = new THREE.HemisphereLight(0xcfe4ff, 0x63705d, 0.85);
  scene.add(hemi);
  const sun = new THREE.DirectionalLight(0xfff2dd, 2.2);
  sun.position.set(-420, 560, 300);
  sun.castShadow = true;
  sun.shadow.mapSize.set(2048, 2048);
  const sc = sun.shadow.camera as THREE.OrthographicCamera;
  sc.left = -700; sc.right = 700; sc.top = 700; sc.bottom = -700; sc.far = 2400;
  scene.add(sun);
}

// ── terrain + buildings ───────────────────────────────────────────────────

/** Bilinear ground-height sampler over the world3d terrain grid (world XY, m). */
export function makeGroundSampler(d: World3D): (x: number, y: number) => number {
  const { n, step, z } = d.terrain;
  const half = d.half_m;
  return (x: number, y: number) => {
    const fc = Math.min(Math.max((x + half) / step, 0), n - 1);
    const fr = Math.min(Math.max((y + half) / step, 0), n - 1);
    const c0 = Math.min(Math.floor(fc), n - 2), r0 = Math.min(Math.floor(fr), n - 2);
    const tc = fc - c0, tr = fr - r0;
    const z00 = z[r0 * n + c0], z01 = z[r0 * n + c0 + 1];
    const z10 = z[(r0 + 1) * n + c0], z11 = z[(r0 + 1) * n + c0 + 1];
    return (z00 * (1 - tc) + z01 * tc) * (1 - tr) + (z10 * (1 - tc) + z11 * tc) * tr;
  };
}

/** Terrain (draped with the ground ortho) + merged building extrusions. */
export function buildTerrain(scene: THREE.Scene, d: World3D) {
  const half = d.half_m;
  scene.fog = new THREE.Fog(SKY, half * 0.9, half * 2.6);

  const { n, z } = d.terrain;
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

  const parts: THREE.BufferGeometry[] = [];
  for (const b of d.buildings) {
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
}

// ── trees ─────────────────────────────────────────────────────────────────
// Low-poly species from Kenney's Nature Kit (CC0), committed in
// public/models/trees/ (see LICENSE.md there). Each GLB is two primitives
// (bark + leaves); we bake them into normalized unit-height geometries and
// draw every tree of a species as two InstancedMeshes — thousands of trees,
// six draw calls. Canopy color varies per instance; broadleaf species get
// green-hued canopies, the pine a darker blue-green.

type TreeSpecies = {
  bark: THREE.BufferGeometry;
  leaf: THREE.BufferGeometry;
  pine: boolean;
};

const TREE_FILES = [
  { file: "tree_default.glb", pine: false },
  { file: "tree_oak.glb", pine: false },
  { file: "tree_pineTallA.glb", pine: true },
];

function bakeSpecies(root: THREE.Object3D, pine: boolean): TreeSpecies | null {
  root.updateMatrixWorld(true);
  const barks: THREE.BufferGeometry[] = [];
  const leafs: THREE.BufferGeometry[] = [];
  root.traverse((o) => {
    if (!(o as THREE.Mesh).isMesh) return;
    const m = o as THREE.Mesh;
    const g = m.geometry.clone().applyMatrix4(m.matrixWorld);
    const name = (Array.isArray(m.material) ? m.material[0] : m.material)?.name ?? "";
    (/leaf/i.test(name) ? leafs : barks).push(g);
  });
  if (!leafs.length || !barks.length) return null;
  const leaf = leafs.length > 1 ? mergeGeometries(leafs, false)! : leafs[0];
  const bark = barks.length > 1 ? mergeGeometries(barks, false)! : barks[0];
  // normalize: base at y=0, height exactly 1 (instances scale by tree height)
  const box = new THREE.Box3().setFromBufferAttribute(
    leaf.attributes.position as THREE.BufferAttribute);
  box.union(new THREE.Box3().setFromBufferAttribute(
    bark.attributes.position as THREE.BufferAttribute));
  const h = Math.max(box.max.y - box.min.y, 1e-3);
  for (const g of [leaf, bark]) {
    g.translate(0, -box.min.y, 0);
    g.scale(1 / h, 1 / h, 1 / h);
    g.computeVertexNormals();
  }
  return { bark, leaf, pine };
}

/** Procedural stand-in species so a broken model fetch still grows a forest. */
function fallbackSpecies(pine: boolean): TreeSpecies {
  const bark = new THREE.CylinderGeometry(0.02, 0.035, 0.45, 5);
  bark.translate(0, 0.225, 0);
  let leaf: THREE.BufferGeometry;
  if (pine) {
    const cones: THREE.BufferGeometry[] = [];
    for (let i = 0; i < 3; i++) {
      const c = new THREE.ConeGeometry(0.28 - i * 0.07, 0.42, 7);
      c.translate(0, 0.38 + i * 0.21, 0);
      cones.push(c);
    }
    leaf = mergeGeometries(cones, false)!;
  } else {
    const blobs: THREE.BufferGeometry[] = [];
    for (const [dx, dy, dz, s] of [[0, 0.68, 0, 0.3], [0.16, 0.56, 0.1, 0.2],
                                   [-0.14, 0.58, -0.08, 0.22], [0.02, 0.82, -0.04, 0.2]]) {
      const b = new THREE.IcosahedronGeometry(s, 1);
      b.translate(dx, dy, dz);
      blobs.push(b);
    }
    leaf = mergeGeometries(blobs, false)!;
  }
  return { bark, leaf, pine };
}

let treeCache: Promise<TreeSpecies[]> | null = null;

/** Load (once) the tree species GLBs; never rejects — falls back procedurally. */
export function loadTreeSpecies(base = "/models/trees"): Promise<TreeSpecies[]> {
  if (treeCache) return treeCache;
  const loader = new GLTFLoader();
  treeCache = Promise.all(
    TREE_FILES.map((t) =>
      loader
        .loadAsync(`${base}/${t.file}`)
        .then((gltf) => bakeSpecies(gltf.scene, t.pine) ?? fallbackSpecies(t.pine))
        .catch(() => fallbackSpecies(t.pine)),
    ),
  );
  return treeCache;
}

const BARK_MAT = () => new THREE.MeshStandardMaterial({
  color: 0x5c4832, roughness: 1.0, flatShading: true,
});
const LEAF_MAT = () => new THREE.MeshStandardMaterial({
  color: 0xffffff, roughness: 1.0, flatShading: true,
});

/** Instance every tree from the world3d `trees` array onto the scene. */
export function buildTrees(scene: THREE.Scene, trees: number[][], species: TreeSpecies[]) {
  if (!trees.length || !species.length) return;
  // deterministic per-tree hash -> species / rotation / width jitter / hue
  const hash = (i: number, salt: number) => {
    let x = (i * 2654435761 + salt * 40503) >>> 0;
    x = ((x >>> 16) ^ x) * 0x45d9f3b >>> 0;
    x = ((x >>> 16) ^ x) >>> 0;
    return (x % 10000) / 10000;
  };
  const pick = (i: number, h: number) => {
    // tall stands lean pine, low scrub leans broadleaf; always mixed
    const r = hash(i, 1);
    if (h >= 13) return r < 0.55 ? 2 : r < 0.8 ? 0 : 1;
    return r < 0.42 ? 0 : r < 0.8 ? 1 : 2;
  };
  const counts = species.map(() => 0);
  const assign = trees.map((t, i) => {
    const s = pick(i, t[3]);
    counts[s]++;
    return s;
  });
  const m = new THREE.Matrix4();
  const q = new THREE.Quaternion();
  const up = new THREE.Vector3(0, 1, 0);
  const col = new THREE.Color();
  species.forEach((sp, si) => {
    if (!counts[si]) return;
    const barks = new THREE.InstancedMesh(sp.bark, BARK_MAT(), counts[si]);
    const leafs = new THREE.InstancedMesh(sp.leaf, LEAF_MAT(), counts[si]);
    let j = 0;
    for (let i = 0; i < trees.length; i++) {
      if (assign[i] !== si) continue;
      const [x, y, gz, h] = trees[i];
      q.setFromAxisAngle(up, hash(i, 2) * Math.PI * 2);
      const wj = 0.8 + hash(i, 3) * 0.45;               // width jitter
      m.compose(w2t(x, y, gz), q, new THREE.Vector3(h * wj, h, h * wj));
      barks.setMatrixAt(j, m);
      leafs.setMatrixAt(j, m);
      if (sp.pine) col.setHSL(0.38 + hash(i, 4) * 0.06, 0.32, 0.16 + hash(i, 5) * 0.07);
      else col.setHSL(0.24 + hash(i, 4) * 0.09, 0.42, 0.2 + hash(i, 5) * 0.1);
      leafs.setColorAt(j, col);
      j++;
    }
    barks.castShadow = true;
    leafs.castShadow = true;
    scene.add(barks, leafs);
  });
}

/** One-call world build: sky, lights, terrain, buildings, trees. */
export async function buildWorldScene(scene: THREE.Scene, d: World3D) {
  addSkyAndLights(scene);
  buildTerrain(scene, d);
  buildTrees(scene, d.trees, await loadTreeSpecies());
  return makeGroundSampler(d);
}
