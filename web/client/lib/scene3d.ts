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

// ── target vehicle: BTR-80A ───────────────────────────────────────────────
// Real textured APC (assets/vehicles/btr80, CC BY 4.0 — attribution ships in
// public/models/vehicles/ATTRIBUTION.md), the same asset the Isaac lane
// drives. Loaded once, normalized to the sim's vehicle frame (nose +X, wheels
// on the ground, true 7.65 m length), then cloned per target with per-clone
// materials so a wreck can char independently. Until the GLB arrives (or if
// it can't), the procedural box tank stands in.
const BTR_LENGTH_M = 7.65;
let btrTemplate: THREE.Group | null = null;
let btrLoading: Promise<THREE.Group | null> | null = null;

function normalizeVehicle(root: THREE.Object3D): THREE.Group {
  const align = new THREE.Group();
  align.add(root);
  let box = new THREE.Box3().setFromObject(root);
  let size = box.getSize(new THREE.Vector3());
  if (size.z > size.x) root.rotation.y = Math.PI / 2;   // long axis -> +X (nose)
  box = new THREE.Box3().setFromObject(align);
  size = box.getSize(new THREE.Vector3());
  align.scale.setScalar(BTR_LENGTH_M / Math.max(size.x, 1e-3));
  box = new THREE.Box3().setFromObject(align);
  align.position.set(-(box.min.x + box.max.x) / 2, -box.min.y,
                     -(box.min.z + box.max.z) / 2);
  const tpl = new THREE.Group();
  tpl.add(align);
  tpl.traverse((o) => { o.castShadow = true; });
  return tpl;
}

/** Kick off (once) the BTR load; resolves null when unavailable. */
export function preloadVehicle(base = "/models/vehicles"): Promise<THREE.Group | null> {
  if (!btrLoading) {
    btrLoading = new GLTFLoader()
      .loadAsync(`${base}/btr80.glb`)
      .then((gltf) => { btrTemplate = normalizeVehicle(gltf.scene); return btrTemplate; })
      .catch(() => null);
  }
  return btrLoading;
}

/** A target vehicle instance: the BTR when loaded (swapped in as it arrives),
 *  else the box tank. Clones get their own materials (wreck tinting). */
export function vehicleModel(): THREE.Group {
  const wrap = new THREE.Group();
  const fill = (tpl: THREE.Group) => {
    const c = tpl.clone(true);
    c.traverse((o) => {
      const mesh = o as THREE.Mesh;
      if (!mesh.isMesh) return;
      mesh.material = Array.isArray(mesh.material)
        ? mesh.material.map((m) => m.clone())
        : mesh.material.clone();
    });
    wrap.add(c);
  };
  if (btrTemplate) fill(btrTemplate);
  else {
    const stub = tankModel();
    wrap.add(stub);
    preloadVehicle().then((tpl) => {
      if (tpl) { wrap.remove(stub); fill(tpl); }
    });
  }
  return wrap;
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
    tex.anisotropy = 16;
    mat.map = tex;
  } else {
    mat.color = new THREE.Color(0x6b7a5e);
  }
  const terrain = new THREE.Mesh(geo, mat);
  terrain.receiveShadow = true;
  scene.add(terrain);

  buildBuildings(scene, d.buildings);
}

// ── buildings ─────────────────────────────────────────────────────────────
// Extrusions split into walls + roofs: walls carry a repeating window-facade
// texture (UVs from ExtrudeGeometry's side walls are in world metres, so one
// tile ≈ one 3 m window bay on every building) and a per-building plaster
// tint; roofs get their own muted per-building color. Reads as a town instead
// of beige boxes, still two draw calls.

const FACADE_M = 3.0;            // metres per window bay, both axes

/** Grayscale-ish facade tile (plaster + one window) — tinted by vertex color. */
function facadeTexture(): THREE.CanvasTexture {
  const S = 128;
  const c = document.createElement("canvas");
  c.width = c.height = S;
  const g = c.getContext("2d")!;
  g.fillStyle = "#cfcbc2";
  g.fillRect(0, 0, S, S);
  // plaster speckle (deterministic)
  let seed = 7;
  const rnd = () => (seed = (seed * 16807) % 2147483647) / 2147483647;
  for (let i = 0; i < 350; i++) {
    g.fillStyle = rnd() < 0.5 ? "rgba(0,0,0,0.045)" : "rgba(255,255,255,0.05)";
    g.fillRect(rnd() * S, rnd() * S, 1 + rnd() * 2, 1 + rnd() * 2);
  }
  // window bay: frame, glazing with a sky-ish gradient, sill shadow
  const wx = S * 0.28, wy = S * 0.22, ww = S * 0.44, wh = S * 0.5;
  g.fillStyle = "#8e8a80";
  g.fillRect(wx - 3, wy - 3, ww + 6, wh + 6);
  const glaze = g.createLinearGradient(0, wy, 0, wy + wh);
  glaze.addColorStop(0, "#3a4550");
  glaze.addColorStop(1, "#232a31");
  g.fillStyle = glaze;
  g.fillRect(wx, wy, ww, wh);
  g.fillStyle = "#8e8a80";                       // mullions
  g.fillRect(wx + ww / 2 - 1, wy, 2, wh);
  g.fillRect(wx, wy + wh / 2 - 1, ww, 2);
  g.fillStyle = "rgba(0,0,0,0.18)";              // sill shadow
  g.fillRect(wx - 3, wy + wh + 3, ww + 6, 3);
  const tex = new THREE.CanvasTexture(c);
  tex.colorSpace = THREE.SRGBColorSpace;
  tex.wrapS = tex.wrapT = THREE.RepeatWrapping;
  tex.repeat.set(1 / FACADE_M, 1 / FACADE_M);    // side-wall UVs are in metres
  tex.anisotropy = 8;
  return tex;
}

/** Split an ExtrudeGeometry into its cap (roof) and side-wall triangles. */
function splitExtrude(g: THREE.BufferGeometry) {
  const out: { caps?: THREE.BufferGeometry; walls?: THREE.BufferGeometry } = {};
  const src = g.index ? g.toNonIndexed() : g;
  for (const grp of src.groups) {
    const ng = new THREE.BufferGeometry();
    for (const name of ["position", "normal", "uv"] as const) {
      const attr = src.getAttribute(name) as THREE.BufferAttribute;
      if (!attr) continue;
      const arr = (attr.array as Float32Array)
        .slice(grp.start * attr.itemSize, (grp.start + grp.count) * attr.itemSize);
      ng.setAttribute(name, new THREE.BufferAttribute(arr, attr.itemSize));
    }
    if (grp.materialIndex === 0) out.caps = ng;
    else out.walls = ng;
  }
  return out;
}

const WALL_TINTS = [0xd6cfc2, 0xcbc6bd, 0xd9d2c4, 0xc2bcae, 0xd0c4b0, 0xbfb9b0];
const ROOF_TINTS = [0x6e5f52, 0x5a5750, 0x7a5646, 0x615c54, 0x54514a, 0x6b6257];

function tintAttr(g: THREE.BufferGeometry, hex: number) {
  const n = (g.getAttribute("position") as THREE.BufferAttribute).count;
  const c = new THREE.Color(hex);
  const arr = new Float32Array(n * 3);
  for (let i = 0; i < n; i++) { arr[i * 3] = c.r; arr[i * 3 + 1] = c.g; arr[i * 3 + 2] = c.b; }
  g.setAttribute("color", new THREE.BufferAttribute(arr, 3));
}

function buildBuildings(scene: THREE.Scene, buildings: World3D["buildings"]) {
  const wallParts: THREE.BufferGeometry[] = [];
  const roofParts: THREE.BufferGeometry[] = [];
  buildings.forEach((b, i) => {
    const shape = new THREE.Shape(b.p.map(([x, y]) => new THREE.Vector2(x, y)));
    const g = new THREE.ExtrudeGeometry(shape, { depth: b.h, bevelEnabled: false });
    g.rotateX(-Math.PI / 2);                   // (x,y,ext) -> (x, ext up, -y)
    g.translate(0, b.z, 0);
    const { caps, walls } = splitExtrude(g);
    if (walls) { tintAttr(walls, WALL_TINTS[i % WALL_TINTS.length]); wallParts.push(walls); }
    if (caps) { tintAttr(caps, ROOF_TINTS[(i * 7 + 3) % ROOF_TINTS.length]); roofParts.push(caps); }
    g.dispose();
  });
  if (wallParts.length) {
    const wallGeo = mergeGeometries(wallParts, false)!;
    wallGeo.computeVertexNormals();
    const wallsMesh = new THREE.Mesh(wallGeo, new THREE.MeshStandardMaterial({
      map: facadeTexture(), vertexColors: true, roughness: 0.9, flatShading: true,
    }));
    wallsMesh.castShadow = true;
    wallsMesh.receiveShadow = true;
    scene.add(wallsMesh);
    wallParts.forEach((p) => p.dispose());
  }
  if (roofParts.length) {
    const roofGeo = mergeGeometries(roofParts, false)!;
    roofGeo.computeVertexNormals();
    const roofMesh = new THREE.Mesh(roofGeo, new THREE.MeshStandardMaterial({
      vertexColors: true, roughness: 1.0, flatShading: true,
    }));
    roofMesh.castShadow = true;
    roofMesh.receiveShadow = true;
    scene.add(roofMesh);
    roofParts.forEach((p) => p.dispose());
  }
}

// ── trees ─────────────────────────────────────────────────────────────────
// Stylized species from Quaternius' Ultimate Stylized Nature pack (CC0),
// committed in public/models/trees/ (see LICENSE.md there): textured bark and
// alpha-cutout leaf cards, decimated to ~1.2k tris per tree. Each GLB's
// primitives are baked into normalized unit-height geometries and every tree
// of a species is drawn as one InstancedMesh per primitive — thousands of
// trees, a handful of draw calls. Foliage keeps the pack's textures, tinted
// per instance and shaded by a baked canopy AO gradient.

type TreePart = { geo: THREE.BufferGeometry; mat: THREE.MeshStandardMaterial; leaf: boolean };
type TreeSpecies = { parts: TreePart[]; pine: boolean; accent?: boolean };

// green broadleaf + pine stands only — no autumn/red species (operator call)
const TREE_FILES: { file: string; pine: boolean; accent?: boolean }[] = [
  { file: "NormalTree_1.glb", pine: false },
  { file: "NormalTree_4.glb", pine: false },
  { file: "BirchTree_1.glb", pine: false },
  { file: "PineTree_1.glb", pine: true },
  { file: "PineTree_3.glb", pine: true },
];

function bakeSpecies(root: THREE.Object3D, pine: boolean): TreeSpecies | null {
  root.updateMatrixWorld(true);
  const raw: { geo: THREE.BufferGeometry; mat: THREE.Material; leaf: boolean }[] = [];
  root.traverse((o) => {
    if (!(o as THREE.Mesh).isMesh) return;
    const m = o as THREE.Mesh;
    const mat = Array.isArray(m.material) ? m.material[0] : m.material;
    raw.push({
      geo: m.geometry.clone().applyMatrix4(m.matrixWorld),
      mat,
      leaf: /leaf|leaves/i.test(mat?.name ?? ""),
    });
  });
  if (!raw.length || !raw.some((r) => r.leaf)) return null;
  // source packs disagree on the up axis (even per file): detect the growth
  // axis from the union bounds, rotate it onto +Y, and make sure the canopy
  // ends up above the trunk — never trust the file's orientation
  const union = (leafOnly?: boolean) => {
    const b = new THREE.Box3();
    for (const r of raw) {
      if (leafOnly !== undefined && r.leaf !== leafOnly) continue;
      b.union(new THREE.Box3().setFromBufferAttribute(
        r.geo.attributes.position as THREE.BufferAttribute));
    }
    return b;
  };
  let box = union();
  const ext = box.getSize(new THREE.Vector3());
  if (ext.z >= ext.x && ext.z >= ext.y) raw.forEach((r) => r.geo.rotateX(-Math.PI / 2));
  else if (ext.x >= ext.y && ext.x >= ext.z) raw.forEach((r) => r.geo.rotateZ(Math.PI / 2));
  const midY = (b: THREE.Box3) => (b.min.y + b.max.y) / 2;
  if (midY(union(true)) < midY(union(false))) raw.forEach((r) => r.geo.rotateX(Math.PI));
  box = union();
  // normalize: base at y=0, height exactly 1 (instances scale by tree height)
  const h = Math.max(box.max.y - box.min.y, 1e-3);
  const parts = raw.map((r) => {
    r.geo.translate(0, -box.min.y, 0);
    r.geo.scale(1 / h, 1 / h, 1 / h);
    if (r.leaf) bakeCanopyAO(r.geo);
    // keep the GLB's textured material (bark map / leaf alpha map), tuned for
    // instanced foliage: alpha-cutout cards visible from both sides, no
    // transparency sorting, canopy AO via vertex colors
    const mat = (r.mat as THREE.MeshStandardMaterial).clone();
    mat.roughness = 1.0;
    mat.metalness = 0.0;
    if (r.leaf) {
      mat.alphaTest = 0.45;
      mat.transparent = false;
      mat.depthWrite = true;
      mat.side = THREE.DoubleSide;
      mat.vertexColors = true;
      mat.color.set(0xffffff);
    }
    return { geo: r.geo, mat, leaf: r.leaf };
  });
  return { parts, pine };
}

/** Vertical ambient-occlusion gradient baked into the canopy's vertex colors:
 *  shaded underside, sunlit crown. Multiplies with the per-instance hue, so
 *  the foliage reads as volume instead of a flat-lit blob. */
function bakeCanopyAO(leaf: THREE.BufferGeometry) {
  const pos = leaf.getAttribute("position") as THREE.BufferAttribute;
  let lo = Infinity, hi = -Infinity;
  for (let i = 0; i < pos.count; i++) {
    const y = pos.getY(i);
    if (y < lo) lo = y;
    if (y > hi) hi = y;
  }
  const span = Math.max(hi - lo, 1e-4);
  const arr = new Float32Array(pos.count * 3);
  for (let i = 0; i < pos.count; i++) {
    const t = (pos.getY(i) - lo) / span;
    const v = 0.62 + 0.38 * Math.pow(t, 0.9);     // shaded underside -> lit crown
    arr[i * 3] = arr[i * 3 + 1] = arr[i * 3 + 2] = v;
  }
  leaf.setAttribute("color", new THREE.BufferAttribute(arr, 3));
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
  bakeCanopyAO(leaf);
  return {
    pine,
    parts: [
      { geo: bark, leaf: false, mat: new THREE.MeshStandardMaterial({
        color: 0x4a3b2c, roughness: 1.0, flatShading: true }) },
      { geo: leaf, leaf: true, mat: new THREE.MeshStandardMaterial({
        color: 0x5a6b3f, roughness: 1.0, flatShading: true, vertexColors: true }) },
    ],
  };
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
        .catch(() => fallbackSpecies(t.pine))
        .then((s) => { s.accent = !!t.accent; return s; }),
    ),
  );
  return treeCache;
}

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
  const broad = species.map((s, i) => (!s.pine && !s.accent ? i : -1)).filter((i) => i >= 0);
  const pines = species.map((s, i) => (s.pine && !s.accent ? i : -1)).filter((i) => i >= 0);
  const accents = species.map((s, i) => (s.accent ? i : -1)).filter((i) => i >= 0);
  const pick = (i: number, h: number) => {
    // a few percent accent trees (the red maple); tall stands lean pine,
    // low scrub leans broadleaf; always mixed
    if (accents.length && hash(i, 6) < 0.05) {
      return accents[Math.floor(hash(i, 7) * accents.length) % accents.length];
    }
    const r = hash(i, 1);
    const pool = h >= 13 ? (r < 0.55 ? pines : broad) : (r < 0.22 ? pines : broad);
    return pool.length
      ? pool[Math.floor(hash(i, 7) * pool.length) % pool.length]
      : 0;
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
    const meshes = sp.parts.map((p) =>
      new THREE.InstancedMesh(p.geo, p.mat, counts[si]));
    let j = 0;
    for (let i = 0; i < trees.length; i++) {
      if (assign[i] !== si) continue;
      const [x, y, gz, h] = trees[i];
      q.setFromAxisAngle(up, hash(i, 2) * Math.PI * 2);
      const wj = 0.8 + hash(i, 3) * 0.45;               // width jitter
      m.compose(w2t(x, y, gz), q, new THREE.Vector3(h * wj, h, h * wj));
      // subtle per-tree tint over the leaf texture: mostly brightness variety
      // with a whisper of hue drift — the textures carry the actual color
      if (sp.pine) col.setHSL(0.35, 0.12, 0.5 + hash(i, 5) * 0.24);
      else col.setHSL(0.22 + hash(i, 4) * 0.08, 0.15, 0.55 + hash(i, 5) * 0.3);
      for (let pi = 0; pi < sp.parts.length; pi++) {
        meshes[pi].setMatrixAt(j, m);
        if (sp.parts[pi].leaf) meshes[pi].setColorAt(j, col);
      }
      j++;
    }
    for (const mesh of meshes) {
      mesh.castShadow = true;
      scene.add(mesh);
    }
  });
}

/** One-call world build: sky, lights, terrain, buildings, trees. */
export async function buildWorldScene(scene: THREE.Scene, d: World3D) {
  addSkyAndLights(scene);
  buildTerrain(scene, d);
  buildTrees(scene, d.trees, await loadTreeSpecies());
  return makeGroundSampler(d);
}
