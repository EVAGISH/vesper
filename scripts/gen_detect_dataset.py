"""Render a COCO detection dataset of the BTR-80A from inside Isaac Sim.

    /isaac-sim/python.sh scripts/gen_detect_dataset.py \
        --images 3000 --out scratch/datasets/btr80 --headless --enable_cameras

Writes scratch/datasets/<name>/{train,valid,test}/*.jpg plus the
_annotations.coco.json each split needs, which is the layout rfdetr trains from
directly.

What the randomisation is for: a detector trained on one lighting rig and one
camera arc learns the rig. Every scene redraws the sky, the sun, the site, the
vehicle count and heading and the clutter; every camera redraws its own
stand-off, azimuth and look-down angle, with an occluder dropped on its line of
sight, because a vehicle behind a tree is the case the chase task has to
survive.

Throughput comes from --cams: moving prims and refitting the scene costs far
more than drawing it, so one randomised scene is photographed from several
independent camera poses in a single render tick instead of being thrown away
after one frame.

Boxes come from Replicator's bounding_box_2d_tight annotator, which bounds
visible pixels only. Pairing it with the loose annotator (whole object,
occluded parts included) gives an occlusion ratio for free, so a target that is
95% hidden is dropped rather than teaching the detector that a wheel is a BTR.

The camera matches the airframe's lens (110 deg) by default, so the training
distribution is the deployment distribution rather than a nicer version of it.
"""
from __future__ import annotations

import argparse
import json
import math
import random
import time
from pathlib import Path

from isaaclab.app import AppLauncher

REPO = Path(__file__).resolve().parents[1]

parser = argparse.ArgumentParser()
parser.add_argument("--images", type=int, default=3000, help="frames to keep")
parser.add_argument("--out", default="scratch/datasets/btr80", help="repo-relative output dir")
parser.add_argument("--sites", nargs="*", default=None,
                    help="site names under assets/, optionally name:weight "
                         "(e.g. vuhledar:6 cornell:1). Default: every built world found, "
                         "equally weighted. Each needs assets/<name>/<name>.usd and "
                         "<name>_map.npz")
parser.add_argument("--vehicle", default="btr80", help="VEHICLE_SPECS key")
parser.add_argument("--cams", type=int, default=6,
                    help="camera poses photographed per randomised scene")
parser.add_argument("--res", type=int, default=640)
parser.add_argument("--hfov", type=float, default=110.0, help="degrees; the airframe's lens")
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--max-vehicles", type=int, default=3)
parser.add_argument("--occluders", type=int, default=6, help="clutter prims per scene, max")
parser.add_argument("--clutter-clear-m", type=float, default=18.0,
                    help="no clutter this close to any camera: a tree a few metres from "
                         "a 110 deg lens is the whole frame, not an occluder")
parser.add_argument("--near", type=float, default=6.0, help="closest stand-off, m")
parser.add_argument("--far", type=float, default=80.0, help="furthest stand-off, m")
parser.add_argument("--min-box-px", type=int, default=14, help="drop boxes smaller than this")
parser.add_argument("--max-occlusion", type=float, default=0.85)
parser.add_argument("--neg-frac", type=float, default=0.08,
                    help="fraction of frames kept with no visible target")
parser.add_argument("--subframes", type=int, default=3, help="RTX ticks per scene")
parser.add_argument("--sky-every", type=int, default=4,
                    help="scenes between HDR swaps; the dome's rotation and intensity "
                         "still change every scene")
parser.add_argument("--flush-every", type=int, default=200)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.enable_cameras = True
app = AppLauncher(args).app

import carb  # noqa: E402
import numpy as np  # noqa: E402
import omni.replicator.core as rep  # noqa: E402
import torch  # noqa: E402
from PIL import Image  # noqa: E402
from pxr import Gf, UsdGeom, UsdLux  # noqa: E402

import isaacsim.core.utils.prims as prim_utils  # noqa: E402
import isaacsim.core.utils.stage as stage_utils  # noqa: E402
from isaaclab.sim import SimulationCfg, SimulationContext  # noqa: E402

from vesper.worlds.heightmap import WorldMap  # noqa: E402

settings = carb.settings.get_settings()
settings.set("/rtx-transient/resourcemanager/enableTextureStreaming", False)
settings.set("/rtx/post/aa/op", 2)
settings.set("/rtx/post/dlss/execMode", 0)
settings.set("/rtx/post/motionblur/enabled", False)

CLASS_NAME = "btr80"
SITE_SPACING_M = 20_000.0      # far enough apart that no camera can see two sites
rng = random.Random(args.seed)
np_rng = np.random.default_rng(args.seed)
out_root = (REPO / args.out).resolve()


# --------------------------------------------------------------------- assets
def discover_sites() -> list[str]:
    """Every world that has both a USD and an exported map, newest first.

    Sites are built by the environments page (web/server/app.py) or
    scripts/build_geo_world.py, so which ones exist is a property of the box,
    not of this script -- hence discovery rather than a hard-coded list.
    """
    found = []
    for d in sorted((REPO / "assets").iterdir()):
        if d.is_dir() and (d / f"{d.name}.usd").exists() and (d / f"{d.name}_map.npz").exists():
            found.append(d.name)
    return found


def sky_files() -> list[Path]:
    # Night domes are excluded: the airframe flies in daylight, and an RGB
    # detector trained on frames where the target is a black shape against
    # black ground learns nothing that transfers.
    skies = [p for p in sorted((REPO / "assets" / "skies").glob("*.hdr"))
             if "night" not in p.name.lower()]
    if not skies:
        print("[warn] no HDRs in assets/skies; run scripts/fetch_skies.py for lighting "
              "variety. Falling back to a plain coloured dome.", flush=True)
    return skies


def tree_files(site_names: list[str]) -> list[Path]:
    """Occluder prototypes, preferring the species the loaded sites already use.

    assets/vegetation/Trees holds NVIDIA's raw tree USDs, and referencing those
    drags in a fresh set of MDL materials and textures that nothing else in the
    scene shares -- minutes of first-tick asset loading for ten prims. Each
    built site ships the species it was scattered with, and those are already
    resident because the world references them, so clutter drawn from there is
    effectively free.
    """
    found = []
    for name in site_names:
        found += sorted((REPO / "assets" / name / "species").glob("*.usd"))
    return found or sorted((REPO / "assets" / "vegetation" / "Trees").glob("*.usd"))


def parse_sites(spec: list[str] | None) -> list[tuple[str, float]]:
    """[(name, weight)] from `name` or `name:weight` arguments.

    Weighting exists because the sites are not interchangeable: inference is
    expected to happen over one of them, so that site should dominate the
    training distribution while the others are there to stop the detector
    keying on any single place's ground and architecture.
    """
    out = []
    for entry in spec or discover_sites():
        name, _, w = entry.partition(":")
        out.append((name, float(w) if w else 1.0))
    return out


site_names = parse_sites(args.sites)
if not site_names:
    raise SystemExit("no built worlds found under assets/. Build one from the environments "
                     "page, or: python3 scripts/build_geo_world.py <name> --lat .. --lon ..")


# --------------------------------------------------------------------- scene
# A SimulationContext has to exist before anything is rendered. Without one the
# app comes up, the scene loads, and every annotator read blocks forever:
# nothing drives the hydra pipeline, so the first frame never completes. It is
# created first because it opens its own stage.
sim = SimulationContext(SimulationCfg(dt=1.0 / 60.0, device=args.device))
stage = stage_utils.get_current_stage()
UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
UsdGeom.SetStageMetersPerUnit(stage, 1.0)
prim_utils.create_prim("/World", "Xform")

SITES, SITE_WEIGHTS = [], []
for i, (name, weight) in enumerate(site_names):
    usd = REPO / "assets" / name / f"{name}.usd"
    npz = REPO / "assets" / name / f"{name}_map.npz"
    if not (usd.exists() and npz.exists()):
        print(f"[warn] skipping site {name}: missing {usd.name} or {npz.name}", flush=True)
        continue
    # Sites are laid out side by side on one stage rather than loaded and
    # unloaded: opening a 20 MB site USD costs far more than the slab of empty
    # space between them, and no camera can see 20 km.
    dx = i * SITE_SPACING_M
    prim_utils.create_prim(f"/World/sites/{name}", "Xform", usd_path=str(usd),
                           translation=(dx, 0.0, 0.0))
    SITES.append({"name": name, "dx": dx, "map": WorldMap(str(npz), device="cpu")})
    SITE_WEIGHTS.append(weight)
if not SITES:
    raise SystemExit("none of the requested sites are built on this machine")

sun = UsdLux.DistantLight.Define(stage, "/World/sun")
sun.CreateAngleAttr(0.53)
sun_xf = UsdGeom.Xformable(sun)
sun_rx, sun_rz = sun_xf.AddRotateXOp(), sun_xf.AddRotateZOp()
dome = UsdLux.DomeLight.Define(stage, "/World/sky")
dome_rz = UsdGeom.Xformable(dome).AddRotateZOp()
SKIES = sky_files()

from vesper.lab.pursuit_env import VEHICLE_SPECS  # noqa: E402
from vesper.worlds import vehicle as V  # noqa: E402

# A render-only twin of the vehicle the sim drives: same mesh, same fitted
# scale, same nose-on-+X frame, but no rigid body. With physics in the scene a
# rigid body's pose belongs to PhysX, and the per-frame USD writes that place
# the targets would move nothing at all.
mesh = Path(VEHICLE_SPECS[args.vehicle]["mesh"])
if not mesh.exists():
    raise SystemExit(f"{mesh} is missing. Fetch and convert the vehicle first:\n"
                     f"  python3 scripts/fetch_objaverse_vehicle.py {args.vehicle}\n"
                     "  /isaac-sim/python.sh scripts/convert_asset.py "
                     f"assets/vehicles/{args.vehicle}/{args.vehicle}.glb --yup "
                     "--collision convexHull --headless")
VEH_USD = V.write_mesh_vehicle_usd(
    mesh, mesh.parent / f"{args.vehicle}_visual.usd", length_m=V.BTR80_LENGTH_M,
    nose_yaw_deg=90.0, hull=V.BTR80_HULL, physics=False)
print(f"[vesper] dataset vehicle (render-only): {VEH_USD}", flush=True)
vehicles = []
for i in range(args.max_vehicles):
    path = f"/World/targets/v{i}"
    prim_utils.create_prim(path, "Xform", usd_path=str(VEH_USD), semantic_label=CLASS_NAME)
    vehicles.append(path)

TREES = tree_files([s['name'] for s in SITES])
occluders = []
for i in range(max(args.occluders, args.cams)):
    path = f"/World/clutter/c{i}"
    prim_utils.create_prim(path, "Xform", usd_path=str(rng.choice(TREES))) if TREES \
        else prim_utils.create_prim(path, "Cube")
    occluders.append(path)

# --- cameras: one render product each, all drawn by the same render tick
APERTURE = 20.955                       # USD's tenths-of-a-mm horizontal aperture
CAMS = []
for i in range(args.cams):
    cam = UsdGeom.Camera.Define(stage, f"/World/cams/cam{i}")
    cam.CreateHorizontalApertureAttr(APERTURE)
    cam.CreateVerticalApertureAttr(APERTURE)
    cam.CreateFocalLengthAttr(APERTURE / (2.0 * math.tan(math.radians(args.hfov) / 2.0)))
    cam.CreateClippingRangeAttr(Gf.Vec2f(0.05, 6000.0))
    xf = UsdGeom.Xformable(cam)
    rp = rep.create.render_product(f"/World/cams/cam{i}", (args.res, args.res))
    annots = {k: rep.AnnotatorRegistry.get_annotator(k)
              for k in ("rgb", "bounding_box_2d_tight", "bounding_box_2d_loose")}
    for a in annots.values():
        a.attach(rp)
    CAMS.append({"t": xf.AddTranslateOp(), "r": xf.AddRotateXYZOp(), "annots": annots})

sites_desc = ", ".join(f"{s['name']}x{w:g}" for s, w in zip(SITES, SITE_WEIGHTS))
print(f"[vesper] sites: {sites_desc} | {len(SKIES)} skies | {len(TREES)} tree species | "
      f"{args.max_vehicles} vehicle slots | {len(occluders)} clutter slots | "
      f"{args.cams} cameras/scene", flush=True)


# --------------------------------------------------------------------- helpers
_OPS: dict[str, tuple] = {}


def set_xform(prim_path, pos, yaw_deg=0.0, scale=1.0, visible=True):
    """Absolute pose on a prim, through one fixed op stack per prim.

    create_prim leaves its own translate op behind, so the op set is replaced
    outright rather than added to -- otherwise repeated AddRotateZOp calls would
    stack a new rotation onto every prim on every frame.
    """
    prim = stage.GetPrimAtPath(prim_path)
    ops = _OPS.get(prim_path)
    if ops is None:
        xf = UsdGeom.Xformable(prim)
        xf.ClearXformOpOrder()
        for attr in list(prim.GetAttributes()):
            if attr.GetName().startswith("xformOp:"):
                prim.RemoveProperty(attr.GetName())
        ops = (xf.AddTranslateOp(), xf.AddRotateZOp(), xf.AddScaleOp())
        _OPS[prim_path] = ops
    ops[0].Set(Gf.Vec3d(*[float(v) for v in pos]))
    ops[1].Set(float(yaw_deg))
    ops[2].Set(Gf.Vec3f(scale, scale, scale))
    UsdGeom.Imageable(prim).GetVisibilityAttr().Set(
        UsdGeom.Tokens.inherited if visible else UsdGeom.Tokens.invisible)


def _at(field_fn, site, x, y) -> float:
    t = torch.tensor([[float(x), float(y)]])
    return float(field_fn(t[:, 0], t[:, 1])[0])


def ground_at(site, x, y):
    return _at(site["map"].ground_at, site, x, y)


def solid_at(site, x, y):
    """Top of whatever is at (x, y): terrain, building or canopy."""
    return _at(site["map"].solid_at, site, x, y)


def drivable(site, x, y) -> bool:
    m = site["map"]
    t = torch.tensor([[float(x), float(y)]])
    return bool(m.is_drivable(t[:, 0], t[:, 1])[0])


def sample_stand(site, tries=60):
    """An open, drivable spot to park the targets on, in site-local metres."""
    half = site["map"].half_m * 0.85
    for _ in range(tries):
        x, y = np_rng.uniform(-half, half, 2)
        if drivable(site, x, y):
            return float(x), float(y)
    return 0.0, 0.0


def look_at(cam, pos, target, roll_deg=0.0):
    """USD camera convention: it looks down its own -Z with +Y up."""
    d = np.asarray(target, float) - np.asarray(pos, float)
    yaw = math.degrees(math.atan2(d[1], d[0]))
    pitch = math.degrees(math.atan2(d[2], math.hypot(d[0], d[1]) + 1e-9))
    cam["t"].Set(Gf.Vec3d(*[float(v) for v in pos]))
    cam["r"].Set(Gf.Vec3f(90.0 + pitch, float(roll_deg), yaw - 90.0))


def render_scene():
    """Tick the renderer; every camera's render product is drawn by the same tick.

    sim.render() rather than rep.orchestrator.step(): the orchestrator waits on
    a capture that never completes in this headless standalone app. Subframes
    matter because the first tick after a pose change can still carry the
    previous frame's accumulation.
    """
    for _ in range(max(2, args.subframes)):
        sim.render()


def boxes_from(data, res):
    """Our class's boxes as a list, clipped to the frame.

    A list, not a dict keyed by semanticId: every vehicle carries the same class
    label and therefore the same semanticId, so keying by it would quietly
    discard all but one APC per frame and teach the detector the rest is
    background.
    """
    rows = data.get("data") if isinstance(data, dict) else data
    info = data.get("info", {}) if isinstance(data, dict) else {}
    labels = info.get("idToLabels", {}) or {}
    out = []
    for r in rows:
        sid = int(r["semanticId"])
        name = labels.get(str(sid), labels.get(sid, {}))
        name = name.get("class", "") if isinstance(name, dict) else str(name)
        if CLASS_NAME not in str(name):
            continue
        out.append((float(np.clip(r["x_min"], 0, res)), float(np.clip(r["y_min"], 0, res)),
                    float(np.clip(r["x_max"], 0, res)), float(np.clip(r["y_max"], 0, res))))
    return out


def area(b):
    return max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])


def intersect(a, b):
    return area((max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])))


def occlusion_of(tight, loose_boxes):
    """1 - visible/whole, pairing a tight box with the loose box it sits inside.

    Matched by overlap rather than by row order: the two annotators enumerate
    instances independently, and a fully hidden vehicle drops out of the tight
    list entirely, which would shift every later pairing by one.
    """
    if not loose_boxes:
        return 0.0
    ref = max(loose_boxes, key=lambda l: intersect(tight, l))
    whole = area(ref)
    return 1.0 - (area(tight) / whole) if whole > 0 else 1.0


# --------------------------------------------------------------------- splits
SPLITS = (("train", 0.85), ("valid", 0.12), ("test", 0.03))
for name, _ in SPLITS:
    (out_root / name).mkdir(parents=True, exist_ok=True)
coco = {name: {"images": [], "annotations": [],
               "categories": [{"id": 1, "name": CLASS_NAME, "supercategory": "vehicle"}]}
        for name, _ in SPLITS}
next_ann = {name: 1 for name, _ in SPLITS}


def split_for(i):
    r = (i * 2654435761 % 1000) / 1000.0        # stable, no shuffle bookkeeping
    acc = 0.0
    for name, frac in SPLITS:
        acc += frac
        if r < acc:
            return name
    return SPLITS[0][0]


def flush():
    for name, _ in SPLITS:
        (out_root / name / "_annotations.coco.json").write_text(json.dumps(coco[name]))


# --------------------------------------------------------------------- loop
print("[vesper] priming the renderer (the first tick compiles shaders for the vehicle "
      "materials and the sky dome; this can take minutes)", flush=True)
sim.reset()
t_prime = time.time()
for _ in range(8):
    sim.render()
print(f"[vesper] renderer ready in {time.time() - t_prime:.0f}s", flush=True)

kept, scenes, negatives, skipped = 0, 0, 0, 0
t_render = t_readback = t_place = 0.0
t_start = time.time()
print(f"[vesper] rendering {args.images} frames -> {out_root}", flush=True)
while kept < args.images:
    scenes += 1
    t_place0 = time.time()
    site = rng.choices(SITES, weights=SITE_WEIGHTS, k=1)[0]
    dx = site["dx"]

    # --- lighting. The HDRs are ~100 MB each and setting the texture forces a
    # reload, which costs more than the render; swapping on a period and
    # spinning/redimming the dome in between buys the variety far cheaper.
    if SKIES:
        if scenes == 1 or scenes % args.sky_every == 0:
            dome.CreateTextureFileAttr().Set(str(rng.choice(SKIES)))
            dome.CreateTextureFormatAttr().Set("latlong")
    else:
        dome.CreateColorAttr(Gf.Vec3f(*np_rng.uniform(0.35, 0.95, 3)))
    dome.CreateIntensityAttr(float(np_rng.uniform(600.0, 2200.0)))
    dome_rz.Set(float(np_rng.uniform(0, 360)))
    sun.CreateIntensityAttr(float(np_rng.uniform(1500.0, 5000.0)))
    sun.CreateColorAttr(Gf.Vec3f(1.0, float(np_rng.uniform(0.88, 1.0)),
                                 float(np_rng.uniform(0.75, 1.0))))
    sun_rx.Set(float(-np_rng.uniform(12.0, 82.0)))
    sun_rz.Set(float(np_rng.uniform(0, 360)))

    # --- targets, in site-local metres then offset onto this site's slab
    stand = sample_stand(site)
    n_veh = rng.randint(1, args.max_vehicles)
    placed = []
    for i, path in enumerate(vehicles):
        if i >= n_veh:
            set_xform(path, (0.0, 0.0, -5000.0), visible=False)
            continue
        x, y = stand
        for _ in range(20):
            cx = stand[0] + float(np_rng.normal(0, 9.0))
            cy = stand[1] + float(np_rng.normal(0, 9.0))
            if drivable(site, cx, cy):
                x, y = cx, cy
                break
        set_xform(path, (x + dx, y, ground_at(site, x, y) + 0.02),
                  yaw_deg=float(np_rng.uniform(0, 360)))
        placed.append((x, y))

    # --- camera poses: each on its own sphere segment around a chosen target
    shots = []
    for ci, cam in enumerate(CAMS):
        aim = placed[rng.randrange(len(placed))]
        aim_z = ground_at(site, *aim) + 1.2
        # Resample until the lens is over open ground. Lifting the camera clear
        # of whatever is beneath it is not enough on a wooded site: standing it
        # 2 m above a tree crown or a roof fills the frame with bark or tiles
        # and leaves a technically-correct box on a target nobody can see.
        for _ in range(12):
            dist = float(math.exp(np_rng.uniform(math.log(args.near), math.log(args.far))))
            az = float(np_rng.uniform(0, 2 * math.pi))
            el = math.radians(float(np_rng.uniform(3.0, 72.0)))
            cx = aim[0] + dist * math.cos(el) * math.cos(az)
            cy = aim[1] + dist * math.cos(el) * math.sin(az)
            cz = aim_z + dist * math.sin(el)
            if solid_at(site, cx, cy) - ground_at(site, cx, cy) < 1.5:
                break
        cz = max(cz, ground_at(site, cx, cy) + 1.5)
        jitter = dist * 0.16                    # target off dead centre
        look_at(cam, (cx + dx, cy, cz),
                (aim[0] + dx + float(np_rng.normal(0, jitter)),
                 aim[1] + float(np_rng.normal(0, jitter)),
                 aim_z + float(np_rng.normal(0, jitter * 0.4))),
                roll_deg=float(np_rng.normal(0, 4.0)))
        shots.append({"cam": cam, "eye": (cx, cy), "aim": aim})

    # --- clutter: one occluder near each camera's target, the rest scattered.
    # It is placed on the far half of the line of sight and offset sideways by
    # about a vehicle's width, so it cuts into the silhouette. Put it near the
    # lens instead and it stops being an occluder and becomes the photograph.
    for i, path in enumerate(occluders):
        if i < len(shots) and rng.random() < 0.6:
            sh = shots[i]
            ex, ey = sh["eye"]
            ax, ay = sh["aim"]
            span = math.hypot(ax - ex, ay - ey)
            # never nearer the camera than clutter_clear_m, and always past halfway
            t_lo = max(0.5, min(0.9, args.clutter_clear_m / max(span, 1e-6)))
            t = float(np_rng.uniform(t_lo, 0.92))
            ux, uy = (ax - ex) / max(span, 1e-6), (ay - ey) / max(span, 1e-6)
            lat = float(np_rng.choice([-1.0, 1.0]) * np_rng.uniform(1.5, 5.0))
            ox = ex + t * (ax - ex) - uy * lat
            oy = ey + t * (ay - ey) + ux * lat
        else:
            ox = stand[0] + float(np_rng.normal(0, 26.0))
            oy = stand[1] + float(np_rng.normal(0, 26.0))
        # A tree that landed on top of any other camera would ruin that frame too.
        too_close = any(math.hypot(ox - sh["eye"][0], oy - sh["eye"][1]) < args.clutter_clear_m
                        for sh in shots)
        set_xform(path, (ox + dx, oy, ground_at(site, ox, oy)),
                  yaw_deg=float(np_rng.uniform(0, 360)),
                  scale=float(np_rng.uniform(0.6, 1.4)),
                  visible=not too_close)

    t_place += time.time() - t_place0
    t_scene = time.time()
    render_scene()
    t_render += time.time() - t_scene

    t_read = time.time()
    for s in shots:
        if kept >= args.images:
            break
        annots = s["cam"]["annots"]
        rgb = annots["rgb"].get_data()
        tb = boxes_from(annots["bounding_box_2d_tight"].get_data(), args.res)
        lb = boxes_from(annots["bounding_box_2d_loose"].get_data(), args.res)

        keep = []
        for box in tb:
            w, h = box[2] - box[0], box[3] - box[1]
            if w < args.min_box_px or h < args.min_box_px:
                continue
            if occlusion_of(box, lb) > args.max_occlusion:
                continue
            keep.append((box, w, h))

        if not keep:
            # A bounded share of empty frames teaches the detector what is not a
            # BTR. Past that share the frame is thrown away -- already rendered,
            # so every discard is pure cost; --neg-frac trades dataset balance
            # against throughput.
            if negatives >= args.neg_frac * args.images:
                skipped += 1
                continue
            negatives += 1

        split = split_for(kept)
        name = f"{CLASS_NAME}_{kept:06d}.jpg"
        img = rgb["data"] if isinstance(rgb, dict) else rgb
        Image.fromarray(np.asarray(img)[..., :3]).save(out_root / split / name, quality=92)
        coco[split]["images"].append(
            {"id": kept + 1, "file_name": name, "width": args.res, "height": args.res,
             "vesper_site": site["name"]})
        for box, w, h in keep:
            coco[split]["annotations"].append(
                {"id": next_ann[split], "image_id": kept + 1, "category_id": 1,
                 "bbox": [round(box[0], 2), round(box[1], 2), round(w, 2), round(h, 2)],
                 "area": round(w * h, 2), "iscrowd": 0})
            next_ann[split] += 1
        kept += 1

        if kept % args.flush_every == 0:
            flush()
            rate = kept / max(1e-9, time.time() - t_start)
            eta = (args.images - kept) / max(rate, 1e-9) / 60
            print(f"  {kept}/{args.images} frames | {rate:.1f} img/s | "
                  f"{scenes} scenes | {negatives} empty | ETA {eta:.0f} min", flush=True)
    t_readback += time.time() - t_read

flush()
secs = time.time() - t_start
total_ann = sum(len(coco[n]["annotations"]) for n, _ in SPLITS)
print(f"[vesper] wrote {kept} images, {total_ann} boxes, {negatives} negatives "
      f"from {scenes} scenes in {secs / 60:.1f} min -> {out_root}", flush=True)
for name, _ in SPLITS:
    print(f"   {name}: {len(coco[name]['images'])} images, "
          f"{len(coco[name]['annotations'])} boxes", flush=True)
# Where the time actually went. Placement and render are paid once per scene;
# readback is paid once per camera, so the split says whether raising --cams
# buys anything or just moves the cost.
print(f"[vesper] {kept / secs:.2f} img/s | {secs / max(scenes, 1):.2f} s/scene "
      f"({kept / max(scenes, 1):.1f} kept per scene, {skipped} discarded) | "
      f"place {t_place:.0f}s render {t_render:.0f}s readback {t_readback:.0f}s "
      f"other {secs - t_place - t_render - t_readback:.0f}s", flush=True)
app.close()
