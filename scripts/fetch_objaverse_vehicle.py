"""Fetch a ground-vehicle mesh from Objaverse and stage it for conversion.

Objaverse mirrors ~800k Sketchfab models on HuggingFace with no credentials and
no API key, which is the only reason this is scriptable at all: Sketchfab's own
/download endpoint is a 401 without an OAuth token, and every other free
catalogue (CGTrader, Free3D, GrabCAD) is behind a login.

    python3 scripts/fetch_objaverse_vehicle.py btr80

writes assets/vehicles/<name>/<name>.glb plus ATTRIBUTION.md (the models are
CC-BY: the credit file is a licence obligation, not decoration). Convert it in
the container afterwards:

    /isaac-sim/python.sh scripts/convert_asset.py \
        assets/vehicles/btr80/btr80.glb --yup --collision convexHull --headless

`--uid <32-hex>` fetches any other Objaverse object. That needs the 20 MB
object-paths index to resolve a uid to its shard, so the registered vehicles
below carry their resolved path and skip the index entirely.
"""
import argparse
import gzip
import json
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
HF = "https://huggingface.co/datasets/allenai/objaverse/resolve/main"

# Geometry is recorded here rather than measured at convert time because these
# numbers were read off the mesh once (extent, which end the bow is on) and are
# properties of the specific model, not of the pipeline. nose_yaw_deg is the
# rotation that puts the bow on +X *after* convert_asset.py --yup: glTF is Y-up
# with this model's bow at +Z, --yup rotates +90 deg about X (so +Z -> -Y), and
# +90 deg about Z carries -Y to +X.
VEHICLES = {
    "btr80": dict(
        uid="5e4b7a1b18c04ac68fcd76036feebfc8",
        path="glbs/000-086/5e4b7a1b18c04ac68fcd76036feebfc8.glb",
        title="BTR 80A",
        author="Alexandr Zhilkin",
        author_url="https://sketchfab.com/allexandr007",
        source="https://sketchfab.com/3d-models/5e4b7a1b18c04ac68fcd76036feebfc8",
        licence="CC BY 4.0 (https://creativecommons.org/licenses/by/4.0/)",
        tris=70585,
        note="Soviet 8x8 APC, 30 mm turret variant. Six 1024^2 camo texture sets.",
    ),
}


def resolve_path(uid: str) -> str:
    """Shard path for an arbitrary uid, via Objaverse's 20 MB path index."""
    cache = ROOT / "assets" / "vehicles" / "objaverse-paths.json.gz"
    if not cache.exists():
        cache.parent.mkdir(parents=True, exist_ok=True)
        print(f"fetching object index -> {cache} (~20 MB, one time)", flush=True)
        urllib.request.urlretrieve(f"{HF}/object-paths.json.gz", cache)
    paths = json.load(gzip.open(cache))
    if uid not in paths:
        sys.exit(f"uid {uid} is not in Objaverse")
    return paths[uid]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("name", nargs="?", default="btr80",
                    help=f"registered vehicle ({', '.join(VEHICLES)}) or a name for --uid")
    ap.add_argument("--uid", help="any Objaverse uid, for a vehicle not in the registry")
    ap.add_argument("--force", action="store_true", help="re-download over an existing glb")
    args = ap.parse_args()

    spec = dict(VEHICLES.get(args.name, {}))
    if args.uid:
        spec.update(uid=args.uid, path=resolve_path(args.uid))
    if not spec:
        sys.exit(f"unknown vehicle {args.name!r}; registered: {', '.join(VEHICLES)}"
                 " (or pass --uid)")

    outdir = ROOT / "assets" / "vehicles" / args.name
    outdir.mkdir(parents=True, exist_ok=True)
    glb = outdir / f"{args.name}.glb"
    if glb.exists() and not args.force:
        print(f"{glb} already present ({glb.stat().st_size / 1e6:.1f} MB); --force to refetch")
    else:
        url = f"{HF}/{spec['path']}"
        print(f"downloading {url}", flush=True)
        urllib.request.urlretrieve(url, glb)
        print(f"wrote {glb} ({glb.stat().st_size / 1e6:.1f} MB)")

    (outdir / "ATTRIBUTION.md").write_text(
        f"# {spec.get('title', args.name)}\n\n"
        f"- Author: {spec.get('author', 'unknown')} {spec.get('author_url', '')}\n"
        f"- Source: {spec.get('source', '')}\n"
        f"- Licence: {spec.get('licence', 'see source')}\n"
        f"- Objaverse uid: {spec['uid']}\n"
        f"- Triangles: {spec.get('tris', '?')}\n\n"
        f"{spec.get('note', '')}\n\n"
        "Redistributed under CC BY: this file is the required attribution and must\n"
        "travel with any build, render or dataset derived from the model.\n")
    print(f"wrote {outdir / 'ATTRIBUTION.md'}")
    print("\nnext, in the container:\n"
          f"  /isaac-sim/python.sh scripts/convert_asset.py {glb.relative_to(ROOT)} "
          "--yup --collision convexHull --headless")


if __name__ == "__main__":
    main()
