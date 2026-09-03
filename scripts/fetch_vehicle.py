"""Bring a custom ground-vehicle model into the pursuit task.

The pursuit target defaults to NVIDIA's stock forklift prop, with a generated
utility-cart proxy as the offline fallback (see vesper.lab.pursuit_env). To use
your own model instead, point this at a glTF/GLB (CC0/CC-BY from Poly Pizza,
Quaternius, a Sketchfab export, ...) or a local file; it stages the source and,
in the container, converts to USD via Isaac Lab's MeshConverter.

    # download + convert (run in the Isaac container so pxr/MeshConverter exist):
    /isaac-sim/python.sh scripts/fetch_vehicle.py --url https://example/van.glb --name van
    # or convert a file you already have:
    /isaac-sim/python.sh scripts/fetch_vehicle.py --file assets/vehicles/van.glb --name van

Then:  export VESPER_VEHICLE=$(pwd)/assets/vehicles/van/van.usd
       (or pass --vehicle <path> to train_pursuit.py / fly_pursuit.py)
"""
import argparse
import subprocess
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser()
parser.add_argument("--url", help="glTF/GLB URL to download")
parser.add_argument("--file", help="local glTF/GLB already on disk")
parser.add_argument("--name", default="vehicle")
parser.add_argument("--no-yup", action="store_true", help="source is already Z-up")
args = parser.parse_args()

outdir = ROOT / "assets" / "vehicles" / args.name
outdir.mkdir(parents=True, exist_ok=True)

if args.file:
    src = Path(args.file)
elif args.url:
    ext = ".glb" if ".glb" in args.url.lower() else ".gltf"
    src = outdir / f"{args.name}{ext}"
    print(f"downloading {args.url} -> {src}", flush=True)
    urllib.request.urlretrieve(args.url, src)
else:
    print("give --url or --file", file=sys.stderr)
    sys.exit(2)

usd = outdir / f"{args.name}.usd"
cmd = [sys.executable, str(ROOT / "scripts" / "convert_asset.py"), str(src), str(usd),
       "--collision", "convexHull"]
if not args.no_yup:
    cmd.append("--yup")
print("convert:", " ".join(cmd), flush=True)
subprocess.run(cmd, check=True)
print(f"\nOK. Set: export VESPER_VEHICLE={usd}", flush=True)
print("The drone will then pursue this vehicle instead of the default forklift.", flush=True)
