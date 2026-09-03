"""Prepare an imported world USD (Unreal/Blender export or convert_asset.py output).

    python3 scripts/prepare_world.py assets/medieval/medieval.usda --spawn 0 0

Runs on the Mac (usd-core) or in the container (Isaac's pxr). Writes
<stem>_world.usda next to the source (meters, Z-up, static colliders on every mesh),
prints bounds and the ground height under the spawn point, and emits the `terrain`
block to paste into a ScenarioSpec (translation moves the spawn to the origin at z=0).
"""
import argparse
import json

from vesper.worlds.prepare import ground_height, write_wrapper

parser = argparse.ArgumentParser()
parser.add_argument("usd")
parser.add_argument("--spawn", type=float, nargs=2, default=(0.0, 0.0), metavar=("X", "Y"),
                    help="spawn point in the world's own (meter, Z-up) coordinates")
parser.add_argument("--no-collision", action="store_true")
args = parser.parse_args()

rep = write_wrapper(args.usd, collision=not args.no_collision)
print(f"wrapper: {rep.wrapper}")
print(f"source units: {rep.meters_per_unit} m/unit, up axis {rep.up_axis}; "
      f"{rep.meshes} meshes, {rep.instancers} instancers ({rep.instances} instances), {rep.collided} colliders")
print(f"bounds (m): {rep.bounds_min_m} -> {rep.bounds_max_m}")
x, y = args.spawn
z = ground_height(rep.wrapper, x, y)
if z is None:
    print(f"no ground under spawn ({x}, {y}) -- pick a point inside the bounds")
    raise SystemExit(1)
print(f"ground at spawn ({x}, {y}): z = {z:.3f}")
terrain = {"usd": rep.wrapper, "translation": [-x, -y, -z], "rotation_xyz_deg": [0.0, 0.0, 0.0], "scale": 1.0}
print("terrain:", json.dumps(terrain))
