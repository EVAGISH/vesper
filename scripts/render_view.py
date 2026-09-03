"""Render still frames of a scenario's world from given camera poses -- no PX4, no flight.

    /isaac-sim/python.sh scripts/render_view.py sloviansk0.json --view 0 0 1.7 0 0 0 --view 50 -30 60 0 40 200

Each --view is X Y Z PITCH YAW ROLL... simplified: X Y Z pitch_deg yaw_deg (camera looks
along +x rotated by yaw, pitched down by pitch). Writes runs/<id>/view_<i>.png. The fast
way to inspect a world's scale, placement and materials before spending a flight on it.
"""
from isaacsim import SimulationApp

app = SimulationApp({"headless": True})

import argparse
import sys
import time

import carb
import numpy as np
import isaacsim.core.utils.numpy.rotations as rot_utils
import omni.usd
from isaacsim.core.api import World
from isaacsim.sensors.camera import Camera
from PIL import Image
from pxr import UsdLux

from vesper.capture import RunCapture
from vesper.scenario import ScenarioSpec
from vesper.worlds.usd_stage import build_world

carb.settings.get_settings().set("/rtx-transient/resourcemanager/enableTextureStreaming", False)

ap = argparse.ArgumentParser()
ap.add_argument("scenario")
ap.add_argument("--view", type=float, nargs=5, action="append", metavar=("X", "Y", "Z", "PITCH", "YAW"), required=True)
ap.add_argument("--hfov", type=float, default=90.0)
ap.add_argument("--settle", type=int, default=30, help="render steps before capturing each view (RTX convergence)")
a = ap.parse_args()

spec = ScenarioSpec.load(a.scenario)
world = World(stage_units_in_meters=1.0)
build_world(world, spec)
stage = omni.usd.get_context().get_stage()
if not spec.terrain:
    world.scene.add_default_ground_plane()
if spec.sky_hdr or not stage.GetPrimAtPath("/World/terrain/sky"):
    UsdLux.DomeLight.Define(stage, "/World/dome_light").CreateIntensityAttr(600.0)
cam = Camera(prim_path="/World/view_cam", position=np.array([0.0, 0.0, 2.0]), resolution=(1280, 720))
world.reset()
cam.initialize()
ap_ = cam.get_horizontal_aperture()
cam.set_focal_length(ap_ / (2.0 * np.tan(np.radians(a.hfov) / 2.0)))
cam.set_clipping_range(0.05, 5000.0)

run = RunCapture(f"view-{spec.world}")
for i, (x, y, z, pitch, yaw) in enumerate(a.view):
    cam.set_world_pose(np.array([x, y, z]), rot_utils.euler_angles_to_quats(np.array([0.0, pitch, yaw]), degrees=True))
    for _ in range(a.settle):
        world.render()
    rgba = cam.get_rgba()
    if rgba is None or not getattr(rgba, "size", 0):
        print(f"view {i}: no image", flush=True); continue
    out = run.dir / f"view_{i}.png"
    Image.fromarray(np.asarray(rgba)[..., :3].astype(np.uint8)).save(out)
    print(f"view {i} ({x},{y},{z} pitch {pitch} yaw {yaw}) -> {out}", flush=True)
run.note(kind="views", views=a.view)
run.finish()
app.close()
