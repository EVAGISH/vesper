"""Step 0 smoke test: basic terrain + a quadrotor + an RTX camera, headless.

Nobody is flying -- the drone is dropped from 2m so physics visibly runs.
Output: runs/<id>/overview.mp4 (+ S3 sync if VESPER_RUNS_BUCKET is set).

Run inside the container on the GPU box:
    /isaac-sim/python.sh scripts/smoke_render.py

Written against the Isaac Sim 5.1 python API; expect to touch up import paths
on first real boot -- that is what Step 0 is for.
"""
from isaacsim import SimulationApp

app = SimulationApp({"headless": True})

import numpy as np
import isaacsim.core.utils.numpy.rotations as rot_utils
from isaacsim.core.api import World
from isaacsim.core.utils.stage import add_reference_to_stage
from isaacsim.sensors.camera import Camera
from isaacsim.storage.native import get_assets_root_path
from pxr import UsdLux
import omni.usd

from vesper.capture import RunCapture

DRONE_USD = "/Isaac/Robots/Crazyflie/cf2x.usd"  # stock asset; ours comes later
SECONDS, FPS, PHYSICS_HZ = 5, 30, 60

world = World(physics_dt=1.0 / PHYSICS_HZ, rendering_dt=1.0 / FPS)
world.scene.add_default_ground_plane()

stage = omni.usd.get_context().get_stage()
dome = UsdLux.DomeLight.Define(stage, "/World/dome_light")
dome.CreateIntensityAttr(1500.0)
sun = UsdLux.DistantLight.Define(stage, "/World/sun")
sun.CreateIntensityAttr(3000.0)

assets_root = get_assets_root_path()
add_reference_to_stage(assets_root + DRONE_USD, "/World/drone")

from isaacsim.core.prims import XFormPrim
XFormPrim("/World/drone").set_world_poses(positions=np.array([[0.0, 0.0, 1.0]]))

# Crazyflie is ~10cm across: camera close in, pitched down, yawed toward origin
camera = Camera(
    prim_path="/World/overview_cam",
    position=np.array([1.1, 1.1, 0.9]),
    orientation=rot_utils.euler_angles_to_quats(np.array([0.0, 25.0, 225.0]), degrees=True),
    resolution=(1280, 720),
)

world.reset()
camera.initialize()

cap = RunCapture("smoke")
cap.note(scene="ground+crazyflie drop", resolution=[1280, 720])

steps_per_frame = PHYSICS_HZ // FPS
for frame in range(SECONDS * FPS):
    for _ in range(steps_per_frame):
        world.step(render=False)
    world.render()
    rgba = camera.get_rgba()
    if rgba is not None and rgba.size:
        cap.add_frame(rgba)

video = cap.finish(fps=FPS)
print(f"wrote {video}")
app.close()
