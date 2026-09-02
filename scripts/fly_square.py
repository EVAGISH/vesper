"""Step 1: Pegasus + PX4 SITL scripted flight -- takeoff, square, land.

A Pegasus Iris multirotor with the real PX4 autopilot in the loop
(auto-launched over MAVLink lockstep). A MAVSDK thread commands the mission
while the sim steps; a fixed camera records to runs/<id>/overview.mp4.

Run inside the container:
    /isaac-sim/python.sh scripts/fly_square.py

First draft against Pegasus v5.1.0 -- expect on-box iteration.
"""
from isaacsim import SimulationApp

app = SimulationApp({"headless": True})

import time

import numpy as np
import isaacsim.core.utils.numpy.rotations as rot_utils
import omni.timeline
import omni.usd
from isaacsim.sensors.camera import Camera
from pxr import UsdLux
from scipy.spatial.transform import Rotation

from pegasus.simulator.logic.backends.px4_mavlink_backend import (
    PX4MavlinkBackend,
    PX4MavlinkBackendConfig,
)
from pegasus.simulator.logic.interface.pegasus_interface import PegasusInterface
from pegasus.simulator.logic.vehicles.multirotor import Multirotor, MultirotorConfig
from pegasus.simulator.params import ROBOTS, SIMULATION_ENVIRONMENTS

from vesper.capture import RunCapture

FPS = 30
MAX_SIM_SECONDS = 150.0   # hard stop; mission signals completion earlier

# --- world + vehicle ---------------------------------------------------------
timeline = omni.timeline.get_timeline_interface()
pg = PegasusInterface()
from isaacsim.core.api import World  # noqa: E402  (after app init)

pg._world = World(**pg._world_settings)
pg.load_environment(SIMULATION_ENVIRONMENTS["Default Environment"])

stage = omni.usd.get_context().get_stage()
UsdLux.DomeLight.Define(stage, "/World/dome_light").CreateIntensityAttr(1500.0)
UsdLux.DistantLight.Define(stage, "/World/sun").CreateIntensityAttr(3000.0)

mavlink_config = PX4MavlinkBackendConfig({
    "vehicle_id": 0,
    "px4_autolaunch": True,
    "px4_dir": "/px4/PX4-Autopilot",
})
config = MultirotorConfig()
config.backends = [PX4MavlinkBackend(mavlink_config)]
Multirotor(
    "/World/quadrotor",
    ROBOTS["Iris"],
    0,
    [0.0, 0.0, 0.07],
    Rotation.identity().as_quat(),
    config=config,
)

camera = Camera(
    prim_path="/World/overview_cam",
    position=np.array([9.0, 7.0, 4.5]),
    orientation=rot_utils.euler_angles_to_quats(np.array([0.0, 8.0, 208.0]), degrees=True),
    resolution=(1280, 720),
)

pg.world.reset()
camera.initialize()

# --- mission subprocess (vanilla asyncio; kit's patched asyncio breaks MAVSDK)
import subprocess
mission_proc = subprocess.Popen(
    ["/isaac-sim/python.sh", "scripts/mission_square.py"],
)

# --- sim loop + capture ------------------------------------------------------
cap = RunCapture("fly-square")
cap.note(scene="pegasus iris + px4 sitl, takeoff-square-land", resolution=[1280, 720])

timeline.play()
render_dt = pg.world.get_rendering_dt()
sim_time, next_frame = 0.0, 0.0
while sim_time < MAX_SIM_SECONDS and mission_proc.poll() is None:
    pg.world.step(render=True)
    sim_time += render_dt
    if sim_time >= next_frame:
        rgba = camera.get_rgba()
        if rgba is not None and getattr(rgba, "size", 0):
            cap.add_frame(rgba)
        next_frame = sim_time + 1.0 / FPS

timeline.stop()
video = cap.finish(fps=FPS)
print(f"wrote {video}")
app.close()
