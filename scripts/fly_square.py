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

import asyncio
import threading
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

SECONDS, FPS = 45, 30

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
    position=np.array([6.0, 6.0, 4.0]),
    orientation=rot_utils.euler_angles_to_quats(np.array([0.0, 22.0, 225.0]), degrees=True),
    resolution=(1280, 720),
)

pg.world.reset()
camera.initialize()

# --- mission thread (MAVSDK -> PX4 over udp:14540) ---------------------------
def mission():
    from mavsdk import System
    from mavsdk.offboard import OffboardError, VelocityNedYaw

    async def run():
        drone = System()
        await drone.connect(system_address="udp://:14540")
        async for state in drone.core.connection_state():
            if state.is_connected:
                break
        async for health in drone.telemetry.health():
            if health.is_armable:
                break
        await drone.action.arm()
        await drone.action.set_takeoff_altitude(3.0)
        await drone.action.takeoff()
        await asyncio.sleep(8)
        try:
            await drone.offboard.set_velocity_ned(VelocityNedYaw(0, 0, 0, 0))
            await drone.offboard.start()
            for vx, vy in [(1.5, 0), (0, 1.5), (-1.5, 0), (0, -1.5)]:
                await drone.offboard.set_velocity_ned(VelocityNedYaw(vx, vy, 0, 0))
                await asyncio.sleep(4)
            await drone.offboard.set_velocity_ned(VelocityNedYaw(0, 0, 0, 0))
            await drone.offboard.stop()
        except OffboardError as e:
            print(f"offboard failed: {e}")
        await drone.action.land()

    asyncio.run(run())

threading.Thread(target=mission, daemon=True).start()

# --- sim loop + capture ------------------------------------------------------
cap = RunCapture("fly-square")
cap.note(scene="pegasus iris + px4 sitl, takeoff-square-land", resolution=[1280, 720])

timeline.play()
t_end = time.time() + SECONDS
frame_dt, next_frame = 1.0 / FPS, 0.0
while time.time() < t_end:
    pg.world.step(render=True)
    now = time.time()
    if now >= next_frame:
        rgba = camera.get_rgba()
        if rgba is not None and getattr(rgba, "size", 0):
            cap.add_frame(rgba)
        next_frame = now + frame_dt

timeline.stop()
video = cap.finish(fps=FPS)
print(f"wrote {video}")
app.close()
