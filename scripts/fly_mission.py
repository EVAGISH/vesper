"""Fly a ScenarioSpec: Pegasus + PX4 SITL, world built from the spec.

Usage:  /isaac-sim/python.sh scripts/fly_mission.py [scenario.json]
(defaults to the step-1 square scenario)

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

import sys

from vesper.capture import RunCapture
from vesper.record import TrajectoryWriter
from vesper.scenario import ScenarioSpec
from vesper.scenario.spec import square_scenario
from vesper.worlds.usd_stage import build_world

FPS = 30
spec = ScenarioSpec.load(sys.argv[1]) if len(sys.argv) > 1 else square_scenario(seed=0)

# --- world + vehicle ---------------------------------------------------------
timeline = omni.timeline.get_timeline_interface()
pg = PegasusInterface()
from isaacsim.core.api import World  # noqa: E402  (after app init)

pg._world = World(**pg._world_settings)
pg.load_environment(SIMULATION_ENVIRONMENTS["Default Environment"])
build_world(pg.world, spec)

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
vehicle = Multirotor(
    "/World/quadrotor",
    ROBOTS["Iris"],
    0,
    [0.0, 0.0, 0.07],
    Rotation.identity().as_quat(),
    config=config,
)

CAM_POS = np.array([16.0, -11.0, 9.0])  # elevated, clear of the building field
camera = Camera(
    prim_path="/World/overview_cam",
    position=CAM_POS,
    resolution=(1280, 720),
)

def aim_camera_at_drone(target):
    d = target - CAM_POS
    yaw = np.degrees(np.arctan2(d[1], d[0]))
    pitch = np.degrees(np.arctan2(-d[2], np.linalg.norm(d[:2])))
    camera.set_world_pose(
        CAM_POS, rot_utils.euler_angles_to_quats(np.array([0.0, pitch, yaw]), degrees=True)
    )

pg.world.reset()
camera.initialize()

# --- run artifacts: capture + scenario + trajectory --------------------------
cap = RunCapture(f"fly-{spec.world}")
spec_path = spec.save(cap.dir / "scenario.json")
traj = TrajectoryWriter(cap.dir)
cap.note(scene="pegasus iris + px4 sitl", resolution=[1280, 720],
         scenario="scenario.json", schema=1)

# --- mission subprocess (vanilla asyncio; kit's patched asyncio breaks MAVSDK)
import subprocess
mission_proc = subprocess.Popen(
    ["/isaac-sim/python.sh", "scripts/mission_waypoints.py", str(spec_path)],
)

# --- sim loop + capture ------------------------------------------------------
timeline.play()
render_dt = pg.world.get_rendering_dt()
sim_time, next_frame = 0.0, 0.0
while sim_time < spec.max_sim_s and mission_proc.poll() is None:
    pg.world.step(render=True)
    sim_time += render_dt
    if sim_time >= next_frame:
        # Pegasus vehicle state is the moving body (the /World/quadrotor xform
        # is the static spawn transform -- reading it gives a constant pose)
        pos = np.asarray(vehicle.state.position, dtype=float).reshape(-1)[:3]
        q = np.asarray(vehicle.state.attitude, dtype=float).reshape(-1)[:4]  # xyzw
        traj.append(sim_time, pos, [q[3], q[0], q[1], q[2]])
        aim_camera_at_drone(pos)
        rgba = camera.get_rgba()
        if rgba is not None and getattr(rgba, "size", 0):
            cap.add_frame(rgba)
        next_frame = sim_time + 1.0 / FPS

timeline.stop()
print(f"trajectory: {traj.close()} ({len(traj)} rows)")
video = cap.finish(fps=FPS)
print(f"wrote {video}")
app.close()
