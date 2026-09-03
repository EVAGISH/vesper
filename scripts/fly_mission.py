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

import carb
# Load textures up front instead of streaming them during flight: Kit's asset streamer
# hit a mutex assertion mid-run on a 21k-tree geo world (crash after 24 min).
_settings = carb.settings.get_settings()
_settings.set("/rtx-transient/resourcemanager/enableTextureStreaming", False)
# RTX defaults to temporal AA, which accumulates across frames. At 18 m/s the
# ground smeared into streaks -- the same world rendered as a static still was
# sharp, which is what gives the motion away. FXAA is spatial only, so a moving
# camera keeps the texture detail it actually samples.
_settings.set("/rtx/post/aa/op", 2)
_settings.set("/rtx/post/dlss/execMode", 0)
# Motion blur is the actual cause of the smeared ground: a static still from the
# same camera height is sharp, and in flight the drone (which is stationary
# relative to the chase camera) stays crisp while only the ground streaks. That
# is motion blur, not texture resolution. A camera sensor wants the instantaneous
# image, so turn it off.
_settings.set("/rtx/post/motionblur/enabled", False)
_settings.set("/rtx/post/motionblur/maxBlurDiameterFraction", 0.0)


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
if spec.terrain is None:
    pg.load_environment(SIMULATION_ENVIRONMENTS["Default Environment"])
build_world(pg.world, spec)  # terrain reference (if any) + prism buildings

stage = omni.usd.get_context().get_stage()
# Only light the scene if the world does not light itself. A geo world ships a
# sun and sky tuned to its ground albedo (1700 / 480); adding a second 3000 sun
# and 1000 dome on top roughly doubled the exposure and washed the aerial
# imagery out to flat pale green.
_world_lights = [pr for pr in stage.Traverse()
                 if pr.GetPath().pathString.startswith("/World/terrain")
                 and "Light" in pr.GetTypeName()]
if _world_lights:
    print(f"lighting: using the world's own ({', '.join(pr.GetName() for pr in _world_lights)})", flush=True)
else:
    UsdLux.DomeLight.Define(stage, "/World/dome_light").CreateIntensityAttr(1000.0)
    UsdLux.DistantLight.Define(stage, "/World/sun").CreateIntensityAttr(3000.0)
    print("lighting: added default sun + dome", flush=True)

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

CAM_POS = np.array(spec.overview_cam or [-12.0, 7.0, 22.0])  # default: high vantage west of the corridor
camera = Camera(
    prim_path="/World/overview_cam",
    position=CAM_POS,
    resolution=(1280, 720),
)

# The FPV camera is a real strapdown sensor: its prim is a CHILD of the moving
# rigid body, so USD transform inheritance carries the airframe's pose into the
# sensor every step. Nothing repositions it by hand. (/World/quadrotor itself is
# only the static spawn transform, which is why this has to hunt for the body.)
def _find_body_prim():
    from pxr import UsdPhysics
    stage = omni.usd.get_context().get_stage()
    best = None
    for prim in stage.Traverse():
        path = prim.GetPath().pathString
        if not path.startswith("/World/quadrotor/"):
            continue
        if prim.HasAPI(UsdPhysics.RigidBodyAPI):
            # prefer the shallowest rigid body (the airframe, not a rotor)
            if best is None or path.count("/") < best.count("/"):
                best = path
    return best

_body = _find_body_prim()
if _body is None:
    raise SystemExit("could not find the vehicle rigid body under /World/quadrotor")
print(f"fpv camera parented to rigid body: {_body}", flush=True)
fpv_camera = Camera(prim_path=f"{_body}/fpv_cam", resolution=(1280, 720))

def pose_chase_camera(pos, q_wxyz, vel):
    """Follow from 9 m behind and 4 m above along the drone's travel direction
    (falls back to heading when hovering); look at the drone."""
    r_body = Rotation.from_quat([q_wxyz[1], q_wxyz[2], q_wxyz[3], q_wxyz[0]])
    fwd = np.array(vel[:2], dtype=float)
    if np.linalg.norm(fwd) < 0.5:
        fwd = r_body.apply(np.array([1.0, 0.0, 0.0]))[:2]
    fwd = fwd / max(np.linalg.norm(fwd), 1e-6)
    cam = pos - 9.0 * np.array([fwd[0], fwd[1], 0.0]) + np.array([0.0, 0.0, 4.0])
    d = pos - cam
    yaw = np.degrees(np.arctan2(d[1], d[0]))
    pitch = np.degrees(np.arctan2(-d[2], np.linalg.norm(d[:2])))
    camera.set_world_pose(cam, rot_utils.euler_angles_to_quats(np.array([0.0, pitch, yaw]), degrees=True))

def aim_camera_at_drone(target):
    d = target - CAM_POS
    yaw = np.degrees(np.arctan2(d[1], d[0]))
    pitch = np.degrees(np.arctan2(-d[2], np.linalg.norm(d[:2])))
    camera.set_world_pose(
        CAM_POS, rot_utils.euler_angles_to_quats(np.array([0.0, pitch, yaw]), degrees=True)
    )

pg.world.reset()
camera.initialize()
fpv_camera.initialize()
# fixed mount: 12 cm forward, 2 cm down, pitched 25 deg down. Set once; the body carries it.
_q = Rotation.from_euler("y", 25, degrees=True).as_quat()          # xyzw
fpv_camera.set_local_pose(np.array([0.12, 0.0, -0.02]),
                          np.array([_q[3], _q[0], _q[1], _q[2]]), camera_axes="world")
# lenses: FPV ~120 deg horizontal (action-cam wide angle); overview ~65 deg. Both see 3 km.
FPV_HFOV_DEG, OVERVIEW_HFOV_DEG = 120.0, 65.0
def set_hfov(cam, hfov_deg):
    ap = cam.get_horizontal_aperture()
    cam.set_focal_length(ap / (2.0 * np.tan(np.radians(hfov_deg) / 2.0)))
    cam.set_clipping_range(0.05, 3000.0)
set_hfov(fpv_camera, FPV_HFOV_DEG)
set_hfov(camera, OVERVIEW_HFOV_DEG)
print(f"hfov: fpv {np.degrees(fpv_camera.get_horizontal_fov()):.0f} deg, overview {np.degrees(camera.get_horizontal_fov()):.0f} deg", flush=True)

# --- run artifacts: capture + scenario + trajectory --------------------------
cap = RunCapture(f"fly-{spec.world}")
spec_path = spec.save(cap.dir / "scenario.json")
traj = TrajectoryWriter(cap.dir)
cap.note(scene="pegasus iris + px4 sitl", resolution=[1280, 720],
         scenario="scenario.json", schema=1, streams={"overview": "chase" if spec.chase_cam else "fixed", "fpv": "120deg"})

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
    want_frame = sim_time + render_dt >= next_frame
    pg.world.step(render=want_frame)          # physics every step, RTX only when a frame is due
    sim_time += render_dt
    if want_frame:
        # Pegasus vehicle state is the moving body (the /World/quadrotor xform
        # is the static spawn transform -- reading it gives a constant pose)
        pos = np.asarray(vehicle.state.position, dtype=float).reshape(-1)[:3]
        q = np.asarray(vehicle.state.attitude, dtype=float).reshape(-1)[:4]  # xyzw
        q_wxyz = [q[3], q[0], q[1], q[2]]
        traj.append(sim_time, pos, q_wxyz)
        if spec.chase_cam:
            pose_chase_camera(pos, q_wxyz, np.asarray(vehicle.state.linear_velocity, dtype=float).reshape(-1)[:3])
        else:
            aim_camera_at_drone(pos)
        rgba = camera.get_rgba()
        if rgba is not None and getattr(rgba, "size", 0):
            cap.add_frame(rgba)
        rgba_fpv = fpv_camera.get_rgba()
        if rgba_fpv is not None and getattr(rgba_fpv, "size", 0):
            cap.add_frame(rgba_fpv, stream="fpv")
        next_frame = sim_time + 1.0 / FPS

timeline.stop()
print(f"trajectory: {traj.close()} ({len(traj)} rows)")
video = cap.finish(fps=FPS)
print(f"wrote {video}")
app.close()
