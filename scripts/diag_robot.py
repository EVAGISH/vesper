"""Does the airframe have the same USD-vs-PhysX split as the tanks?

The drone is an Isaac Lab Articulation under /World/envs/env_0/Robot, which is
the hierarchy that is supposed to publish its pose to the renderer. That
assumption is exactly what made the tank bug take so long to find, so measure it
rather than assume it.

Runs with the default use_fabric (True), which is what training uses. Reports
the robot's USD transform against its PhysX pose after a reset, after a
write_root_pose_to_sim teleport, and after a hand-written USD xform, and
photographs each state from a camera aimed where PhysX says the drone is.

    /isaac-sim/python.sh scripts/diag_robot.py --headless --enable_cameras
"""
from __future__ import annotations

import argparse
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--no-fabric", action="store_true", help="set sim.use_fabric = False")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.enable_cameras = True
app = AppLauncher(args).app

import carb  # noqa: E402

carb.settings.get_settings().set("/rtx-transient/resourcemanager/enableTextureStreaming", False)

import numpy as np  # noqa: E402
import torch  # noqa: E402
from PIL import Image  # noqa: E402

from isaacsim.core.utils.extensions import enable_extension  # noqa: E402

enable_extension("isaacsim.sensors.camera")
from isaacsim.sensors.camera import Camera  # noqa: E402
import isaacsim.core.utils.numpy.rotations as rot_utils  # noqa: E402
import isaacsim.core.utils.stage as stage_utils  # noqa: E402
from pxr import Gf, UsdGeom  # noqa: E402

from vesper.lab.chase_env import ChaseEnv, ChaseEnvCfg  # noqa: E402

cfg = ChaseEnvCfg()
cfg.scene.num_envs = 1
cfg.scene.env_spacing = 0.0
cfg.n_targets = 1
cfg.camera = False
cfg.vehicle_cycle_s = 10_000.0
cfg.require_tree_colliders = False
if args.no_fabric:
    cfg.sim.use_fabric = False
env = ChaseEnv(cfg, render_mode="rgb_array", seed=23)
out = Path("runs") / "diag-robot"
out.mkdir(parents=True, exist_ok=True)

stage = stage_utils.get_current_stage()
ROBOT = "/World/envs/env_0/Robot"
prim = stage.GetPrimAtPath(ROBOT)
print(f"ROBOT prim valid={prim.IsValid()} type={prim.GetTypeName()} "
      f"use_fabric={getattr(cfg.sim, 'use_fabric', 'n/a')}", flush=True)

cam = Camera(prim_path="/World/robot_cam", position=np.array([0.0, 0.0, 200.0]), resolution=(960, 720))
cam.initialize()
cam.set_focal_length(cam.get_horizontal_aperture() / (2.0 * np.tan(np.radians(45.0) / 2.0)))
cam.set_clipping_range(0.05, 6000.0)


def report(tag):
    for _ in range(25):
        env.sim.render()
    phys = env._robot.data.root_pos_w[0].cpu().numpy()
    xf = UsdGeom.XformCache().GetLocalToWorldTransform(prim).ExtractTranslation()
    print(f"{tag}: USD xform={[round(v, 2) for v in xf]}  PHYSX={phys.round(2).tolist()}", flush=True)
    eye = phys + np.array([7.0, 6.0, 2.5])
    d = phys.astype(float) - eye
    cam.set_world_pose(eye.astype(float), rot_utils.euler_angles_to_quats(
        np.array([0.0, np.degrees(np.arctan2(-d[2], np.linalg.norm(d[:2]))),
                  np.degrees(np.arctan2(d[1], d[0]))]), degrees=True))
    for _ in range(12):
        env.sim.render()
    Image.fromarray(np.asarray(cam.get_rgba()[..., :3], dtype=np.uint8)).save(out / f"{tag}.png")


env.vision_reset()
env.vision_step(torch.tensor([[0.0, 0.0, 0.0, -1.0]], device=env.device))
report("a_after_reset")

# a teleport of the kind every render script does
p = torch.as_tensor(np.array([2.0, 2.0], np.float32), device=env.device).view(1, 2)
gz = float(env.world.ground_at(p[:, 0], p[:, 1])[0])
target = np.array([12.0, 2.0, gz + 6.0], dtype=np.float64)
env._robot.write_root_pose_to_sim(torch.tensor(
    [[*target.tolist(), 1.0, 0.0, 0.0, 0.0]], device=env.device, dtype=torch.float32))
env._robot.write_root_velocity_to_sim(torch.zeros(1, 6, device=env.device))
env.vision_step(torch.tensor([[0.0, 0.0, 0.0, -1.0]], device=env.device))
report("b_after_teleport")

# the same hand-written USD xform that fixed the tanks
xf = UsdGeom.Xformable(prim)
ops = {o.GetOpName(): o for o in xf.GetOrderedXformOps()}
print(f"robot xformOps: {list(ops)}", flush=True)
t_op = ops.get("xformOp:translate") or xf.AddTranslateOp()
phys = env._robot.data.root_pos_w[0].cpu().numpy()
t_op.Set(Gf.Vec3d(*(float(v) for v in phys)))
report("c_usd_xform")

print("ROBOT_DIAG_DONE", flush=True)
env.close()
app.close()
