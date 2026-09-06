"""Which write actually moves the tank in the renderer?

diag_tank.py showed the split: PhysX has the tank on the site, USD still has it
at its (0,0,40) spawn pose, and the RTX path draws the USD/Fabric pose. So the
task, the reward and the range readouts are all correct while every camera looks
at empty ground.

This tries three ways of moving it and photographs each from the same place:

  a  write_root_pose_to_sim only            -- what the env does today
  b  a + the USD xformOps set directly
  c  a + isaacsim XFormPrim.set_world_poses -- the fabric-aware writer

    /isaac-sim/python.sh scripts/diag_tank_pose.py --headless --enable_cameras
"""
from __future__ import annotations

import argparse
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
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
from pxr import Gf, Usd, UsdGeom  # noqa: E402

from vesper.lab.chase_env import ChaseEnv, ChaseEnvCfg  # noqa: E402

cfg = ChaseEnvCfg()
cfg.scene.num_envs = 1
cfg.scene.env_spacing = 0.0
cfg.n_targets = 1
cfg.camera = False
cfg.vehicle_cycle_s = 10_000.0
cfg.require_tree_colliders = False
env = ChaseEnv(cfg, render_mode="rgb_array", seed=23)
out = Path("runs") / "diag-pose"
out.mkdir(parents=True, exist_ok=True)

stage = stage_utils.get_current_stage()
TANK = "/World/vehicles/v0/Tank"
prim = stage.GetPrimAtPath(TANK)
bcache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_])

# somewhere flat, central and definitely on the site
target_xy = np.array([2.0, 2.0], dtype=np.float32)
p = torch.as_tensor(target_xy, device=env.device).view(1, 2)
gz = float(env.world.ground_at(p[:, 0], p[:, 1])[0])
pose = torch.tensor([[float(target_xy[0]), float(target_xy[1]), gz + 0.08, 1.0, 0.0, 0.0, 0.0]],
                    device=env.device)
aim = np.array([target_xy[0], target_xy[1], gz + 1.4], dtype=np.float64)
eye = aim + np.array([11.0, 9.0, 5.0])

cam = Camera(prim_path="/World/pose_cam", position=np.array([0.0, 0.0, 200.0]), resolution=(960, 720))
cam.initialize()
cam.set_focal_length(cam.get_horizontal_aperture() / (2.0 * np.tan(np.radians(50.0) / 2.0)))
cam.set_clipping_range(0.05, 6000.0)
d = aim - eye
cam.set_world_pose(eye, rot_utils.euler_angles_to_quats(
    np.array([0.0, np.degrees(np.arctan2(-d[2], np.linalg.norm(d[:2]))),
              np.degrees(np.arctan2(d[1], d[0]))]), degrees=True))


def report(tag):
    for _ in range(25):
        env.sim.render()
    b = bcache.ComputeWorldBound(prim).ComputeAlignedRange()
    xf = UsdGeom.XformCache().GetLocalToWorldTransform(prim).ExtractTranslation()
    phys = env._vehicles.data.root_pos_w[0].cpu().numpy()
    print(f"{tag}: USD xform={[round(v, 2) for v in xf]} "
          f"bound_mid={[round(v, 2) for v in b.GetMidpoint()]} "
          f"PHYSX={phys.round(2).tolist()}", flush=True)
    Image.fromarray(np.asarray(cam.get_rgba()[..., :3], dtype=np.uint8)).save(out / f"{tag}.png")


# ---- a: what the env does today
env._vehicles.write_root_pose_to_sim(pose)
env._vehicles.write_root_velocity_to_sim(torch.zeros(1, 6, device=env.device))
env.vision_step(torch.tensor([[0.0, 0.0, 0.0, -1.0]], device=env.device))
report("a_physx_only")

# ---- b: also set the USD xformOps by hand
xf = UsdGeom.Xformable(prim)
ops = {op.GetOpName(): op for op in xf.GetOrderedXformOps()}
t_op = ops.get("xformOp:translate") or xf.AddTranslateOp()
t_op.Set(Gf.Vec3d(float(target_xy[0]), float(target_xy[1]), gz + 0.08))
o_op = ops.get("xformOp:orient")
if o_op is not None:
    o_op.Set(Gf.Quatd(1.0, 0.0, 0.0, 0.0) if o_op.GetTypeName() == "quatd"
             else Gf.Quatf(1.0, 0.0, 0.0, 0.0))
print(f"xformOps present: {list(ops)}", flush=True)
report("b_usd_xform")

# ---- c: the fabric-aware writer
try:
    from isaacsim.core.prims import XFormPrim
    view = XFormPrim(TANK)
    view.set_world_poses(positions=np.array([[target_xy[0], target_xy[1], gz + 0.08]]),
                         orientations=np.array([[1.0, 0.0, 0.0, 0.0]]))
    report("c_xformprim")
except Exception as exc:  # noqa: BLE001
    print(f"c_xformprim failed: {type(exc).__name__}: {exc}", flush=True)

print("POSE_DIAG_DONE", flush=True)
env.close()
app.close()
