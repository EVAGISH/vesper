"""Find out why the tank never reaches the renderer.

Prints, for the spawned vehicle prim: its USD subtree, each gprim's type,
computed visibility and purpose, the USD-side world bound, and the PhysX-side
root pose. Then renders three views -- aimed where PhysX says the tank is,
aimed where USD says it is, and a wide shot that would catch it at its spawn
altitude -- so a disagreement between the two is visible rather than inferred.

    /isaac-sim/python.sh scripts/diag_tank.py --headless --enable_cameras
"""
from __future__ import annotations

import argparse
import math
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
from pxr import Usd, UsdGeom  # noqa: E402
import isaacsim.core.utils.stage as stage_utils  # noqa: E402

from vesper.lab.chase_env import ChaseEnv, ChaseEnvCfg  # noqa: E402

cfg = ChaseEnvCfg()
cfg.scene.num_envs = 1
cfg.scene.env_spacing = 0.0
cfg.n_targets = 1
cfg.camera = False
cfg.vehicle_cycle_s = 10_000.0
cfg.require_tree_colliders = False
env = ChaseEnv(cfg, render_mode="rgb_array", seed=23)

out = Path("runs") / "diag-tank"
out.mkdir(parents=True, exist_ok=True)

# ------------------------------------------------------------------ the asset
from vesper.lab.pursuit_env import resolve_vehicle  # noqa: E402

spec = resolve_vehicle(cfg.vehicle_model)
src = Usd.Stage.Open(spec["usd"])
kinds = {}
for p in src.Traverse():
    kinds[p.GetTypeName()] = kinds.get(p.GetTypeName(), 0) + 1
print(f"ASSET {spec['usd']}", flush=True)
print(f"ASSET prim types: {kinds}", flush=True)

# ------------------------------------------------------------------ the stage
stage = stage_utils.get_current_stage()
xcache = UsdGeom.XformCache()
bcache = UsdGeom.BBoxCache(Usd.TimeCode.Default(),
                           [UsdGeom.Tokens.default_, UsdGeom.Tokens.render])
tank_prim = stage.GetPrimAtPath("/World/vehicles/v0/Tank")
print(f"STAGE /World/vehicles/v0/Tank valid={tank_prim.IsValid()} "
      f"active={tank_prim.IsActive() if tank_prim.IsValid() else '-'} "
      f"instanceable={tank_prim.IsInstanceable() if tank_prim.IsValid() else '-'}", flush=True)

if tank_prim.IsValid():
    n = 0
    for p in Usd.PrimRange(tank_prim, Usd.TraverseInstanceProxies()):
        img = UsdGeom.Imageable(p)
        vis = img.ComputeVisibility() if img else "-"
        pur = img.ComputePurpose() if img else "-"
        if p.IsA(UsdGeom.Gprim) and n < 8:
            b = bcache.ComputeWorldBound(p).ComputeAlignedRange()
            print(f"  GPRIM {p.GetPath()} type={p.GetTypeName()} vis={vis} purpose={pur} "
                  f"bound={[round(v,2) for v in b.GetMin()]}..{[round(v,2) for v in b.GetMax()]}",
                  flush=True)
            n += 1
    root_b = bcache.ComputeWorldBound(tank_prim).ComputeAlignedRange()
    xf = xcache.GetLocalToWorldTransform(tank_prim).ExtractTranslation()
    print(f"USD  xform translate = {[round(v,2) for v in xf]}", flush=True)
    print(f"USD  world bound     = {[round(v,2) for v in root_b.GetMin()]}"
          f" .. {[round(v,2) for v in root_b.GetMax()]}", flush=True)

phys = env._vehicles.data.root_pos_w[0].cpu().numpy()
print(f"PHYSX root_pos_w     = {phys.round(2).tolist()}", flush=True)
print(f"USE_FABRIC           = {getattr(cfg.sim, 'use_fabric', 'n/a')}", flush=True)

# ------------------------------------------------------------------ renders
cam = Camera(prim_path="/World/diag_cam", position=np.array([0.0, 0.0, 200.0]),
             resolution=(960, 720))
cam.initialize()
ap_h = cam.get_horizontal_aperture()
cam.set_focal_length(ap_h / (2.0 * np.tan(np.radians(50.0) / 2.0)))
cam.set_clipping_range(0.05, 6000.0)


def shot(name, pos, target):
    delta = np.asarray(target, float) - np.asarray(pos, float)
    yaw = np.degrees(np.arctan2(delta[1], delta[0]))
    pitch = np.degrees(np.arctan2(-delta[2], np.linalg.norm(delta[:2]) + 1e-6))
    cam.set_world_pose(np.asarray(pos, float),
                       rot_utils.euler_angles_to_quats(np.array([0.0, pitch, yaw]), degrees=True))
    for _ in range(20):
        env.sim.render()
    Image.fromarray(np.asarray(cam.get_rgba()[..., :3], dtype=np.uint8)).save(out / name)
    print(f"shot {name} from {np.round(pos,1).tolist()} at {np.round(target,1).tolist()}", flush=True)


shot("a_physx.png", phys + np.array([12.0, 10.0, 6.0]), phys + np.array([0.0, 0.0, 1.4]))
if tank_prim.IsValid():
    c = np.array(root_b.GetMidpoint(), dtype=float) if not root_b.IsEmpty() else phys
    shot("b_usd.png", c + np.array([12.0, 10.0, 6.0]), c)
# the spawn pose: init_state.pos is (0,0,40), so a render stuck there shows here
shot("c_spawn.png", np.array([35.0, 30.0, 55.0]), np.array([0.0, 0.0, 40.0]))
print("DIAG_DONE", flush=True)
env.close()
app.close()
