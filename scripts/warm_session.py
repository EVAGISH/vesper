"""Warm session: load the world ONCE, then fly on command forever.

The fix for the 1-2 minute cold start. A persistent sim process holds the world,
the drones and a policy in memory and steps continuously; resetting an episode or
deploying a new policy is a command over HTTP, not a fresh Isaac boot -- ~1 s
instead of minutes. It is the backbone both app modes ride: Operations reads the
live map + feeds and sends deploy/reset; the feeds never go dark.

    /isaac-sim/python.sh scripts/warm_session.py --num_envs 16 --policy runs/friend-checkpoints/search.pt

Serves on VESPER_LIVE_PORT (8180):
    GET  /streams          cameras being published
    GET  /fpv.mjpeg        drone's own view (nadir, cone lens) -- annotated
    GET  /overview.mjpeg   chase of drone 0
    GET  /state            {t, drones:[{x,y,z}], targets:[{x,y,found,reached}], policy, found, reached}
    POST /command          {"kind":"reset"} | {"kind":"deploy","policy":"runs/<id>/<f>.pt"}
"""
import argparse
import os

os.environ.setdefault("VESPER_LIVE_PORT", "8180")

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--num_envs", type=int, default=16)
parser.add_argument("--targets", type=int, default=3)
parser.add_argument("--arena", type=float, default=300.0)
parser.add_argument("--hfov", type=float, default=75.0)
parser.add_argument("--policy", default=None, help="initial checkpoint (optional)")
parser.add_argument("--world", default=None, help="world USD (default the Cornell site)")
parser.add_argument("--map", default=None, help="world map npz (default beside the USD)")
parser.add_argument("--every", type=int, default=2, help="render 1 frame per N control steps")
parser.add_argument("--cameras", action="store_true",
                    help="also render + publish drone camera feeds. Off by default: "
                         "rendering RTX cameras in this env on the 16k-tree world "
                         "hits a CUDA illegal-access crash (~2 min in). The AO map "
                         "(/state) needs no rendering and is solid without this.")
parser.add_argument("--seed", type=int, default=7)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.headless = True
args.enable_cameras = args.cameras
app = AppLauncher(args).app

import carb  # noqa: E402
carb.settings.get_settings().set("/rtx-transient/resourcemanager/enableTextureStreaming", False)

import numpy as np  # noqa: E402
import torch  # noqa: E402
import isaacsim.core.utils.numpy.rotations as rot_utils  # noqa: E402

if args.cameras:
    from isaacsim.core.utils.extensions import enable_extension
    enable_extension("isaacsim.sensors.camera")
    from isaacsim.sensors.camera import Camera

from vesper.capture.live import LiveFrameServer  # noqa: E402
from vesper.lab.ppo import ActorCritic, RunningNorm  # noqa: E402
from vesper.lab.search_env import SearchEnv, SearchEnvCfg  # noqa: E402

cfg = SearchEnvCfg()
if args.world:
    cfg.world_usd = args.world
if args.map:
    cfg.world_map = args.map
# The RTX-render crash on the 16k-tree world faults inside omni.physx.fabric's
# GPU sync. Fabric is a throughput optimization we don't need for a live render
# session -- disabling it renders the drone feeds cleanly (same fix as fly_search).
if args.cameras and not os.environ.get("VESPER_KEEP_FABRIC"):
    try:
        cfg.sim.use_fabric = False
    except Exception:
        pass
cfg.scene.num_envs = args.num_envs
cfg.scene.env_spacing = 0.0
cfg.n_targets = args.targets
cfg.search = {"arena_half": args.arena}
env = SearchEnv(cfg, render_mode="rgb_array", seed=args.seed)


class Policy:
    """A swappable actor. Zero action until a checkpoint is deployed."""
    def __init__(self, path=None):
        self.name = "none"
        self.ac = self.norm = None
        if path:
            self.load(path)

    def load(self, path):
        ck = torch.load(path, map_location=env.device)
        ac = ActorCritic(ck["obs_dim"], ck["act_dim"]).to(env.device)
        ac.load_state_dict(ck["ac"]); ac.eval()
        norm = RunningNorm(ck["obs_dim"]).to(env.device)
        norm.load_state_dict(ck["norm"])
        self.ac, self.norm, self.name = ac, norm, path.split("/")[-1]

    @torch.no_grad()
    def act(self, obs):
        if self.ac is None:
            return torch.zeros(env.num_envs, env.num_actions, device=env.device)
        return self.ac.actor(self.norm(obs))


policy = Policy(args.policy)
srv = LiveFrameServer(int(os.environ["VESPER_LIVE_PORT"]), run_id="warm-session")

fpv = chase = None
obs = env.ppo_reset()
if args.cameras:
    fpv = Camera(prim_path="/World/fpv_cam", position=np.array([0.0, 0.0, 200.0]), resolution=(900, 900))
    chase = Camera(prim_path="/World/chase_cam", position=np.array([0.0, 0.0, 200.0]), resolution=(1280, 720))
    for cam, hf in ((fpv, 2.0 * env.task.cfg.fov_half_deg), (chase, args.hfov)):
        cam.initialize()
        ap = cam.get_horizontal_aperture()
        cam.set_focal_length(ap / (2.0 * np.tan(np.radians(hf) / 2.0)))
        cam.set_clipping_range(0.05, 6000.0)

dt = cfg.sim.dt * cfg.decimation
t0 = 0.0
step = 0
print("[warm] world loaded; ready for commands on /command", flush=True)


def look_at(cam, pos, target):
    d = target - pos
    yaw = np.degrees(np.arctan2(d[1], d[0]))
    pitch = np.degrees(np.arctan2(-d[2], np.linalg.norm(d[:2]) + 1e-6))
    cam.set_world_pose(pos, rot_utils.euler_angles_to_quats(
        np.array([0.0, pitch, yaw]), degrees=True))


def apply_commands():
    global obs, t0, step
    for cmd in srv.drain_commands():
        kind = cmd.get("kind")
        if kind == "reset":
            obs = env.ppo_reset()
            t0, step = 0.0, 0
            print("[warm] reset", flush=True)
        elif kind == "deploy":
            p = cmd.get("policy")
            try:
                policy.load(p)
                obs = env.ppo_reset()
                t0, step = 0.0, 0
                print(f"[warm] deployed {policy.name}", flush=True)
            except Exception as e:                          # noqa: BLE001
                print(f"[warm] deploy failed: {e}", flush=True)


while app.is_running():
    apply_commands()
    act = policy.act(obs)
    obs, rew, done, info = env.ppo_step(act)
    step += 1
    t = step * dt

    # publish world state for the AO map (all drones, env-0 targets)
    drones = env._robot.data.root_pos_w.cpu().numpy()
    tp = env.target_pos[0].cpu().numpy()
    known = env.task.known[0].cpu().numpy()
    reached = env.task.reached[0].cpu().numpy()
    srv.set_state({
        "t": round(float(t), 1),
        "policy": policy.name,
        "found": int(known.sum()), "reached": int(reached.sum()), "targets": int(args.targets),
        "drones": [{"x": round(float(x), 1), "y": round(float(y), 1), "z": round(float(z), 1)}
                   for x, y, z in drones],
        "vehicles": [{"x": round(float(tp[k][0]), 1), "y": round(float(tp[k][1]), 1),
                      "found": bool(known[k]), "reached": bool(reached[k])}
                     for k in range(args.targets)],
    })

    if args.cameras and step % args.every == 0:
        d0 = drones[0]
        look_at(fpv, d0 + np.array([0.0, 0.0, -0.6]), d0 + np.array([0.0, 0.0, -50.0]))
        look_at(chase, d0 + np.array([-45.0, -45.0, 30.0]), d0)
        env.sim.render()
        srv.publish(np.asarray(fpv.get_rgba()[..., :3], dtype=np.uint8), "fpv")
        srv.publish(np.asarray(chase.get_rgba()[..., :3], dtype=np.uint8), "overview")

    if bool(done[0].item()):
        obs = env.ppo_reset()
        step = 0
