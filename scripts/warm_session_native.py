"""Warm session on the native sim: the UI backbone with no Isaac, no droplet.

Runs the search world on vesper.native.NativeSearchEnv in this process, on this
machine -- no GPU box, no firewall, no RTX, no cold start at all (the world is a
2.6 MB raster). Serves the same /state, /command, /streams contract as
scripts/warm_session.py, so the web client points at localhost and works
unchanged -- including the video downlink: fpv and overview are raymarched from
the world rasters (vesper.native.camera), the same lens and cone the task
flies. RTX-grade footage remains the Isaac session's job on the droplet.

    .venv/bin/python scripts/warm_session_native.py --policy runs/<id>/search.pt

Serves on VESPER_LIVE_PORT (8180):
    GET  /streams          {"run": ..., "streams": ["fpv", "overview"]}
    GET  /fpv.mjpeg        drone 0's own camera, synthetic (task lens)
    GET  /overview.mjpeg   chase view trailing drone 0
    GET  /state            {t, drones:[{x,y,z,linked}], vehicles:[{x,y,found,reached,pending}],
                            policy, found, reached, pending, manual, teleop_age_s,
                            comms_denied, comms:{n,half,grid,denied_frac}} -- `grid` is the
                            mission's confirmed-connectivity map over the AO, one digit per
                            cell (0 unknown, 1 confirmed link, 2 confirmed dead zone),
                            row 0 = south, revealed as the drones fly. found/reached are
                            RELAYED reports: a sighting made while the lead is jammed stays
                            `pending` (off the operator's map) until it regains link
    POST /command          {"kind":"reset"} | {"kind":"deploy","policy":"runs/<id>/<f>.pt"}
                           {"kind":"manual","on":true|false}   hand drone 0 to the operator
                           {"kind":"teleop","axes":[fwd,left,up]} in [-1,1], body frame;
                           re-sent every ~100 ms by the page. Older than --deadman_s: hover.
"""
import argparse
import math
import os
import time

import numpy as np
import torch

from vesper.capture.live import LiveFrameServer
from vesper.lab.ppo import load_policy
from vesper.native import NativeSearchEnv, NativeSearchEnvCfg

parser = argparse.ArgumentParser()
parser.add_argument("--num_envs", type=int, default=16)
parser.add_argument("--targets", type=int, default=3)
parser.add_argument("--arena", type=float, default=300.0)
parser.add_argument("--policy", default=None, help="initial checkpoint (optional)")
parser.add_argument("--map", default=None, help="world map npz (default the Cornell site)")
parser.add_argument("--groups", type=int, default=1,
                    help="vehicle sets shared by groups of envs; 1 = every drone hunts the same three")
parser.add_argument("--seed", type=int, default=7)
parser.add_argument("--device", default="cpu", help="cpu is plenty for a live session")
parser.add_argument("--speed", type=float, default=1.0,
                    help="sim seconds per wall second; the native sim is far faster than "
                         "realtime, so the loop sleeps to hold this")
parser.add_argument("--teleop_gain", type=float, default=0.6,
                    help="full stick = tanh^-1(gain) of the action range; 0.6 is ~13 m/s")
parser.add_argument("--deadman_s", type=float, default=0.7,
                    help="manual mode hovers when no teleop command arrived for this long")
parser.add_argument("--no_feeds", action="store_true",
                    help="skip the synthetic camera downlink (state only)")
parser.add_argument("--every", type=int, default=4,
                    help="render 1 frame per N control steps (4 = ~6 fps at 25 Hz)")
parser.add_argument("--render_device", default=None,
                    help="device for the raymarcher (default: mps if available, else cpu)")
parser.add_argument("--ortho", default=None,
                    help="ground orthophoto for the terrain texture (default beside the map)")
parser.add_argument("--reach_radius", type=float, default=None,
                    help="range that counts as neutralizing a target (m); wider registers the "
                         "policy's close passes as strikes (demo climax) without retraining")
parser.add_argument("--episode_s", type=float, default=None,
                    help="episode length (default 75, a training value); longer lets a full "
                         "search->detect->neutralize mission complete before any rollover")
args = parser.parse_args()

os.environ.setdefault("VESPER_LIVE_PORT", "8180")

cfg = NativeSearchEnvCfg()
cfg.num_envs = args.num_envs
cfg.n_targets = args.targets
cfg.search = {"arena_half": args.arena}
if args.reach_radius is not None:
    cfg.search["reach_radius"] = args.reach_radius
if args.episode_s is not None:
    cfg.episode_length_s = args.episode_s
cfg.n_groups = args.groups
if args.map:
    cfg.world_map = args.map
env = NativeSearchEnv(cfg, device=args.device, seed=args.seed)
# world name for the client (assets/<name>/<name>_map.npz convention)
from pathlib import Path as _P
world_name = _P(cfg.world_map).stem.replace("_map", "")

# --- comms reveal grid: the mission's confirmed-connectivity map over the AO.
# The FIELD is static (baked raster, see export_world_map.py); what changes is
# what the drones have CONFIRMED by flying there. A coarse grid keeps /state
# small: 64x64 digits ~4 KB. Row 0 = south (raster convention, row indexes +y).
from vesper.worlds.heightmap import LINK_THRESHOLD
COMMS_N = 64
_ao_half = float(args.arena)
_cc = torch.as_tensor((np.arange(COMMS_N) + 0.5) / COMMS_N * (2 * _ao_half) - _ao_half,
                      dtype=torch.float32, device=env.device)
_cx, _cy = torch.meshgrid(_cc, _cc, indexing="xy")               # [row=y, col=x]
comms_small = env.world.comms_at(_cx.reshape(-1), _cy.reshape(-1)) \
    .reshape(COMMS_N, COMMS_N).cpu().numpy()
comms_connected = comms_small >= LINK_THRESHOLD
comms_denied_frac = float(1.0 - comms_connected.mean())
comms_seen = np.zeros((COMMS_N, COMMS_N), np.uint8)              # 0 unknown / 1 link / 2 dead
_comms_cell = 2 * _ao_half / COMMS_N

# relay gating: a sighting (or strike) made while the lead is jammed is a
# PENDING report -- the operator only learns of it when the drone regains link,
# so a tank found deep in a dead zone stays off the map until the drone climbs
# or flies back out of the interference shadow. Dies with the drone: an episode
# rollover drops whatever it never relayed.
relayed_found = np.zeros(args.targets, bool)
relayed_reach = np.zeros(args.targets, bool)
prev_ep0 = 0


def stamp_comms(drones_xy):
    """Mark the cells around each drone as confirmed (link or dead zone).

    Cell status comes from the baked field at the cell, not the drone's own
    reading, so a drone skirting a dead zone confirms the zone correctly."""
    for x, y in drones_xy:
        c = int((x + _ao_half) / _comms_cell)
        r = int((y + _ao_half) / _comms_cell)
        if c < -1 or c > COMMS_N or r < -1 or r > COMMS_N:       # far outside the AO
            continue
        r0, r1 = max(0, r - 1), min(COMMS_N, r + 2)
        c0, c1 = max(0, c - 1), min(COMMS_N, c + 2)
        comms_seen[r0:r1, c0:c1] = np.where(comms_connected[r0:r1, c0:c1], 1, 2)


class Policy:
    """A swappable actor. Zero action until a checkpoint is deployed."""

    def __init__(self, path=None):
        self.name = "none"
        self.ac = self.norm = None
        if path:
            self.load(path)

    def load(self, path):
        ck = torch.load(path, map_location=env.device)
        if ck["obs_dim"] != env.num_obs:
            raise ValueError(f"policy expects {ck['obs_dim']}-wide observations, env gives {env.num_obs}")
        ac, norm = load_policy(ck, env.device)
        self.ac, self.norm, self.name = ac, norm, path.split("/")[-1]

    @torch.no_grad()
    def act(self, obs):
        if self.ac is None:
            return torch.zeros(env.num_envs, env.num_actions, device=env.device)
        return self.ac.actor(self.norm(obs))


try:
    policy = Policy(args.policy)
except Exception as e:                                       # noqa: BLE001
    print(f"[warm] initial policy not loaded ({e}); flying with zero action until a deploy", flush=True)
    policy = Policy(None)
srv = LiveFrameServer(int(os.environ["VESPER_LIVE_PORT"]), run_id="warm-session-native")

obs = env.ppo_reset()
dt = env._dt
t0 = 0.0
step = 0
manual = False
teleop = {"axes": [0.0, 0.0, 0.0], "t": 0.0}
stick = math.atanh(min(max(args.teleop_gain, 0.05), 0.99))

# --- synthetic downlink: the raymarched fpv + chase views (vesper.native.camera)
fpv_cam = chase_cam = None
if not args.no_feeds:
    import numpy as np
    from pathlib import Path

    from vesper.native.camera import RasterCamera
    from vesper.worlds.heightmap import WorldMap

    rdev = args.render_device or ("mps" if torch.backends.mps.is_available() else "cpu")
    ortho = args.ortho
    if ortho is None:
        cand = Path(env.cfg.world_map).parent / "ground.png"
        ortho = str(cand) if cand.exists() else None
    # the renderer keeps its own WorldMap on the render device (2.6 MB copy)
    rworld = WorldMap(env.cfg.world_map, device=rdev) if rdev != env.device else env.world
    fpv_cam = RasterCamera(rworld, ortho_path=ortho, res=(640, 640), samples=128,
                           fov_half_deg=env.task.cfg.fov_half_deg, device=rdev)
    chase_cam = RasterCamera(rworld, ortho_path=ortho, res=(768, 432), samples=128,
                             fov_half_deg=37.5, device=rdev)
    chase_dir = np.array([1.0, 0.0])
    # rendering runs on its own thread with a latest-pose slot: the sim loop
    # never waits on the raymarcher, so 25 Hz realtime holds regardless of how
    # long a frame takes; the downlink simply runs at whatever fps that is
    import threading
    _feed_slot = {"req": None}
    _feed_cv = threading.Condition()

    def _feed_worker():
        while True:
            with _feed_cv:
                while _feed_slot["req"] is None:
                    _feed_cv.wait()
                req = _feed_slot["req"]
                _feed_slot["req"] = None
            try:
                render_feeds(*req)
            except Exception as e:                       # noqa: BLE001
                print(f"[warm] feed render failed: {e}", flush=True)

    threading.Thread(target=_feed_worker, daemon=True).start()
    print(f"[warm] synthetic downlink on ({rdev}, threaded, poses 1/{args.every} steps)", flush=True)

print(f"[warm] native world loaded ({env.world.n}x{env.world.n} raster); "
      f"ready for commands on /command", flush=True)


def render_feeds(d0, quat0, v_xy, tp, others, fleet):
    """Raymarch fpv (drone 0's lens, exactly the task's cone) + a trailing chase.

    Runs on the feed thread from a pose snapshot; touches nothing of the env.
    """
    global chase_dir
    srv.publish(fpv_cam.render(d0, quat0, env.task.cfg.cam_pitch_deg,
                               targets=tp, drones=others), "fpv")
    # trail 18 m behind and 6 m above along the (smoothed) travel direction
    speed = float(np.linalg.norm(v_xy))
    if speed > 0.5:
        chase_dir = 0.9 * chase_dir + 0.1 * (v_xy / speed)
        chase_dir /= np.linalg.norm(chase_dir) + 1e-6
    cpos = d0 + np.array([-18.0 * chase_dir[0], -18.0 * chase_dir[1], 6.0])
    cyaw = math.atan2(d0[1] - cpos[1], d0[0] - cpos[0])
    cpitch = math.degrees(math.atan2(cpos[2] - d0[2],
                                     float(np.linalg.norm((d0 - cpos)[:2])) + 1e-6))
    cquat = [math.cos(cyaw / 2), 0.0, 0.0, math.sin(cyaw / 2)]
    srv.publish(chase_cam.render(cpos, cquat, cpitch, targets=tp, drones=fleet), "overview")


def apply_commands():
    global obs, t0, step, manual
    for cmd in srv.drain_commands():
        kind = cmd.get("kind")
        if kind == "manual":
            manual = bool(cmd.get("on", True))
            teleop["axes"] = [0.0, 0.0, 0.0]
            print(f"[warm] manual {'ON: drone 0 is the operator' if manual else 'off: policy flies'}", flush=True)
        elif kind == "teleop":
            ax = cmd.get("axes") or [0.0, 0.0, 0.0]
            try:
                teleop["axes"] = [min(max(float(v), -1.0), 1.0) for v in ax[:3]] + [0.0] * (3 - len(ax[:3]))
                teleop["t"] = time.time()
            except (TypeError, ValueError):
                pass
        elif kind == "reset":
            obs = env.ppo_reset()
            t0, step = 0.0, 0
            comms_seen[:] = 0                               # fresh mission, fresh reveal
            relayed_found[:] = relayed_reach[:] = False
            print("[warm] reset", flush=True)
        elif kind == "deploy":
            p = cmd.get("policy")
            try:
                policy.load(p)
                obs = env.ppo_reset()
                t0, step = 0.0, 0
                comms_seen[:] = 0
                relayed_found[:] = relayed_reach[:] = False
                print(f"[warm] deployed {policy.name}", flush=True)
            except Exception as e:                          # noqa: BLE001
                print(f"[warm] deploy failed: {e}", flush=True)


try:
    while True:
        tick = time.time()
        apply_commands()
        act = policy.act(obs)
        if manual:
            # drone 0 belongs to the operator: body-frame stick through the same
            # action the policy uses, so WASD flies exactly what the policy would
            fresh = (time.time() - teleop["t"]) < args.deadman_s
            ax = teleop["axes"] if fresh else [0.0, 0.0, 0.0]
            act[0] = torch.tensor(ax, device=env.device) * stick
        obs, rew, done, info = env.ppo_step(act)
        step += 1
        t = step * dt

        # publish world state for the AO map + 3D view (all drones, env-0 targets)
        pos, vel, quat, _ = env.flight_state()
        drones = pos.cpu().numpy()
        quats = quat.cpu().numpy()
        tp = env.target_pos[0].cpu().numpy()
        headings = env.veh_heading[env.group[0].item()].cpu().numpy()
        known = env.task.known[0].cpu().numpy()
        reached = env.task.reached[0].cpu().numpy()
        v0 = vel[0]
        linked = env.linked.cpu().numpy()
        stamp_comms(drones[:, :2])
        # relay gate: the lead's episode rolled over -> unrelayed contacts die
        # with it; while linked, everything it knows commits to the operator
        ep0 = int(env.episode_length_buf[0].item())
        if ep0 < prev_ep0:
            relayed_found[:] = relayed_reach[:] = False
        prev_ep0 = ep0
        if linked[0]:
            relayed_found |= known
            relayed_reach |= reached
        srv.set_state({
            "t": round(float(t), 1),
            "world": world_name,
            "policy": policy.name,
            "manual": manual,
            "teleop_age_s": round(time.time() - teleop["t"], 2) if teleop["t"] else None,
            "drone0": {"speed": round(float(v0[:2].norm()), 1), "vz": round(float(v0[2]), 1),
                       "agl": round(float(info["agl"][0]), 1)},
            "found": int(relayed_found.sum()), "reached": int(relayed_reach.sum()),
            "targets": int(args.targets),
            "pending": int((known & ~relayed_found).sum()),   # contacts awaiting relay
            "comms_denied": round(comms_denied_frac, 3),
            "comms": {"n": COMMS_N, "half": _ao_half, "denied_frac": round(comms_denied_frac, 3),
                      "grid": (comms_seen + ord("0")).astype(np.uint8).tobytes().decode("ascii")},
            "drones": [{"x": round(float(p[0]), 1), "y": round(float(p[1]), 1),
                        "z": round(float(p[2]), 1), "linked": bool(lk),
                        "q": [round(float(v), 3) for v in q]}
                       for p, q, lk in zip(drones, quats, linked)],
            "vehicles": [{"x": round(float(tp[k][0]), 1), "y": round(float(tp[k][1]), 1),
                          "z": round(float(tp[k][2]), 1), "hdg": round(float(headings[k]), 2),
                          "found": bool(relayed_found[k]), "reached": bool(relayed_reach[k]),
                          "pending": bool(known[k] and not relayed_found[k])}
                         for k in range(args.targets)],
        })

        if fpv_cam is not None and step % args.every == 0:
            pos, vel, quat, _ = env.flight_state()
            with _feed_cv:
                _feed_slot["req"] = (pos[0].cpu().numpy(), quat[0].cpu().tolist(),
                                     vel[0, :2].cpu().numpy(), env.target_pos[0].cpu().tolist(),
                                     pos[1:].cpu().tolist(), pos.cpu().tolist())
                _feed_cv.notify()

        # NB: env.step() already auto-resets each drone individually as it
        # finishes (staggered), so we must NOT full-reset here -- doing that
        # teleported all 16 drones at once every time the lead finished. The
        # mission clock follows the lead's own episode via its buffer.
        step = int(env.episode_length_buf[0].item())

        # the native sim steps in microseconds; hold the commanded pace
        rest = dt / max(args.speed, 1e-3) - (time.time() - tick)
        if rest > 0:
            time.sleep(rest)
except KeyboardInterrupt:
    print("\n[warm] bye", flush=True)
