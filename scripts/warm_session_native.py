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
    GET  /state            {t, drones:[{x,y,z,linked,expended}], vehicles:[{x,y,found,reached,pending}],
                            -- a drone flagged `expended` was consumed by its own
                            strike (loitering munition) and is gone until the
                            next sortie fields a fresh fleet --
                            policy, found, reached, pending, manual, teleop_age_s,
                            comms_denied, comms:{n,half,grid,denied_frac}} -- `grid` is the
                            mission's confirmed-connectivity map over the AO, one digit per
                            cell (0 unknown, 1 confirmed link, 2 confirmed dead zone),
                            row 0 = south, revealed as the drones fly. found/reached are
                            RELAYED reports: a sighting made while the lead is jammed stays
                            `pending` (off the operator's map) until it regains link.
                            `requests` is the strike-approval queue (env 0's mission):
                            [{target, status, x, y, detected_t, uplink, disengaged}] with
                            status one of PENDING_UPLINK / AWAITING_APPROVAL / APPROVED /
                            DENIED, and `strikes` {pending, awaiting, approved, denied} the
                            HUD tally. Any group-0 drone's detection raises the request and
                            any linked drone relays it; --auto_approve_s N approves it N s
                            after uplink (unattended demos). A target only neutralizes once
                            its request is APPROVED; one un-approved past --disengage_s is
                            masked from the actor's belief (disengaged: true) so the drones
                            search on instead of orbiting it, re-engaging on approval.
    POST /command          {"kind":"reset"} | {"kind":"deploy","policy":"runs/<id>/<f>.pt"}
                           {"kind":"manual","on":true|false}   hand drone 0 to the operator
                           {"kind":"teleop","axes":[fwd,left,up]} in [-1,1], body frame;
                           re-sent every ~100 ms by the page. Older than --deadman_s: hover.
                           {"kind":"approve","target":i} | {"kind":"deny","target":i}
                           the operator's strike decision for target i
                           {"kind":"record","renderers":"three,tactical"}   dump the
                           mission buffered since the last reset to a fresh
                           runs/<id>/ (replay.json + trajectory.parquet +
                           manifest) and kick scripts/render_dispatch.py in the
                           background -- the Live->Runs loop. `renderers` is the
                           render FLAG: any of tactical (2D top-down), three
                           (on-device three.js chase+fpv, the default fast lane)
                           and isaac (photoreal on the droplet, the hero shot);
                           unset falls back to --renderers. A manual record is
                           the KEEPER: never pruned.

Missions also auto-save: when the lead's episode completes (all-cleared or the
episode_s rollover) the buffered mission lands in runs/ by itself IF it scored
at least one neutralization -- tagged auto in the manifest and capped at the
last AUTO_KEEP, so the Runs tab fills with real sorties without flooding.
"""
import argparse
import copy
import json
import math
import os
import shutil
import subprocess
import sys
import threading
import time

import numpy as np
import torch

from vesper.capture.live import LiveFrameServer
from vesper.lab.ppo import load_policy
from vesper.native import NativeSearchEnv, NativeSearchEnvCfg
from vesper.record.trajectory import TrajectoryWriter

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
parser.add_argument("--auto_approve_s", type=float, default=None,
                    help="unattended demo: a strike request AWAITING_APPROVAL this many "
                         "seconds auto-promotes to APPROVED (default off: operator decides)")
parser.add_argument("--transit", action="store_true",
                    help="OPT-IN scripted transit layer: drones with no pursuable target "
                         "fly straight to the stalest coverage cell of their own sector, "
                         "the policy taking back over on detection. Off by default -- the "
                         "search behavior should be the policy's own (retrain with "
                         "w_cover_stale / w_yaw_rate), not choreography")
parser.add_argument("--renderers", default="three,tactical",
                    help="comma list from {tactical,three,isaac}: which replay renderers a "
                         "manual RECORD SORTIE kicks off (the record command's own "
                         "'renderers' field overrides; auto-saves always render tactical "
                         "only). three = fast on-device three.js chase+fpv; isaac = "
                         "photoreal on the GPU box (the hero shot)")
parser.add_argument("--disengage_s", type=float, default=20.0,
                    help="a target whose strike request stays un-approved this long is shown "
                         "to the actor as already struck, so its drones break off and keep "
                         "searching instead of orbiting an ungranted strike; re-engaged the "
                         "moment it is APPROVED (<= 0 disables)")
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

# --- human-in-the-loop strike approval (env 0's mission, the one on the map).
# A detection RAISES a strike request; the request reaches the operator only
# while the lead holds an RF link (PENDING_UPLINK until then -- a tank found in
# a dead zone forces the drone to egress before it can call the strike in);
# and the target can only be neutralized once the operator APPROVES. The gate
# itself lives in SearchTask.strike_hold: a [N,K] mask the task ANDs out of
# `touching`, None during training, so the policy and reward are untouched --
# the drone still dives, the kill just does not register until cleared hot.
REQ_NONE, REQ_PENDING, REQ_AWAITING, REQ_APPROVED, REQ_DENIED = (
    "NONE", "PENDING_UPLINK", "AWAITING_APPROVAL", "APPROVED", "DENIED")
req_status = [REQ_NONE] * args.targets
req_detected_t = [0.0] * args.targets
req_await_t = [0.0] * args.targets      # when the request reached AWAITING_APPROVAL
mission_t = [0.0]                       # the lead's mission clock, for the helpers below
strike_hold = torch.ones(env.num_envs, args.targets, dtype=torch.bool, device=env.device)
_grp0 = (env.group == 0)         # every env hunting vehicle set 0 shares the gate
env.task.strike_hold = strike_hold
# disengage-on-hold: False masks the target out of the actor's belief (it reads
# as already struck) so drones search on instead of orbiting an ungranted strike
engage = torch.ones(env.num_envs, args.targets, dtype=torch.bool, device=env.device)
env.task.engage_mask = engage

# --- loitering-munition bookkeeping. A registered strike is a kamikaze impact:
# the drone dives into the tank and both are destroyed. `expended` marks the
# consumed airframes -- hidden from the map, the comms reveal, the relay lane and
# the request raising until the next sortie fields a fresh fleet. `wrecked` pins
# the kill at FLEET level: the striker's episode ends on impact (SearchCfg.
# expend_on_reach) and its auto-reset clears its own `reached` row, which would
# otherwise let the "destroyed" tank drive off again (_drive_vehicles recomputes
# wrecks from task.reached every step).
expended = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
wrecked = torch.zeros(args.targets, dtype=torch.bool, device=env.device)


def sync_wrecks():
    """Re-assert fleet-level kills into every group-0 drone's belief, so the
    wreck stays a wreck and every surviving drone reads it as already struck."""
    if bool(wrecked.any()):
        env.task.reached[_grp0] |= wrecked
        env.task.known[_grp0] |= wrecked


# --- shared fleet coverage (VES-24): union the group's recency grid into each
# drone's actor obs, so the frontier-seeking policy spreads instead of every
# drone re-sweeping the same central ground. The teacher already reads a G*G
# recency grid in its obs and was trained to fly toward low-recency (stale)
# cells; here we replace each drone's OWN grid with the group MAX (a cell any
# drone swept recently reads as covered for all), so the profitable heading is
# ground NOBODY has recently looked at. Each drone still closes on its NEAREST
# stale cell, and they start from different positions, so the fleet fans out
# instead of herding. Deploy-only: no retrain, the obs slot is unchanged.
# obs layout (search_task.privileged): [self 12 | targets 8K | recency G*G | tail 3]
_G = env.task.cfg.grid
_REC0 = 12 + 8 * args.targets            # recency block start column
_REC1 = _REC0 + _G * _G                  # recency block end column
_shared_cov = os.environ.get("VESPER_SHARED_COVERAGE", "1") != "0"


def apply_shared_coverage(o):
    """Overwrite each group-0 drone's recency obs with the group-union recency."""
    if not _shared_cov or not bool(_grp0.any()):
        return o
    live = _grp0 & ~expended
    if not bool(live.any()):
        return o
    union = env.task.recency[live].amax(dim=0)      # [G*G] most-recent sweep by anyone
    o = o.clone()
    o[_grp0, _REC0:_REC1] = union
    return o


# --- strike commit: the deterministic terminal dive on an APPROVED request.
# The RL policy finds and shadows targets but loiters OUTSIDE the kill sphere,
# so an approval used to open the strike_hold gate and then nothing walked
# through it -- approved targets survived while drones circled (the HITL half of
# project memory circling-root-cause). Approval is a commitment: the nearest
# surviving drone is put into a STRIKE state and flown by a deterministic
# terminal-guidance override -- through the exact same body-frame action path
# the policy uses -- straight into the target until `touching` fires, the tank
# dies and the airframe is expended. Search stays the policy's job; only the
# committed strike is scripted, and only for the one servicing drone.
STRIKE_CRUISE_AGL = 60.0        # en-route height over ground until the terminal dive
STRIKE_DIVE_AT = 60.0           # horizontal range where the dive onto the target begins
STRIKE_GAIN = 1.5               # pre-tanh stick per look_ahead of remaining offset
strike_tgt = torch.full((env.num_envs,), -1, dtype=torch.long, device=env.device)


def sync_strike_commits():
    """Keep exactly one live servicing drone per APPROVED, still-alive target."""
    pos = env.flight_state()[0]
    for i in torch.nonzero(strike_tgt >= 0).flatten().tolist():
        k = int(strike_tgt[i])
        # call the dive off when the approval lapses, the target dies (to this
        # drone or another), or the airframe is spent
        if req_status[k] != REQ_APPROVED or bool(wrecked[k]) or bool(expended[i]):
            strike_tgt[i] = -1
    for k in range(args.targets):
        if req_status[k] != REQ_APPROVED or bool(wrecked[k]) or bool((strike_tgt == k).any()):
            continue
        cand = _grp0 & ~expended & (strike_tgt < 0)
        cand = cand.clone()
        # the lead is never conscripted: its episode owns the vehicle set, so
        # expending it ends the whole sortie (and in manual it is the operator's)
        cand[0] = False
        if not bool(cand.any()):
            continue
        d = (pos[:, :2] - env.target_pos[:, k, :2]).norm(dim=1)
        d = torch.where(cand, d, torch.full_like(d, 1e9))
        i = int(d.argmin())
        strike_tgt[i] = k
        print(f"[warm] strike committed: drone {i} diving on target {k} "
              f"({float(d[i]):.0f} m out)", flush=True)


def strike_override(act):
    """Terminal guidance for drones in STRIKE: cruise to the target, then dive
    into it. Same body-frame/tanh action path the policy flies.

    The descent is thrust-aware: a naive "setpoint 20 m below" saturates the
    SE(3) vertical accel demand past -g, the world thrust vector inverts, the
    airframe chases an upside-down attitude and the task terminates it as a
    flip -- which is exactly how the first version lost every diver en route.
    So the vertical offset is floored each step at what keeps a_z above
    A_Z_MIN given the current sink rate, and a sink-rate brake caps the dive
    at VZ_MAX. The kill needs slant < reach_radius, not a ballistic impact.
    """
    idx = torch.nonzero(strike_tgt >= 0).flatten()
    if not len(idx):
        return act
    pos, vel, quat, _ = env.flight_state()
    tgt = env.target_pos[idx, strike_tgt[idx]]                    # [M,3]
    d = tgt - pos[idx]
    horiz = d[:, :2].norm(dim=1)
    yaw = torch.atan2(2 * (quat[idx, 0] * quat[idx, 3] + quat[idx, 1] * quat[idx, 2]),
                      1 - 2 * (quat[idx, 2] ** 2 + quat[idx, 3] ** 2))
    c, s = torch.cos(yaw), torch.sin(yaw)
    la = env.task.cfg.look_ahead
    lim = 0.97 * la
    agl = pos[idx, 2] - env.world.ground_at(pos[idx, 0], pos[idx, 1])
    # desired vertical offset (metres): hold cruise height en route, close on
    # the hull inside STRIKE_DIVE_AT
    off_z = torch.where(horiz > STRIKE_DIVE_AT, STRIKE_CRUISE_AGL - agl, d[:, 2])
    # thrust-aware floor: kp*off_z - kv*vz >= A_Z_MIN keeps f_z comfortably
    # positive; the brake overrides everything when the sink rate hits VZ_MAX
    A_Z_MIN, VZ_MAX = -5.5, 12.0
    vz = vel[idx, 2]
    off_z = torch.maximum(off_z, (A_Z_MIN + env.ctrl.kv * vz) / env.ctrl.kp)
    off_z = torch.where(vz < -VZ_MAX, torch.zeros_like(off_z), off_z)
    off_f = (c * d[:, 0] + s * d[:, 1]).clamp(-lim, lim)
    off_l = (-s * d[:, 0] + c * d[:, 1]).clamp(-lim, lim)
    off_z = off_z.clamp(-lim, lim)
    act = act.clone()
    act[idx, 0] = torch.atanh(off_f / la)          # tanh(a)*la = off, exactly
    act[idx, 1] = torch.atanh(off_l / la)
    act[idx, 2] = torch.atanh(off_z / la)
    return act


def sync_strike_hold():
    """Push the approval state into the task's neutralization gate."""
    for k, s in enumerate(req_status):
        strike_hold[:, k] = s != REQ_APPROVED
    strike_hold[~_grp0] = False   # other groups (if any) are not the operator's


def sync_engage():
    """Mask long-held targets out of the actor's belief (disengage-on-hold).

    A request un-approved for --disengage_s reads to the policy as a struck
    target, so its drones break off and resume the search; an APPROVED request
    re-engages immediately, whatever its age."""
    if args.disengage_s <= 0:
        return
    for k, s in enumerate(req_status):
        held = (s in (REQ_PENDING, REQ_AWAITING, REQ_DENIED)
                and mission_t[0] - req_detected_t[k] >= args.disengage_s)
        engage[_grp0, k] = not held


def reset_requests():
    req_status[:] = [REQ_NONE] * args.targets
    req_detected_t[:] = [0.0] * args.targets
    req_await_t[:] = [0.0] * args.targets
    engage[:] = True
    sync_strike_hold()


sync_strike_hold()

# --- scripted transit layer: navigator between engagements.
# The policy is a good closer (detect->strike in ~8 s) but its learned "search"
# is a saturated orbit (project memory: circling-root-cause). So a drone whose
# actor currently sees no pursuable target is flown by script instead: straight
# to the stalest coverage cell of ITS OWN sector -- the AO's cells are dealt
# round-robin across the fleet, so 16 drones sweep 16 disjoint slices instead
# of all re-scanning the same ground. The moment a target is visible to the
# actor (known, unreached, engaged) the policy takes the stick back.
# Actions go through the exact same body-frame/tanh path the policy uses.
TRANSIT_GAIN = 1.2               # pre-tanh stick -> ~21 m setpoint offset, ~17 m/s
TRANSIT_AGL = 65.0               # cruise height over ground: wide footprint, safe over trees
_G2 = env.task.cell_xy.shape[0]
_cellsector = torch.arange(_G2, device=env.device) % env.num_envs
_sector = _cellsector.unsqueeze(0) == torch.arange(env.num_envs,
                                                   device=env.device).unsqueeze(1)  # [N,G2]
transit_wp = env.task.cell_xy[torch.zeros(env.num_envs, dtype=torch.long,
                                          device=env.device)].clone()   # [N,2]
transit_next = torch.zeros(env.num_envs, device=env.device)  # steps until a re-pick


def transit_override(act):
    """Replace the action rows of no-target drones with a transect command."""
    pursuable = (env.task.known & ~env.task.reached & engage).any(dim=1)
    script = ~pursuable
    if not bool(script.any()):
        return act
    pos, _, quat, _ = env.flight_state()
    # re-pick when arrived or on a slow clock (a swept sector should move on)
    transit_next.sub_(1)
    d_wp = (transit_wp - pos[:, :2]).norm(dim=1)
    repick = script & ((d_wp < 35.0) | (transit_next <= 0))
    if bool(repick.any()):
        stale = env.task.recency + (~_sector).float() * 1e6   # own sector only
        cell = stale.argmin(dim=1)
        transit_wp[repick] = env.task.cell_xy[cell[repick]]
        transit_next[repick] = 10.0 / dt
    d = transit_wp - pos[:, :2]
    dirw = d / d.norm(dim=1, keepdim=True).clamp(min=1e-6)
    yaw = torch.atan2(2 * (quat[:, 0] * quat[:, 3] + quat[:, 1] * quat[:, 2]),
                      1 - 2 * (quat[:, 2] ** 2 + quat[:, 3] ** 2))
    c, s = torch.cos(yaw), torch.sin(yaw)
    fwd = (c * dirw[:, 0] + s * dirw[:, 1]) * TRANSIT_GAIN
    left = (-s * dirw[:, 0] + c * dirw[:, 1]) * TRANSIT_GAIN
    agl = pos[:, 2] - env.world.ground_at(pos[:, 0], pos[:, 1])
    up = ((TRANSIT_AGL - agl) / 25.0).clamp(-0.9, 0.9)
    act = act.clone()
    act[script, 0] = fwd[script]
    act[script, 1] = left[script]
    act[script, 2] = up[script]
    return act


# --- sortie recorder: the mission since the last reset, buffered in the same
# frame schema replay.json carries (vesper.native.replay), so a {"kind":"record"}
# command can dump it straight into a fresh runs/<id>/ and hand it to
# scripts/render_replay.py for tactical.mp4. Bounded: one frame per REC_EVERY
# control steps, and when the buffer hits REC_CAP it thins itself 2x and doubles
# the stride -- a long session keeps the whole mission at coarser sampling in a
# few MB of RAM instead of growing without limit.
REC_EVERY = 2                     # 25 Hz control -> 12.5 Hz logged
REC_CAP = 4500                    # ~6 min at 12.5 Hz before the first thinning
AUTO_KEEP = 10                    # auto-saved sorties kept; older ones are pruned
rec_buf: list[dict] = []
rec_every = REC_EVERY
rec_count = 0
rec_t = 0.0                       # mission clock: resets with each mission
manual_recorded = False           # this mission was kept by hand -> skip the auto-save
last_record: dict = {}
_REPO = _P(__file__).resolve().parents[1]

def _yaws_np(q):
    """World yaw per drone from wxyz quats [N,4] (matches vesper.native.replay)."""
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    return np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


def reset_recording():
    global rec_t, rec_every, rec_count, manual_recorded
    rec_buf.clear()
    rec_t, rec_every, rec_count = 0.0, REC_EVERY, 0
    manual_recorded = False


def _prune_auto_runs(keep=AUTO_KEEP):
    """Cap auto-saved sorties at `keep`, oldest deleted first. Only ever touches
    runs the session itself auto-saved (dir suffix AND manifest auto:true), so
    training / manual / flight runs are untouchable by construction."""
    autos = []
    for d in (_REPO / "runs").iterdir():
        if not d.is_dir() or not d.name.endswith("-auto-sortie"):
            continue
        try:
            m = json.loads((d / "manifest.json").read_text())
        except (OSError, json.JSONDecodeError):
            continue
        if m.get("kind") == "sortie" and m.get("auto") is True:
            autos.append((m.get("started") or 0, d))
    autos.sort()
    for _, d in autos[:-keep]:
        shutil.rmtree(d, ignore_errors=True)
        print(f"[warm] pruned old auto sortie runs/{d.name}", flush=True)


def dump_recording(frames, frame_dt, auto=False, renderers=None):
    """Write the buffered mission as a run: replay.json (vesper.native.replay
    schema) + trajectory.parquet + manifest.json into runs/<id>/, then kick off
    scripts/render_dispatch.py detached so the videos appear without stalling
    the sim loop. Runs on its own thread over a snapshot; touches no env state.

    auto=True is the end-of-episode auto-save: tagged {kind:"sortie", auto:true}
    in the manifest, pruned to the last AUTO_KEEP, and rendered tactical-only
    (cheap). auto=False is the operator's RECORD SORTIE keeper: never pruned,
    rendered by `renderers` — the record command's choice, else --renderers
    (default the fast on-device three.js lane + tactical; "isaac" adds the
    photoreal render on the droplet, the hero-shot lane)."""
    global last_record
    rends = ["tactical"] if auto else [r.strip() for r in
                                       (renderers or args.renderers).split(",") if r.strip()]
    run_id = time.strftime("%Y%m%d-%H%M%S-" + ("auto-sortie" if auto else "live-sortie"))
    run_dir = _REPO / "runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    now = time.time()
    dur = frames[-1]["t"] - frames[0]["t"]
    manifest = {"name": "auto sortie" if auto else "live sortie", "scene": world_name,
                "kind": "sortie", "auto": auto, "started": now - dur, "finished": now}
    if "isaac" in rends:
        manifest["isaac"] = "rendering"           # photoreal is on its way (or pending)
    (run_dir / "manifest.json").write_text(json.dumps(manifest))
    (run_dir / "replay.json").write_text(json.dumps({
        "world": world_name, "half_m": float(env.world.half_m), "dt": frame_dt,
        "targets": int(args.targets), "frames": frames}, separators=(",", ":")))
    tw = TrajectoryWriter(run_dir)
    for f in frames:                              # lead pose; yaw-only quat
        h = f["hdg"][0]
        tw.append(f["t"], f["d"][0], [math.cos(h / 2), 0.0, 0.0, math.sin(h / 2)])
    tw.close()
    if not auto:
        last_record = {"run": run_id, "at": round(now, 1)}
    print(f"[warm] recorded {len(frames)} frames -> runs/{run_id}"
          f" ({'auto' if auto else 'manual keeper'}; renderers: {','.join(rends)})", flush=True)
    subprocess.Popen([sys.executable, "scripts/render_dispatch.py", str(run_dir),
                      "--renderers", ",".join(rends)],
                     cwd=_REPO, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                     start_new_session=True)
    if auto:
        _prune_auto_runs()


# the most recent COMPLETED mission (frames, frame_dt): a manual RECORD SORTIE
# that lands right after an episode rollover would otherwise keep only the few
# seconds buffered since the reset -- a 7-second clip of drones taking off --
# so record falls back to this when the live buffer is still a stub
last_mission = None


def finish_mission():
    """The lead's episode just completed (all-cleared or the episode_s rollover).
    Auto-save the buffered mission when it is interesting -- >=1 neutralization
    relayed to the operator -- unless the operator already kept it by hand, then
    clear the buffer so the next mission records fresh."""
    global last_mission
    frames = list(rec_buf)
    interesting = len(frames) >= 12 and any(t[3] for t in frames[-1]["tg"])
    if len(frames) >= 12:
        last_mission = (frames, dt * rec_every)
    if interesting and not manual_recorded:
        threading.Thread(target=dump_recording, args=(frames, dt * rec_every, True),
                         daemon=True).start()
    reset_recording()


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
    global obs, t0, step, manual, manual_recorded
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
            expended[:] = False
            wrecked[:] = False
            strike_tgt[:] = -1
            reset_requests()
            reset_recording()
            print("[warm] reset", flush=True)
        elif kind == "deploy":
            p = cmd.get("policy")
            try:
                policy.load(p)
                obs = env.ppo_reset()
                t0, step = 0.0, 0
                comms_seen[:] = 0
                relayed_found[:] = relayed_reach[:] = False
                expended[:] = False
                wrecked[:] = False
                strike_tgt[:] = -1
                reset_requests()
                reset_recording()
                print(f"[warm] deployed {policy.name}", flush=True)
            except Exception as e:                          # noqa: BLE001
                print(f"[warm] deploy failed: {e}", flush=True)
        elif kind == "record":
            frames = list(rec_buf)                          # snapshot; buffer keeps rolling
            fdt = dt * rec_every
            # the mission just rolled over and the fresh buffer is a stub:
            # the operator means "save what I just watched" -> the completed one
            if last_mission and len(frames) * fdt < 20.0 \
                    and len(last_mission[0]) * last_mission[1] > len(frames) * fdt:
                print(f"[warm] record: only {len(frames) * fdt:.0f}s buffered since the "
                      f"rollover -> saving the last completed mission "
                      f"({len(last_mission[0]) * last_mission[1]:.0f}s) instead", flush=True)
                frames, fdt = last_mission
            if len(frames) < 12:
                print("[warm] record ignored: nothing buffered yet", flush=True)
            else:
                # renderer FLAG: the UI's choice rides the command ("renderers":
                # "three" | "isaac" | ... or a list); unset -> --renderers default
                req = cmd.get("renderers") or cmd.get("renderer")
                if isinstance(req, list):
                    req = ",".join(str(r) for r in req)
                if req:
                    req = ",".join(r for r in str(req).split(",")
                                   if r.strip() in ("tactical", "three", "isaac")) or None
                manual_recorded = True                      # keeper exists; skip the auto-save
                threading.Thread(target=dump_recording,
                                 args=(frames, fdt, False, req),
                                 daemon=True).start()
        elif kind in ("approve", "deny"):
            try:
                i = int(cmd.get("target", -1))
            except (TypeError, ValueError):
                i = -1
            # only a request the operator has actually seen can be decided;
            # a DENIED (or even APPROVED) one may be re-decided while it stands
            if 0 <= i < args.targets and req_status[i] in (REQ_AWAITING, REQ_APPROVED, REQ_DENIED):
                req_status[i] = REQ_APPROVED if kind == "approve" else REQ_DENIED
                sync_strike_hold()
                sync_engage()          # an approval re-engages the drones at once
                print(f"[warm] strike on target {i} {req_status[i]}", flush=True)
            else:
                print(f"[warm] {kind} ignored (target {cmd.get('target')}, "
                      f"status {req_status[i] if 0 <= i < args.targets else '?'})", flush=True)


try:
    while True:
        tick = time.time()
        apply_commands()
        act = policy.act(apply_shared_coverage(obs))
        if args.transit:
            act = transit_override(act)   # manual override below still wins for drone 0
        sync_strike_commits()
        act = strike_override(act)        # approved strikes: committed terminal dives
        if manual:
            # drone 0 belongs to the operator: body-frame stick through the same
            # action the policy uses, so WASD flies exactly what the policy would
            fresh = (time.time() - teleop["t"]) < args.deadman_s
            ax = teleop["axes"] if fresh else [0.0, 0.0, 0.0]
            act[0] = torch.tensor(ax, device=env.device) * stick
        if bool(expended.any()):
            act[expended] = 0.0        # a spent airframe flies nothing
        obs, rew, done, info = env.ppo_step(act)
        step += 1
        t = step * dt

        # --- loitering-munition accounting, BEFORE anything reads task state.
        # env.step already auto-reset whoever finished, so `info["strike"]` is
        # the only record of a kill that just registered.
        ep0 = int(env.episode_length_buf[0].item())
        rolled = ep0 < prev_ep0                   # the lead's episode just ended
        st = info["strike"]                       # [N,K] kills registered this step
        striker = st.any(dim=1)
        new_wreck = st[_grp0].any(dim=0)
        if rolled:
            # sortie over (lead expended on its strike / all cleared / rollover).
            # A kill on this final step would be lost with the fleet reset --
            # stamp it into the buffered mission before the auto-save.
            if bool(new_wreck.any()) and rec_buf:
                fin = copy.deepcopy(rec_buf[-1])
                for _k in torch.nonzero(new_wreck).flatten().tolist():
                    fin["tg"][_k][2] = fin["tg"][_k][3] = 1
                fin["dead"] = sorted(set(fin.get("dead", []))
                                     | set(torch.nonzero(striker).flatten().tolist()))
                rec_buf.append(fin)
            finish_mission()                      # auto-save the completed sortie
            relayed_found[:] = relayed_reach[:] = False
            reset_requests()                      # requests die with the sortie
            expended[:] = False                   # a new sortie fields a fresh fleet
            wrecked[:] = False
            strike_tgt[:] = -1
        elif bool(st.any()):
            for _k in torch.nonzero(new_wreck & ~wrecked).flatten().tolist():
                lag = (f", {t - req_await_t[_k]:.1f}s after approval"
                       if req_status[_k] == REQ_APPROVED else "")
                print(f"[warm] impact on target {_k} at t={t:.1f}s: tank destroyed, "
                      f"striking drone expended{lag}", flush=True)
            expended |= striker
            wrecked |= new_wreck
        # an episode that ended without a kill (crash/timeout) aborts any dive
        # that drone was flying; sync_strike_commits recommits the nearest
        aborted = done & (strike_tgt >= 0) & ~striker
        if bool(aborted.any()):
            for i in torch.nonzero(aborted).flatten().tolist():
                cause = ("crash" if bool(info["crash"][i]) else
                         "oob" if bool(info["oob"][i]) else
                         "flip" if bool(info["flip"][i]) else "rollover")
                print(f"[warm] dive aborted: drone {i} lost to {cause} en route "
                      f"to target {int(strike_tgt[i])}", flush=True)
            strike_tgt[aborted] = -1
        prev_ep0 = ep0
        sync_wrecks()

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
        expended_np = expended.cpu().numpy()
        stamp_comms(drones[~expended_np][:, :2])   # spent airframes confirm nothing
        # relay gate: a contact (or confirmed kill) commits to the operator's
        # map as soon as ANY surviving group-0 drone holds a link -- the same
        # relay lane the strike requests ride, not just the lead's own radio.
        # A non-lead loitering munition that kills deep in a dead zone stays
        # PENDING until some drone climbs back into coverage, then registers.
        if bool((linked & ~expended_np).any()):
            relayed_found |= env.task.known[_grp0 & ~expended].any(dim=0).cpu().numpy()
            relayed_reach |= (env.task.reached[_grp0].any(dim=0).cpu().numpy()
                              | wrecked.cpu().numpy())

        # engagement: ANY surviving group-0 drone's detection raises a strike
        # request (the hold gates the whole group, so the whole group's eyes
        # count); it reaches the operator once ANY surviving drone holds a link
        # (the relay lane, not just the lead's own radio); approval opens the gate
        mission_t[0] = float(t)
        known_any = env.task.known[_grp0 & ~expended].any(dim=0).cpu().numpy()
        any_link = bool((linked & ~expended_np).any())
        for k in range(args.targets):
            if req_status[k] == REQ_NONE and known_any[k]:
                req_status[k] = REQ_PENDING
                req_detected_t[k] = float(t)
                print(f"[warm] strike request raised on target {k} "
                      f"({'uplinked' if any_link else 'NO LINK -- egressing to call it in'})",
                      flush=True)
            if req_status[k] == REQ_PENDING and any_link:
                req_status[k] = REQ_AWAITING
                req_await_t[k] = float(t)
                print(f"[warm] request {k} uplinked; awaiting operator approval", flush=True)
            if (args.auto_approve_s is not None and req_status[k] == REQ_AWAITING
                    and t - req_await_t[k] >= args.auto_approve_s):
                req_status[k] = REQ_APPROVED
                print(f"[warm] strike on target {k} AUTO-APPROVED "
                      f"(--auto_approve_s {args.auto_approve_s:g})", flush=True)
        sync_strike_hold()
        sync_engage()
        n_by = {s: sum(1 for x in req_status if x == s) for s in
                (REQ_PENDING, REQ_AWAITING, REQ_APPROVED, REQ_DENIED)}

        # buffer this step for RECORD SORTIE (operator's picture: relayed truth)
        rec_t += dt
        rec_count += 1
        if rec_count % rec_every == 0:
            rec_buf.append({
                "t": round(rec_t, 2),
                "d": [[round(float(p[0]), 1), round(float(p[1]), 1), round(float(p[2]), 1)]
                      for p in drones],
                "hdg": [round(float(h), 3) for h in _yaws_np(quats)],
                "tg": [[round(float(tp[k][0]), 1), round(float(tp[k][1]), 1),
                        int(relayed_found[k]), int(relayed_reach[k])]
                       for k in range(args.targets)],
                "agl": round(float(info["agl"][0]), 1),
                "dead": [int(i) for i in np.nonzero(expended_np)[0]],
            })
            if len(rec_buf) > REC_CAP:            # thin 2x, keep the whole mission
                rec_buf[:] = rec_buf[1::2]
                rec_every *= 2

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
            # strike-approval queue: every live request, plus the HUD tally
            "requests": [{"target": k, "status": req_status[k],
                          "x": round(float(tp[k][0]), 1), "y": round(float(tp[k][1]), 1),
                          "detected_t": round(req_detected_t[k], 1),
                          "uplink": any_link,
                          "disengaged": not bool(engage[0, k]),
                          "committed": (int((strike_tgt == k).nonzero()[0])
                                        if bool((strike_tgt == k).any()) else None)}
                         for k in range(args.targets) if req_status[k] != REQ_NONE],
            "strikes": {"pending": n_by[REQ_PENDING], "awaiting": n_by[REQ_AWAITING],
                        "approved": n_by[REQ_APPROVED], "denied": n_by[REQ_DENIED]},
            # the last RECORD SORTIE dump ({run, at}), so the UI can confirm it
            "last_record": last_record or None,
            "comms_denied": round(comms_denied_frac, 3),
            "comms": {"n": COMMS_N, "half": _ao_half, "denied_frac": round(comms_denied_frac, 3),
                      "grid": (comms_seen + ord("0")).astype(np.uint8).tobytes().decode("ascii")},
            "expended": int(expended_np.sum()),   # airframes consumed by strikes
            "drones": [{"x": round(float(p[0]), 1), "y": round(float(p[1]), 1),
                        "z": round(float(p[2]), 1), "linked": bool(lk),
                        "expended": bool(ex),
                        "q": [round(float(v), 3) for v in q]}
                       for p, q, lk, ex in zip(drones, quats, linked, expended_np)],
            "vehicles": [{"x": round(float(tp[k][0]), 1), "y": round(float(tp[k][1]), 1),
                          "z": round(float(tp[k][2]), 1), "hdg": round(float(headings[k]), 2),
                          "found": bool(relayed_found[k]), "reached": bool(relayed_reach[k]),
                          "pending": bool(known[k] and not relayed_found[k])}
                         for k in range(args.targets)],
        })

        if fpv_cam is not None and step % args.every == 0:
            pos, vel, quat, _ = env.flight_state()
            alive = pos[~expended]                 # spent airframes carry no glyph
            with _feed_cv:
                _feed_slot["req"] = (pos[0].cpu().numpy(), quat[0].cpu().tolist(),
                                     vel[0, :2].cpu().numpy(), env.target_pos[0].cpu().tolist(),
                                     pos[1:][~expended[1:]].cpu().tolist(), alive.cpu().tolist())
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
