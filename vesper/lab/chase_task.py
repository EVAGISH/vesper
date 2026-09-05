"""Chase: find a forklift on the site with your own camera and hit it, fast.

This is the task the end-to-end vision policy trains on. It is deliberately
smaller than the search task (vesper.lab.search_task): there is no belief, no
coverage grid, no assigned target. Some forklifts drive around the site; the
drone launches from the launch zone, has to *see* one -- the camera is the only
way -- fly to it through the trees and buildings, and touch it. The touch ends
the episode, and the sooner it comes the more it pays.

What is real and what is privileged:

  real (the actor's inputs)   the camera frame (RGB + depth) and the airframe's
                              own instruments (vesper.lab.frames.proprio)
  real (events)               a touch is a PhysX contact between the airframe
                              and a forklift; a crash is a contact with
                              anything else. The env reads both off a contact
                              sensor and hands them in.
  privileged (training only)  the reward's shaping: distance closed on the
                              nearest forklift, clearance from obstacles read
                              from the site map, whether a forklift is in frame
                              (from the segmentation mask), the safe zones.
                              The critic and the state-based teacher see the
                              privileged vector; the actor never does.

Safe zones are places the drone should not be. A forklift inside one is
protected -- no sighting bonus, no touch reward, touching it ends nothing -- and
the drone itself pays a per-step penalty that ramps up as it approaches the
boundary and ends the episode if it goes well inside.

Fallbacks keep the task CPU-testable and usable without a renderer: with no
contact sensor a touch is a small radius and a crash is the map's solid
layer; with no segmentation a sighting is a geometric cone with line of sight.

Frame: world metres, x east, y north, z up. Shapes: N drones, K forklifts.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from vesper.lab.frames import PROPRIO_DIM, camera_axis, proprio, tilt_from_quat, yaw_from_quat


@dataclass
class ChaseCfg:
    n_targets: int = 6                 # forklifts on the site, shared by every drone
    arena_half: float = 300.0          # square about the world centre the episode lives in

    # --- drone spawn ---
    spawn_alt_min: float = 25.0        # above local ground (m)
    spawn_alt_max: float = 60.0

    # --- camera: body-fixed, pitched forward and down (see search_task for why) ---
    cam_pitch_deg: float = 40.0
    fov_half_deg: float = 55.0
    sight_px: int = 4                  # mask pixels in frame that count as "seen"
    detect_range: float = 200.0        # geometric fallback only

    # --- guidance action: body-frame velocity command ---
    look_ahead: float = 25.0

    # --- events ---
    touch_radius: float = 2.5          # fallback when there is no contact sensor (m)
    min_clearance: float = 1.0         # fallback crash: below solid + this
    clear_min: float = 6.0             # clearance below which the shaping penalty starts (m)
    tilt_limit: float = 1.4
    ceiling: float = 150.0
    oob_margin: float = 40.0

    # --- reward ---
    # Ordering, best to worst: touch early > touch late > see one > fly around
    # > time out > crash. The progress term is the only dense signal before the
    # first sighting is rewarded, and it is what makes the early episodes learn
    # anything at all; it is privileged and training-only.
    r_touch: float = 100.0             # touching an unprotected forklift
    r_touch_speed: float = 100.0       # extra, scaled by the fraction of the episode left
    r_sight: float = 10.0              # first frame each unprotected forklift is in view
    r_touch_protected: float = 0.0     # touching a forklift inside a safe zone (penalty if < 0)
    # The safe zone is a place the drone should not be, not merely a place where
    # targets do not count. The penalty ramps up over `safe_margin_m` *outside*
    # the boundary so there is a gradient pushing away from it rather than a
    # cliff the policy can only find by crossing; it is flat and maximal inside;
    # and going more than `safe_breach_m` in ends the episode the way leaving
    # the arena does. Note the actor has no map: on one fixed site it learns the
    # boundary from landmarks, and `geofence` in the observation is the fallback
    # if that proves too slow.
    w_safe: float = 1.0                # per step, scaled by the ramp below
    safe_margin_m: float = 25.0        # ramp width outside the boundary
    safe_breach_m: float = 15.0        # this far inside ends the episode
    r_safe: float = 80.0               # one-off penalty on a breach
    w_progress: float = 1.0            # per metre closed on the nearest unprotected forklift
    w_time: float = 0.05               # per step
    w_clear: float = 0.5               # per step, scaled by (1 - clearance / clear_min)+
    r_crash: float = 100.0
    r_oob: float = 60.0
    r_flip: float = 60.0


class ChaseTask:
    """Reward, termination and observation vectors for N drones chasing K forklifts."""

    def __init__(self, world, cfg: ChaseCfg, num_envs: int, dt: float, max_steps: int,
                 device="cpu", generator=None):
        self.world, self.cfg, self.dt = world, cfg, float(dt)
        self.n, self.k = int(num_envs), int(cfg.n_targets)
        self.max_steps = int(max_steps)
        self.device = device
        self.gen = generator
        self.seen = torch.zeros(self.n, self.k, dtype=torch.bool, device=device)
        self.prev_dist = torch.zeros(self.n, device=device)
        self.cos_fov = math.cos(math.radians(cfg.fov_half_deg))
        self.pitch = math.radians(cfg.cam_pitch_deg)
        # clearance probes: three rings of eight, in metres
        ring = torch.tensor([[math.cos(a), math.sin(a)] for a in torch.arange(8) * (math.pi / 4)],
                            device=device)
        self.probe = torch.cat([ring * 3.0, ring * 6.0, ring * 10.0], dim=0)      # [24,2]
        self.probe_r = self.probe.norm(dim=1)                                      # [24]
        self.obs_dim = PROPRIO_DIM
        self.priv_dim = 12 + 6 * self.k + 2

    # ------------------------------------------------------------------ reset
    def reset(self, env_ids):
        self.seen[env_ids] = False
        self.prev_dist[env_ids] = 0.0

    # ------------------------------------------------------------------ helpers
    def detect(self, drone_pos, quat, target_pos):
        """Geometric fallback sighting: in the camera cone, in range, line of sight clear."""
        rel = target_pos - drone_pos.unsqueeze(1)
        slant = rel.norm(dim=2)
        axis = camera_axis(quat, self.pitch)
        in_cone = (rel * axis.unsqueeze(1)).sum(dim=2) / slant.clamp(min=1e-6) >= self.cos_fov
        p0 = drone_pos.unsqueeze(1).expand(-1, self.k, -1).reshape(-1, 3)
        clear, _ = self.world.trace(p0, target_pos.reshape(-1, 3))
        return in_cone & clear.view(self.n, self.k) & (slant < self.cfg.detect_range)

    def clearance(self, drone_pos):
        """[N] metres to the nearest solid thing: the ground or a building or a
        hard tree below, or one of those rising above the airframe within a
        probe ring. Read from the site map: privileged, for the reward only."""
        x, y, z = drone_pos[:, 0], drone_pos[:, 1], drone_pos[:, 2]
        below = z - self.world.solid_at(x, y)
        px = x.unsqueeze(1) + self.probe[:, 0].unsqueeze(0)
        py = y.unsqueeze(1) + self.probe[:, 1].unsqueeze(0)
        top = self.world.solid_at(px, py)                                          # [N,24]
        hit = top > (z.unsqueeze(1) - 1.0)                                         # rises to the airframe
        side = torch.where(hit, self.probe_r.unsqueeze(0), torch.full_like(top, 1e3)).amin(dim=1)
        return torch.minimum(below, side).clamp(min=0.0)

    # ------------------------------------------------------------------ step
    def step(self, drone_pos, drone_vel, quat, ang_vel_b, target_pos, step_count,
             touched=None, crashed=None, seen_px=None, protected=None):
        """One control step. All world-frame [N,...]; target_pos [N,K,3].

        touched [N,K] bool     contact with each forklift this step (None: radius fallback)
        crashed [N] bool       contact with anything that is not a forklift (None: map fallback)
        seen_px [N,K] int      mask pixels per forklift in the frame (None: geometric fallback)
        protected [N,K] bool   forklift inside a safe zone (None: none are)
        Returns (proprio obs [N,11], reward [N], terminated [N], info).
        """
        cfg, dev = self.cfg, drone_pos.device
        rel = target_pos - drone_pos.unsqueeze(1)
        slant = rel.norm(dim=2)
        if protected is None:
            protected = torch.zeros(self.n, self.k, dtype=torch.bool, device=dev)
        live = ~protected

        # --- events ---------------------------------------------------------
        if touched is None:
            touched = slant < cfg.touch_radius
        touch_live = (touched & live).any(dim=1)
        touch_prot = (touched & protected).any(dim=1) & ~touch_live
        if seen_px is None:
            visible = self.detect(drone_pos, quat, target_pos)
        else:
            visible = seen_px >= cfg.sight_px
        new_sight = visible & live & ~self.seen
        self.seen |= visible

        ground = self.world.ground_at(drone_pos[:, 0], drone_pos[:, 1])
        agl = (drone_pos[:, 2] - ground).clamp(min=0.0)
        clear = self.clearance(drone_pos)

        # --- the safe zone, as it applies to the drone itself. The zone is the
        # whole column: flying over it at altitude is still being over it.
        s_out, s_in = self.world.safe_field(drone_pos[:, 0], drone_pos[:, 1])
        in_safe = s_in > 0.0
        near = (1.0 - s_out / cfg.safe_margin_m).clamp(0.0, 1.0)
        safe_ramp = torch.where(in_safe, torch.ones_like(near), near)
        breach = s_in > cfg.safe_breach_m
        if crashed is None:
            solid = self.world.solid_at(drone_pos[:, 0], drone_pos[:, 1])
            crashed = drone_pos[:, 2] < solid + cfg.min_clearance
        crashed = crashed & ~touch_live
        rxy = drone_pos[:, :2].abs().amax(dim=1)
        oob = (rxy > cfg.arena_half + cfg.oob_margin) | (drone_pos[:, 2] > ground + cfg.ceiling)
        flip = tilt_from_quat(quat) > cfg.tilt_limit
        terminated = touch_live | crashed | oob | flip | breach

        # --- progress on the nearest unprotected forklift ---------------------
        d_live = torch.where(live, slant, torch.full_like(slant, 1e6))
        nearest, _ = d_live.min(dim=1)
        has = live.any(dim=1)
        progress = torch.where(has & (self.prev_dist > 0), self.prev_dist - nearest,
                               torch.zeros_like(nearest)).clamp(-3.0, 3.0)
        self.prev_dist = torch.where(has, nearest, torch.zeros_like(nearest))

        # --- reward -----------------------------------------------------------
        time_frac = (step_count.float() / self.max_steps).clamp(0, 1)
        left = 1.0 - time_frac
        r = cfg.w_progress * progress - cfg.w_time
        r = r - cfg.w_clear * (1.0 - clear / cfg.clear_min).clamp(min=0.0)
        r = r + cfg.r_sight * new_sight.float().sum(dim=1)
        r = r + (cfg.r_touch + cfg.r_touch_speed * left) * touch_live.float()
        r = r + cfg.r_touch_protected * touch_prot.float()
        r = r - cfg.w_safe * safe_ramp
        r = r - cfg.r_safe * breach.float()
        r = r - cfg.r_crash * crashed.float() - cfg.r_oob * oob.float() - cfg.r_flip * flip.float()

        # --- what the belief head is trained on: the relative vector to the
        # nearest forklift *that is actually in frame*. Asking the network to
        # place one it cannot see would teach it to hallucinate.
        d_vis = torch.where(visible & live, slant, torch.full_like(slant, 1e6))
        vis_d, vis_i = d_vis.min(dim=1)
        b_ok = vis_d < 1e5
        b_rel = torch.gather(rel, 1, vis_i.view(-1, 1, 1).expand(-1, 1, 3)).squeeze(1) / 100.0
        b_rel = torch.where(b_ok.unsqueeze(1), b_rel, torch.zeros_like(b_rel))

        obs = proprio(drone_vel, quat, ang_vel_b, agl, time_frac)
        info = {
            "belief_target": b_rel, "belief_ok": b_ok,
            "touch": touch_live, "touch_protected": touch_prot,
            "time_to_touch": torch.where(touch_live, step_count.float() * self.dt,
                                         torch.full_like(nearest, float("nan"))),
            "seen": (self.seen & live).any(dim=1),
            "visible": visible, "crash": crashed, "oob": oob, "flip": flip,
            "agl": agl, "clearance": clear, "nearest": nearest,
            "in_safe": in_safe, "safe_breach": breach, "geofence": self.geofence(drone_pos),
        }
        return obs, r, terminated, info

    def geofence(self, drone_pos):
        """[N] the zone signal, in [-1, 1]: 0 at the boundary, +1 well outside,
        -1 well inside. Privileged as it stands (it needs the map). Appending it
        to the proprio vector turns the safe zone into a geofence receiver the
        actor can actually sense -- the fallback if landmark learning is slow."""
        cfg = self.cfg
        s_out, s_in = self.world.safe_field(drone_pos[:, 0], drone_pos[:, 1])
        return ((s_out - s_in) / cfg.safe_margin_m).clamp(-1.0, 1.0)

    # ------------------------------------------------------------------ obs
    def privileged(self, drone_pos, drone_vel, quat, ang_vel_b, target_pos, visible, protected,
                   time_frac):
        """[N, 12 + 6K + 2] -- the truth, for the critic and a teacher only."""
        A = self.cfg.arena_half
        ground = self.world.ground_at(drone_pos[:, 0], drone_pos[:, 1])
        agl = (drone_pos[:, 2] - ground).clamp(min=0.0)
        yaw = yaw_from_quat(quat)
        self_block = torch.stack([
            drone_pos[:, 0] / A, drone_pos[:, 1] / A, agl / 100.0,
            drone_vel[:, 0] / 15.0, drone_vel[:, 1] / 15.0, drone_vel[:, 2] / 15.0,
            tilt_from_quat(quat), torch.sin(yaw), torch.cos(yaw),
            ang_vel_b[:, 0], ang_vel_b[:, 1], ang_vel_b[:, 2],
        ], dim=1)
        rel = target_pos - drone_pos.unsqueeze(1)
        tgt = torch.cat([
            rel[..., :2] / A, rel[..., 2:3] / 50.0,
            rel.norm(dim=2, keepdim=True) / A,
            visible.float().unsqueeze(2), protected.float().unsqueeze(2),
        ], dim=2).reshape(self.n, -1)
        return torch.cat([self_block, tgt, self.geofence(drone_pos).unsqueeze(1),
                          time_frac.unsqueeze(1)], dim=1)
