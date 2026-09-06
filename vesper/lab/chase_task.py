"""Chase: find a tank on the site with your own camera and detonate near it.

This is the task the end-to-end vision policy trains on. It is deliberately
smaller than the search task (vesper.lab.search_task): there is no belief, no
coverage grid, no assigned target. Some tanks drive around the site; the drone
launches from the launch zone, has to *see* one -- the camera is the only way --
fly close and choose when to detonate. A tank inside the small blast radius is
marked hit, and the sooner that happens the more it pays.

What is real and what is privileged:

  real (the actor's inputs)   the camera frame (RGB + depth) and the airframe's
                              own instruments (vesper.lab.frames.proprio)
  real (events)               detonation is the policy's fourth action. A hit is
                              a tank inside its blast radius; any physical
                              contact before detonation is a crash.
  privileged (training only)  the reward's shaping: distance closed on the
                              nearest tank, clearance from obstacles read from
                              the site map, whether a tank is in frame
                              (from the segmentation mask), the safe zones.
                              The critic and the state-based teacher see the
                              privileged vector; the actor never does.

Safe zones are friendly ground. The drone launches from a pad inside one, and
every step it spends over friendly ground costs, so leaving is the first thing
worth learning; the penalty is monotone in the distance to the boundary, so it
points the way out the whole time. A tank inside one is protected: detonating
near it yields no hit reward because a friendly vehicle is not a target.

Fallbacks keep the task CPU-testable and usable without a renderer: with no
contact sensor a crash is the map's solid layer; with no segmentation a
sighting is a geometric cone with line of sight.

Frame: world metres, x east, y north, z up. Shapes: N drones, K tanks.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from vesper.lab.frames import PROPRIO_DIM, camera_axis, proprio, tilt_from_quat, yaw_from_quat


@dataclass
class ChaseCfg:
    n_targets: int = 12                # tanks on the site, shared by every drone
    # The whole site, not a box inside it: the Cornell raster is 1200 m square,
    # and with oob_margin the hard limit lands exactly on its edge, so the drone
    # is bounded by the terrain rather than by an invisible wall inside it.
    arena_half: float = 590.0

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
    blast_radius: float = 4.0          # 3D radius around the drone at detonation (m)
    min_clearance: float = 1.0         # fallback crash: below solid + this
    clear_min: float = 6.0             # clearance below which the shaping penalty starts (m)
    tilt_limit: float = 1.4
    ceiling: float = 150.0
    oob_margin: float = 10.0

    # --- reward ---
    # Ordering, best to worst: hit early > hit late > see one > fly around
    # > time out > crash. The progress term is the only dense signal before the
    # first sighting is rewarded, and it is what makes the early episodes learn
    # anything at all; it is privileged and training-only.
    r_hit: float = 100.0               # detonating within radius of an unprotected tank
    r_hit_speed: float = 100.0         # extra, scaled by the fraction of the episode left
    r_sight: float = 10.0              # first frame each unprotected tank is in view
    r_hit_protected: float = 0.0       # protected tank in the radius (penalty if < 0)
    r_miss: float = 100.0              # prevents immediate empty detonation gaming time costs
    # Friendly ground. The drone launches inside it and every step spent there
    # costs, so the first thing worth learning is to leave -- and the penalty is
    # monotone in a signed distance to the boundary, so there is a gradient
    # pointing out the whole way rather than a flat plateau with no direction in
    # it: worst deep inside, half at the boundary, zero `safe_margin_m` beyond.
    # Nothing about it terminates the episode; the drone starts there.
    # The actor has no map, so on one fixed site it learns the boundary from
    # landmarks; `geofence` in the observation is the fallback if that is slow.
    w_safe: float = 0.5                # per step, scaled by the ramp below
    safe_margin_m: float = 25.0        # ramp width outside the boundary
    safe_depth_m: float = 50.0         # depth inside at which the penalty is maximal
    w_progress: float = 1.0            # per metre closed on the nearest unprotected tank
    w_time: float = 0.05               # per step
    w_clear: float = 0.5               # per step, scaled by (1 - clearance / clear_min)+
    r_crash: float = 100.0
    r_oob: float = 60.0
    r_flip: float = 60.0


class ChaseTask:
    """Reward, termination and observation vectors for N drones chasing K tanks."""

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
             detonated=None, crashed=None, seen_px=None, protected=None):
        """One control step. All world-frame [N,...]; target_pos [N,K,3].

        detonated [N] bool     the drone chose to explode this step
        crashed [N] bool       physical contact before detonation (None: map fallback)
        seen_px [N,K] int      mask pixels per tank in the frame (None: geometric fallback)
        protected [N,K] bool   tank inside a safe zone (None: none are)
        Returns (proprio obs [N,11], reward [N], terminated [N], info).
        """
        cfg, dev = self.cfg, drone_pos.device
        rel = target_pos - drone_pos.unsqueeze(1)
        slant = rel.norm(dim=2)
        if protected is None:
            protected = torch.zeros(self.n, self.k, dtype=torch.bool, device=dev)
        live = ~protected

        # --- events ---------------------------------------------------------
        if detonated is None:
            detonated = torch.zeros(self.n, dtype=torch.bool, device=dev)
        else:
            detonated = detonated.to(device=dev, dtype=torch.bool)
        hit_mask = detonated.unsqueeze(1) & (slant <= cfg.blast_radius)
        hit_live_mask = hit_mask & live
        hit = hit_live_mask.any(dim=1)
        hit_protected = (hit_mask & protected).any(dim=1) & ~hit
        miss = detonated & ~hit
        if seen_px is None:
            visible = self.detect(drone_pos, quat, target_pos)
        else:
            visible = seen_px >= cfg.sight_px
        new_sight = visible & live & ~self.seen
        self.seen |= visible

        ground = self.world.ground_at(drone_pos[:, 0], drone_pos[:, 1])
        agl = (drone_pos[:, 2] - ground).clamp(min=0.0)
        clear = self.clearance(drone_pos)

        # --- friendly ground, as it applies to the drone itself. The zone is the
        # whole column: being over it at altitude is still being over it.
        s_out, s_in = self.world.safe_field(drone_pos[:, 0], drone_pos[:, 1])
        in_safe = s_in > 0.0
        safe_ramp = torch.where(
            in_safe,
            0.5 + 0.5 * (s_in / cfg.safe_depth_m).clamp(0.0, 1.0),      # 0.5 at the line -> 1 deep in
            0.5 * (1.0 - s_out / cfg.safe_margin_m).clamp(0.0, 1.0),    # 0.5 at the line -> 0 well out
        )
        if crashed is None:
            solid = self.world.solid_at(drone_pos[:, 0], drone_pos[:, 1])
            crashed = drone_pos[:, 2] < solid + cfg.min_clearance
        # If the policy detonates on the same step as a contact, the deliberate
        # action owns the outcome. Otherwise touching a tank is just a crash.
        crashed = crashed & ~detonated
        rxy = drone_pos[:, :2].abs().amax(dim=1)
        oob = (rxy > cfg.arena_half + cfg.oob_margin) | (drone_pos[:, 2] > ground + cfg.ceiling)
        flip = tilt_from_quat(quat) > cfg.tilt_limit
        terminated = detonated | crashed | oob | flip

        # --- progress on the nearest unprotected tank -------------------------
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
        r = r + (cfg.r_hit + cfg.r_hit_speed * left) * hit.float()
        r = r + cfg.r_hit_protected * hit_protected.float()
        r = r - cfg.r_miss * miss.float()
        r = r - cfg.w_safe * safe_ramp
        r = r - cfg.r_crash * crashed.float() - cfg.r_oob * oob.float() - cfg.r_flip * flip.float()

        # --- what the belief head is trained on: the relative vector to the
        # nearest tank *that is actually in frame*. Asking the network to
        # place one it cannot see would teach it to hallucinate.
        d_vis = torch.where(visible & live, slant, torch.full_like(slant, 1e6))
        vis_d, vis_i = d_vis.min(dim=1)
        b_ok = vis_d < 1e5
        b_rel = torch.gather(rel, 1, vis_i.view(-1, 1, 1).expand(-1, 1, 3)).squeeze(1) / 100.0
        b_rel = torch.where(b_ok.unsqueeze(1), b_rel, torch.zeros_like(b_rel))

        obs = proprio(drone_vel, quat, ang_vel_b, agl, time_frac)
        info = {
            "belief_target": b_rel, "belief_ok": b_ok,
            "detonated": detonated, "hit": hit, "hit_mask": hit_live_mask,
            "hit_protected": hit_protected, "miss": miss,
            "time_to_hit": torch.where(hit, step_count.float() * self.dt,
                                       torch.full_like(nearest, float("nan"))),
            # The auto-reset moves the airframe before render scripts regain
            # control, so carry the actual blast transform in the step info.
            "explosion_pos": drone_pos.clone(), "explosion_quat": quat.clone(),
            "seen": (self.seen & live).any(dim=1),
            "visible": visible, "crash": crashed, "oob": oob, "flip": flip,
            "agl": agl, "clearance": clear, "nearest": nearest,
            "in_safe": in_safe, "safe_cost": safe_ramp, "geofence": self.geofence(drone_pos),
        }
        return obs, r, terminated, info

    def geofence(self, drone_pos):
        """[N] the zone signal, in [-1, 1]: 0 at the boundary, +1 well outside,
        -1 deep in friendly ground. Privileged as it stands (it needs the map).
        Appending it to the proprio vector turns the boundary into a geofence
        receiver the actor can sense -- the fallback if landmarks are slow."""
        cfg = self.cfg
        s_out, s_in = self.world.safe_field(drone_pos[:, 0], drone_pos[:, 1])
        return ((s_out / cfg.safe_margin_m).clamp(max=1.0)
                - (s_in / cfg.safe_depth_m).clamp(max=1.0))

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
