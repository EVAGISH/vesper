"""Search-and-reach: the drone does not know where the vehicles are.

The pursuit task (vesper.lab.pursuit_task) hands the policy a relative vector to
its target every step. That is a homing problem, not a search problem. Here the
drone starts somewhere random over a real 1.2 km site with several tanks
scattered across it -- some driving in the open, some parked under tree canopy,
some painted to blend in -- and the only way it learns where any of them are is
by looking.

The look is a body-fixed camera pitched forward and down, the way a real
airframe carries one: it sees where it is flying, and it sees the ground ahead.
There are two ways a sighting is decided, and the task supports both:

  pixels    the env renders that camera and hands `step()` how many pixels of
            each vehicle's segmentation mask are in the frame. This is the real
            thing: occlusion, foliage and range are whatever the renderer says.
  geometry  no renderer: a cone around the camera axis, limited slant range,
            blocked by terrain and buildings, attenuated by foliage, degraded by
            a target's own contrast. Cheap, runs at thousands of envs, and is
            what a state-based teacher policy trains against.

The policy is GPS-denied. Two observation vectors come out of here:

  proprio     what the airframe can measure on its own: body-frame velocity
              (optical flow / VIO class), gravity direction in the body frame
              (attitude from the IMU), body rates, rangefinder height, clock.
              Together with the camera image this is all the actor gets.
  privileged  world position, the belief (per target: ever seen, last fix, its
              staleness) and a coverage grid of what has already been swept.
              For the critic and for a state-based teacher only.

The reward pays for sweeping ground, for first sightings, for closing on a
known target, and most of all for clearing every target quickly. It is a
training-time signal and may use the truth freely.

Everything here is pure torch over a vesper.worlds.heightmap.WorldMap, so the
whole task -- sensor, belief, reward, termination -- runs and is tested on a Mac
CPU against a synthetic map. The Isaac env is a thin shell around it.

Frame: world metres, x east, y north, z up. Body: x forward, y left, z up.
Shapes: N envs, K targets.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch


@dataclass
class SearchCfg:
    n_targets: int = 3
    arena_half: float = 300.0          # search box half-extent about the world centre (m)
    grid: int = 8                      # coverage cells per side

    # --- drone spawn ---
    spawn_alt_min: float = 35.0        # above local ground (m)
    spawn_alt_max: float = 90.0

    # --- camera: body-fixed, pitched forward and down ---
    # 40 deg below the horizon with a 110 deg lens spans from 15 deg above the
    # horizon (the drone sees where it is going) to 5 deg past nadir (a target
    # it is descending onto stays in frame all the way in). The same numbers
    # drive the rendered TiledCamera in the env and the geometric cone here, so
    # a teacher trained on geometry and a student trained on pixels look at the
    # same patch of ground.
    cam_pitch_deg: float = 40.0        # camera axis below horizontal
    fov_half_deg: float = 55.0         # half-angle of the lens cone
    detect_range: float = 220.0        # slant range on a plain target in clear air (m)
    min_detect_range: float = 15.0     # you can always see it from close enough
    canopy_k: float = 0.15             # extinction per density-weighted metre of foliage
    miss_p: float = 0.04               # per-step dropout on an otherwise valid detection
    fix_noise_m: float = 2.5           # error on a reported fix
    stale_tau_s: float = 8.0           # a fix this old tells you little about a moving target
    sight_px: int = 6                  # pixel mode: mask pixels in frame that count as a sighting

    # --- coverage memory ---
    recency_tau_s: float = 25.0        # how fast a swept cell goes stale in the observation

    # --- guidance action: body-frame velocity command ---
    look_ahead: float = 25.0           # tanh(action) * look_ahead = setpoint offset (m)

    # --- success / failure geometry ---
    reach_radius: float = 10.0         # 3D range to a vehicle that counts as reaching it
                                       # (the tank is several metres long; 8 m is 'arrived at it',
                                       #  and it is what makes the first reaches happen at all)
    min_clearance: float = 1.5         # below solid_top + this (and not reaching) = crash
    tilt_limit: float = 1.4            # rad from upright
    ceiling: float = 200.0             # above local ground
    oob_margin: float = 40.0           # beyond arena_half + this = out of bounds

    # --- reward ---
    # Ordering the shaping has to preserve, best to worst:
    #   clear every vehicle fast > clear them slowly > find some > sweep ground
    #   > run out of time > crash.
    # The first version of these weights paid 192 for sweeping the box and 150 for
    # three first sightings, both reachable by flying high and fast, against a
    # reach bonus the policy had to *discover* by descending onto a 8 m sphere.
    # It never did: 130 iterations, thousands of episodes, exactly zero reaches,
    # coverage pinned at 0.6. Sweeping and sighting are now worth roughly a third
    # of what they were, closing is worth more per metre, and w_proximity adds a
    # dense pull over the last few tens of metres so the final approach has a
    # gradient instead of being a lottery.
    w_progress: float = 1.2            # per metre closed on the nearest known target
    w_time: float = 0.02               # per-step cost: the pressure to finish
    w_cover: float = 1.2               # per coverage cell swept for the first time
    w_proximity: float = 1.0           # per step, w_proximity / (1 + dist/20)
    w_foliage: float = 0.05            # per step spent inside a crown (branch strikes)
    r_detect: float = 30.0             # first sighting of a vehicle
    r_reach: float = 120.0             # reaching one
    r_speed: float = 80.0              # extra on a reach, scaled by episode left
    r_complete: float = 150.0          # all vehicles reached, scaled by episode left
    r_crash: float = 80.0              # flew into ground or a building
    r_oob: float = 60.0
    r_flip: float = 60.0


from vesper.lab.frames import (PROPRIO_DIM, camera_axis, proprio as _proprio, quat_mul, seg_counts,  # noqa: F401,E402
                               seg_lookup, sensor_pose, tilt_from_quat, yaw_from_quat)
from vesper.lab.frames import setpoint as _setpoint  # noqa: E402


def setpoint(drone_pos_w, action, yaw, cfg: SearchCfg):
    """Body-frame guidance action -> world setpoint (see vesper.lab.frames)."""
    return _setpoint(drone_pos_w, action, yaw, cfg.look_ahead)


class SearchTask:
    """Belief, coverage, reward and termination for N parallel searches.

    Owns only tensors; the caller owns the physics. Call `reset(env_ids)` when
    environments restart and `step(...)` once per control step.
    """

    def __init__(self, world, cfg: SearchCfg, num_envs: int, dt: float, max_steps: int,
                 device="cpu", generator=None):
        self.world, self.cfg, self.dt = world, cfg, float(dt)
        self.n, self.k = int(num_envs), int(cfg.n_targets)
        self.max_steps = int(max_steps)
        self.device = device
        self.gen = generator
        g, n, k = cfg.grid, self.n, self.k

        self.known = torch.zeros(n, k, dtype=torch.bool, device=device)
        self.reached = torch.zeros(n, k, dtype=torch.bool, device=device)
        self.fix = torch.zeros(n, k, 3, device=device)
        self.fix_age = torch.zeros(n, k, device=device)
        self.contrast = torch.ones(n, k, device=device)
        self.visited = torch.zeros(n, g * g, dtype=torch.bool, device=device)
        self.recency = torch.zeros(n, g * g, device=device)
        self.prev_dist = torch.zeros(n, device=device)

        self.tan_fov = math.tan(math.radians(cfg.fov_half_deg))
        self.cos_fov = math.cos(math.radians(cfg.fov_half_deg))
        self.pitch = math.radians(cfg.cam_pitch_deg)
        self.set_arena(cfg.arena_half)
        self.obs_dim = 12 + 8 * k + g * g + 3          # privileged width
        self.proprio_dim = PROPRIO_DIM

    def set_arena(self, half: float):
        """Resize the search box, rebuilding the coverage grid over it.

        The curriculum grows the box during training, and the grid is defined in
        world metres -- leaving it at the old size would keep scoring coverage of
        a box that no longer exists.
        """
        g, dev = self.cfg.grid, self.device
        self.cfg.arena_half = float(half)
        c = (torch.arange(g, device=dev) + 0.5) / g * (2 * half) - half
        cx, cy = torch.meshgrid(c, c, indexing="xy")
        self.cell_xy = torch.stack([cx.reshape(-1), cy.reshape(-1)], dim=1)      # [G*G,2]
        # A cell counts as swept when the camera footprint touches it, not when
        # it happens to contain the cell's centre: with 75 m cells and a 40 m
        # footprint the centre test credits a low pass with nothing at all, and
        # the policy sees no reason to ever fly low.
        self.cell_reach = 0.70711 * (2 * half) / g

    # ------------------------------------------------------------------ reset
    def reset(self, env_ids, contrast=None):
        self.known[env_ids] = False
        self.reached[env_ids] = False
        self.fix[env_ids] = 0.0
        self.fix_age[env_ids] = 0.0
        self.visited[env_ids] = False
        self.recency[env_ids] = 0.0
        self.prev_dist[env_ids] = 0.0
        if contrast is not None:
            self.contrast[env_ids] = contrast

    # ------------------------------------------------------------------ sensor
    def detect(self, drone_pos, quat, target_pos):
        """Geometric sighting. Returns (visible [N,K], slant [N,K]).

        A target is seen when it is inside the camera cone, inside the effective
        range for its contrast and the foliage in the way, with unbroken line of
        sight over terrain and buildings -- and the sensor does not happen to miss
        it this frame.
        """
        cfg = self.cfg
        rel = target_pos - drone_pos.unsqueeze(1)                     # [N,K,3]
        slant = rel.norm(dim=2)
        axis = camera_axis(quat, self.pitch)                          # [N,3]
        cos_off = (rel * axis.unsqueeze(1)).sum(dim=2) / slant.clamp(min=1e-6)
        in_cone = cos_off >= self.cos_fov

        p0 = drone_pos.unsqueeze(1).expand(-1, self.k, -1).reshape(-1, 3)
        p1 = target_pos.reshape(-1, 3)
        clear, foliage = self.world.trace(p0, p1)
        clear = clear.view(self.n, self.k)
        foliage = foliage.view(self.n, self.k)

        transmit = torch.exp(-cfg.canopy_k * foliage)
        r_eff = (cfg.detect_range * self.contrast * transmit).clamp(min=cfg.min_detect_range)
        hit = in_cone & clear & (slant < r_eff) & ~self.reached
        if cfg.miss_p > 0:
            keep = torch.rand(hit.shape, device=hit.device, generator=self.gen) >= cfg.miss_p
            hit = hit & keep
        return hit, slant

    def footprint(self, drone_xy, agl, quat):
        """Ground patch the camera covers: (centre [N,2], radius [N]).

        The true intersection of a tilted cone with the terrain is a distorted
        ellipse that runs to the horizon when the lens reaches above it. For
        paying coverage a disc is enough: centred where the camera axis meets
        the ground, radius the nadir footprint, both capped at detect range.
        """
        cfg = self.cfg
        axis = camera_axis(quat, self.pitch)
        down = (-axis[:, 2]).clamp(min=0.2)
        horiz = axis[:, :2]
        hn = horiz.norm(dim=1).clamp(min=1e-6)
        ahead = (agl / down * hn).clamp(max=cfg.detect_range * math.cos(self.pitch))
        centre = drone_xy + horiz / hn.unsqueeze(1) * ahead.unsqueeze(1)
        radius = torch.minimum(agl * self.tan_fov, torch.full_like(agl, cfg.detect_range))
        return centre, radius

    # ------------------------------------------------------------------ step
    def step(self, drone_pos, drone_vel, quat, ang_vel_b, target_pos, step_count, seen_px=None):
        """One control step of belief + reward. All inputs world-frame [N,...].

        `seen_px` [N,K]: pixels of each target's mask in this env's rendered
        frame. Given, a sighting is decided by the renderer; None falls back to
        the geometric cone. Returns (privileged obs [N,obs_dim], reward [N],
        terminated [N], info dict).
        """
        cfg, dev = self.cfg, drone_pos.device
        if seen_px is None:
            visible, slant = self.detect(drone_pos, quat, target_pos)
        else:
            slant = (target_pos - drone_pos.unsqueeze(1)).norm(dim=2)
            visible = (seen_px >= cfg.sight_px) & ~self.reached

        # --- belief update -------------------------------------------------
        new_find = visible & ~self.known
        self.known |= visible
        noise = torch.randn(target_pos.shape, device=dev, generator=self.gen) * cfg.fix_noise_m
        seen3 = visible.unsqueeze(2)
        self.fix = torch.where(seen3, target_pos + noise, self.fix)
        self.fix_age = torch.where(visible, torch.zeros_like(self.fix_age), self.fix_age + self.dt)

        # --- reaching ------------------------------------------------------
        touching = (slant < cfg.reach_radius) & ~self.reached
        new_reach = touching
        self.reached |= touching
        self.known |= touching
        all_done = self.reached.all(dim=1)

        # --- coverage ------------------------------------------------------
        ground = self.world.ground_at(drone_pos[:, 0], drone_pos[:, 1])
        agl = (drone_pos[:, 2] - ground).clamp(min=0.0)
        centre, foot = self.footprint(drone_pos[:, :2], agl, quat)
        d_cell = (self.cell_xy.unsqueeze(0) - centre.unsqueeze(1)).norm(dim=2)
        swept = d_cell <= (foot + self.cell_reach).unsqueeze(1)
        fresh = swept & ~self.visited
        self.visited |= swept
        self.recency = self.recency * math.exp(-self.dt / cfg.recency_tau_s)
        self.recency = torch.where(swept, torch.ones_like(self.recency), self.recency)

        # --- progress toward the nearest known, unreached target ------------
        pursuable = self.known & ~self.reached
        big = torch.full_like(slant, 1e6)
        fix_rel = self.fix - drone_pos.unsqueeze(1)
        fix_dist = torch.where(pursuable, fix_rel.norm(dim=2), big)
        nearest_d, nearest_i = fix_dist.min(dim=1)
        has_target = pursuable.any(dim=1)
        progress = torch.where(has_target & (self.prev_dist > 0),
                               self.prev_dist - nearest_d, torch.zeros_like(nearest_d))
        # A step cannot really close more than ~1 m at these speeds, so clamp:
        # when the nearest known target changes (a new sighting, or one just
        # reached) the baseline belongs to a different vehicle and the raw
        # difference is meaningless.
        progress = progress.clamp(-3.0, 3.0)
        self.prev_dist = torch.where(has_target, nearest_d, torch.zeros_like(nearest_d))

        # --- failures ------------------------------------------------------
        solid = self.world.solid_at(drone_pos[:, 0], drone_pos[:, 1])
        reaching_now = touching.any(dim=1)
        crash = (drone_pos[:, 2] < solid + cfg.min_clearance) & ~reaching_now
        # The arena is a square and vehicles spawn anywhere in it, so the bound has
        # to be square too. A radial test at arena_half + margin puts the limit at
        # 340 m while the box's own corners are at 424 m: a drone flying to a
        # corner of its own search area was being killed for leaving it, and an
        # eval put 49% of all episodes on exactly that.
        rxy = drone_pos[:, :2].abs().amax(dim=1)
        oob = (rxy > cfg.arena_half + cfg.oob_margin) | (drone_pos[:, 2] > ground + cfg.ceiling)
        tilt = tilt_from_quat(quat)
        flip = tilt > cfg.tilt_limit
        terminated = crash | oob | flip | all_done

        # --- reward --------------------------------------------------------
        time_frac = (step_count.float() / self.max_steps).clamp(0, 1)
        left = (1.0 - time_frac).clamp(0.0, 1.0)
        canopy_top = self.world.canopy_at(drone_pos[:, 0], drone_pos[:, 1])
        in_foliage = (drone_pos[:, 2] < canopy_top) & (drone_pos[:, 2] > ground)

        r = cfg.w_progress * progress
        r = r - cfg.w_time
        r = r + cfg.w_cover * fresh.float().sum(dim=1)
        # dense homing on the nearest known target: without it the last 50 m of
        # the approach carries almost no gradient and is never explored
        r = r + torch.where(has_target,
                            cfg.w_proximity / (1.0 + nearest_d / 20.0),
                            torch.zeros_like(nearest_d))
        r = r + cfg.r_detect * new_find.float().sum(dim=1)
        r = r + (cfg.r_reach + cfg.r_speed * left.unsqueeze(1)).mul(new_reach.float()).sum(dim=1)
        r = r - cfg.w_foliage * in_foliage.float()
        r = r + cfg.r_complete * left * all_done.float()
        r = r - cfg.r_crash * crash.float()
        r = r - cfg.r_oob * oob.float()
        r = r - cfg.r_flip * flip.float()

        obs = self.privileged(drone_pos, drone_vel, quat, ang_vel_b, agl, time_frac)
        info = {
            "intercept": all_done,                                  # episode fully cleared
            "time_to_intercept": torch.where(all_done, step_count.float() * self.dt,
                                             torch.full_like(nearest_d, float("nan"))),
            "found": self.known.float().mean(dim=1),
            "cleared": self.reached.float().mean(dim=1),
            "coverage": self.visited.float().mean(dim=1),
            "crash": crash, "oob": oob, "flip": flip,
            "visible": visible, "agl": agl, "nearest_known": nearest_d,
        }
        return obs, r, terminated, info

    # ------------------------------------------------------------------ obs
    def proprio(self, drone_vel, quat, ang_vel_b, agl, time_frac):
        """[N, 11] -- see vesper.lab.frames.proprio."""
        return _proprio(drone_vel, quat, ang_vel_b, agl, time_frac)

    def privileged(self, drone_pos, drone_vel, quat, ang_vel_b, agl, time_frac):
        """[N, 12 + 8K + G*G + 3] -- world pose plus the belief and coverage.

        Never a target's true position: what the sensor has reported, and only
        that. For the critic and the state-based teacher, not the actor.
        """
        cfg = self.cfg
        A = cfg.arena_half
        yaw = yaw_from_quat(quat)
        self_block = torch.stack([
            drone_pos[:, 0] / A, drone_pos[:, 1] / A, agl / 100.0,
            drone_vel[:, 0] / 15.0, drone_vel[:, 1] / 15.0, drone_vel[:, 2] / 15.0,
            tilt_from_quat(quat), torch.sin(yaw), torch.cos(yaw),
            ang_vel_b[:, 0], ang_vel_b[:, 1], ang_vel_b[:, 2],
        ], dim=1)

        rel = self.fix - drone_pos.unsqueeze(1)
        dist = rel.norm(dim=2, keepdim=True)
        live = (self.known & ~self.reached).float().unsqueeze(2)
        stale = torch.exp(-self.fix_age / cfg.stale_tau_s).unsqueeze(2)
        tgt = torch.cat([
            self.known.float().unsqueeze(2),
            self.reached.float().unsqueeze(2),
            stale,
            rel[..., :2] / A * live,
            rel[..., 2:3] / 50.0 * live,
            dist / A * live,
            (self.fix_age.unsqueeze(2) / 30.0).clamp(max=2.0) * live,
        ], dim=2).reshape(self.n, -1)

        tail = torch.stack([
            time_frac,
            self.reached.float().mean(dim=1),
            self.known.float().mean(dim=1),
        ], dim=1)
        return torch.cat([self_block, tgt, self.recency, tail], dim=1)
