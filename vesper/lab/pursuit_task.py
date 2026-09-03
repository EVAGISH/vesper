"""Pursuit task: the reward/observation/termination math for an aerial pursuit
of a moving ground vehicle -- the drone must reach the vehicle as fast as it can.

Pure torch, no Isaac import, so the whole task -- target motion, observation
assembly, shaped reward, terminal conditions -- is unit-testable on CPU. The
Isaac Lab env (vesper.lab.pursuit_env.PursuitEnv) is a thin shell that feeds
PhysX state into these functions and applies the guidance action.

Coordinates are env-local (drone/target positions relative to each env origin),
world axes (x east, y north, z up). One target per env.
"""
from dataclasses import dataclass

import torch


@dataclass
class PursuitCfg:
    # --- arena / spawn ---
    arena_radius: float = 45.0        # target stays within this xy radius of env origin
    spawn_alt: float = 18.0           # drone spawn height (m)
    spawn_jitter_xy: float = 6.0      # drone spawn xy noise (m)
    target_min_r: float = 20.0        # target spawned at least this far from origin (xy)
    target_max_r: float = 40.0        # ... and at most this far
    target_h: float = 1.1             # vehicle origin height at spawn; it settles under gravity (m)

    # --- moving target (constant-velocity with heading jitter; bounces at arena edge) ---
    target_speed: float = 4.0         # m/s (0 -> parked vehicle)
    # The vehicle steers at a bounded rate rather than teleporting its heading:
    # the old per-step heading noise (0.03 rad at 50 Hz = 1.5 rad/s) demanded
    # more turn rate than any hull could deliver, so the model lagged ~40 deg
    # behind its own velocity and visibly crabbed.
    target_yaw_rate: float = 1.2      # rad/s cap on steering (3.3 m radius at 4 m/s)
    target_turn_jitter: float = 0.06  # rad/s of steering-rate noise per step
    target_turn_decay: float = 0.995  # steering relaxes back toward straight

    # --- guidance action ---
    look_ahead: float = 10.0          # tanh(action) * look_ahead = setpoint offset from drone (m)

    # --- intercept / failure geometry ---
    intercept_radius: float = 3.0     # 3D range to target center that counts as reaching it (m)
    ground_z: float = 0.0
    min_clearance: float = 0.4        # drone below ground_z+this (and no intercept) = ground crash
    tilt_limit: float = 1.4           # |tilt| from upright (rad) beyond which = loss of control
    oob_margin: float = 15.0          # drone beyond arena_radius+this from origin (xy) = out of bounds

    # --- reward weights: minimise time-to-intercept ---
    # The shaping is deliberately balanced so the ordering of outcomes is
    #   fast intercept  >  slow intercept  >  time-out  >  crash / tumble / leave arena.
    # A per-step cost alone would make dying early the cheapest way to stop the
    # bleeding, so the failure penalties sit well above the worst-case time cost
    # (0.05 x 750 steps = 37.5 over a full 15 s episode).
    w_progress: float = 1.0           # per metre of range closed this step (rewards closing SPEED)
    w_time: float = 0.05              # per-step cost -- the pressure to finish fast
    w_proximity: float = 0.5          # dense homing bonus w_proximity/(1+dist)
    r_intercept: float = 150.0        # terminal: reached the vehicle
    r_speed_bonus: float = 120.0      # extra, scaled by how much of the episode was left
    r_ground: float = 60.0            # terminal penalty: flew into the ground
    r_oob: float = 60.0               # terminal penalty: left the arena
    r_flip: float = 60.0              # terminal penalty: tumbled


def sample_targets(n, cfg: PursuitCfg, device, generator=None):
    """Initial target position [n,3] and velocity [n,3] (env-local)."""
    def rand(*shape):
        return torch.rand(*shape, device=device, generator=generator)

    ang = rand(n) * (2 * torch.pi)
    r = cfg.target_min_r + rand(n) * (cfg.target_max_r - cfg.target_min_r)
    pos = torch.zeros(n, 3, device=device)
    pos[:, 0] = r * torch.cos(ang)
    pos[:, 1] = r * torch.sin(ang)
    pos[:, 2] = cfg.ground_z + cfg.target_h
    heading = rand(n) * (2 * torch.pi)
    vel = torch.zeros(n, 3, device=device)
    vel[:, 0] = cfg.target_speed * torch.cos(heading)
    vel[:, 1] = cfg.target_speed * torch.sin(heading)
    return pos, vel


def tilt_from_quat(quat):
    """Angle [n] between body +z and world +z, from wxyz quat."""
    w, x, y, z = quat.unbind(dim=1)
    bz_z = 1 - 2 * (x * x + y * y)          # world-z component of body z axis
    return torch.arccos(bz_z.clamp(-1.0, 1.0))


def observations(drone_pos, drone_vel, quat, ang_vel_body, target_pos, target_vel):
    """[n,17]: rel-to-target(3), drone vel(3), target vel(3), quat(4), ang_vel_body(3), dist(1)."""
    rel = target_pos - drone_pos
    dist = rel.norm(dim=1, keepdim=True)
    return torch.cat([rel, drone_vel, target_vel, quat, ang_vel_body, dist], dim=1)


def setpoint(drone_pos_w, action, cfg: PursuitCfg):
    """Guidance action [n,3] in [-1,1] -> world setpoint for the inner SE3 loop."""
    return drone_pos_w + torch.tanh(action) * cfg.look_ahead


def evaluate(drone_pos, drone_vel, quat, target_pos, prev_dist, cfg: PursuitCfg, time_frac=None):
    """Reward [n] and terminal flags for one step. Returns (reward, terminated, info)
    where info has boolean masks intercept/ground/oob/flip and the current dist."""
    rel = target_pos - drone_pos
    dist = rel.norm(dim=1)
    tilt = tilt_from_quat(quat)
    rxy = drone_pos[:, :2].norm(dim=1)

    intercept = dist < cfg.intercept_radius
    ground = (drone_pos[:, 2] < cfg.ground_z + cfg.min_clearance) & ~intercept
    oob = (rxy > cfg.arena_radius + cfg.oob_margin) | (drone_pos[:, 2] > cfg.spawn_alt * 4)
    flip = tilt > cfg.tilt_limit
    terminated = intercept | ground | oob | flip

    reward = cfg.w_progress * (prev_dist - dist)
    reward = reward - cfg.w_time
    reward = reward + cfg.w_proximity / (1.0 + dist)
    # hitting sooner is worth strictly more than hitting later
    if time_frac is None:
        speed_bonus = torch.zeros_like(dist)
    else:
        speed_bonus = cfg.r_speed_bonus * (1.0 - time_frac).clamp(0.0, 1.0)
    reward = reward + (cfg.r_intercept + speed_bonus) * intercept.float()
    reward = reward - cfg.r_ground * ground.float()
    reward = reward - cfg.r_oob * oob.float()
    reward = reward - cfg.r_flip * flip.float()

    info = {"dist": dist, "intercept": intercept, "ground": ground, "oob": oob, "flip": flip, "tilt": tilt}
    return reward, terminated, info
