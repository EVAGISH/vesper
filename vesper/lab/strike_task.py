"""Strike task: the reward/observation/termination math for a loitering-munition
drone whose goal is to reach (crash into) a ground vehicle.

Pure torch, no Isaac import, so the whole task -- target motion, observation
assembly, shaped reward, terminal conditions -- is unit-testable on CPU. The
Isaac Lab env (vesper.lab.strike_env.StrikeEnv) is a thin shell that feeds
PhysX state into these functions and applies the guidance action.

Coordinates are env-local (drone/target positions relative to each env origin),
world axes (x east, y north, z up). One target per env.
"""
from dataclasses import dataclass

import torch


@dataclass
class StrikeCfg:
    # --- arena / spawn ---
    arena_radius: float = 45.0        # target stays within this xy radius of env origin
    spawn_alt: float = 18.0           # drone spawn height (m)
    spawn_jitter_xy: float = 6.0      # drone spawn xy noise (m)
    target_min_r: float = 20.0        # target spawned at least this far from origin (xy)
    target_max_r: float = 40.0        # ... and at most this far
    target_h: float = 1.1             # target center height above ground (hull center, m)

    # --- moving target (constant-velocity with heading jitter; bounces at arena edge) ---
    target_speed: float = 4.0         # m/s (0 -> static armor)
    target_turn_std: float = 0.15     # rad/step heading noise

    # --- guidance action ---
    look_ahead: float = 10.0          # tanh(action) * look_ahead = setpoint offset from drone (m)

    # --- hit / failure geometry ---
    hit_radius: float = 3.0           # 3D distance to target center that counts as a strike (m)
    ground_z: float = 0.0
    min_clearance: float = 0.4        # drone below ground_z+this (and not a hit) = ground crash
    tilt_limit: float = 1.4           # |tilt| from upright (rad) beyond which = loss of control
    oob_margin: float = 15.0          # drone beyond arena_radius+this from origin (xy) = out of bounds

    # --- reward weights: minimise time-to-hit ---
    # The shaping is deliberately balanced so the ordering of outcomes is
    #   fast hit  >  slow hit  >  time-out  >  crash / tumble / leave arena.
    # A per-step cost alone would make dying early the cheapest way to stop the
    # bleeding, so the failure penalties sit well above the worst-case time cost
    # (0.05 x 750 steps = 37.5 over a full 15 s episode).
    w_progress: float = 1.0           # per metre of range closed this step (rewards closing SPEED)
    w_time: float = 0.05              # per-step cost -- the pressure to finish fast
    w_proximity: float = 0.5          # dense homing bonus w_proximity/(1+dist)
    r_hit: float = 150.0              # terminal: reached the vehicle
    r_hit_speed: float = 120.0        # extra, scaled by how much of the episode was left
    r_ground: float = 60.0            # terminal penalty: flew into the ground
    r_oob: float = 60.0               # terminal penalty: left the arena
    r_flip: float = 60.0              # terminal penalty: tumbled


def sample_targets(n, cfg: StrikeCfg, device, generator=None):
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


def step_targets(pos, vel, cfg: StrikeCfg, dt, generator=None):
    """Advance targets one policy step: heading jitter + arena-edge bounce. In place-safe (returns new)."""
    n = pos.shape[0]
    device = pos.device
    if cfg.target_speed > 0 and cfg.target_turn_std > 0:
        dtheta = torch.randn(n, device=device, generator=generator) * cfg.target_turn_std
        c, s = torch.cos(dtheta), torch.sin(dtheta)
        vx, vy = vel[:, 0].clone(), vel[:, 1].clone()
        vel = vel.clone()
        vel[:, 0] = c * vx - s * vy
        vel[:, 1] = s * vx + c * vy
    pos = pos + vel * dt
    # bounce back toward origin if beyond arena radius
    rxy = pos[:, :2].norm(dim=1, keepdim=True)
    out = (rxy > cfg.arena_radius).squeeze(1)
    if out.any():
        inward = -pos[out, :2] / rxy[out].clamp(min=1e-6)
        speed = vel[out, :2].norm(dim=1, keepdim=True)
        vel = vel.clone()
        vel[out, :2] = inward * speed
        pos = pos.clone()
        pos[out, :2] = pos[out, :2] * (cfg.arena_radius / rxy[out]).clamp(max=1.0)
    pos = pos.clone()
    pos[:, 2] = cfg.ground_z + cfg.target_h
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


def setpoint(drone_pos_w, action, cfg: StrikeCfg):
    """Guidance action [n,3] in [-1,1] -> world setpoint for the inner SE3 loop."""
    return drone_pos_w + torch.tanh(action) * cfg.look_ahead


def evaluate(drone_pos, drone_vel, quat, target_pos, prev_dist, cfg: StrikeCfg, time_frac=None):
    """Reward [n] and terminal flags for one step. Returns (reward, terminated, info)
    where info has boolean masks hit/ground/oob/flip and the current dist."""
    rel = target_pos - drone_pos
    dist = rel.norm(dim=1)
    tilt = tilt_from_quat(quat)
    rxy = drone_pos[:, :2].norm(dim=1)

    hit = dist < cfg.hit_radius
    ground = (drone_pos[:, 2] < cfg.ground_z + cfg.min_clearance) & ~hit
    oob = (rxy > cfg.arena_radius + cfg.oob_margin) | (drone_pos[:, 2] > cfg.spawn_alt * 4)
    flip = tilt > cfg.tilt_limit
    terminated = hit | ground | oob | flip

    reward = cfg.w_progress * (prev_dist - dist)
    reward = reward - cfg.w_time
    reward = reward + cfg.w_proximity / (1.0 + dist)
    # hitting sooner is worth strictly more than hitting later
    if time_frac is None:
        speed_bonus = torch.zeros_like(dist)
    else:
        speed_bonus = cfg.r_hit_speed * (1.0 - time_frac).clamp(0.0, 1.0)
    reward = reward + (cfg.r_hit + speed_bonus) * hit.float()
    reward = reward - cfg.r_ground * ground.float()
    reward = reward - cfg.r_oob * oob.float()
    reward = reward - cfg.r_flip * flip.float()

    info = {"dist": dist, "hit": hit, "ground": ground, "oob": oob, "flip": flip, "tilt": tilt}
    return reward, terminated, info
