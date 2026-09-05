"""NativeSearchEnv: the search task with no Isaac underneath.

The Isaac SearchEnv is, by its own docstring, a thin shell: the sensor, belief,
reward and termination are pure torch (vesper.lab.search_task), the flight
forces are pure torch (vesper.dynamics, our Pegasus port), the inner loop is
pure torch (vesper.control.se3), and the world is a raster
(vesper.worlds.heightmap). PhysX's remaining jobs were integrating one free
rigid body per drone and letting the tanks ride the terrain. This env does
those two things itself -- the drone through vesper.dynamics.ReferenceIntegrator,
the vehicles kinematically on the ground raster -- and so runs anywhere torch
runs: a Mac CPU, MPS, or the droplet's CUDA. No Kit, no PhysX, no RTX.

What is faithful: the same dynamics code, the same SE(3) gains, the same
100 Hz inner loop under 25 Hz guidance, the same task tensors, the same role
table and steering rules (vesper.lab.ground), the same spawn logic, the same
observation and reward numbers.

What is approximate, and accepted:
  * no contact. A drone below solid_top + min_clearance is a crash *because the
    task says so* -- exactly the termination Isaac training used -- but there is
    no tumbling wreck afterwards, and a hull cannot physically wedge against a
    trunk (the stuck-recovery branch simply never fires).
  * vehicles ride ground_z + a fixed hull clearance instead of settling on
    suspension; no rendered camera exists, so sightings are always the
    geometric cone (`seen_px` stays None). The vision lane stays on Isaac.

PPO contract (vesper.lab.ppo): `num_envs`, `device`, `num_obs`, `num_actions`,
`ppo_reset() -> obs [N,obs]`, `ppo_step(act) -> (obs, reward, done, info)` with
auto-reset, same as the Isaac env's glue.
"""
from __future__ import annotations

import math
import os
from dataclasses import dataclass, field

import torch

from vesper.control.se3 import SE3Controller, quat_to_rot
from vesper.dynamics import GustField, MultirotorDynamics, MultirotorParams
from vesper.dynamics.reference_integrator import ReferenceIntegrator
from vesper.lab import ground as GD
from vesper.lab import search_task as T
from vesper.worlds.heightmap import LINK_THRESHOLD, WorldMap

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
CORNELL_MAP = os.path.join(REPO, "assets", "cornell", "cornell_map.npz")

VEH_CLEARANCE = 1.0        # hull ride height over ground_z (m); Isaac lets suspension settle here


@dataclass
class NativeSearchEnvCfg:
    num_envs: int = 1024
    n_targets: int = 3
    n_groups: int = 0                   # 0 -> one vehicle set per env
    episode_length_s: float = 75.0
    dt: float = 1 / 100                 # inner-loop step
    decimation: int = 4                 # 25 Hz guidance over a 100 Hz inner loop
    accel_limit: float = 11.0           # same lean cap as the Isaac env
    yaw_follow_tau_s: float = 0.6
    yaw_follow_min_speed: float = 1.5
    search: dict | None = None          # SearchCfg overrides
    world_map: str = CORNELL_MAP
    ppo_key: str = "privileged"         # "privileged" trains the state teacher, "policy" the proprio actor
    wind_mean: tuple = (0.0, 0.0, 0.0)  # m/s world frame; gusts on top when gust_std > 0
    gust_std: float = 0.0
    action_space: int = 3


class NativeSearchEnv:
    cfg: NativeSearchEnvCfg

    def __init__(self, cfg: NativeSearchEnvCfg, device="cpu", seed=0):
        self.cfg = cfg
        self.device = device
        self.num_envs = int(cfg.num_envs)
        self.tcfg = T.SearchCfg(**(cfg.search or {}))
        self.tcfg.n_targets = cfg.n_targets
        self.k = int(cfg.n_targets)
        N, K, dev = self.num_envs, self.k, device

        self.gen = torch.Generator(device=dev)
        self.gen.manual_seed(int(seed))
        self.world = WorldMap(cfg.world_map, device=dev)
        self.params = MultirotorParams()
        self.dynamics = MultirotorDynamics(self.params, N, device=dev)
        self.ctrl = SE3Controller(self.params, N, device=dev)
        self.ctrl.accel_limit = cfg.accel_limit
        self.body = ReferenceIntegrator(self.dynamics, N, dt=cfg.dt, device=dev)

        self._dt = cfg.dt * cfg.decimation
        self.max_episode_length = math.ceil(cfg.episode_length_s / self._dt)
        self.task = T.SearchTask(self.world, self.tcfg, N, self._dt,
                                 self.max_episode_length, device=dev, generator=self.gen)

        self.G = int(cfg.n_groups) if cfg.n_groups and cfg.n_groups > 0 else N
        self.G = min(self.G, N)
        self.group = torch.arange(N, device=dev) % self.G       # whose vehicles env i hunts

        self.episode_length_buf = torch.zeros(N, dtype=torch.long, device=dev)
        # each drone's RF reading at its position, refreshed every step; the
        # field is a baked raster (see export_world_map.py) so this is a lookup
        self.comms_now = torch.ones(N, device=dev)
        self.yaw_des = torch.zeros(N, device=dev)
        self._setpoint = torch.zeros(N, 3, device=dev)
        self.wind_world = torch.as_tensor(cfg.wind_mean, dtype=torch.float32,
                                          device=dev).expand(N, 3).clone()
        self.gust = (GustField(N, cfg.wind_mean, gust_std=cfg.gust_std, dt=cfg.dt,
                               device=dev, generator=self.gen)
                     if cfg.gust_std > 0 else None)

        # vehicle fleet: every env owns a set like the Isaac env's rigid objects;
        # sets beyond the first G are dormant (parked out of the arena, speed 0)
        self.veh_pos = torch.zeros(N, K, 3, device=dev)
        self.veh_vel = torch.zeros(N, K, 2, device=dev)          # last applied ground velocity
        self.veh_heading = torch.zeros(N, K, device=dev)
        self.veh_turn_rate = torch.zeros(N, K, device=dev)
        self.veh_speed = torch.zeros(N, K, device=dev)           # cruise speed for the role
        self.veh_speed_cmd = torch.zeros(N, K, device=dev)       # ramped command
        self.veh_stuck_s = torch.zeros(N, K, device=dev)
        self.veh_on_road = torch.zeros(N, K, dtype=torch.bool, device=dev)
        self.role = torch.zeros(N, K, dtype=torch.long, device=dev)
        self._probe = torch.tensor(GD.PROBE, device=dev)
        self._dormant_xy = (self.world.half_m - 15.0, self.world.half_m - 15.0)

        r = GD.ROLES
        self._role_speed = torch.tensor([x[1] for x in r], device=dev)
        self._role_contrast = torch.tensor([x[2] for x in r], device=dev)
        self._role_layer = [x[3] for x in r]
        self._role_on_road = torch.tensor([x[4] for x in r], device=dev, dtype=torch.bool)

    # ---------------------------------------------------------------- state
    @property
    def vehicle_pos(self):
        """[N,K,3] world positions of every vehicle set, dormant ones included."""
        return self.veh_pos

    @property
    def target_pos(self):
        """[N,K,3] the vehicles env i is hunting: its group's set."""
        return self.veh_pos[self.group]

    def flight_state(self):
        """(pos, vel_w, quat wxyz, ang_vel_body), each [N,...] -- the Isaac names."""
        return self.body.pos, self.body.vel, self.body.quat, self.body.ang_vel

    @property
    def linked(self):
        """[N] bool: which drones currently hold a link to the base station."""
        return self.comms_now >= LINK_THRESHOLD

    # ---------------------------------------------------------------- vehicles
    def _drive_vehicles(self):
        """Kinematic driving on the ground raster, steering from vesper.lab.ground."""
        actual = self.veh_vel.norm(dim=2)
        sp = GD.steer(self, self.world, self.tcfg.arena_half, actual, self._dt,
                      gen=self.gen, probe=self._probe)
        vel = torch.stack([sp * torch.cos(self.veh_heading),
                           sp * torch.sin(self.veh_heading)], dim=2)
        self.veh_vel = vel
        self.veh_pos[..., :2] += vel * self._dt
        self.veh_pos[..., 2] = self.world.ground_at(self.veh_pos[..., 0],
                                                    self.veh_pos[..., 1]) + VEH_CLEARANCE

    # ---------------------------------------------------------------- reset
    def _reset_idx(self, env_ids):
        n = len(env_ids)
        if n == 0:
            return
        g, dev, c = self.gen, self.device, self.tcfg
        env_ids = torch.as_tensor(env_ids, device=dev)
        rep = env_ids[env_ids < self.G]                       # envs whose vehicle set is live
        dormant = env_ids[env_ids >= self.G]

        if len(rep):
            m = len(rep)
            # roles: a fresh shuffle per episode, always at least one 'open' tank
            # (the only class an untrained policy stumbles onto -- see SearchEnv)
            perm = torch.argsort(torch.rand(m, len(GD.ROLES), device=dev, generator=g),
                                 dim=1)[:, :self.k]
            has_open = (perm == 0).any(dim=1)
            slot = torch.randint(0, self.k, (m,), device=dev, generator=g)
            perm[~has_open, slot[~has_open]] = 0
            self.role[rep] = perm
            self.veh_speed[rep] = self._role_speed[perm] * (
                0.7 + 0.6 * torch.rand(m, self.k, device=dev, generator=g))
            self.veh_speed_cmd[rep] = 0.0
            self.veh_stuck_s[rep] = 0.0
            self.veh_on_road[rep] = self._role_on_road[perm]

            # placement by the role's spawn layer, falling back to open drivable ground
            xy_open, _ = self.world.sample_mask_xy(self.world.drivable, m * self.k, c.arena_half, g)
            xy = xy_open.view(m, self.k, 2).clone()
            heading = torch.rand(m, self.k, device=dev, generator=g) * (2 * torch.pi)
            for li, name in enumerate(self._role_layer):
                want = perm == li
                if name == "drivable" or not want.any():
                    continue
                mask = getattr(self.world, name)
                # exact sampling: road/concealed/parking are a few percent of the
                # arena, and 24 rejection rounds still miss ~1 in 6 -- the exact
                # sampler draws the same uniform-over-mask distribution and never
                # falls back when the layer exists (the Isaac env still rejects;
                # converging it is a droplet-side change)
                cand, ok = self.world.sample_cells_xy(mask, m * self.k, generator=g,
                                                      half=c.arena_half)
                ok = ok & (cand.abs().amax(dim=1) <= c.arena_half)
                cand = cand.view(m, self.k, 2)
                use = want & ok.view(m, self.k)
                xy = torch.where(use.unsqueeze(2), cand, xy)
                if name == "road":
                    ry = self.world.yaw_at(self.world.road_yaw, cand[..., 0], cand[..., 1])
                    flip = (torch.rand(m, self.k, device=dev, generator=g) < 0.5).float() * torch.pi
                    heading = torch.where(use, ry + flip, heading)
                elif name == "parking":
                    py = self.world.yaw_at(self.world.park_yaw, cand[..., 0], cand[..., 1])
                    flip = (torch.rand(m, self.k, device=dev, generator=g) < 0.5).float() * torch.pi
                    heading = torch.where(use, py + flip, heading)
            self.veh_heading[rep] = heading
            self.veh_turn_rate[rep] = 0.0
            self.veh_vel[rep] = 0.0
            gz = self.world.ground_at(xy[..., 0], xy[..., 1])
            self.veh_pos[rep, :, 0] = xy[..., 0]
            self.veh_pos[rep, :, 1] = xy[..., 1]
            self.veh_pos[rep, :, 2] = gz + VEH_CLEARANCE

        if len(dormant):
            m = len(dormant)
            self.veh_speed[dormant] = 0.0
            self.veh_speed_cmd[dormant] = 0.0
            self.veh_on_road[dormant] = False
            self.veh_vel[dormant] = 0.0
            dx, dy = self._dormant_xy
            xs = dx + 6.0 * torch.arange(self.k, device=dev)
            gz = self.world.ground_at(xs, torch.full((self.k,), dy, device=dev))
            self.veh_pos[dormant, :, 0] = xs
            self.veh_pos[dormant, :, 1] = dy
            self.veh_pos[dormant, :, 2] = gz + VEH_CLEARANCE

        # every env inherits the role of the set it hunts (its own, for a rep)
        self.role[env_ids] = self.role[self.group[env_ids]]

        # drone: anywhere over the arena, well clear of whatever is beneath it
        dxy = (torch.rand(n, 2, device=dev, generator=g) * 2 - 1) * (c.arena_half * 0.9)
        solid = self.world.solid_at(dxy[:, 0], dxy[:, 1])
        grnd = self.world.ground_at(dxy[:, 0], dxy[:, 1])
        alt = c.spawn_alt_min + torch.rand(n, device=dev, generator=g) * (c.spawn_alt_max - c.spawn_alt_min)
        dz = torch.maximum(grnd + alt, solid + 20.0)
        dyaw = torch.rand(n, device=dev, generator=g) * (2 * torch.pi)
        self.body.pos[env_ids] = torch.stack([dxy[:, 0], dxy[:, 1], dz], dim=1)
        self.body.vel[env_ids] = 0.0
        self.body.ang_vel[env_ids] = 0.0
        quat = torch.zeros(n, 4, device=dev)
        quat[:, 0] = torch.cos(dyaw / 2)
        quat[:, 3] = torch.sin(dyaw / 2)
        self.body.quat[env_ids] = quat
        self.yaw_des[env_ids] = dyaw
        self.episode_length_buf[env_ids] = 0

        self.task.reset(env_ids, contrast=self._role_contrast[self.role[env_ids]])

    # ---------------------------------------------------------------- stepping
    def _observations(self):
        pos, vel, quat, avb = self.flight_state()
        grnd = self.world.ground_at(pos[:, 0], pos[:, 1])
        agl = (pos[:, 2] - grnd).clamp(min=0.0)
        tf = (self.episode_length_buf.float() / self.max_episode_length).clamp(0, 1)
        return {"policy": self.task.proprio(vel, quat, avb, agl, tf),
                "privileged": self.task.privileged(pos, vel, quat, avb, agl, tf)}

    def reset(self):
        self._reset_idx(torch.arange(self.num_envs, device=self.device))
        self.comms_now = self.world.comms_at(self.body.pos[:, 0], self.body.pos[:, 1])
        return self._observations(), {}

    def step(self, actions: torch.Tensor):
        """(obs dict, reward [N], terminated [N], truncated [N], info). Auto-resets."""
        cfg = self.cfg
        pos, vel, quat, avb = self.flight_state()

        # guidance: world setpoint from the body-frame action, nose onto the travel
        yaw = T.yaw_from_quat(quat)
        self._setpoint = T.setpoint(pos, actions, yaw, self.tcfg)
        speed = vel[:, :2].norm(dim=1)
        target = torch.atan2(vel[:, 1], vel[:, 0])
        err = torch.atan2(torch.sin(target - self.yaw_des), torch.cos(target - self.yaw_des))
        a = min(1.0, self._dt / max(cfg.yaw_follow_tau_s, 1e-3))
        follow = speed > cfg.yaw_follow_min_speed
        self.yaw_des = torch.where(follow, self.yaw_des + a * err, self.yaw_des)

        self._drive_vehicles()

        # inner loop: SE(3) -> rotor omegas -> wrench -> integrate, at 100 Hz
        for _ in range(cfg.decimation):
            wind = self.gust.step() if self.gust is not None else self.wind_world
            omega = self.ctrl.compute(self.body.pos, self.body.vel - wind, self.body.quat,
                                      self.body.ang_vel, self._setpoint, yaw_des=self.yaw_des)
            self.body.step(omega, wind)

        self.episode_length_buf += 1
        pos, vel, quat, avb = self.flight_state()
        self.comms_now = self.world.comms_at(pos[:, 0], pos[:, 1])
        _, reward, term, info = self.task.step(pos, vel, quat, avb, self.target_pos,
                                               self.episode_length_buf)
        trunc = self.episode_length_buf >= self.max_episode_length - 1
        if self.G < self.num_envs:
            # a group lives and dies with the env that owns its vehicles
            owner_done = (term | trunc)[: self.G]
            trunc = trunc | owner_done[self.group]

        done = term | trunc
        if done.any():
            self._reset_idx(torch.nonzero(done).flatten())
        return self._observations(), reward, term, trunc, info

    def close(self):
        pass

    # ---------------------------------------------------------------- ppo glue
    @property
    def num_obs(self):
        return self.task.obs_dim if self.cfg.ppo_key == "privileged" else self.task.proprio_dim

    @property
    def num_actions(self):
        return self.cfg.action_space

    def ppo_reset(self):
        obs, _ = self.reset()
        return obs[self.cfg.ppo_key]

    def ppo_step(self, action):
        obs, rew, term, trunc, info = self.step(action)
        return obs[self.cfg.ppo_key], rew, term | trunc, info
