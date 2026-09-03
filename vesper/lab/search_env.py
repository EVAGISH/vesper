"""SearchEnv: search a real site for several ground vehicles, then run them down.

One drone and `n_targets` forklifts per environment, all of them dropped at
random on the Cornell world (vesper.worlds.geo output) every reset. The world
itself is loaded once as a global prim and shared by every environment -- the
rule from STACK.md: thousands of drones, one world -- so env_spacing is zero and
the environments are separated by collision filtering, not by distance.

Three things make this a search rather than a chase:

  * the forklifts are placed by concealment class -- one driving in the open, one
    painted down (low contrast), one crawling under tree canopy -- and which slot
    gets which role is reshuffled every episode, so the slot index tells the
    policy nothing;
  * the policy never sees a target's true position. It sees what its downward
    camera reports (vesper.lab.search_task), which terrain, buildings and foliage
    can all take away from it;
  * the vehicles drive on the real terrain under a velocity controller that
    steers around water, buildings and slopes it cannot climb.

Action is guidance -- a look-ahead setpoint for the SE3 inner loop -- the same
contract the pursuit task uses.
"""
import os

import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, RigidObject
from isaaclab.sim import SimulationCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass

from vesper.control.se3 import SE3Controller
from vesper.lab import search_task as T
from vesper.lab.pursuit_env import resolve_vehicle, vehicle_cfg
from vesper.lab.vesper_quad import VesperQuadEnv, VesperQuadEnvCfg
from vesper.worlds.heightmap import WorldMap

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
CORNELL_USD = os.path.join(REPO, "assets", "cornell", "cornell.usd")
CORNELL_MAP = os.path.join(REPO, "assets", "cornell", "cornell_map.npz")

# (name, ground speed m/s, optical contrast, spawn mask)
#   open        a forklift going about its business in plain sight
#   camouflaged same behaviour, painted to blend into the ground
#   concealed   crawling under tree canopy, plain paint but hard to see through leaves
#   parked      shut down next to the buildings, no motion cue at all
ROLES = [
    ("open", 4.5, 1.00, "drivable"),
    ("camouflaged", 3.5, 0.32, "drivable"),
    ("concealed", 1.2, 0.90, "concealed"),
    ("parked", 0.0, 0.75, "drivable"),
]


@configclass
class SearchEnvCfg(VesperQuadEnvCfg):
    episode_length_s = 75.0
    decimation = 4                      # 25 Hz guidance over a 100 Hz inner loop
    # match the render interval to the decimation: nothing renders in a
    # headless training run, and a mismatch makes Isaac Lab warn every boot
    sim: SimulationCfg = SimulationCfg(dt=1 / 100, render_interval=4)
    action_space = 3
    observation_space = 0               # filled from SearchTask.obs_dim at runtime
    search: dict | None = None
    world_usd: str = CORNELL_USD
    world_map: str = CORNELL_MAP
    vehicle_model: str | None = None
    # 14 m/s^2 asks for a 55 deg lean and 12.5% of evaluated episodes ended in a
    # tumble; 11 caps the steady-state lean near 48 deg and still flies hard.
    accel_limit: float = 11.0
    n_targets: int = 3
    terrain: TerrainImporterCfg = TerrainImporterCfg(
        prim_path="/World/ground", terrain_type="usd", usd_path=CORNELL_USD, collision_group=-1,
    )


class SearchEnv(VesperQuadEnv):
    cfg: SearchEnvCfg

    def __init__(self, cfg: SearchEnvCfg, render_mode=None, seed=0, **kwargs):
        self.tcfg = T.SearchCfg(**(cfg.search or {}))
        self.tcfg.n_targets = cfg.n_targets
        self.k = cfg.n_targets
        self._seed = int(seed)
        cfg.terrain.usd_path = cfg.world_usd
        self._veh_spec = resolve_vehicle(cfg.vehicle_model)
        self._veh_yaw_offset = float(self._veh_spec["yaw_offset"])
        # observation width is a property of the task, not a magic number
        g = self.tcfg.grid
        cfg.observation_space = 12 + 8 * self.k + g * g + 3
        super().__init__(cfg, render_mode, **kwargs)

        self.gen = torch.Generator(device=self.device); self.gen.manual_seed(self._seed)
        self.world = WorldMap(cfg.world_map, device=self.device)
        self.ctrl = SE3Controller(self.params, self.num_envs, device=self.device)
        self.ctrl.accel_limit = cfg.accel_limit
        self._dt = self.cfg.sim.dt * self.cfg.decimation
        self.task = T.SearchTask(self.world, self.tcfg, self.num_envs, self._dt,
                                 int(self.max_episode_length), device=self.device, generator=self.gen)
        self.spawn_offsets = torch.zeros(self.num_envs, 3, device=self.device)
        self.veh_heading = torch.zeros(self.num_envs, self.k, device=self.device)
        self.veh_turn_rate = torch.zeros(self.num_envs, self.k, device=self.device)
        self.veh_speed = torch.zeros(self.num_envs, self.k, device=self.device)
        self.role = torch.zeros(self.num_envs, self.k, dtype=torch.long, device=self.device)
        self._setpoint = torch.zeros(self.num_envs, 3, device=self.device)
        self._reward = torch.zeros(self.num_envs, device=self.device)
        self._term = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._evaluated = False
        # probe fan used to steer vehicles away from ground they cannot drive on
        self._probe = torch.tensor([0.0, 0.5, -0.5, 1.0, -1.0, 1.8, -1.8, 3.14159],
                                   device=self.device)

    # ---------------------------------------------------------------- scene
    def _setup_scene(self):
        self._robot = Articulation(self.cfg.robot)
        self.scene.articulations["robot"] = self._robot
        self._vehicles = []
        for i in range(self.k):
            vcfg = vehicle_cfg(self._veh_spec)
            vcfg.prim_path = f"/World/envs/env_.*/Vehicle_{i}"
            vcfg.init_state.pos = (60.0 + 30.0 * i, 0.0, 40.0)
            obj = RigidObject(vcfg)
            self.scene.rigid_objects[f"vehicle_{i}"] = obj
            self._vehicles.append(obj)
        self.cfg.terrain.num_envs = self.scene.cfg.num_envs
        self.cfg.terrain.env_spacing = self.scene.cfg.env_spacing
        self._terrain = self.cfg.terrain.class_type(self.cfg.terrain)
        # One world, many drones: every environment sits on the same origin and
        # they are kept apart by collision filtering instead of by spacing.
        self._terrain.env_origins[:] = 0.0
        self.scene.clone_environments(copy_from_source=False)
        self.scene.filter_collisions(global_prim_paths=[self.cfg.terrain.prim_path])
        # the site USD carries its own sun and sky; nothing more to light it with

    # ---------------------------------------------------------------- targets
    @property
    def target_pos(self):
        """[N,K,3] world positions of every vehicle."""
        return torch.stack([v.data.root_pos_w for v in self._vehicles], dim=1)

    def _drive_vehicles(self):
        """Velocity-controlled driving that respects the map.

        Heading is the integral of a bounded steering rate (so the hull can
        actually deliver it), nudged by two constraints: stay inside the arena,
        and do not drive into water, a building or a slope. z is left to gravity
        and contacts so the vehicle rides the real terrain.
        """
        c = self.tcfg
        pos = self.target_pos                                    # [N,K,3]
        flat = pos.reshape(-1, 3)

        noise = torch.randn(self.num_envs, self.k, device=self.device, generator=self.gen)
        self.veh_turn_rate = (self.veh_turn_rate * 0.995 + noise * 0.05).clamp(-1.0, 1.0)
        self.veh_heading = self.veh_heading + self.veh_turn_rate * self._dt

        # --- look ahead along a fan of candidate headings and pick drivable ground
        probe_h = self.veh_heading.unsqueeze(2) + self._probe.view(1, 1, -1)     # [N,K,P]
        ahead = 14.0
        px = flat[:, 0].view(self.num_envs, self.k, 1) + ahead * torch.cos(probe_h)
        py = flat[:, 1].view(self.num_envs, self.k, 1) + ahead * torch.sin(probe_h)
        r, col = self.world.nearest_cell(px, py)
        ok = self.world.drivable[r, col]                                        # [N,K,P]
        inside = (px.abs() < c.arena_half) & (py.abs() < c.arena_half)
        score = ok * inside.float() - 0.03 * self._probe.abs().view(1, 1, -1)   # prefer straight on
        best = score.argmax(dim=2)
        blocked = score.gather(2, best.unsqueeze(2)).squeeze(2) < 0.5
        chosen = probe_h.gather(2, best.unsqueeze(2)).squeeze(2)
        self.veh_heading = torch.where(blocked, chosen, self.veh_heading)
        # nothing drivable in any direction: turn back toward the arena centre
        home = torch.atan2(-flat[:, 1].view(self.num_envs, self.k), -flat[:, 0].view(self.num_envs, self.k))
        lost = score.amax(dim=2) < 0.5
        self.veh_heading = torch.where(lost, home, self.veh_heading)
        self.veh_turn_rate = torch.where(blocked | lost, torch.zeros_like(self.veh_turn_rate),
                                         self.veh_turn_rate)

        for i, v in enumerate(self._vehicles):
            vel = v.data.root_vel_w.clone()
            sp = self.veh_speed[:, i]
            vel[:, 0] = sp * torch.cos(self.veh_heading[:, i])
            vel[:, 1] = sp * torch.sin(self.veh_heading[:, i])
            vel[:, 3:5] = 0.0                                     # no roll/pitch rate
            q = v.data.root_quat_w
            yaw = torch.atan2(2 * (q[:, 0] * q[:, 3] + q[:, 1] * q[:, 2]),
                              1 - 2 * (q[:, 2] ** 2 + q[:, 3] ** 2))
            want = self.veh_heading[:, i] + self._veh_yaw_offset
            err = torch.atan2(torch.sin(want - yaw), torch.cos(want - yaw))
            vel[:, 5] = (err / self._dt).clamp(-3.6, 3.6)
            # a stopped vehicle stays stopped: no yaw servo chatter on a parked hull
            moving = (sp > 0.05).float()
            vel[:, 5] = vel[:, 5] * moving
            v.write_root_velocity_to_sim(vel)

    # ---------------------------------------------------------------- control
    def _pre_physics_step(self, actions):
        self._action = actions
        self._drive_vehicles()
        self._setpoint = T.setpoint(self._robot.data.root_pos_w, actions, self.tcfg)
        self._evaluated = False

    def _apply_action(self):
        d = self._robot.data
        vel_w = d.root_lin_vel_w - self.wind_world
        omega = self.ctrl.compute(d.root_pos_w, vel_w, d.root_quat_w, d.root_ang_vel_b, self._setpoint)
        from vesper.control.se3 import quat_to_rot
        R = quat_to_rot(d.root_quat_w)
        v_body = (R.transpose(1, 2) @ vel_w.unsqueeze(2)).squeeze(2)
        force_b, torque_b = self.dynamics.wrench(omega, v_body)
        self._thrust[:, 0, :] = force_b
        self._moment[:, 0, :] = torque_b
        self._robot.set_external_force_and_torque(self._thrust, self._moment, body_ids=self._body_id)

    # ---------------------------------------------------------------- task
    def _evaluate(self):
        if self._evaluated:
            return
        d = self._robot.data
        _, self._reward, self._term, info = self.task.step(
            d.root_pos_w, d.root_lin_vel_w, d.root_quat_w, d.root_ang_vel_b,
            self.target_pos, self.episode_length_buf)
        self.extras.update(info)
        self._evaluated = True

    def _get_observations(self):
        """Rebuilt from live state every call, never cached.

        DirectRLEnv calls this *after* _reset_idx has run for whichever
        environments finished, so a cached observation from _evaluate would hand
        the policy the last frame of the episode that just ended as the first
        frame of the next one. The belief update stays in _evaluate (which runs
        from _get_dones, before the reset); only the read-out happens here.
        """
        d = self._robot.data
        ground = self.world.ground_at(d.root_pos_w[:, 0], d.root_pos_w[:, 1])
        agl = (d.root_pos_w[:, 2] - ground).clamp(min=0.0)
        tf = (self.episode_length_buf.float() / self.max_episode_length).clamp(0, 1)
        return {"policy": self.task.observations(d.root_pos_w, d.root_lin_vel_w,
                                                 d.root_quat_w, d.root_ang_vel_b, agl, tf)}

    def _get_rewards(self):
        self._evaluate()
        return self._reward

    def _get_dones(self):
        self._evaluate()
        return self._term, self.episode_length_buf >= self.max_episode_length - 1

    # ---------------------------------------------------------------- reset
    def _reset_idx(self, env_ids):
        n = len(env_ids)
        g, dev, c = self.gen, self.device, self.tcfg

        # --- roles: a fresh shuffle per episode so slot 0 is not always the easy one.
        # Role 0 (a forklift driving in the open) is always one of them: it is the
        # only class a policy that cannot yet search reliably will ever stumble
        # onto, so guaranteeing one per episode is what keeps the reach reward
        # reachable at all early in training.
        perm = torch.argsort(torch.rand(n, len(ROLES), device=dev, generator=g), dim=1)[:, :self.k]
        has_open = (perm == 0).any(dim=1)
        slot = torch.randint(0, self.k, (n,), device=dev, generator=g)
        perm[~has_open, slot[~has_open]] = 0
        self.role[env_ids] = perm
        speed = torch.tensor([r[1] for r in ROLES], device=dev)
        contrast = torch.tensor([r[2] for r in ROLES], device=dev)
        hides = torch.tensor([r[3] == "concealed" for r in ROLES], device=dev)
        self.veh_speed[env_ids] = speed[perm] * (0.7 + 0.6 * torch.rand(n, self.k, device=dev, generator=g))

        # --- vehicle placement, by role
        want_hidden = hides[perm]                                            # [n,K]
        xy_open, _ = self.world.sample_mask_xy(self.world.drivable, n * self.k, c.arena_half, g)
        xy_hide, hid_ok = self.world.sample_mask_xy(self.world.concealed, n * self.k, c.arena_half, g)
        xy_open = xy_open.view(n, self.k, 2)
        xy_hide = xy_hide.view(n, self.k, 2)
        use_hide = want_hidden & hid_ok.view(n, self.k)                      # fall back if no cover
        xy = torch.where(use_hide.unsqueeze(2), xy_hide, xy_open)
        heading = torch.rand(n, self.k, device=dev, generator=g) * (2 * torch.pi)
        self.veh_heading[env_ids] = heading
        self.veh_turn_rate[env_ids] = 0.0

        gz = self.world.ground_at(xy[..., 0], xy[..., 1])
        for i, v in enumerate(self._vehicles):
            root = torch.zeros(n, 13, device=dev)
            root[:, 0] = xy[:, i, 0]
            root[:, 1] = xy[:, i, 1]
            root[:, 2] = gz[:, i] + c.min_clearance + 0.4        # settles under gravity
            yaw = heading[:, i] + self._veh_yaw_offset
            root[:, 3] = torch.cos(yaw / 2)
            root[:, 6] = torch.sin(yaw / 2)
            v.write_root_pose_to_sim(root[:, :7], env_ids)
            v.write_root_velocity_to_sim(root[:, 7:], env_ids)

        # --- drone: anywhere over the arena, well clear of whatever is beneath it
        dxy = (torch.rand(n, 2, device=dev, generator=g) * 2 - 1) * (c.arena_half * 0.9)
        solid = self.world.solid_at(dxy[:, 0], dxy[:, 1])
        ground = self.world.ground_at(dxy[:, 0], dxy[:, 1])
        alt = c.spawn_alt_min + torch.rand(n, device=dev, generator=g) * (c.spawn_alt_max - c.spawn_alt_min)
        dz = torch.maximum(ground + alt, solid + 20.0)
        default = self._robot.data.default_root_state[env_ids]
        self.spawn_offsets[env_ids] = torch.stack([dxy[:, 0], dxy[:, 1], dz], dim=1) - default[:, :3]
        super()._reset_idx(env_ids)
        # Random heading, so the policy cannot assume a starting look direction.
        # Built from the pose we just asked for rather than read back out of
        # root_state_w, which is a cached buffer and does not refresh until the
        # next physics step.
        dyaw = torch.rand(n, device=dev, generator=g) * (2 * torch.pi)
        pose = torch.zeros(n, 7, device=dev)
        pose[:, 0] = dxy[:, 0]; pose[:, 1] = dxy[:, 1]; pose[:, 2] = dz
        pose[:, 3] = torch.cos(dyaw / 2)
        pose[:, 6] = torch.sin(dyaw / 2)
        self._robot.write_root_pose_to_sim(pose, env_ids)

        self.task.reset(env_ids, contrast=contrast[perm])

    # ---------------------------------------------------------------- ppo glue
    @property
    def num_obs(self):
        return self.cfg.observation_space

    @property
    def num_actions(self):
        return self.cfg.action_space

    def ppo_reset(self):
        obs, _ = self.reset()
        return obs["policy"]

    def ppo_step(self, action):
        obs, rew, term, trunc, info = self.step(action)
        return obs["policy"], rew, term | trunc, info
