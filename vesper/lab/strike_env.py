"""StrikeEnv: one drone, one physically simulated ground vehicle, per environment.

Architecture note: every environment holds exactly one drone and one vehicle and
is fully independent (own spawn, own target, own reward, own reset). Isaac Lab
clones those environments side by side inside a single PhysX scene because that
is what makes 4096 of them step on one GPU -- it is a vectorized batch of
separate sims, not one shared arena with many drones in it.

The vehicle is a PhysX rigid body (assets/vehicles/tank.usd: 42 t, hull and
track colliders). Gravity and contacts hold it on the terrain; a velocity
controller drives it. So it is physically present -- it rests on slopes, is
collidable, and cannot pass through anything -- though its drive is a velocity
command rather than a wheel/track/suspension model.

Action is guidance: the policy emits a look-ahead setpoint and the conventional
SE3 inner loop flies to it, so the policy learns interception on top of a
trusted stabilizer instead of relearning how to hover.
"""
import os

import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, RigidObject, RigidObjectCfg
from isaaclab.utils import configclass

from vesper.control.se3 import SE3Controller
from vesper.lab.vesper_quad import VesperQuadEnv, VesperQuadEnvCfg
from vesper.lab import strike_task as T

TANK_USD = os.environ.get(
    "VESPER_TARGET_USD",
    os.path.join(os.path.dirname(__file__), "..", "..", "assets", "vehicles", "tank.usd"),
)


def vehicle_cfg(usd_path: str = None) -> RigidObjectCfg:
    return RigidObjectCfg(
        prim_path="/World/envs/env_.*/Vehicle",
        spawn=sim_utils.UsdFileCfg(
            usd_path=os.path.abspath(usd_path or TANK_USD),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False, max_linear_velocity=40.0, max_angular_velocity=4.0,
                linear_damping=0.2, angular_damping=1.0,
            ),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(25.0, 0.0, 0.6)),
    )


@configclass
class StrikeEnvCfg(VesperQuadEnvCfg):
    episode_length_s = 15.0
    action_space = 3
    observation_space = 17
    strike: dict | None = None
    vehicle: RigidObjectCfg | None = None
    accel_limit: float = 12.0          # SE3 horizontal accel budget (stock 6) -- fly it hard


class StrikeEnv(VesperQuadEnv):
    cfg: StrikeEnvCfg

    def __init__(self, cfg: StrikeEnvCfg, render_mode=None, seed=0, **kwargs):
        self.tcfg = T.StrikeCfg(**(cfg.strike or {}))
        self._seed = int(seed)
        if cfg.vehicle is None:
            cfg.vehicle = vehicle_cfg()
        super().__init__(cfg, render_mode, **kwargs)
        self.gen = torch.Generator(device=self.device); self.gen.manual_seed(self._seed)
        self.ctrl = SE3Controller(self.params, self.num_envs, device=self.device)
        self.ctrl.accel_limit = cfg.accel_limit
        self.spawn_offsets = torch.zeros(self.num_envs, 3, device=self.device)
        self.veh_heading = torch.zeros(self.num_envs, device=self.device)
        self.prev_dist = torch.full((self.num_envs,), self.tcfg.target_max_r, device=self.device)
        self._setpoint = torch.zeros(self.num_envs, 3, device=self.device)
        self._reward = torch.zeros(self.num_envs, device=self.device)
        self._term = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._hit = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._evaluated = False
        self._dt = self.cfg.sim.dt * self.cfg.decimation

    # ---- target state comes from the physics body, not an integrated tensor ----
    @property
    def target_pos(self):
        return self._vehicle.data.root_pos_w - self.scene.env_origins

    @property
    def target_vel(self):
        return self._vehicle.data.root_lin_vel_w

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

    # ---- scene: vehicle must exist before clone_environments ----
    def _setup_scene(self):
        self._robot = Articulation(self.cfg.robot)
        self.scene.articulations["robot"] = self._robot
        self._vehicle = RigidObject(self.cfg.vehicle)
        self.scene.rigid_objects["vehicle"] = self._vehicle
        self.cfg.terrain.num_envs = self.scene.cfg.num_envs
        self.cfg.terrain.env_spacing = self.scene.cfg.env_spacing
        self._terrain = self.cfg.terrain.class_type(self.cfg.terrain)
        self.scene.clone_environments(copy_from_source=False)
        self.scene.filter_collisions(global_prim_paths=[self.cfg.terrain.prim_path])
        dome = sim_utils.DomeLightCfg(intensity=900.0, color=(0.72, 0.80, 0.95))
        dome.func("/World/Light", dome)
        sun = sim_utils.DistantLightCfg(intensity=2600.0, angle=0.6, color=(1.0, 0.96, 0.88))
        sun.func("/World/Sun", sun, orientation=(0.88, 0.20, 0.42, 0.0))

    # ---- control ----
    def _pre_physics_step(self, actions):
        self._action = actions
        self._drive_vehicles()
        self._setpoint = T.setpoint(self._robot.data.root_pos_w, actions, self.tcfg)
        self._evaluated = False

    def _drive_vehicles(self):
        """Velocity-controlled driving; z is left to gravity/contacts so the hull
        actually rides the terrain instead of floating along a scripted path."""
        c = self.tcfg
        if c.target_speed <= 0:
            return
        self.veh_heading += torch.randn(self.num_envs, device=self.device, generator=self.gen) * c.target_turn_std
        # steer back toward the arena before driving out of it
        p = self.target_pos
        r = p[:, :2].norm(dim=1)
        out = r > c.arena_radius
        if out.any():
            inward = torch.atan2(-p[out, 1], -p[out, 0])
            self.veh_heading[out] = inward
        vel = self._vehicle.data.root_vel_w.clone()
        vel[:, 0] = c.target_speed * torch.cos(self.veh_heading)
        vel[:, 1] = c.target_speed * torch.sin(self.veh_heading)
        vel[:, 3:5] = 0.0                                    # no roll/pitch rate
        vel[:, 5] = 0.0
        self._vehicle.write_root_velocity_to_sim(vel)

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

    # ---- task ----
    def _evaluate(self):
        if self._evaluated:
            return
        pos, vel, quat, _ = self.flight_state()
        frac = (self.episode_length_buf.float() / self.max_episode_length).clamp(0, 1)
        self._reward, self._term, info = T.evaluate(
            pos, vel, quat, self.target_pos, self.prev_dist, self.tcfg, time_frac=frac)
        self._hit = info["hit"]
        self.prev_dist = info["dist"]
        self.extras["hit"] = self._hit
        self.extras["dist"] = info["dist"]
        self.extras["time_to_hit"] = torch.where(
            self._hit, self.episode_length_buf.float() * self._dt, torch.full_like(info["dist"], float("nan")))
        self._evaluated = True

    def _get_observations(self):
        pos, vel, quat, avb = self.flight_state()
        return {"policy": T.observations(pos, vel, quat, avb, self.target_pos, self.target_vel)}

    def _get_rewards(self):
        self._evaluate()
        return self._reward

    def _get_dones(self):
        self._evaluate()
        return self._term, self.episode_length_buf >= self.max_episode_length - 1

    def _reset_idx(self, env_ids):
        k = len(env_ids)
        g, dev = self.gen, self.device
        jit = (torch.rand(k, 2, device=dev, generator=g) - 0.5) * 2 * self.tcfg.spawn_jitter_xy
        self.spawn_offsets[env_ids, :2] = jit
        self.spawn_offsets[env_ids, 2] = self.tcfg.spawn_alt
        super()._reset_idx(env_ids)

        tp, _ = T.sample_targets(k, self.tcfg, dev, g)
        heading = torch.rand(k, device=dev, generator=g) * (2 * torch.pi)
        self.veh_heading[env_ids] = heading
        root = torch.zeros(k, 13, device=dev)
        root[:, :3] = tp + self.scene.env_origins[env_ids]
        root[:, 2] = self.scene.env_origins[env_ids][:, 2] + self.tcfg.target_h
        root[:, 3] = torch.cos(heading / 2)              # w
        root[:, 6] = torch.sin(heading / 2)              # z
        self._vehicle.write_root_pose_to_sim(root[:, :7], env_ids)
        self._vehicle.write_root_velocity_to_sim(root[:, 7:], env_ids)

        dpos = self._robot.data.root_pos_w[env_ids] - self.scene.env_origins[env_ids]
        self.prev_dist[env_ids] = (tp - dpos).norm(dim=1)
