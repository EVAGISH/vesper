"""PursuitEnv: one drone, one physically simulated ground vehicle, per environment.

Architecture note: every environment holds exactly one drone and one vehicle and
is fully independent (own spawn, own target, own reward, own reset). Isaac Lab
clones those environments side by side inside a single PhysX scene because that
is what makes 4096 of them step on one GPU -- it is a vectorized batch of
separate sims, not one shared arena with many drones in it.

The vehicle is a real PhysX rigid body, not a scripted path: gravity and contacts
hold it on the terrain and a velocity controller drives it, so it rests on slopes,
is collidable, and cannot pass through anything -- though its drive is a velocity
command rather than a wheel/suspension model. Two models are available (see
VEHICLE_SPECS): NVIDIA's stock forklift prop, and a generated utility-cart proxy
for when the Isaac asset server is unreachable. Pick with VESPER_VEHICLE.

Action is guidance: the policy emits a look-ahead setpoint and the conventional
SE3 inner loop flies to it, so the policy learns pursuit on top of a trusted
stabilizer instead of relearning how to hover.
"""
import math
import os

import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, RigidObject, RigidObjectCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from pxr import Usd, UsdGeom, UsdPhysics

from vesper.control.se3 import SE3Controller
from vesper.lab.vesper_quad import VesperQuadEnv, VesperQuadEnvCfg
from vesper.lab import pursuit_task as T

_CART_USD = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "assets", "vehicles", "utility_cart.usd"))

# yaw_offset rotates the model so its nose points +X, which is the heading
# convention the drive controller and the reset pose both use.
VEHICLE_SPECS = {
    # NVIDIA's stock forklift prop: a plain Xform, not an articulation, modelled
    # Z-up in metres at 1.2 x 3.5 x 2.2 m with long axis Y (hence the -90 deg
    # yaw). Its colliders are already convexHull, so nothing needs
    # re-approximating, but it has no RigidBodyAPI -- see make_rigid_wrapper.
    "forklift": {"usd": f"{ISAAC_NUCLEUS_DIR}/Props/Forklift/forklift.usd",
                 "yaw_offset": -math.pi / 2, "mass": 2700.0,
                 "needs_rigid_wrapper": True},
    # Generated fallback (vesper.worlds.vehicle): box collider, cheap to clone,
    # and it authors its own RigidBodyAPI and mass.
    "cart": {"usd": _CART_USD, "yaw_offset": 0.0},
}
DEFAULT_VEHICLE = os.environ.get("VESPER_VEHICLE", "forklift")


def make_rigid_wrapper(src_usd: str, out_path: str, mass: float | None = None) -> str:
    """Wrap a prop USD in a prim that carries RigidBodyAPI, and return its path.

    Isaac Lab's USD spawner only *modifies* rigid-body properties -- it never
    applies the schema (schemas.define_* does, but spawn_from_usd calls
    schemas.modify_*). Stock props like the forklift ship with colliders but no
    RigidBodyAPI, so spawning one straight from UsdFileCfg dies with "Failed to
    find a rigid body when resolving ...". Authoring a thin local layer that
    references the asset and applies the schema is the same trick geo.py uses
    for tree species, and it keeps the fix in USD rather than in Isaac internals.
    """
    out = os.path.abspath(out_path)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    if os.path.exists(out):
        os.remove(out)
    stage = Usd.Stage.CreateNew(out)
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    root = UsdGeom.Xform.Define(stage, "/Vehicle")
    stage.SetDefaultPrim(root.GetPrim())
    root.GetPrim().GetReferences().AddReference(src_usd)
    UsdPhysics.RigidBodyAPI.Apply(root.GetPrim())
    if mass is not None:
        # Author mass here too: modify_mass_properties would likewise skip a
        # prim with no MassAPI, and PhysX would infer ~9 t from hull volume.
        UsdPhysics.MassAPI.Apply(root.GetPrim()).CreateMassAttr(float(mass))
    stage.GetRootLayer().Save()
    return out


def resolve_vehicle(name_or_path: str = None) -> dict:
    """Vehicle spec from a VEHICLE_SPECS key or a bare USD path."""
    key = name_or_path or DEFAULT_VEHICLE
    spec = dict(VEHICLE_SPECS.get(key, {"usd": os.path.abspath(key), "yaw_offset": 0.0}))
    if key == "cart" and not os.path.exists(spec["usd"]):
        from vesper.worlds.vehicle import write_vehicle_usd
        write_vehicle_usd(spec["usd"])
    if spec.pop("needs_rigid_wrapper", False):
        spec["usd"] = make_rigid_wrapper(
            spec["usd"], os.path.join(os.path.dirname(_CART_USD), f"{key}_rb.usd"),
            spec.pop("mass", None))
    return spec


def vehicle_cfg(spec: dict) -> RigidObjectCfg:
    kw = dict(
        usd_path=spec["usd"],
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False, max_linear_velocity=40.0, max_angular_velocity=4.0,
            linear_damping=0.2, angular_damping=1.0,
        ),
    )
    return RigidObjectCfg(
        prim_path="/World/envs/env_.*/Vehicle",
        spawn=sim_utils.UsdFileCfg(**kw),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(25.0, 0.0, 0.6)),
    )


@configclass
class PursuitEnvCfg(VesperQuadEnvCfg):
    episode_length_s = 15.0
    action_space = 3
    observation_space = 17
    pursuit: dict | None = None
    vehicle: RigidObjectCfg | None = None
    vehicle_model: str | None = None    # VEHICLE_SPECS key or a USD path
    accel_limit: float = 12.0          # SE3 horizontal accel budget (stock 6) -- fly it hard


class PursuitEnv(VesperQuadEnv):
    cfg: PursuitEnvCfg

    def __init__(self, cfg: PursuitEnvCfg, render_mode=None, seed=0, **kwargs):
        self.tcfg = T.PursuitCfg(**(cfg.pursuit or {}))
        self._seed = int(seed)
        spec = resolve_vehicle(cfg.vehicle_model)
        self._veh_yaw_offset = float(spec["yaw_offset"])
        if cfg.vehicle is None:
            cfg.vehicle = vehicle_cfg(spec)
        super().__init__(cfg, render_mode, **kwargs)
        self.gen = torch.Generator(device=self.device); self.gen.manual_seed(self._seed)
        self.ctrl = SE3Controller(self.params, self.num_envs, device=self.device)
        self.ctrl.accel_limit = cfg.accel_limit
        self.spawn_offsets = torch.zeros(self.num_envs, 3, device=self.device)
        self.veh_heading = torch.zeros(self.num_envs, device=self.device)
        self.veh_turn_rate = torch.zeros(self.num_envs, device=self.device)
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
        # Steer at a bounded rate: the heading is the integral of a steering rate
        # the hull can actually deliver, not a random walk it has to chase.
        noise = torch.randn(self.num_envs, device=self.device, generator=self.gen)
        self.veh_turn_rate = (self.veh_turn_rate * c.target_turn_decay
                              + noise * c.target_turn_jitter).clamp(-c.target_yaw_rate,
                                                                    c.target_yaw_rate)
        self.veh_heading = self.veh_heading + self.veh_turn_rate * self._dt
        # steer back toward the arena before driving out of it
        p = self.target_pos
        r = p[:, :2].norm(dim=1)
        out = r > c.arena_radius
        if out.any():
            inward = torch.atan2(-p[out, 1], -p[out, 0])
            self.veh_heading[out] = inward
            self.veh_turn_rate[out] = 0.0
        vel = self._vehicle.data.root_vel_w.clone()
        vel[:, 0] = c.target_speed * torch.cos(self.veh_heading)
        vel[:, 1] = c.target_speed * torch.sin(self.veh_heading)
        vel[:, 3:5] = 0.0                                    # no roll/pitch rate
        # Steer the hull to face where it is going. Without this the vehicle keeps
        # whatever yaw it spawned with and crabs sideways across the terrain.
        q = self._vehicle.data.root_quat_w
        yaw = torch.atan2(2 * (q[:, 0] * q[:, 3] + q[:, 1] * q[:, 2]),
                          1 - 2 * (q[:, 2] ** 2 + q[:, 3] ** 2))
        want = self.veh_heading + self._veh_yaw_offset
        err = torch.atan2(torch.sin(want - yaw), torch.cos(want - yaw))
        # Correction authority, not a physical turn limit: the steering rate above
        # is already capped at target_yaw_rate, so the servo needs headroom above
        # it to close residual error instead of trailing at exactly the cap.
        lim = 3.0 * c.target_yaw_rate
        vel[:, 5] = (err / self._dt).clamp(-lim, lim)
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
        self._hit = info["intercept"]
        self.prev_dist = info["dist"]
        self.extras["intercept"] = self._hit
        self.extras["dist"] = info["dist"]
        self.extras["time_to_intercept"] = torch.where(
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
        self.veh_turn_rate[env_ids] = 0.0
        root = torch.zeros(k, 13, device=dev)
        root[:, :3] = tp + self.scene.env_origins[env_ids]
        root[:, 2] = self.scene.env_origins[env_ids][:, 2] + self.tcfg.target_h
        yaw = heading + self._veh_yaw_offset             # model nose -> travel direction
        root[:, 3] = torch.cos(yaw / 2)                  # w
        root[:, 6] = torch.sin(yaw / 2)                  # z
        self._vehicle.write_root_pose_to_sim(root[:, :7], env_ids)
        self._vehicle.write_root_velocity_to_sim(root[:, 7:], env_ids)

        dpos = self._robot.data.root_pos_w[env_ids] - self.scene.env_origins[env_ids]
        self.prev_dist[env_ids] = (tp - dpos).norm(dim=1)
