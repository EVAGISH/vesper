"""VesperQuad: Isaac Lab DirectRLEnv flying our torch-ported Iris model.

PhysX integrates rigid bodies; per-step body wrench (thrust + drag + moments)
comes from vesper.dynamics (the Pegasus port). Actions are normalized rotor
speeds [N,4] in [0,1] -- whoever steps the env (SE3 controller now, a policy
later) owns the outer loop.
"""
import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, ArticulationCfg
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.envs import DirectRLEnv, DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass

from vesper.dynamics import MultirotorDynamics, MultirotorParams


def iris_cfg() -> ArticulationCfg:
    from pegasus.simulator.params import ROBOTS

    return ArticulationCfg(
        prim_path="/World/envs/env_.*/Robot",
        spawn=sim_utils.UsdFileCfg(
            usd_path=ROBOTS["Iris"],
            rigid_props=sim_utils.RigidBodyPropertiesCfg(disable_gravity=False),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(enabled_self_collisions=False),
        ),
        init_state=ArticulationCfg.InitialStateCfg(pos=(0.0, 0.0, 0.1)),
        actuators={"rotors": ImplicitActuatorCfg(joint_names_expr=[".*"], stiffness=0.0, damping=0.0)},
    )


@configclass
class VesperQuadEnvCfg(DirectRLEnvCfg):
    episode_length_s = 60.0
    decimation = 2
    action_space = 4
    observation_space = 13
    state_space = 0
    sim: SimulationCfg = SimulationCfg(dt=1 / 100, render_interval=2)
    terrain: TerrainImporterCfg = TerrainImporterCfg(
        prim_path="/World/ground", terrain_type="plane", collision_group=-1,
    )
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=1024, env_spacing=6.0, replicate_physics=True)
    robot: ArticulationCfg | None = None  # filled at runtime (needs pegasus import)
    city_buildings: list | None = None    # [{"center":[n,e],"size":[dn,de],"height":h}]


class VesperQuadEnv(DirectRLEnv):
    cfg: VesperQuadEnvCfg

    def __init__(self, cfg: VesperQuadEnvCfg, render_mode=None, **kwargs):
        if cfg.robot is None:
            cfg.robot = iris_cfg()
        super().__init__(cfg, render_mode, **kwargs)
        self.params = MultirotorParams()
        self.dynamics = MultirotorDynamics(self.params, self.num_envs, device=self.device)
        self._thrust = torch.zeros(self.num_envs, 1, 3, device=self.device)
        self._moment = torch.zeros(self.num_envs, 1, 3, device=self.device)
        self._omega = torch.zeros(self.num_envs, 4, device=self.device)
        self.wind_world = torch.zeros(self.num_envs, 3, device=self.device)
        self._body_id = self._robot.find_bodies("body")[0]

    def _setup_scene(self):
        self._robot = Articulation(self.cfg.robot)
        self.scene.articulations["robot"] = self._robot
        self.cfg.terrain.num_envs = self.scene.cfg.num_envs
        self.cfg.terrain.env_spacing = self.scene.cfg.env_spacing
        self._terrain = self.cfg.terrain.class_type(self.cfg.terrain)
        self.scene.clone_environments(copy_from_source=False)
        self.scene.filter_collisions(global_prim_paths=[self.cfg.terrain.prim_path])
        light = sim_utils.DomeLightCfg(intensity=1500.0)
        light.func("/World/Light", light)
        if self.cfg.city_buildings:
            for i, b in enumerate(self.cfg.city_buildings):
                (n, e), (dn, de), h = b["center"], b["size"], b["height"]
                shade = 0.45 + 0.3 * ((i * 37) % 10) / 10
                cub = sim_utils.CuboidCfg(
                    size=(de, dn, h),
                    collision_props=sim_utils.CollisionPropertiesCfg(),
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(shade, shade * 0.95, shade * 0.9)),
                )
                cub.func(f"/World/city/b{i:03d}", cub, translation=(e, n, h / 2))

    def _pre_physics_step(self, actions: torch.Tensor):
        self._omega = actions.clamp(0.0, 1.0) * self.params.omega_max

    def _apply_action(self):
        quat = self._robot.data.root_quat_w
        vel_w = self._robot.data.root_lin_vel_w - self.wind_world
        from vesper.control.se3 import quat_to_rot
        R = quat_to_rot(quat)
        v_body = (R.transpose(1, 2) @ vel_w.unsqueeze(2)).squeeze(2)
        force_b, torque_b = self.dynamics.wrench(self._omega, v_body)
        self._thrust[:, 0, :] = force_b
        self._moment[:, 0, :] = torque_b
        self._robot.set_external_force_and_torque(self._thrust, self._moment, body_ids=self._body_id)

    # state access for the outer loop (positions relative to each env origin)
    def flight_state(self):
        d = self._robot.data
        return (d.root_pos_w - self.scene.env_origins, d.root_lin_vel_w,
                d.root_quat_w, d.root_ang_vel_b)

    def _get_observations(self):
        pos, vel, quat, avb = self.flight_state()
        return {"policy": torch.cat([pos, vel, quat, avb], dim=1)[:, :13]}

    def _get_rewards(self):
        return torch.zeros(self.num_envs, device=self.device)

    def _get_dones(self):
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        return torch.zeros_like(time_out), time_out

    def _reset_idx(self, env_ids):
        super()._reset_idx(env_ids)
        root = self._robot.data.default_root_state[env_ids].clone()
        root[:, :3] += self.scene.env_origins[env_ids]
        if hasattr(self, "spawn_offsets"):
            root[:, :3] += self.spawn_offsets[env_ids]
        self._robot.write_root_pose_to_sim(root[:, :7], env_ids)
        self._robot.write_root_velocity_to_sim(root[:, 7:], env_ids)
