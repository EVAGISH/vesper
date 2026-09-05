"""ChaseEnv: N drones launch, find a tank with their camera, and detonate near it.

The task itself is vesper.lab.chase_task; the vehicles are vesper.lab.vehicles.
This file is the Isaac Lab shell around them: one shared site (a global prim,
env_spacing 0, drones kept apart by collision filtering), K tanks that belong
to nobody and can be hit by everybody, a body-fixed camera per drone that
renders RGB, depth and instance segmentation in one pass, and an airframe
contact sensor for crashes.

Observation dict:
  policy       [N,11] (+1)   proprio -- the actor's vector input; with
                             `geofence` a 12th value, the signed distance to the
                             nearest safe zone, is appended (see ChaseEnvCfg)
  pixels       [N,H,W,3]     uint8 RGB from the tilted camera
  depth        [N,H,W,1]     stereo-class depth in [0,1] (vesper.sensors.depth)
  privileged   [N,priv_dim]  truth, for the critic and the state teacher
`ppo_key` picks which of these the flat PPO trainer sees ("privileged" for the
teacher); the recurrent vision trainer reads the whole dict.

The fourth policy output is the detonation trigger. At detonation time, the task
marks every tank inside a small 3D blast radius as hit. PhysX contacts before
detonation are crashes rather than successes.
"""
import math
import os

import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, RigidObject
from isaaclab.sim import SimulationCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass

from vesper.control.se3 import SE3Controller, quat_to_rot
from vesper.lab import chase_task as T
from vesper.lab.frames import PROPRIO_DIM, seg_counts, seg_lookup, setpoint, yaw_from_quat
from vesper.lab.pursuit_env import resolve_vehicle, vehicle_cfg
from vesper.lab.vehicles import TankDriver
from vesper.lab.vesper_quad import VesperQuadEnv, VesperQuadEnvCfg, iris_cfg
from vesper.sensors.depth import DepthModel
from vesper.worlds.heightmap import WorldMap
from vesper.worlds.zones import Zones, find_zones

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
CORNELL_USD = os.path.join(REPO, "assets", "cornell", "cornell.usd")
CORNELL_MAP = os.path.join(REPO, "assets", "cornell", "cornell_map.npz")

VEHICLE_ROOT = "/World/vehicles"
VEHICLE_SEMANTIC = "vehicle"
VEHICLE_PATH_RX = r"/World/vehicles/v(\d+)/Tank"
CRASH_FORCE_N = 2.0          # airframe contact force that counts as a crash


@configclass
class ChaseEnvCfg(VesperQuadEnvCfg):
    episode_length_s = 90.0          # ~1200 m of path: enough to cross the site
    decimation = 4                                  # 25 Hz guidance over a 100 Hz inner loop
    sim: SimulationCfg = SimulationCfg(dt=1 / 100, render_interval=4)
    action_space = 4                                # body forward / left / up + detonate
    observation_space = PROPRIO_DIM
    state_space = 0                                 # priv_dim, filled at runtime
    chase: dict | None = None                       # ChaseCfg overrides
    world_usd: str = CORNELL_USD
    world_map: str = CORNELL_MAP
    require_tree_colliders: bool = True              # reject stale visual-only site exports
    zones: str | None = None                        # zones.json; default: beside the map or <site>_zones.json
    vehicle_model: str | None = None
    n_targets: int = 12
    vehicle_cycle_s: float = 180.0                  # tanks are re-placed this often
    accel_limit: float = 11.0
    yaw_follow_tau_s: float = 0.6
    yaw_follow_min_speed: float = 1.5
    detonate_threshold: float = 0.5
    # --- camera ---
    camera: bool = False
    cam_res: int = 96
    cam_offset: tuple = (0.12, 0.0, -0.04)
    depth_max_m: float = 20.0
    # The safe zone is a place the drone is penalised for being in, but the actor
    # has no map: on one fixed site it has to learn the boundary from landmarks.
    # Setting this appends the signed distance to the zone to the proprio vector,
    # which is the geofence receiver a real airframe would carry -- the fallback
    # if landmark learning turns out to be too slow.
    geofence: bool = False
    ppo_key: str = "privileged"
    terrain: TerrainImporterCfg = TerrainImporterCfg(
        prim_path="/World/ground", terrain_type="usd", usd_path=CORNELL_USD, collision_group=-1,
    )


class ChaseEnv(VesperQuadEnv):
    cfg: ChaseEnvCfg

    def __init__(self, cfg: ChaseEnvCfg, render_mode=None, seed=0, **kwargs):
        self.tcfg = T.ChaseCfg(**(cfg.chase or {}))
        self.tcfg.n_targets = cfg.n_targets
        self.k = cfg.n_targets
        self._seed = int(seed)
        cfg.terrain.usd_path = cfg.world_usd
        self._veh_spec = resolve_vehicle(cfg.vehicle_model)
        self._veh_yaw_offset = float(self._veh_spec["yaw_offset"])
        self._veh_target_height = float(self._veh_spec.get("target_height", 0.0))
        cfg.observation_space = PROPRIO_DIM + (1 if cfg.geofence else 0)
        cfg.state_space = 12 + 6 * self.k + 2
        if cfg.robot is None:
            cfg.robot = iris_cfg()
        cfg.robot.spawn.activate_contact_sensors = True      # the airframe reports what it hits
        super().__init__(cfg, render_mode, **kwargs)

        N, dev = self.num_envs, self.device
        self.gen = torch.Generator(device=dev); self.gen.manual_seed(self._seed)
        self.world = WorldMap(cfg.world_map, device=dev)
        if cfg.require_tree_colliders and not self.world.has_tree_solids:
            raise RuntimeError(
                f"{cfg.world_map} has visual-only trees; rebuild its species wrappers "
                "with colliders and rerun scripts/export_world_map.py"
            )
        zpath = cfg.zones or find_zones(cfg.world_map, REPO)
        if zpath:
            self.world.attach_zones(Zones.load(zpath))
        self.zones_path = str(zpath) if zpath else None
        self.ctrl = SE3Controller(self.params, N, device=dev)
        self.ctrl.accel_limit = cfg.accel_limit
        self._dt = self.cfg.sim.dt * self.cfg.decimation
        self.task = T.ChaseTask(self.world, self.tcfg, N, self._dt, int(self.max_episode_length),
                                device=dev, generator=self.gen)
        self.driver = TankDriver(self.world, self.k, self.tcfg.arena_half, device=dev, generator=self.gen)
        self.depth_model = DepthModel(max_range=cfg.depth_max_m, generator=self.gen)
        self.spawn_offsets = torch.zeros(N, 3, device=dev)
        self.yaw_des = torch.zeros(N, device=dev)
        self._setpoint = torch.zeros(N, 3, device=dev)
        self._reward = torch.zeros(N, device=dev)
        self._term = torch.zeros(N, dtype=torch.bool, device=dev)
        self._detonated = torch.zeros(N, dtype=torch.bool, device=dev)
        self._evaluated = False
        self._seg_table = None
        self._seg_labels_n = -1
        self._veh_step = 0
        self._veh_cycle = int(cfg.vehicle_cycle_s / self._dt)
        self._place_vehicles()

    # ---------------------------------------------------------------- scene
    def _setup_scene(self):
        self._robot = Articulation(self.cfg.robot)
        self.scene.articulations["robot"] = self._robot
        # K tanks that belong to no environment: global prims every drone can hit
        import isaacsim.core.utils.prims as prim_utils
        prim_utils.create_prim(VEHICLE_ROOT, "Xform")
        for i in range(self.k):
            prim_utils.create_prim(f"{VEHICLE_ROOT}/v{i}", "Xform")
        vcfg = vehicle_cfg(self._veh_spec)
        vcfg.prim_path = f"{VEHICLE_ROOT}/v.*/Tank"
        vcfg.init_state.pos = (0.0, 0.0, 40.0)
        vcfg.spawn.semantic_tags = [("class", VEHICLE_SEMANTIC)]
        self._vehicles = RigidObject(vcfg)
        self.scene.rigid_objects["vehicles"] = self._vehicles
        # Any airframe contact before detonation is a crash.
        from isaaclab.sensors import ContactSensor, ContactSensorCfg
        self._contact = ContactSensor(ContactSensorCfg(
            prim_path="/World/envs/env_.*/Robot/body",
            update_period=0.0, history_length=0,
        ))
        self.scene.sensors["contact"] = self._contact
        self._cam = self._make_camera() if self.cfg.camera else None
        if self._cam is not None:
            self.scene.sensors["cam"] = self._cam
        self.cfg.terrain.num_envs = self.scene.cfg.num_envs
        self.cfg.terrain.env_spacing = self.scene.cfg.env_spacing
        self._terrain = self.cfg.terrain.class_type(self.cfg.terrain)
        self._terrain.env_origins[:] = 0.0
        self.scene.clone_environments(copy_from_source=False)
        self.scene.filter_collisions(global_prim_paths=[self.cfg.terrain.prim_path, VEHICLE_ROOT])

    def _make_camera(self):
        """Body-fixed TiledCamera, pitched forward-down, RGB + depth + segmentation."""
        from isaaclab.sensors import TiledCamera, TiledCameraCfg
        half = math.radians(self.tcfg.fov_half_deg)
        aperture = 20.955
        focal = aperture / (2.0 * math.tan(half))
        p = math.radians(self.tcfg.cam_pitch_deg) / 2.0
        cfg = TiledCameraCfg(
            prim_path="/World/envs/env_.*/Robot/body/cam",
            offset=TiledCameraCfg.OffsetCfg(pos=tuple(self.cfg.cam_offset),
                                            rot=(math.cos(p), 0.0, math.sin(p), 0.0),
                                            convention="world"),
            data_types=["rgb", "depth", "instance_segmentation_fast"],
            colorize_instance_segmentation=False,
            spawn=sim_utils.PinholeCameraCfg(focal_length=focal, horizontal_aperture=aperture,
                                             clipping_range=(0.2, 1500.0)),
            width=self.cfg.cam_res, height=self.cfg.cam_res,
        )
        return TiledCamera(cfg)

    # ---------------------------------------------------------------- vehicles
    @property
    def vehicle_pos(self):
        """[K,3]"""
        return self._vehicles.data.root_pos_w

    @property
    def target_pos(self):
        """[N,K,3] -- every drone hunts every tank."""
        p = self.vehicle_pos.clone()
        p[:, 2] += self._veh_target_height
        return p.unsqueeze(0).expand(self.num_envs, -1, -1)

    def _place_vehicles(self):
        ids = torch.arange(self.k, device=self.device)
        self.driver.assign_roles(ids)
        xy, heading = self.driver.place(ids)
        gz = self.world.ground_at(xy[:, 0], xy[:, 1])
        root = torch.zeros(self.k, 13, device=self.device)
        root[:, 0] = xy[:, 0]; root[:, 1] = xy[:, 1]
        root[:, 2] = gz + float(self._veh_spec.get("ground_clearance", 0.08))
        yaw = heading + self._veh_yaw_offset
        root[:, 3] = torch.cos(yaw / 2); root[:, 6] = torch.sin(yaw / 2)
        self._vehicles.write_root_pose_to_sim(root[:, :7])
        self._vehicles.write_root_velocity_to_sim(root[:, 7:])
        self._veh_step = 0

    def _drive_vehicles(self):
        # the arena can shrink under a curriculum; tanks placed outside the box
        # the drone is confined to are tanks it can never reach
        self.driver.half = self.tcfg.arena_half
        d = self._vehicles.data
        q = d.root_quat_w
        yaw = yaw_from_quat(q) - self._veh_yaw_offset
        speed = d.root_lin_vel_w[:, :2].norm(dim=1)
        v_xy, yaw_rate = self.driver.command(d.root_pos_w[:, :2], yaw, speed, self._dt)
        vel = d.root_vel_w.clone()
        vel[:, 0:2] = v_xy
        vel[:, 3:5] = 0.0
        vel[:, 5] = yaw_rate
        self._vehicles.write_root_velocity_to_sim(vel)
        self._veh_step += 1
        if self._veh_step >= self._veh_cycle:
            self._place_vehicles()

    def protected(self):
        """[N,K] tank inside a safe zone (shared, so the same row for every drone)."""
        p = self.vehicle_pos
        return self.world.in_safe(p[:, 0], p[:, 1]).unsqueeze(0).expand(self.num_envs, -1)

    # ---------------------------------------------------------------- control
    def _pre_physics_step(self, actions):
        self._action = actions
        self._detonated = actions[:, 3] > self.cfg.detonate_threshold
        self._drive_vehicles()
        d = self._robot.data
        yaw = yaw_from_quat(d.root_quat_w)
        self._setpoint = setpoint(d.root_pos_w, actions[:, :3], yaw, self.tcfg.look_ahead)
        v = d.root_lin_vel_w
        speed = v[:, :2].norm(dim=1)
        target = torch.atan2(v[:, 1], v[:, 0])
        err = torch.atan2(torch.sin(target - self.yaw_des), torch.cos(target - self.yaw_des))
        a = min(1.0, self._dt / max(self.cfg.yaw_follow_tau_s, 1e-3))
        self.yaw_des = torch.where(speed > self.cfg.yaw_follow_min_speed, self.yaw_des + a * err, self.yaw_des)
        self._evaluated = False

    def _apply_action(self):
        d = self._robot.data
        vel_w = d.root_lin_vel_w - self.wind_world
        omega = self.ctrl.compute(d.root_pos_w, vel_w, d.root_quat_w, d.root_ang_vel_b,
                                  self._setpoint, yaw_des=self.yaw_des)
        R = quat_to_rot(d.root_quat_w)
        v_body = (R.transpose(1, 2) @ vel_w.unsqueeze(2)).squeeze(2)
        force_b, torque_b = self.dynamics.wrench(omega, v_body)
        self._thrust[:, 0, :] = force_b
        self._moment[:, 0, :] = torque_b
        self._robot.set_external_force_and_torque(self._thrust, self._moment, body_ids=self._body_id)

    # ---------------------------------------------------------------- sensors
    def _crashed(self):
        """[N] physical airframe contact before a deliberate detonation."""
        cd = self._contact.data
        return cd.net_forces_w[:, 0].norm(dim=1) > CRASH_FORCE_N

    def _seen_px(self):
        if self._cam is None:
            return None
        seg = self._cam.data.output.get("instance_segmentation_fast")
        if seg is None:
            return None
        labels = ((self._cam.data.info or {}).get("instance_segmentation_fast") or {}).get("idToLabels") or {}
        if len(labels) != self._seg_labels_n:
            # every tank is "env 0": the table maps id -> 0 * K + slot
            self._seg_table = seg_lookup(labels, r"v(\d+)/Tank", self.k).to(self.device)
            self._seg_table = torch.where(self._seg_table >= 0, self._seg_table % self.k, self._seg_table)
            self._seg_labels_n = len(labels)
        if self._seg_table is None or self._seg_table.numel() <= 1:
            return torch.zeros(self.num_envs, self.k, dtype=torch.long, device=self.device)
        group = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        return seg_counts(seg, self._seg_table, group, self.k)

    def pixels(self):
        return None if self._cam is None else self._cam.data.output.get("rgb")

    def depth(self):
        if self._cam is None:
            return None
        raw = self._cam.data.output.get("depth")
        return None if raw is None else self.depth_model(raw)

    # ---------------------------------------------------------------- task
    def _evaluate(self):
        if self._evaluated:
            return
        d = self._robot.data
        crashed = self._crashed()
        self._last_visible = None
        _, self._reward, self._term, info = self.task.step(
            d.root_pos_w, d.root_lin_vel_w, d.root_quat_w, d.root_ang_vel_b,
            self.target_pos, self.episode_length_buf,
            detonated=self._detonated, crashed=crashed,
            seen_px=self._seen_px(), protected=self.protected())
        # Flat PPO's generic episode collector calls successes "intercepts".
        # Keep that adapter contract while exposing the task-native hit fields.
        info["intercept"] = info["hit"]
        info["time_to_intercept"] = info["time_to_hit"]
        self._last_visible = info["visible"]
        self.extras.update(info)
        self._evaluated = True

    def _get_observations(self):
        d = self._robot.data
        ground = self.world.ground_at(d.root_pos_w[:, 0], d.root_pos_w[:, 1])
        agl = (d.root_pos_w[:, 2] - ground).clamp(min=0.0)
        tf = (self.episode_length_buf.float() / self.max_episode_length).clamp(0, 1)
        from vesper.lab.frames import proprio
        vis = (self._last_visible if getattr(self, "_last_visible", None) is not None
               else torch.zeros(self.num_envs, self.k, dtype=torch.bool, device=self.device))
        pr = proprio(d.root_lin_vel_w, d.root_quat_w, d.root_ang_vel_b, agl, tf)
        if self.cfg.geofence:
            pr = torch.cat([pr, self.task.geofence(d.root_pos_w).unsqueeze(1)], dim=1)
        obs = {"policy": pr,
               "privileged": self.task.privileged(d.root_pos_w, d.root_lin_vel_w, d.root_quat_w,
                                                  d.root_ang_vel_b, self.target_pos, vis,
                                                  self.protected(), tf)}
        px = self.pixels()
        if px is not None:
            obs["pixels"] = px
            obs["depth"] = self.depth()
        return obs

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
        env_ids = torch.as_tensor(env_ids, device=dev)
        # launch zone, at a random altitude, clear of whatever is beneath
        dxy, _ = self.world.sample_cells_xy(self.world.launch, n, g, half=c.arena_half * 0.95)
        solid = self.world.solid_at(dxy[:, 0], dxy[:, 1])
        ground = self.world.ground_at(dxy[:, 0], dxy[:, 1])
        alt = c.spawn_alt_min + torch.rand(n, device=dev, generator=g) * (c.spawn_alt_max - c.spawn_alt_min)
        dz = torch.maximum(ground + alt, solid + 15.0)
        default = self._robot.data.default_root_state[env_ids]
        self.spawn_offsets[env_ids] = torch.stack([dxy[:, 0], dxy[:, 1], dz], dim=1) - default[:, :3]
        super()._reset_idx(env_ids)
        dyaw = torch.rand(n, device=dev, generator=g) * (2 * torch.pi)
        pose = torch.zeros(n, 7, device=dev)
        pose[:, 0] = dxy[:, 0]; pose[:, 1] = dxy[:, 1]; pose[:, 2] = dz
        pose[:, 3] = torch.cos(dyaw / 2); pose[:, 6] = torch.sin(dyaw / 2)
        self._robot.write_root_pose_to_sim(pose, env_ids)
        self.yaw_des[env_ids] = dyaw
        self.task.reset(env_ids)

    # ---------------------------------------------------------------- ppo glue
    @property
    def num_obs(self):
        return self.cfg.state_space if self.cfg.ppo_key == "privileged" else self.cfg.observation_space

    @property
    def num_actions(self):
        return self.cfg.action_space

    def ppo_reset(self):
        obs, _ = self.reset()
        return obs[self.cfg.ppo_key]

    def ppo_step(self, action):
        obs, rew, term, trunc, info = self.step(action)
        return obs[self.cfg.ppo_key], rew, term | trunc, info

    def vision_reset(self):
        obs, _ = self.reset()
        return obs

    def vision_step(self, action):
        obs, rew, term, trunc, info = self.step(action)
        return obs, rew, term | trunc, info
