"""SearchEnv: search a real site for several ground vehicles, then run them down.

One drone per environment and `n_targets` tanks per *group* of environments,
all of them dropped at random on the Cornell world (vesper.worlds.geo output).
The world itself is loaded once as a global prim and shared by every environment
-- the rule from STACK.md: thousands of drones, one world -- so env_spacing is
zero and the environments are separated by collision filtering, not by distance.

Three things make this a search rather than a chase:

  * the tanks are placed by concealment class -- driving on the roads in the
    open, painted down, crawling under tree canopy, parked against a building
    -- and which slot gets which role is reshuffled every episode, so the slot
    index tells the policy nothing;
  * the policy never sees a target's true position, nor its own. Its actor
    gets the body-fixed camera and what the airframe measures on its own
    (vesper.lab.search_task.proprio); the belief, coverage and world pose go to
    a separate privileged vector for the critic and for a state-based teacher;
  * the vehicles drive on the real terrain under a velocity controller that
    follows roads, parks along facades, and steers around water, buildings and
    slopes it cannot climb.

Groups. With a rendered camera every environment sees the whole shared world,
so the world cannot hold one set of vehicles per environment: at 256 envs that
would carpet the site with 768 tanks, every one of them in somebody's frame.
`n_groups` = G puts vehicles in the first G environments only; environment i
hunts the vehicles of environment i % G, and the G members of a group start and
end their episodes together so nobody holds a belief about a layout that has
been reshuffled under it. G = num_envs (the default) is the old one-set-per-env
behaviour and is right whenever nothing is rendered.

Action is guidance -- a body-frame velocity command for the SE3 inner loop,
which yaws the airframe to face its travel so the forward camera looks where
the drone is going.
"""
import math
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

# Role table and hull limits are shared with the native env (vesper.native),
# which cannot import this module (isaaclab above); they live in ground.py and
# are re-exported here so existing imports keep working.
from vesper.lab.ground import (ROLES, VEH_ACCEL, VEH_LAT_ACCEL, VEH_STUCK_S,  # noqa: F401,E402
                               VEH_TURN_MAX, VEHICLE_PATH_RX, VEHICLE_SEMANTIC)


@configclass
class SearchEnvCfg(VesperQuadEnvCfg):
    episode_length_s = 75.0
    decimation = 4                      # 25 Hz guidance over a 100 Hz inner loop
    # match the render interval to the decimation: nothing renders in a
    # headless training run, and a mismatch makes Isaac Lab warn every boot
    sim: SimulationCfg = SimulationCfg(dt=1 / 100, render_interval=4)
    action_space = 3                    # body-frame forward / left / up
    observation_space = T.PROPRIO_DIM   # the actor's proprio vector ("policy")
    state_space = 0                     # filled from SearchTask.obs_dim at runtime ("privileged")
    search: dict | None = None
    world_usd: str = CORNELL_USD
    world_map: str = CORNELL_MAP
    vehicle_model: str | None = None
    # 14 m/s^2 asks for a 55 deg lean and 12.5% of evaluated episodes ended in a
    # tumble; 11 caps the steady-state lean near 48 deg and still flies hard.
    accel_limit: float = 11.0
    n_targets: int = 3
    n_groups: int = 0                   # 0 -> one vehicle set per env (see module doc)
    yaw_follow_tau_s: float = 0.6       # how quickly the nose swings onto the velocity
    yaw_follow_min_speed: float = 1.5   # below this the heading is held, not chased
    # --- camera ---
    camera: bool = False                # render the body-fixed camera (needs --enable_cameras)
    cam_res: int = 128                  # square tiles
    cam_offset: tuple = (0.12, 0.0, -0.04)   # camera position in the body frame (m)
    # which key ppo_step() returns as the observation: "privileged" trains the
    # state-based teacher, "policy" is the honest proprio vector (+ "pixels")
    ppo_key: str = "privileged"
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
        # observation widths are properties of the task, not magic numbers
        g = self.tcfg.grid
        cfg.observation_space = T.PROPRIO_DIM
        cfg.state_space = 12 + 8 * self.k + g * g + 3
        self.G = int(cfg.n_groups) if cfg.n_groups and cfg.n_groups > 0 else int(cfg.scene.num_envs)
        self.G = min(self.G, int(cfg.scene.num_envs))
        super().__init__(cfg, render_mode, **kwargs)

        N, dev = self.num_envs, self.device
        self.gen = torch.Generator(device=dev); self.gen.manual_seed(self._seed)
        self.world = WorldMap(cfg.world_map, device=dev)
        self.ctrl = SE3Controller(self.params, N, device=dev)
        self.ctrl.accel_limit = cfg.accel_limit
        self._dt = self.cfg.sim.dt * self.cfg.decimation
        self.task = T.SearchTask(self.world, self.tcfg, N, self._dt,
                                 int(self.max_episode_length), device=dev, generator=self.gen)
        self.group = torch.arange(N, device=dev) % self.G        # whose vehicles env i hunts
        self.spawn_offsets = torch.zeros(N, 3, device=dev)
        self.yaw_des = torch.zeros(N, device=dev)
        self.veh_heading = torch.zeros(N, self.k, device=dev)
        self.veh_turn_rate = torch.zeros(N, self.k, device=dev)
        self.veh_speed = torch.zeros(N, self.k, device=dev)      # cruise speed for the role
        self.veh_speed_cmd = torch.zeros(N, self.k, device=dev)  # ramped command
        self.veh_stuck_s = torch.zeros(N, self.k, device=dev)
        self.veh_on_road = torch.zeros(N, self.k, dtype=torch.bool, device=dev)
        self.role = torch.zeros(N, self.k, dtype=torch.long, device=dev)
        self._setpoint = torch.zeros(N, 3, device=dev)
        self._reward = torch.zeros(N, device=dev)
        self._term = torch.zeros(N, dtype=torch.bool, device=dev)
        self._evaluated = False
        self._seg_table = None
        self._seg_labels_n = -1
        # probe fan used to steer vehicles away from ground they cannot drive on
        self._probe = torch.tensor([0.0, 0.5, -0.5, 1.0, -1.0, 1.8, -1.8, 3.14159], device=dev)
        # a dormant vehicle set (envs beyond the first G) is parked out of the
        # arena, invisible, and never driven
        self._dormant_xy = torch.tensor([self.world.half_m - 15.0, self.world.half_m - 15.0], device=dev)

    # ---------------------------------------------------------------- scene
    def _setup_scene(self):
        self._robot = Articulation(self.cfg.robot)
        self.scene.articulations["robot"] = self._robot
        self._vehicles = []
        for i in range(self.k):
            vcfg = vehicle_cfg(self._veh_spec)
            vcfg.prim_path = f"/World/envs/env_.*/Vehicle_{i}"
            vcfg.init_state.pos = (60.0 + 30.0 * i, 0.0, 40.0)
            # the semantic tag is what the instance segmentation keys on; without
            # it a tank is just more scenery to the camera
            vcfg.spawn.semantic_tags = [("class", VEHICLE_SEMANTIC)]
            obj = RigidObject(vcfg)
            self.scene.rigid_objects[f"vehicle_{i}"] = obj
            self._vehicles.append(obj)
        if self.cfg.camera:
            self._cam = self._make_camera()
            self.scene.sensors["cam"] = self._cam
        else:
            self._cam = None
        self.cfg.terrain.num_envs = self.scene.cfg.num_envs
        self.cfg.terrain.env_spacing = self.scene.cfg.env_spacing
        self._terrain = self.cfg.terrain.class_type(self.cfg.terrain)
        # One world, many drones: every environment sits on the same origin and
        # they are kept apart by collision filtering instead of by spacing.
        self._terrain.env_origins[:] = 0.0
        self.scene.clone_environments(copy_from_source=False)
        self.scene.filter_collisions(global_prim_paths=[self.cfg.terrain.prim_path])
        self._hide_dormant_vehicles()
        # the site USD carries its own sun and sky; nothing more to light it with

    def _make_camera(self):
        """Body-fixed TiledCamera on the airframe, pitched forward-down.

        Intrinsics come from the task config so the rendered lens and the
        geometric cone are one and the same: a square tile whose full angle is
        2 * fov_half_deg. RGB for the policy, instance segmentation for the
        reward's notion of "in frame".
        """
        from isaaclab.sensors import TiledCamera, TiledCameraCfg
        half = math.radians(self.tcfg.fov_half_deg)
        aperture = 20.955                                  # mm, Isaac's default sensor width
        focal = aperture / (2.0 * math.tan(half))
        p = math.radians(self.tcfg.cam_pitch_deg) / 2.0
        cfg = TiledCameraCfg(
            prim_path="/World/envs/env_.*/Robot/body/cam",
            offset=TiledCameraCfg.OffsetCfg(pos=tuple(self.cfg.cam_offset),
                                            rot=(math.cos(p), 0.0, math.sin(p), 0.0),
                                            convention="world"),
            data_types=["rgb", "instance_segmentation_fast"],
            colorize_instance_segmentation=False,
            spawn=sim_utils.PinholeCameraCfg(focal_length=focal, horizontal_aperture=aperture,
                                             clipping_range=(0.2, 1500.0)),
            width=self.cfg.cam_res, height=self.cfg.cam_res,
        )
        return TiledCamera(cfg)

    def _hide_dormant_vehicles(self):
        """Vehicles of envs beyond the first G exist (the rigid-object view wants
        one per env) but are nobody's target: make them invisible to every camera."""
        if self.G >= self.scene.cfg.num_envs:
            return
        from pxr import UsdGeom
        import isaacsim.core.utils.stage as stage_utils
        stage = stage_utils.get_current_stage()
        for i in range(self.G, self.scene.cfg.num_envs):
            for k in range(self.k):
                prim = stage.GetPrimAtPath(f"/World/envs/env_{i}/Vehicle_{k}")
                if prim:
                    UsdGeom.Imageable(prim).MakeInvisible()

    # ---------------------------------------------------------------- targets
    @property
    def vehicle_pos(self):
        """[N,K,3] world positions of every vehicle set, dormant ones included."""
        return torch.stack([v.data.root_pos_w for v in self._vehicles], dim=1)

    @property
    def target_pos(self):
        """[N,K,3] the vehicles env i is hunting: its group's set."""
        return self.vehicle_pos[self.group]

    def _drive_vehicles(self):
        """Velocity-controlled driving that respects the map.

        Heading is the integral of a bounded steering rate (so the hull can
        actually deliver it), nudged by three constraints: stay inside the
        arena, do not drive into water, a building or a slope, and -- for the
        roles that drive on roads -- prefer road ahead over lawn. Speed ramps
        from rest, the turn rate is capped by a lateral-acceleration budget so
        a fast tank cannot spin on the spot, and a hull that has been
        pushing against something it cannot see for a couple of seconds
        (a trunk, another vehicle) turns away and tries again. z is left to
        gravity and contacts so the vehicle rides the real terrain.
        """
        c, dt = self.tcfg, self._dt
        pos = self.vehicle_pos                                   # [N,K,3]
        flat = pos.reshape(-1, 3)
        N, K = self.num_envs, self.k

        # --- speed: ramp toward the role's cruise, stuck detection on the actual hull speed
        actual = torch.stack([v.data.root_lin_vel_w[:, :2].norm(dim=1) for v in self._vehicles], 1)
        self.veh_speed_cmd = torch.minimum(self.veh_speed_cmd + VEH_ACCEL * dt, self.veh_speed)
        want_move = self.veh_speed_cmd > 0.5
        slow = actual < 0.3 * self.veh_speed_cmd
        self.veh_stuck_s = torch.where(want_move & slow, self.veh_stuck_s + dt, torch.zeros_like(self.veh_stuck_s))
        stuck = self.veh_stuck_s > VEH_STUCK_S
        if stuck.any():
            sign = torch.where(torch.rand(N, K, device=self.device, generator=self.gen) < 0.5, -1.0, 1.0)
            self.veh_heading = torch.where(stuck, self.veh_heading + sign * (math.pi / 2), self.veh_heading)
            self.veh_speed_cmd = torch.where(stuck, torch.zeros_like(self.veh_speed_cmd), self.veh_speed_cmd)
            self.veh_stuck_s = torch.where(stuck, torch.zeros_like(self.veh_stuck_s), self.veh_stuck_s)

        # --- wander, bounded by what the hull can steer at this speed
        turn_max = torch.minimum(torch.full_like(self.veh_speed_cmd, VEH_TURN_MAX),
                                 VEH_LAT_ACCEL / self.veh_speed_cmd.clamp(min=0.5))
        noise = torch.randn(N, K, device=self.device, generator=self.gen)
        self.veh_turn_rate = (self.veh_turn_rate * 0.995 + noise * 0.05).clamp(-1.0, 1.0) * turn_max
        self.veh_heading = self.veh_heading + self.veh_turn_rate * dt

        # --- look ahead along a fan of candidate headings and pick drivable ground
        probe_h = self.veh_heading.unsqueeze(2) + self._probe.view(1, 1, -1)     # [N,K,P]
        ahead = 14.0
        px = flat[:, 0].view(N, K, 1) + ahead * torch.cos(probe_h)
        py = flat[:, 1].view(N, K, 1) + ahead * torch.sin(probe_h)
        r, col = self.world.nearest_cell(px, py)
        ok = self.world.drivable[r, col]                                        # [N,K,P]
        road = self.world.road[r, col]
        inside = (px.abs() < c.arena_half) & (py.abs() < c.arena_half)
        score = ok * inside.float() - 0.03 * self._probe.abs().view(1, 1, -1)   # prefer straight on
        score = score + 0.6 * road * self.veh_on_road.unsqueeze(2).float()      # and road, if that is the job
        best = score.argmax(dim=2)
        best_score = score.gather(2, best.unsqueeze(2)).squeeze(2)
        straight = score[..., 0]
        # leave the current heading only when it is blocked, or when a road-
        # follower's own heading is off the road and a probe finds one
        blocked = straight < 0.5
        # a road-follower whose own heading is drifting off the road takes the
        # probe that stays on it; on a straight road every probe scores alike
        # and the 0.3 margin keeps it from twitching between neighbours
        leave = blocked | (self.veh_on_road & (best_score > straight + 0.3))
        chosen = probe_h.gather(2, best.unsqueeze(2)).squeeze(2)
        self.veh_heading = torch.where(leave, chosen, self.veh_heading)
        # nothing drivable in any direction: turn back toward the arena centre
        home = torch.atan2(-flat[:, 1].view(N, K), -flat[:, 0].view(N, K))
        lost = score.amax(dim=2) < 0.5
        self.veh_heading = torch.where(lost, home, self.veh_heading)
        self.veh_turn_rate = torch.where(leave | lost, torch.zeros_like(self.veh_turn_rate),
                                         self.veh_turn_rate)

        for i, v in enumerate(self._vehicles):
            vel = v.data.root_vel_w.clone()
            sp = self.veh_speed_cmd[:, i]
            # slow into a turn: at the full steering rate the hull drops to half
            # speed, which is what keeps a tank from drifting off a bend
            sp = sp * (1.0 - 0.5 * (self.veh_turn_rate[:, i].abs() / VEH_TURN_MAX).clamp(max=1.0))
            vel[:, 0] = sp * torch.cos(self.veh_heading[:, i])
            vel[:, 1] = sp * torch.sin(self.veh_heading[:, i])
            vel[:, 3:5] = 0.0                                     # no roll/pitch rate
            q = v.data.root_quat_w
            yaw = torch.atan2(2 * (q[:, 0] * q[:, 3] + q[:, 1] * q[:, 2]),
                              1 - 2 * (q[:, 2] ** 2 + q[:, 3] ** 2))
            want = self.veh_heading[:, i] + self._veh_yaw_offset
            err = torch.atan2(torch.sin(want - yaw), torch.cos(want - yaw))
            vel[:, 5] = (err / dt).clamp(-3.6, 3.6)
            # a stopped vehicle stays stopped: no yaw servo chatter on a parked hull
            moving = (self.veh_speed[:, i] > 0.05).float()
            vel[:, 5] = vel[:, 5] * moving
            v.write_root_velocity_to_sim(vel)

    # ---------------------------------------------------------------- control
    def _pre_physics_step(self, actions):
        self._action = actions
        self._drive_vehicles()
        d = self._robot.data
        yaw = T.yaw_from_quat(d.root_quat_w)
        self._setpoint = T.setpoint(d.root_pos_w, actions, yaw, self.tcfg)
        # the nose follows the velocity: a body-fixed camera has to look where the
        # drone is going, and a policy without a compass has no world yaw to ask for
        v = d.root_lin_vel_w
        speed = v[:, :2].norm(dim=1)
        target = torch.atan2(v[:, 1], v[:, 0])
        err = torch.atan2(torch.sin(target - self.yaw_des), torch.cos(target - self.yaw_des))
        a = min(1.0, self._dt / max(self.cfg.yaw_follow_tau_s, 1e-3))
        follow = speed > self.cfg.yaw_follow_min_speed
        self.yaw_des = torch.where(follow, self.yaw_des + a * err, self.yaw_des)
        self._evaluated = False

    def _apply_action(self):
        d = self._robot.data
        vel_w = d.root_lin_vel_w - self.wind_world
        omega = self.ctrl.compute(d.root_pos_w, vel_w, d.root_quat_w, d.root_ang_vel_b,
                                  self._setpoint, yaw_des=self.yaw_des)
        from vesper.control.se3 import quat_to_rot
        R = quat_to_rot(d.root_quat_w)
        v_body = (R.transpose(1, 2) @ vel_w.unsqueeze(2)).squeeze(2)
        force_b, torque_b = self.dynamics.wrench(omega, v_body)
        self._thrust[:, 0, :] = force_b
        self._moment[:, 0, :] = torque_b
        self._robot.set_external_force_and_torque(self._thrust, self._moment, body_ids=self._body_id)

    # ---------------------------------------------------------------- camera
    def _seen_px(self):
        """[N,K] pixels of each of the env's targets in its own frame, or None
        when there is no camera (the task then falls back to geometry)."""
        if self._cam is None:
            return None
        out = self._cam.data.output
        seg = out.get("instance_segmentation_fast")
        if seg is None:
            return None
        info = (self._cam.data.info or {}).get("instance_segmentation_fast") or {}
        labels = info.get("idToLabels") or {}
        if len(labels) != self._seg_labels_n:
            self._seg_table = T.seg_lookup(labels, VEHICLE_PATH_RX, self.k, groups=self.G).to(self.device)
            self._seg_labels_n = len(labels)
        if self._seg_table is None or self._seg_table.numel() <= 1:
            return torch.zeros(self.num_envs, self.k, dtype=torch.long, device=self.device)
        return T.seg_counts(seg, self._seg_table, self.group, self.k)

    def pixels(self):
        """[N,H,W,3] uint8 -- the actor's view, or None without a camera."""
        if self._cam is None:
            return None
        return self._cam.data.output.get("rgb")

    # ---------------------------------------------------------------- task
    def _evaluate(self):
        if self._evaluated:
            return
        d = self._robot.data
        _, self._reward, self._term, info = self.task.step(
            d.root_pos_w, d.root_lin_vel_w, d.root_quat_w, d.root_ang_vel_b,
            self.target_pos, self.episode_length_buf, seen_px=self._seen_px())
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
        obs = {"policy": self.task.proprio(d.root_lin_vel_w, d.root_quat_w, d.root_ang_vel_b, agl, tf),
               "privileged": self.task.privileged(d.root_pos_w, d.root_lin_vel_w, d.root_quat_w,
                                                  d.root_ang_vel_b, agl, tf)}
        px = self.pixels()
        if px is not None:
            obs["pixels"] = px
        return obs

    def _get_rewards(self):
        self._evaluate()
        return self._reward

    def _get_dones(self):
        self._evaluate()
        trunc = self.episode_length_buf >= self.max_episode_length - 1
        if self.G < self.num_envs:
            # a group lives and dies with the env that owns its vehicles: when that
            # one ends, the layout is about to be reshuffled, so everyone hunting
            # it starts over too
            owner_done = (self._term | trunc)[: self.G]
            trunc = trunc | owner_done[self.group]
        return self._term, trunc

    # ---------------------------------------------------------------- reset
    def _reset_idx(self, env_ids):
        n = len(env_ids)
        g, dev, c = self.gen, self.device, self.tcfg
        env_ids = torch.as_tensor(env_ids, device=dev)
        rep = env_ids[env_ids < self.G]                       # envs whose vehicle set is live
        dormant = env_ids[env_ids >= self.G]

        speed = torch.tensor([r[1] for r in ROLES], device=dev)
        contrast = torch.tensor([r[2] for r in ROLES], device=dev)
        layer = [r[3] for r in ROLES]
        on_road = torch.tensor([r[4] for r in ROLES], device=dev)

        if len(rep):
            m = len(rep)
            # --- roles: a fresh shuffle per episode so slot 0 is not always the easy one.
            # Role 0 (a tank driving in the open) is always one of them: it is the
            # only class a policy that cannot yet search reliably will ever stumble
            # onto, so guaranteeing one per episode is what keeps the reach reward
            # reachable at all early in training.
            perm = torch.argsort(torch.rand(m, len(ROLES), device=dev, generator=g), dim=1)[:, :self.k]
            has_open = (perm == 0).any(dim=1)
            slot = torch.randint(0, self.k, (m,), device=dev, generator=g)
            perm[~has_open, slot[~has_open]] = 0
            self.role[rep] = perm
            self.veh_speed[rep] = speed[perm] * (0.7 + 0.6 * torch.rand(m, self.k, device=dev, generator=g))
            self.veh_speed_cmd[rep] = 0.0
            self.veh_stuck_s[rep] = 0.0
            self.veh_on_road[rep] = on_road[perm]

            # --- vehicle placement, by role's spawn layer, each falling back to
            # open drivable ground when the layer is empty inside the arena
            xy_open, _ = self.world.sample_mask_xy(self.world.drivable, m * self.k, c.arena_half, g)
            xy = xy_open.view(m, self.k, 2).clone()
            heading = torch.rand(m, self.k, device=dev, generator=g) * (2 * torch.pi)
            for li, name in enumerate(layer):
                want = perm == li
                if name == "drivable" or not want.any():
                    continue
                mask = getattr(self.world, name)
                cand, ok = self.world.sample_mask_xy(mask, m * self.k, c.arena_half, g)
                cand = cand.view(m, self.k, 2)
                use = want & ok.view(m, self.k)
                xy = torch.where(use.unsqueeze(2), cand, xy)
                if name == "road":
                    # along the road, either way
                    ry = self.world.yaw_at(self.world.road_yaw, cand[..., 0], cand[..., 1])
                    flip = (torch.rand(m, self.k, device=dev, generator=g) < 0.5).float() * torch.pi
                    heading = torch.where(use, ry + flip, heading)
                elif name == "parking":
                    py = self.world.yaw_at(self.world.park_yaw, cand[..., 0], cand[..., 1])
                    flip = (torch.rand(m, self.k, device=dev, generator=g) < 0.5).float() * torch.pi
                    heading = torch.where(use, py + flip, heading)
            self.veh_heading[rep] = heading
            self.veh_turn_rate[rep] = 0.0
            gz = self.world.ground_at(xy[..., 0], xy[..., 1])
            for i, v in enumerate(self._vehicles):
                root = torch.zeros(m, 13, device=dev)
                root[:, 0] = xy[:, i, 0]
                root[:, 1] = xy[:, i, 1]
                root[:, 2] = gz[:, i] + c.min_clearance + 0.4        # settles under gravity
                yaw = heading[:, i] + self._veh_yaw_offset
                root[:, 3] = torch.cos(yaw / 2)
                root[:, 6] = torch.sin(yaw / 2)
                v.write_root_pose_to_sim(root[:, :7], rep)
                v.write_root_velocity_to_sim(root[:, 7:], rep)

        if len(dormant):
            # parked in a corner outside the arena, never driven, hidden from cameras
            m = len(dormant)
            self.veh_speed[dormant] = 0.0
            self.veh_speed_cmd[dormant] = 0.0
            self.veh_on_road[dormant] = False
            gz = self.world.ground_at(self._dormant_xy[0].expand(m), self._dormant_xy[1].expand(m))
            for i, v in enumerate(self._vehicles):
                root = torch.zeros(m, 13, device=dev)
                root[:, 0] = self._dormant_xy[0] + 6.0 * i
                root[:, 1] = self._dormant_xy[1]
                root[:, 2] = gz + 1.0
                root[:, 3] = 1.0
                v.write_root_pose_to_sim(root[:, :7], dormant)
                v.write_root_velocity_to_sim(root[:, 7:], dormant)

        # every env inherits the role of the set it hunts (its own, for a rep)
        self.role[env_ids] = self.role[self.group[env_ids]]

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
        self.yaw_des[env_ids] = dyaw

        self.task.reset(env_ids, contrast=contrast[self.role[env_ids]])

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
