"""StrikeEnv: VesperQuad wired for the crash-into-a-vehicle task.

Action is guidance, not raw rotors: the policy emits a 3-vector [-1,1] that
vesper.lab.strike_task turns into a look-ahead setpoint, and the conventional
SE3 inner loop flies to it. So the policy learns *where to aim and when to
dive* (lead pursuit of a moving armor target) on top of a trusted stabilizer,
which converges in minutes rather than the tens of GPU-hours raw-rotor PPO
needs just to hover. A raw-rotor variant is a drop-in (see STACK/STEPS).

The target is a ground vehicle (real USD via VESPER_TARGET_USD, else a built
olive-drab tank proxy). Reward/termination math lives in strike_task (CPU-tested).
"""
import os

import torch

import isaaclab.sim as sim_utils
from isaaclab.utils import configclass

from vesper.control.se3 import SE3Controller
from vesper.lab.vesper_quad import VesperQuadEnv, VesperQuadEnvCfg
from vesper.lab import strike_task as T


@configclass
class StrikeEnvCfg(VesperQuadEnvCfg):
    episode_length_s = 20.0
    action_space = 3
    observation_space = 17
    spawn_targets: bool = False       # author tank visuals (eval only; training is headless)
    strike: dict | None = None        # overrides for StrikeCfg fields


class StrikeEnv(VesperQuadEnv):
    cfg: StrikeEnvCfg

    def __init__(self, cfg: StrikeEnvCfg, render_mode=None, seed=0, **kwargs):
        # These must exist before super().__init__ -> _setup_scene runs.
        self.tcfg = T.StrikeCfg(**(cfg.strike or {}))
        self._seed = int(seed)
        self._target_xf = []
        super().__init__(cfg, render_mode, **kwargs)
        # _setup_scene has already sampled target_pos/target_vel and (if asked) built visuals.
        self.gen = torch.Generator(device=self.device)
        self.gen.manual_seed(self._seed)
        self.ctrl = SE3Controller(self.params, self.num_envs, device=self.device)
        self.spawn_offsets = torch.zeros(self.num_envs, 3, device=self.device)
        self.prev_dist = torch.full((self.num_envs,), self.tcfg.target_max_r, device=self.device)
        self._setpoint = torch.zeros(self.num_envs, 3, device=self.device)
        self._reward = torch.zeros(self.num_envs, device=self.device)
        self._evaluated = False
        self._dt = self.cfg.sim.dt * self.cfg.decimation
        self._term = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._hit = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

    # ---- PPO-facing adapter (num_obs/num_actions/reset/step/done) ----
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

    # ---- scene ----
    def _setup_scene(self):
        super()._setup_scene()
        # scene (and env_origins) now exist; seed the target state before any step.
        n = self.scene.num_envs
        g = torch.Generator(device=self.device).manual_seed(self._seed)
        self.target_pos, self.target_vel = T.sample_targets(n, self.tcfg, self.device, g)
        if self.cfg.spawn_targets:
            self._build_targets()

    def _build_targets(self):
        import omni.usd
        from pxr import UsdGeom, Gf
        stage = omni.usd.get_context().get_stage()
        usd = os.environ.get("VESPER_TARGET_USD")
        origins = self.scene.env_origins.cpu().numpy()
        for i in range(self.scene.num_envs):
            path = f"/World/targets/t{i:04d}"
            xf = UsdGeom.Xform.Define(stage, path)
            op = xf.AddTranslateOp()
            p = origins[i] + self.target_pos[i].cpu().numpy()
            op.Set(Gf.Vec3d(float(p[0]), float(p[1]), float(p[2])))
            self._target_xf.append(op)
            if usd:
                sim_utils.UsdFileCfg(usd_path=usd).func(f"{path}/veh", sim_utils.UsdFileCfg(usd_path=usd))
            else:
                self._tank_proxy(path)

    def _tank_proxy(self, path):
        drab = (0.29, 0.33, 0.19)
        dark = (0.20, 0.23, 0.14)
        hull = sim_utils.CuboidCfg(size=(6.5, 3.2, 1.1),
                                   visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=drab, roughness=0.9))
        hull.func(f"{path}/hull", hull, translation=(0.0, 0.0, 0.0))
        turret = sim_utils.CuboidCfg(size=(3.0, 2.4, 0.9),
                                     visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=dark, roughness=0.9))
        turret.func(f"{path}/turret", turret, translation=(0.3, 0.0, 0.9))
        barrel = sim_utils.CylinderCfg(radius=0.12, height=4.2, axis="X",
                                       visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=dark))
        barrel.func(f"{path}/barrel", barrel, translation=(3.4, 0.0, 1.0))
        for sy in (-1.6, 1.6):
            trk = sim_utils.CuboidCfg(size=(6.7, 0.7, 0.6),
                                      visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.09, 0.09, 0.09)))
            trk.func(f"{path}/trk{'p' if sy>0 else 'm'}", trk, translation=(0.0, sy, -0.45))

    def update_target_visuals(self):
        if not self._target_xf:
            return
        from pxr import Gf
        origins = self.scene.env_origins
        p = (origins + self.target_pos).cpu().numpy()
        for i, op in enumerate(self._target_xf):
            op.Set(Gf.Vec3d(float(p[i, 0]), float(p[i, 1]), float(p[i, 2])))

    # ---- control loop ----
    def _pre_physics_step(self, actions):
        self._action = actions
        self.target_pos, self.target_vel = T.step_targets(self.target_pos, self.target_vel,
                                                           self.tcfg, self._dt, self.gen)
        pos_w = self._robot.data.root_pos_w
        self._setpoint = T.setpoint(pos_w, actions, self.tcfg)
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

    # ---- task ----
    def _evaluate(self):
        if self._evaluated:
            return
        pos, vel, quat, _ = self.flight_state()
        self._reward, self._term, info = T.evaluate(pos, vel, quat, self.target_pos, self.prev_dist, self.tcfg)
        self._hit = info["hit"]
        self.prev_dist = info["dist"]
        self.extras["hit"] = self._hit
        self.extras["dist"] = info["dist"]
        self._evaluated = True

    def _get_observations(self):
        pos, vel, quat, avb = self.flight_state()
        obs = T.observations(pos, vel, quat, avb, self.target_pos, self.target_vel)
        return {"policy": obs}

    def _get_rewards(self):
        self._evaluate()
        return self._reward

    def _get_dones(self):
        self._evaluate()
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        return self._term, time_out

    def _reset_idx(self, env_ids):
        k = len(env_ids)
        jit = (torch.rand(k, 2, device=self.device, generator=self.gen) - 0.5) * 2 * self.tcfg.spawn_jitter_xy
        self.spawn_offsets[env_ids, :2] = jit
        self.spawn_offsets[env_ids, 2] = self.tcfg.spawn_alt
        super()._reset_idx(env_ids)
        tp, tv = T.sample_targets(k, self.tcfg, self.device, self.gen)
        self.target_pos[env_ids] = tp
        self.target_vel[env_ids] = tv
        dpos = self._robot.data.root_pos_w[env_ids] - self.scene.env_origins[env_ids]
        self.prev_dist[env_ids] = (tp - dpos).norm(dim=1)
