"""Batched torch port of Pegasus Simulator's multirotor model (v5.1.0).

Faithful to pegasus.simulator.logic.{thrusters.quadratic_thrust_curve,
dynamics.linear_drag, vehicles.multirotor}, Iris defaults:
  T_i = k_t * omega_i^2                 (omega clamped to [0, omega_max])
  tau_yaw = sum(k_m * omega_i^2 * dir)  (dirs [-1,-1,1,1])
  F_drag_body = -diag(cd) @ v_body      (cd = [0.5, 0.3, 0.0])
Roll/pitch torques arise from rotor positions x thrusts (Pegasus lets PhysX
do this by applying forces at rotor prims; here it is explicit cross products).
All tensors are batched [N, ...]; runs on CPU or GPU.
"""
from dataclasses import dataclass, field

import torch


@dataclass
class MultirotorParams:
    mass: float = 1.5                       # kg (3DR Iris)
    inertia: tuple = (0.029125, 0.029125, 0.055225)
    k_thrust: float = 8.54858e-6
    k_moment: float = 1e-6
    omega_max: float = 1100.0
    rot_dirs: tuple = (-1.0, -1.0, 1.0, 1.0)
    # rotor (x, y) body positions, m -- PX4 iris layout, order matches rot_dirs
    rotor_xy: tuple = ((0.13, -0.22), (-0.13, 0.20), (0.13, 0.22), (-0.13, -0.20))
    drag_coeffs: tuple = (0.50, 0.30, 0.0)

    @property
    def hover_omega(self) -> float:
        return (self.mass * 9.81 / (4 * self.k_thrust)) ** 0.5


class MultirotorDynamics:
    """omega commands + body velocity -> body-frame force and torque."""

    def __init__(self, params: MultirotorParams, num_envs: int, device="cpu"):
        self.p = params
        self.device = device
        p = params
        self.rotor_xy = torch.tensor(p.rotor_xy, device=device)          # [4,2]
        self.rot_dirs = torch.tensor(p.rot_dirs, device=device)          # [4]
        self.cd = torch.tensor(p.drag_coeffs, device=device)             # [3]
        self.inertia = torch.tensor(p.inertia, device=device)            # [3]

    def wrench(self, omega: torch.Tensor, v_body: torch.Tensor):
        """omega [N,4] rad/s, v_body [N,3] -> force [N,3], torque [N,3] (body frame)."""
        p = self.p
        w = omega.clamp(0.0, p.omega_max)
        thrust = p.k_thrust * w * w                                       # [N,4]
        f_z = thrust.sum(dim=1)                                           # [N]
        drag = -self.cd * v_body                                          # [N,3]
        force = drag.clone()
        force[:, 2] += f_z
        # roll: +y rotor pos lifts -> negative roll torque; tau = r x F, F=+z
        tau_x = (self.rotor_xy[:, 1] * thrust).sum(dim=1)
        tau_y = (-self.rotor_xy[:, 0] * thrust).sum(dim=1)
        tau_z = (p.k_moment * w * w * self.rot_dirs).sum(dim=1)
        torque = torch.stack([tau_x, tau_y, tau_z], dim=1)
        return force, torque
