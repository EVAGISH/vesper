"""Minimal batched rigid-body integrator -- TEST HARNESS ONLY.

Exists so dynamics+control can be verified closed-loop on a CPU with no
Isaac; PhysX remains the simulation truth everywhere else (STACK.md).
"""
import torch

from .multirotor import MultirotorDynamics

G = 9.81


class ReferenceIntegrator:
    def __init__(self, dyn: MultirotorDynamics, num_envs: int, dt=0.005, device="cpu"):
        self.dyn, self.dt = dyn, dt
        self.pos = torch.zeros(num_envs, 3, device=device)
        self.vel = torch.zeros(num_envs, 3, device=device)
        self.quat = torch.zeros(num_envs, 4, device=device)
        self.quat[:, 0] = 1.0
        self.ang_vel = torch.zeros(num_envs, 3, device=device)  # body frame

    def _rot(self):
        from vesper.control.se3 import quat_to_rot
        return quat_to_rot(self.quat)

    def step(self, omega_cmd, wind_world=None):
        p = self.dyn.p
        R = self._rot()
        v_body = (R.transpose(1, 2) @ self.vel.unsqueeze(2)).squeeze(2)
        if wind_world is not None:
            v_body = v_body - (R.transpose(1, 2) @ wind_world.unsqueeze(2)).squeeze(2)
        force_b, torque_b = self.dyn.wrench(omega_cmd, v_body)
        acc = (R @ force_b.unsqueeze(2)).squeeze(2) / p.mass
        acc[:, 2] -= G
        self.vel = self.vel + acc * self.dt
        self.pos = self.pos + self.vel * self.dt
        inertia = self.dyn.inertia
        ang_acc = (torque_b - torch.cross(self.ang_vel, inertia * self.ang_vel, dim=1)) / inertia
        self.ang_vel = self.ang_vel + ang_acc * self.dt
        # quaternion kinematics: qdot = 0.5 * q * [0, w]
        w, x, y, z = self.quat.unbind(1)
        wx, wy, wz = self.ang_vel.unbind(1)
        dq = 0.5 * torch.stack([
            -x * wx - y * wy - z * wz,
            w * wx + y * wz - z * wy,
            w * wy - x * wz + z * wx,
            w * wz + x * wy - y * wx,
        ], dim=1)
        self.quat = self.quat + dq * self.dt
        self.quat = self.quat / self.quat.norm(dim=1, keepdim=True)
