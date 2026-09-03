"""Batched SE(3) (geometric) position controller + rotor allocation.

The 'conventional, never trained' inner loop for the throughput lane: PD
position -> desired acceleration -> desired attitude -> PD attitude ->
body wrench -> per-rotor omega via the mixer inverse. Torch, [N, ...] batched.
Porting PX4's exact cascade remains an open refinement (STACK.md section 3).
"""
import torch

from vesper.dynamics.multirotor import MultirotorParams

G = 9.81


def quat_to_rot(q):  # q [N,4] wxyz -> R [N,3,3]
    w, x, y, z = q.unbind(dim=1)
    return torch.stack([
        1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y),
        2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x),
        2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y),
    ], dim=1).reshape(-1, 3, 3)


class SE3Controller:
    def __init__(self, params: MultirotorParams, num_envs: int, device="cpu",
                 kp=4.0, kv=4.5, kr=25.0, kw=9.0, accel_limit=6.0):
        self.p = params
        self.device = device
        self.kp, self.kv, self.kr, self.kw = kp, kv, kr, kw
        self.accel_limit = accel_limit   # horizontal accel budget (m/s^2)
        self.inertia = torch.tensor(params.inertia, device=device)
        # mixer: [T, tau_x, tau_y, tau_z] = A @ omega^2
        rx = torch.tensor(params.rotor_xy, device=device)
        dirs = torch.tensor(params.rot_dirs, device=device)
        k, m = params.k_thrust, params.k_moment
        A = torch.stack([
            torch.full((4,), k, device=device),
            k * rx[:, 1], -k * rx[:, 0], m * dirs,
        ])
        self.mix_inv = torch.linalg.inv(A)                              # [4,4]

    def compute(self, pos, vel, quat, ang_vel_body, target_pos):
        """All [N,3] / quat [N,4] wxyz -> rotor omegas [N,4]."""
        N = pos.shape[0]
        p = self.p
        a_des = self.kp * (target_pos - pos) - self.kv * vel
        a_des = a_des.clamp(-self.accel_limit, self.accel_limit)
        f_world = p.mass * (a_des + torch.tensor([0.0, 0.0, G], device=pos.device))
        R = quat_to_rot(quat)
        b3 = R[:, :, 2]
        thrust = (f_world * b3).sum(dim=1).clamp(min=0.1)

        # desired frame: z along f_world, yaw -> world +x
        z_d = f_world / f_world.norm(dim=1, keepdim=True).clamp(min=1e-6)
        x_c = torch.tensor([1.0, 0.0, 0.0], device=pos.device).expand(N, 3)
        y_d = torch.cross(z_d, x_c, dim=1)
        y_d = y_d / y_d.norm(dim=1, keepdim=True).clamp(min=1e-6)
        x_d = torch.cross(y_d, z_d, dim=1)
        R_d = torch.stack([x_d, y_d, z_d], dim=2)

        eR_mat = R_d.transpose(1, 2) @ R - R.transpose(1, 2) @ R_d
        e_R = 0.5 * torch.stack([eR_mat[:, 2, 1], eR_mat[:, 0, 2], eR_mat[:, 1, 0]], dim=1)
        tau = self.inertia * (-self.kr * e_R - self.kw * ang_vel_body)

        wrench = torch.cat([thrust.unsqueeze(1), tau], dim=1)           # [N,4]
        omega_sq = (self.mix_inv @ wrench.T).T.clamp(min=0.0)
        return omega_sq.sqrt().clamp(0.0, p.omega_max)
