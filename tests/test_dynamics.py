import torch

from vesper.dynamics import MultirotorDynamics, MultirotorParams


def test_hover_balance():
    p = MultirotorParams()
    dyn = MultirotorDynamics(p, num_envs=2)
    omega = torch.full((2, 4), p.hover_omega)
    force, torque = dyn.wrench(omega, torch.zeros(2, 3))
    assert abs(force[0, 2].item() - p.mass * 9.81) < 1e-3
    assert torch.allclose(torque, torch.zeros(2, 3), atol=1e-4)  # symmetric layout


def test_drag_opposes_velocity():
    dyn = MultirotorDynamics(MultirotorParams(), num_envs=1)
    force, _ = dyn.wrench(torch.zeros(1, 4), torch.tensor([[2.0, -1.0, 0.5]]))
    assert force[0, 0] < 0 and force[0, 1] > 0
    assert force[0, 2] == 0.0  # cd_z = 0, no thrust


def test_yaw_torque_sign():
    p = MultirotorParams()
    dyn = MultirotorDynamics(p, num_envs=1)
    omega = torch.tensor([[800.0, 800.0, 0.0, 0.0]])  # only dir=-1 rotors
    _, torque = dyn.wrench(omega, torch.zeros(1, 3))
    assert torque[0, 2] < 0
