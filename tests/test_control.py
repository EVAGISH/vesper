"""Closed loop: SE(3) controller + torch dynamics flies to a waypoint on CPU."""
import torch

from vesper.control import SE3Controller
from vesper.dynamics import GustField, MultirotorDynamics, MultirotorParams
from vesper.dynamics.reference_integrator import ReferenceIntegrator


def fly(target, seconds=12.0, wind=None, n=3):
    p = MultirotorParams()
    dyn = MultirotorDynamics(p, num_envs=n)
    sim = ReferenceIntegrator(dyn, num_envs=n, dt=0.005)
    ctrl = SE3Controller(p, num_envs=n)
    tgt = torch.tensor(target).expand(n, 3)
    for _ in range(int(seconds / 0.005)):
        omega = ctrl.compute(sim.pos, sim.vel, sim.quat, sim.ang_vel, tgt)
        w = wind.step() if wind else None
        sim.step(omega, wind_world=w)
    return sim


def test_reaches_waypoint():
    sim = fly([3.0, 2.0, 3.0])
    err = (sim.pos - torch.tensor([3.0, 2.0, 3.0])).norm(dim=1)
    assert err.max().item() < 0.3, f"final error {err.max().item():.2f}m"
    assert sim.vel.norm(dim=1).max().item() < 0.3


def test_flies_in_wind():
    gen = torch.Generator().manual_seed(7)
    wind = GustField(3, [2.0, 0.0, 0.0], gust_std=1.0, dt=0.005, generator=gen)
    sim = fly([3.0, 2.0, 3.0], wind=wind)
    err = (sim.pos - torch.tensor([3.0, 2.0, 3.0])).norm(dim=1)
    assert err.max().item() < 0.8, f"final error in wind {err.max().item():.2f}m"


def test_determinism():
    a, b = fly([1.0, 1.0, 2.0], seconds=3.0), fly([1.0, 1.0, 2.0], seconds=3.0)
    assert torch.equal(a.pos, b.pos)


def test_holds_a_commanded_heading():
    """With yaw_des the airframe turns to face it while holding position."""
    import math
    from vesper.control.se3 import quat_to_rot
    p = MultirotorParams()
    n = 2
    dyn = MultirotorDynamics(p, num_envs=n)
    sim = ReferenceIntegrator(dyn, num_envs=n, dt=0.005)
    ctrl = SE3Controller(p, num_envs=n)
    tgt = torch.tensor([[0.0, 0.0, 3.0]]).expand(n, 3)
    yaw = torch.tensor([math.pi / 2, -math.pi / 4])
    for _ in range(int(8.0 / 0.005)):
        omega = ctrl.compute(sim.pos, sim.vel, sim.quat, sim.ang_vel, tgt, yaw_des=yaw)
        sim.step(omega)
    R = quat_to_rot(sim.quat)
    nose = R[:, :, 0]                                   # body +x in world
    got = torch.atan2(nose[:, 1], nose[:, 0])
    err = torch.atan2(torch.sin(got - yaw), torch.cos(got - yaw)).abs()
    assert err.max().item() < 0.05, f"heading error {err.tolist()}"
    assert (sim.pos - tgt).norm(dim=1).max().item() < 0.3
