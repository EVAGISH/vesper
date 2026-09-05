"""The forklift driver, integrated kinematically on the synthetic map: no physics."""
import math

import numpy as np
import pytest
import torch

from vesper.lab.vehicles import ROLE_INDEX, ForkliftDriver
from vesper.worlds.heightmap import WorldMap


@pytest.fixture
def world(tmp_path):
    """300 m square, flat; a 40 m block at 40<x<60 and |y|<30; a road along y=0
    west of the block and a north-south road at x=-50."""
    n, cell, half = 151, 2.0, 150.0
    ground = np.zeros((n, n), np.float32)
    obstacle = ground.copy()
    xs = np.linspace(-half, half, n)
    X, Y = np.meshgrid(xs, xs)
    obstacle[(X > 40) & (X < 60) & (np.abs(Y) < 30)] = 12.0
    drivable = ((obstacle - ground) < 0.5).astype(np.uint8)
    drivable[:5] = drivable[-5:] = drivable[:, :5] = drivable[:, -5:] = 0
    road = np.zeros((n, n), np.uint8); road_yaw = np.zeros((n, n), np.float32)
    road[(np.abs(Y) < 4) & (X < 35)] = 1
    road[(np.abs(X + 50) < 4)] = 1; road_yaw[(np.abs(X + 50) < 4)] = math.pi / 2
    parking = np.zeros((n, n), np.uint8); park_yaw = np.zeros((n, n), np.float32)
    parking[(X > 36) & (X <= 40) & (np.abs(Y) < 30)] = 1
    park_yaw[(X > 36) & (X <= 40) & (np.abs(Y) < 30)] = math.pi / 2
    p = tmp_path / "w_map.npz"
    np.savez(p, half_m=np.float32(half), cell=np.float32(cell), ground_z=ground,
             obstacle_z=obstacle, canopy_z=ground, canopy_d=np.zeros_like(ground),
             drivable=drivable, concealed=np.zeros_like(drivable), road=road, road_yaw=road_yaw,
             parking=parking, park_yaw=park_yaw)
    return WorldMap(p)


def drive(world, roles, seconds=60.0, dt=0.04, seed=0):
    g = torch.Generator().manual_seed(seed)
    k = len(roles)
    d = ForkliftDriver(world, k, arena_half=140.0, generator=g)
    d.assign_roles(torch.arange(k), torch.tensor(roles))
    xy, heading = d.place(torch.arange(k))
    yaw = heading.clone()
    speed = torch.zeros(k)
    path = [xy.clone()]
    for _ in range(int(seconds / dt)):
        v, yaw_rate = d.command(xy, yaw, speed, dt)
        xy = xy + v * dt
        yaw = yaw + yaw_rate * dt
        speed = v.norm(dim=1)
        path.append(xy.clone())
    return d, torch.stack(path)                     # [T,K,2]


def test_roles_spawn_on_their_layers(world):
    d, path = drive(world, [ROLE_INDEX["cruise"], ROLE_INDEX["parked"], ROLE_INDEX["cruise"]], seconds=0.04)
    p0 = path[0]
    r, c = world.nearest_cell(p0[:, 0], p0[:, 1])
    assert world.road[r[0], c[0]] == 1 and world.road[r[2], c[2]] == 1
    assert world.parking[r[1], c[1]] == 1
    # parked heading runs along the wall (north-south)
    assert abs(math.sin(float(d.heading[1]))) > 0.95


def test_cruisers_move_stay_drivable_and_prefer_roads(world):
    d, path = drive(world, [ROLE_INDEX["cruise"]] * 4, seconds=90.0)
    travelled = (path[1:] - path[:-1]).norm(dim=2).sum(dim=0)
    assert (travelled > 100.0).all(), travelled.tolist()          # ~1.5 m/s average at least
    flat = path.reshape(-1, 2)
    r, c = world.nearest_cell(flat[:, 0], flat[:, 1])
    assert float(world.drivable[r, c].mean()) > 0.97, "drove into the block or off the edge"
    assert float(world.road[r, c].mean()) > 0.5, "road followers should spend most of the time on roads"
    # speed ramps: nothing moves 4.5 m/s in the first half second
    v0 = (path[12] - path[0]).norm(dim=1) / (12 * 0.04)
    assert (v0 < 1.5).all()


def test_parked_stays_put_and_turn_rate_is_bounded(world):
    d, path = drive(world, [ROLE_INDEX["parked"], ROLE_INDEX["cruise"]], seconds=30.0)
    assert (path[-1, 0] - path[0, 0]).norm() < 0.01
    # heading change per step for the cruiser never exceeds the lateral budget
    steps = path[1:, 1] - path[:-1, 1]
    h = torch.atan2(steps[:, 1], steps[:, 0])
    dh = torch.atan2(torch.sin(h[1:] - h[:-1]), torch.cos(h[1:] - h[:-1])).abs()
    fast = steps[1:].norm(dim=1) / 0.04 > 3.0
    if fast.any():
        assert float(dh[fast].max()) / 0.04 < 1.5      # rad/s, generous over the 0.83 budget at 3 m/s
