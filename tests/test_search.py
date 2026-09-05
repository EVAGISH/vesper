"""CPU tests for the search task: map sampling, occlusion, belief, reward.

No Isaac, no GPU, no assets -- the world is a synthetic raster built in-process,
so every mechanism the policy depends on is checked before a GPU is booted.
"""
import math

import numpy as np
import pytest
import torch

from vesper.lab.search_task import (PROPRIO_DIM, SearchCfg, SearchTask, camera_axis, seg_counts,
                                    seg_lookup, sensor_pose, setpoint, tilt_from_quat)
from vesper.worlds.heightmap import WorldMap


@pytest.fixture
def world(tmp_path):
    """200 m square: flat at z=0, a 40 m tower of building at x>40, a wood at y<-40."""
    n, cell, half = 101, 2.0, 100.0
    ground = np.zeros((n, n), np.float32)
    obstacle = ground.copy()
    canopy = ground.copy()
    dens = np.zeros((n, n), np.float32)
    xs = np.linspace(-half, half, n)
    X, Y = np.meshgrid(xs, xs)                       # rows +y, cols +x
    obstacle[(X > 40) & (X < 60)] = 40.0
    wood = Y < -40
    canopy[wood] = 15.0
    dens[wood] = 0.8
    drivable = ((obstacle - ground) < 0.5).astype(np.uint8)
    concealed = (drivable & wood).astype(np.uint8)
    p = tmp_path / "w_map.npz"
    np.savez(p, half_m=np.float32(half), cell=np.float32(cell), ground_z=ground,
             obstacle_z=obstacle, canopy_z=canopy, canopy_d=dens,
             drivable=drivable, concealed=concealed)
    return WorldMap(p)


def test_sampling_and_masks(world):
    assert world.n == 101 and world.half_m == 100.0 and world.cell == 2.0
    z = world.ground_at(torch.tensor([0.0, 30.0]), torch.tensor([0.0, 10.0]))
    assert torch.allclose(z, torch.zeros(2))
    solid = world.solid_at(torch.tensor([50.0]), torch.tensor([0.0]))
    assert solid.item() == pytest.approx(40.0)
    assert world.canopy_at(torch.tensor([0.0]), torch.tensor([-60.0])).item() == pytest.approx(15.0)


def test_line_of_sight_blocked_by_building(world):
    # across the building at 10 m: blocked. over the top at 60 m: clear.
    lo0 = torch.tensor([[0.0, 0.0, 10.0]]); lo1 = torch.tensor([[90.0, 0.0, 10.0]])
    hi0 = torch.tensor([[0.0, 0.0, 60.0]]); hi1 = torch.tensor([[90.0, 0.0, 60.0]])
    assert not world.trace(lo0, lo1)[0].item()
    assert world.trace(hi0, hi1)[0].item()


def test_foliage_accumulates_only_in_the_wood(world):
    over = torch.tensor([[0.0, -60.0, 80.0]])
    under = torch.tensor([[0.0, -60.0, 1.0]])
    clear, foliage = world.trace(over, under)
    assert clear.item()                       # leaves do not block, they attenuate
    assert 8.0 < foliage.item() < 15.0        # ~15 m of canopy at density 0.8
    open_over = torch.tensor([[0.0, 60.0, 80.0]])
    open_under = torch.tensor([[0.0, 60.0, 1.0]])
    assert world.trace(open_over, open_under)[1].item() == pytest.approx(0.0)


def test_mask_sampling_lands_in_the_mask(world):
    g = torch.Generator().manual_seed(0)
    xy, ok = world.sample_mask_xy(world.concealed, 64, 90.0, g)
    assert ok.float().mean() > 0.5
    assert (xy[ok][:, 1] < -40.0).all()        # concealed cells are all in the wood


def _task(world, k=2, **kw):
    cfg = SearchCfg(n_targets=k, arena_half=90.0, grid=4, miss_p=0.0, fix_noise_m=0.0, **kw)
    return SearchTask(world, cfg, num_envs=2, dt=0.04, max_steps=100), cfg


def _state(n=2):
    quat = torch.zeros(n, 4); quat[:, 0] = 1.0
    return quat, torch.zeros(n, 3), torch.zeros(n, 3)


def test_detection_needs_the_target_in_the_forward_cone(world):
    task, cfg = _task(world)
    quat, _, _ = _state()                                # level, nose to +x
    drone = torch.tensor([[0.0, 0.0, 60.0], [0.0, 0.0, 60.0]])
    # target 0 ahead and below (in the forward-pitched cone), target 1 behind
    tgt = torch.tensor([[[40.0, 0.0, 1.0], [-80.0, 0.0, 1.0]]] * 2)
    vis, slant = task.detect(drone, quat, tgt)
    assert vis[:, 0].all() and not vis[:, 1].any()
    # turn the airframe round and the behind target is the one in view
    about = torch.tensor([[0.0, 0.0, 0.0, 1.0]] * 2)      # yaw 180 deg
    vis2, _ = task.detect(drone, about, tgt)
    assert vis2[:, 1].all() and not vis2[:, 0].any()


def test_camera_axis_is_pitched_forward_and_down():
    level = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
    a = camera_axis(level, math.radians(40.0))[0]
    assert a[0] > 0 and a[2] < 0 and abs(a[1]) < 1e-6
    assert a[2].item() == pytest.approx(-math.sin(math.radians(40.0)), abs=1e-5)
    # nadir (5 deg past the edge of a 55 deg half-angle at 40 deg pitch) is in view
    nadir = torch.tensor([0.0, 0.0, -1.0])
    assert (a @ nadir).item() > math.cos(math.radians(55.0))


def test_pixel_sightings_count_only_the_envs_own_vehicles():
    # env 0's targets are env 0's vehicles; env 1 shares them (group 0), env 2 is its own
    labels = {"0": "BACKGROUND", "1": "UNLABELLED", "5": "/World/envs/env_0/Vehicle_1",
              "7": "/World/envs/env_2/Vehicle_0", "9": "/World/trees/t00001",
              "11": "/World/envs/env_0/Vehicle_0"}
    table = seg_lookup(labels, r"env_(\d+)/Vehicle_(\d+)", n_slots=2)
    assert table[5].item() == 1 and table[11].item() == 0 and table[7].item() == 4
    assert table[9].item() == -1 and table[0].item() == -1
    seg = torch.zeros(3, 4, 4, dtype=torch.int32)
    seg[0, :2, :2] = 5            # env 0 sees 4 px of its vehicle 1
    seg[0, 3, 3] = 7              # ...and 1 px of env 2's vehicle: scenery
    seg[1, 0, :3] = 11            # env 1 (group 0) sees 3 px of vehicle 0
    seg[2, 1, 1] = 7              # env 2 sees 1 px of its own vehicle 0
    group = torch.tensor([0, 0, 2])
    c = seg_counts(seg, table, group, 2)
    assert c.tolist() == [[0, 4], [3, 0], [1, 0]]


def test_pixel_mode_decides_sightings(world):
    task, cfg = _task(world, sight_px=4)
    quat, vel, avb = _state()
    drone = torch.tensor([[0.0, 0.0, 60.0]] * 2)
    tgt = torch.tensor([[[-80.0, 0.0, 1.0], [-80.0, 30.0, 1.0]]] * 2)   # both behind the lens
    step = torch.zeros(2, dtype=torch.long)
    px = torch.tensor([[10, 3]] * 2)
    task.step(drone, vel, quat, avb, tgt, step, seen_px=px)
    assert task.known[:, 0].all() and not task.known[:, 1].any()


def test_proprio_is_body_frame_and_carries_no_heading(world):
    task, cfg = _task(world)
    n = 2
    quat = torch.tensor([[1.0, 0.0, 0.0, 0.0], [math.cos(math.pi / 4), 0.0, 0.0, math.sin(math.pi / 4)]])
    vel = torch.tensor([[15.0, 0.0, 0.0], [0.0, 15.0, 0.0]])   # each flying along its own nose
    avb = torch.zeros(n, 3)
    agl = torch.full((n,), 50.0)
    o = task.proprio(vel, quat, avb, agl, torch.zeros(n))
    assert o.shape == (n, PROPRIO_DIM) and o.shape[1] == task.proprio_dim
    # the two drones differ only by heading, which the vector must not encode
    assert torch.allclose(o[0], o[1], atol=1e-5)
    assert o[0, 0].item() == pytest.approx(1.0, abs=1e-5)      # forward speed
    assert o[0, 5].item() == pytest.approx(1.0, abs=1e-5)      # gravity along body z when level


def test_footprint_lies_ahead_of_the_drone(world):
    task, cfg = _task(world)
    quat, _, _ = _state()
    centre, radius = task.footprint(torch.zeros(2, 2), torch.full((2,), 50.0), quat)
    assert centre[0, 0] > 20.0 and abs(centre[0, 1]) < 1e-4
    assert radius[0] > centre[0, 0], "nadir stays inside the swept disc"


def test_camouflage_shortens_the_range(world):
    task, cfg = _task(world)
    task.contrast[:] = torch.tensor([1.0, 0.2])
    # both directly below at 100 m: plain is inside 220 m, camo's range is 44 m
    drone = torch.tensor([[0.0, 0.0, 100.0]] * 2)
    tgt = torch.tensor([[[0.0, 0.0, 1.0], [0.0, 0.0, 1.0]]] * 2)
    vis, _ = task.detect(drone, _state()[0], tgt)
    assert vis[:, 0].all() and not vis[:, 1].any()


def test_canopy_shortens_the_range(world):
    task, cfg = _task(world)
    # same geometry, one over open ground and one over the wood
    drone = torch.tensor([[0.0, 60.0, 120.0], [0.0, -60.0, 120.0]])
    tgt = torch.tensor([[[0.0, 60.0, 1.0]] * 2, [[0.0, -60.0, 1.0]] * 2])
    vis, _ = task.detect(drone, _state()[0], tgt)
    assert vis[0, 0].item(), "open target at 119 m should be seen"
    assert not vis[1, 0].item(), "the same target under canopy should not be"


def test_belief_latches_and_goes_stale(world):
    task, cfg = _task(world)
    quat, vel, avb = _state()
    drone = torch.tensor([[0.0, 0.0, 60.0]] * 2)
    tgt = torch.tensor([[[0.0, 0.0, 1.0], [0.0, 30.0, 1.0]]] * 2)
    step = torch.zeros(2, dtype=torch.long)
    task.step(drone, vel, quat, avb, tgt, step)
    assert task.known[:, 0].all()
    fix0 = task.fix[:, 0].clone()
    # fly away: the fix is remembered, its age grows
    away = torch.tensor([[0.0, 0.0, 300.0]] * 2)
    for i in range(5):
        task.step(away, vel, quat, avb, tgt, step + i)
    assert task.known[:, 0].all()
    assert torch.allclose(task.fix[:, 0], fix0)
    assert task.fix_age[:, 0].min() > 0.15


def test_reward_ordering_fast_beats_slow_beats_crash(world):
    quat, vel, avb = _state(2)
    tgt = torch.tensor([[[0.0, 0.0, 1.0], [10.0, 0.0, 1.0]]] * 2)

    def run(step_at_reach, crash=False):
        task, cfg = _task(world)
        # the crash case flies into the ground well away from either vehicle:
        # touching one at ground level is a reach, not a crash, by design
        drone = (torch.tensor([[0.0, 0.0, 3.0]] * 2) if not crash
                 else torch.tensor([[-70.0, 70.0, 0.5]] * 2))
        s = torch.full((2,), step_at_reach, dtype=torch.long)
        _, r, term, info = task.step(drone, vel, quat, avb, tgt, s)
        return float(r[0]), bool(term[0])

    fast, _ = run(5)
    slow, _ = run(90)
    crash_r, crash_t = run(5, crash=True)
    assert fast > slow > 0
    assert crash_r < 0 and crash_t


def test_coverage_pays_once_and_grows_with_altitude(world):
    task, cfg = _task(world)
    quat, vel, avb = _state()
    tgt = torch.full((2, 2, 3), 500.0)          # far away, nothing to detect
    step = torch.zeros(2, dtype=torch.long)
    low = torch.tensor([[0.0, 0.0, 12.0]] * 2)
    _, r1, _, _ = task.step(low, vel, quat, avb, tgt, step)
    _, r2, _, _ = task.step(low, vel, quat, avb, tgt, step)
    assert r1[0] > r2[0], "a cell already swept must not pay again"
    covered_low = task.visited.float().mean().item()

    task2, _ = _task(world)
    high = torch.tensor([[0.0, 0.0, 150.0]] * 2)
    task2.step(high, vel, quat, avb, tgt, step)
    assert task2.visited.float().mean().item() > covered_low


def test_observation_never_leaks_an_unseen_target(world):
    task, cfg = _task(world)
    quat, vel, avb = _state()
    drone = torch.tensor([[0.0, 0.0, 60.0]] * 2)
    # one target visible below, one hidden behind the lens
    tgt = torch.tensor([[[0.0, 0.0, 1.0], [-85.0, 0.0, 1.0]]] * 2)
    step = torch.zeros(2, dtype=torch.long)
    task.step(drone, vel, quat, avb, tgt, step)
    agl = torch.full((2,), 60.0)
    obs = task.privileged(drone, vel, quat, avb, agl, torch.zeros(2))
    assert obs.shape == (2, task.obs_dim)
    assert torch.isfinite(obs).all()
    # slot 1 was never seen: every one of its five belief fields must be zero
    base = 12 + 8
    assert obs[:, base:base + 8][:, [0, 3, 4, 5, 6, 7]].abs().max() == 0.0
    assert (obs[:, 12] == 1.0).all()            # slot 0 known


def test_setpoint_is_bounded_and_body_framed():
    cfg = SearchCfg(look_ahead=25.0)
    pos = torch.zeros(4, 3)
    a = torch.randn(4, 3) * 10
    sp = setpoint(pos, a, torch.zeros(4), cfg)
    assert sp.abs().max() <= 25.0 + 1e-5
    # "forward" with the nose pointing +y is +y in the world
    fwd = torch.tensor([[10.0, 0.0, 0.0]])
    sp = setpoint(torch.zeros(1, 3), fwd, torch.tensor([math.pi / 2]), cfg)
    assert sp[0, 1].item() == pytest.approx(25.0, abs=1e-3) and abs(sp[0, 0].item()) < 1e-3
    # "left" with the nose pointing +x is +y
    left = torch.tensor([[0.0, 10.0, 0.0]])
    sp = setpoint(torch.zeros(1, 3), left, torch.zeros(1), cfg)
    assert sp[0, 1].item() == pytest.approx(25.0, abs=1e-3)


def test_tilt_from_quat_upright_and_over():
    upright = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
    assert tilt_from_quat(upright).item() == pytest.approx(0.0, abs=1e-6)
    rolled = torch.tensor([[math.cos(math.pi / 4), math.sin(math.pi / 4), 0.0, 0.0]])
    assert tilt_from_quat(rolled).item() == pytest.approx(math.pi / 2, abs=1e-5)


def test_set_arena_rebuilds_the_coverage_grid(world):
    task, cfg = _task(world)
    small = task.cell_xy.abs().max().item()
    task.set_arena(45.0)
    assert task.cfg.arena_half == 45.0
    assert task.cell_xy.abs().max().item() < small
    assert task.cell_xy.shape == (cfg.grid * cfg.grid, 2)
    # and the task still steps cleanly at the new size
    quat, vel, avb = _state()
    tgt = torch.zeros(2, 2, 3)
    obs, r, term, info = task.step(torch.tensor([[0.0, 0.0, 40.0]] * 2), vel, quat, avb, tgt,
                                   torch.zeros(2, dtype=torch.long))
    assert torch.isfinite(obs).all()


def test_out_of_bounds_is_square_like_the_arena(world):
    """A corner of the search box is inside it; a radial bound would kill it."""
    task, cfg = _task(world)                       # arena_half 90
    quat, vel, avb = _state()
    tgt = torch.full((2, 2, 3), 500.0)
    step = torch.zeros(2, dtype=torch.long)
    corner = torch.tensor([[89.0, 89.0, 60.0]] * 2)      # radius 126 m, inside the box
    _, _, term, info = task.step(corner, vel, quat, avb, tgt, step)
    assert not info["oob"].any(), "the box's own corner must not be out of bounds"
    outside = torch.tensor([[cfg.arena_half + cfg.oob_margin + 5.0, 0.0, 60.0]] * 2)
    _, _, _, info2 = task.step(outside, vel, quat, avb, tgt, step)
    assert info2["oob"].all()


def test_sensor_pose_looks_where_the_geometric_cone_does():
    from vesper.control.se3 import quat_to_rot
    n = 3
    ang = torch.tensor([0.0, math.pi / 2, -2.0])
    quat = torch.stack([torch.cos(ang / 2), torch.zeros(n), torch.zeros(n), torch.sin(ang / 2)], dim=1)
    pos = torch.randn(n, 3)
    cam_pos, cam_q = sensor_pose(pos, quat, 40.0, offset=(0.12, 0.0, -0.04))
    # the camera's +x is the cone axis the task uses for detection
    fwd = quat_to_rot(cam_q)[:, :, 0]
    assert torch.allclose(fwd, camera_axis(quat, math.radians(40.0)), atol=1e-5)
    # and it sits just ahead of and below the airframe origin, in the body frame
    assert torch.allclose(cam_pos - pos, quat_to_rot(quat) @ torch.tensor([0.12, 0.0, -0.04]), atol=1e-6)
