"""CPU tests for the chase task, zones and the depth sensor model."""
import math

import numpy as np
import pytest
import torch

from vesper.lab.chase_task import ChaseCfg, ChaseTask
from vesper.lab.frames import PROPRIO_DIM
from vesper.sensors.depth import DepthModel
from vesper.worlds.heightmap import WorldMap
from vesper.worlds.zones import Zones, rasterize


@pytest.fixture
def world(tmp_path):
    """200 m square: flat at z=0, a 40 m block of building at 40<x<60, a wood at y<-40."""
    n, cell, half = 101, 2.0, 100.0
    ground = np.zeros((n, n), np.float32)
    obstacle = ground.copy()
    canopy = ground.copy()
    dens = np.zeros((n, n), np.float32)
    xs = np.linspace(-half, half, n)
    X, Y = np.meshgrid(xs, xs)
    obstacle[(X > 40) & (X < 60)] = 40.0
    wood = Y < -40
    canopy[wood] = 15.0; dens[wood] = 0.8
    drivable = ((obstacle - ground) < 0.5).astype(np.uint8)
    concealed = (drivable & wood).astype(np.uint8)
    p = tmp_path / "w_map.npz"
    np.savez(p, half_m=np.float32(half), cell=np.float32(cell), ground_z=ground,
             obstacle_z=obstacle, canopy_z=canopy, canopy_d=dens, drivable=drivable, concealed=concealed)
    return WorldMap(p)


def _task(world, k=2, n=2, **kw):
    cfg = ChaseCfg(n_targets=k, arena_half=90.0, **kw)
    return ChaseTask(world, cfg, num_envs=n, dt=0.04, max_steps=100), cfg


def _state(n=2):
    quat = torch.zeros(n, 4); quat[:, 0] = 1.0
    return quat, torch.zeros(n, 3), torch.zeros(n, 3)


def test_touch_ends_the_episode_and_pays_more_when_early(world):
    quat, vel, avb = _state()
    tgt = torch.tensor([[[0.0, 0.0, 1.0], [50.0, 0.0, 1.0]]] * 2)
    drone = torch.tensor([[0.0, 0.0, 30.0]] * 2)
    touched = torch.tensor([[True, False], [True, False]])

    def run(step):
        task, _ = _task(world)
        _, r, term, info = task.step(drone, vel, quat, avb, tgt, torch.full((2,), step), touched=touched)
        return float(r[0]), bool(term[0]), bool(info["touch"][0])

    early, t1, touched1 = run(5)
    late, t2, _ = run(90)
    assert t1 and t2 and touched1
    assert early > late > 50.0


def test_protected_forklift_pays_nothing_and_does_not_end(world):
    task, cfg = _task(world)
    quat, vel, avb = _state()
    tgt = torch.tensor([[[0.0, 0.0, 1.0], [50.0, 0.0, 1.0]]] * 2)
    drone = torch.tensor([[0.0, 0.0, 30.0]] * 2)
    touched = torch.tensor([[True, False]] * 2)
    prot = torch.tensor([[True, False]] * 2)
    _, r, term, info = task.step(drone, vel, quat, avb, tgt, torch.zeros(2, dtype=torch.long),
                                 touched=touched, protected=prot)
    assert not term.any() and not info["touch"].any() and info["touch_protected"].all()
    assert r[0] < 15.0                                  # no touch bonus: only the sighting of #1


def test_sighting_pays_once_and_only_for_unprotected(world):
    task, cfg = _task(world)
    quat, vel, avb = _state()
    tgt = torch.tensor([[[30.0, 0.0, 1.0], [-80.0, 0.0, 1.0]]] * 2)   # one ahead, one behind
    drone = torch.tensor([[0.0, 0.0, 30.0]] * 2)
    step = torch.zeros(2, dtype=torch.long)
    no = torch.zeros(2, 2, dtype=torch.bool)
    _, r1, _, info = task.step(drone, vel, quat, avb, tgt, step, touched=no)
    assert info["visible"][0].tolist() == [True, False]
    _, r2, _, _ = task.step(drone, vel, quat, avb, tgt, step, touched=no)
    assert r1[0] > r2[0] + cfg.r_sight * 0.9, "the first sighting pays, the second frame does not"
    task2, _ = _task(world)
    _, r3, _, _ = task2.step(drone, vel, quat, avb, tgt, step, touched=no,
                             protected=torch.tensor([[True, False]] * 2))
    assert abs(float(r3[0] - r2[0])) < 1.0, "a protected forklift in view pays no sighting"


def test_pixel_sightings_override_geometry(world):
    task, cfg = _task(world, sight_px=4)
    quat, vel, avb = _state()
    tgt = torch.tensor([[[-80.0, 0.0, 1.0], [-80.0, 30.0, 1.0]]] * 2)   # both behind the lens
    drone = torch.tensor([[0.0, 0.0, 30.0]] * 2)
    no = torch.zeros(2, 2, dtype=torch.bool)
    _, _, _, info = task.step(drone, vel, quat, avb, tgt, torch.zeros(2, dtype=torch.long),
                              touched=no, seen_px=torch.tensor([[10, 3]] * 2))
    assert info["visible"][0].tolist() == [True, False]


def test_crash_and_clearance(world):
    task, cfg = _task(world)
    quat, vel, avb = _state()
    tgt = torch.full((2, 2, 3), 500.0)
    step = torch.zeros(2, dtype=torch.long)
    no = torch.zeros(2, 2, dtype=torch.bool)
    # next to the 40 m block at 4 m: clearance small, penalty paid; in the open: none
    near = torch.tensor([[38.0, 0.0, 20.0], [-50.0, 0.0, 20.0]])
    _, r, term, info = task.step(near, vel, quat, avb, tgt, step, touched=no)
    assert info["clearance"][0] < 6.0 < info["clearance"][1]
    assert r[0] < r[1] and not term.any()
    # a contact with the world from the sensor is a crash regardless of the map
    _, r2, term2, info2 = task.step(torch.tensor([[-50.0, 0.0, 20.0]] * 2), vel, quat, avb, tgt, step,
                                    touched=no, crashed=torch.tensor([True, False]))
    assert term2[0] and not term2[1] and r2[0] < -50
    # and the map fallback catches a drone below the ground
    _, _, term3, _ = task.step(torch.tensor([[-50.0, 0.0, 0.5]] * 2), vel, quat, avb, tgt, step, touched=no)
    assert term3.all()


def test_observation_widths(world):
    task, cfg = _task(world, k=3)
    quat, vel, avb = _state()
    tgt = torch.zeros(2, 3, 3)
    obs, _, _, info = task.step(torch.tensor([[0.0, 0.0, 30.0]] * 2), vel, quat, avb, tgt,
                                torch.zeros(2, dtype=torch.long), touched=torch.zeros(2, 3, dtype=torch.bool))
    assert obs.shape == (2, PROPRIO_DIM)
    priv = task.privileged(torch.tensor([[0.0, 0.0, 30.0]] * 2), vel, quat, avb, tgt, info["visible"],
                           torch.zeros(2, 3, dtype=torch.bool), torch.zeros(2))
    assert priv.shape == (2, task.priv_dim) and torch.isfinite(priv).all()


def test_zones_rasterise_and_attach(world, tmp_path):
    z = Zones(launch=[[-90, -90], [-30, -90], [-30, -30], [-90, -30]],
              safe=[[[40, 60], [90, 60], [90, 90], [40, 90]]])
    p = z.save(tmp_path / "zones.json")
    z2 = Zones.load(p)
    world.attach_zones(z2)
    assert world.launch[5, 5] == 1 and world.launch[50, 50] == 0          # (-90,-90) in, (0,0) out
    assert world.in_safe(torch.tensor([60.0]), torch.tensor([75.0])).item()
    assert not world.in_safe(torch.tensor([0.0]), torch.tensor([0.0])).item()
    g = torch.Generator().manual_seed(0)
    xy, ok = world.sample_cells_xy(world.launch, 256, g)
    assert ok.all() and (xy[:, 0] <= -29).all() and (xy[:, 1] <= -29).all()
    assert (xy[:, 0] >= -91).all() and xy[:, 0].std() > 10, "spread over the whole pad"
    m = rasterize([[[0, 0], [10, 0], [10, 10]]], 11, 0.0, 1.0)
    assert m[0, 0] == 1 and m[9, 9] == 1 and m[9, 0] == 0


def test_depth_model_normalises_clips_and_holes():
    g = torch.Generator().manual_seed(1)
    dm = DepthModel(max_range=20.0, noise_frac=0.0, hole_p=0.0, hole_p_far=0.0, generator=g)
    d = torch.tensor([[[5.0, 20.0, 100.0, float("inf")]]])
    out = dm(d)
    assert out.shape == (1, 1, 4, 1)
    assert out[0, 0, :, 0].tolist() == pytest.approx([0.25, 1.0, 1.0, 1.0])
    dm2 = DepthModel(max_range=20.0, noise_frac=0.02, hole_p=0.1, hole_p_far=0.0, generator=g)
    big = dm2(torch.full((4, 32, 32), 10.0))
    frac_holes = float((big == 0).float().mean())
    assert 0.05 < frac_holes < 0.15
    assert abs(float(big[big > 0].mean()) - 0.5) < 0.01


def test_belief_target_is_the_nearest_visible_forklift(world):
    task, cfg = _task(world, k=3)
    quat, vel, avb = _state()
    # #0 far ahead, #1 near ahead, #2 near but behind the lens
    tgt = torch.tensor([[[90.0, 0.0, 1.0], [40.0, 0.0, 1.0], [-10.0, 0.0, 1.0]]] * 2)
    drone = torch.tensor([[0.0, 0.0, 30.0]] * 2)
    no = torch.zeros(2, 3, dtype=torch.bool)
    _, _, _, info = task.step(drone, vel, quat, avb, tgt, torch.zeros(2, dtype=torch.long), touched=no)
    assert info["belief_ok"].all()
    assert info["belief_target"][0].tolist() == pytest.approx([0.40, 0.0, -0.29], abs=0.01)
    # nothing in frame: the target is zeroed and masked out of the loss
    behind = torch.tensor([[[-90.0, 0.0, 1.0]] * 3] * 2)
    task2, _ = _task(world, k=3)
    _, _, _, info2 = task2.step(drone, vel, quat, avb, behind, torch.zeros(2, dtype=torch.long), touched=no)
    assert not info2["belief_ok"].any()
    assert float(info2["belief_target"].abs().max()) == 0.0


def test_protected_forklift_is_not_a_belief_target(world):
    task, cfg = _task(world, k=2)
    quat, vel, avb = _state()
    # both in frame and clear of the building (which starts at x=40)
    tgt = torch.tensor([[[20.0, 0.0, 1.0], [35.0, 0.0, 1.0]]] * 2)
    drone = torch.tensor([[0.0, 0.0, 30.0]] * 2)
    no = torch.zeros(2, 2, dtype=torch.bool)
    _, _, _, info = task.step(drone, vel, quat, avb, tgt, torch.zeros(2, dtype=torch.long), touched=no,
                              protected=torch.tensor([[True, False]] * 2))
    assert info["belief_ok"].all()
    assert info["belief_target"][0, 0].item() == pytest.approx(0.35, abs=0.01)   # the unprotected one
