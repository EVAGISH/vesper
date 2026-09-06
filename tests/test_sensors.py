import torch

from vesper.sensors import PrismRayCaster, RangeNoise

BOX = [{"center": [5.0, 0.0], "size": [2.0, 2.0], "height": 4.0}]  # north 4..6, east -1..1


def test_ray_hits_box_ahead():
    rc = PrismRayCaster(BOX, num_rays=4, max_range=50.0)  # rays: +x(east), +y(north), -x, -y
    pos = torch.tensor([[0.0, 0.0, 2.0]])
    r = rc.cast(pos, yaw=torch.zeros(1))
    assert abs(r[0, 1].item() - 4.0) < 1e-4   # +y (north) ray hits front face at 4m
    assert r[0, 0].item() == 50.0 and r[0, 2].item() == 50.0 and r[0, 3].item() == 50.0


def test_flies_over():
    rc = PrismRayCaster(BOX, num_rays=4, max_range=50.0)
    r = rc.cast(torch.tensor([[0.0, 0.0, 5.0]]), yaw=torch.zeros(1))  # above 4m roof
    assert r[0, 1].item() == 50.0


def test_visibility_clips():
    rc = PrismRayCaster(BOX, num_rays=4, max_range=50.0)
    r = rc.cast(torch.tensor([[0.0, 0.0, 2.0]]), yaw=torch.zeros(1), visibility=3.0)
    assert (r == 3.0).all()  # fog wall closer than the building


def test_yaw_rotates_rays():
    import math
    rc = PrismRayCaster(BOX, num_rays=4, max_range=50.0)
    r = rc.cast(torch.tensor([[0.0, 0.0, 2.0]]), yaw=torch.tensor([math.pi / 2]))
    assert abs(r[0, 0].item() - 4.0) < 1e-4   # body +x now points north


def test_range_noise_seeded():
    g1, g2 = torch.Generator().manual_seed(3), torch.Generator().manual_seed(3)
    r = torch.full((2, 4), 10.0)
    a = RangeNoise(std=0.5, dropout_p=0.2, generator=g1).apply(r, 50.0)
    b = RangeNoise(std=0.5, dropout_p=0.2, generator=g2).apply(r, 50.0)
    assert torch.equal(a, b)
