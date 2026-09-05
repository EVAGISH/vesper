"""CPU tests for the map layers a vehicle needs: roads, parking strips, hard trees."""
import math

import numpy as np
import pytest
from shapely.geometry import LineString

from vesper.worlds.rasters import (chamfer_distance, parking_from_buildings, rasterize_roads,
                                   splat_tree_solids)


def test_roads_are_buffered_to_class_width_and_carry_a_direction():
    n, half, cell = 101, 100.0, 2.0
    roads = [(LineString([(-80, 0), (80, 0)]), {"highway": "service"}),      # east-west, 5.5 m
             (LineString([(0, -80), (0, 80)]), {"highway": "footway"})]      # ignored
    mask, yaw = rasterize_roads(roads, n, half, cell)
    assert mask[50, 50] == 1 and mask[50, 10] == 1
    assert mask[20, 50] == 0, "footways are not roads for a forklift"
    rows = np.flatnonzero(mask[:, 30])
    assert 2 <= len(rows) <= 4                                 # ~5.5 m wide at 2 m cells
    assert yaw[50, 30] == pytest.approx(0.0, abs=1e-4)
    m2, y2 = rasterize_roads([(LineString([(0, -80), (0, 80)]), {"highway": "residential"})], n, half, cell)
    assert y2[20, 50] == pytest.approx(math.pi / 2, abs=1e-4)


def test_chamfer_distance_is_close_to_euclidean():
    mask = np.zeros((41, 41), np.uint8); mask[20, 20] = 1
    d = chamfer_distance(mask, 1.0)
    assert d[20, 20] == 0.0
    assert d[20, 30] == pytest.approx(10.0, rel=0.02)
    assert d[30, 30] == pytest.approx(math.sqrt(200), rel=0.08)


def test_parking_hugs_the_facade_and_runs_along_it():
    n, cell = 61, 2.0
    building = np.zeros((n, n), np.uint8); building[20:40, 20:40] = 1      # 40 m square block
    drivable = 1 - building
    park, yaw, bdist = parking_from_buildings(building, drivable, cell, near_m=(1.0, 7.0))
    assert park[30, 42] == 1 and park[30, 45] == 0                          # 4 m off the east wall
    assert park[30, 30] == 0, "not inside the building"
    # east wall runs north-south, so a parked heading there is ~pi/2
    assert abs(math.sin(yaw[30, 42])) > 0.95
    # north wall runs east-west
    assert abs(math.cos(yaw[42, 30])) > 0.95


def test_tree_solids_are_a_conservative_crown_disc():
    n, half, cell = 51, 50.0, 2.0
    ground = np.zeros((n, n), np.float32)
    trees = [(0.0, 0.0, 20.0, 6.0)]                                        # 20 m tree, 6 m crown
    tz, trunks = splat_tree_solids(trees, ground, n, half, cell)
    assert trunks[25, 25] == 1.0 and trunks.sum() == 1.0
    assert tz[25, 25] == pytest.approx(20.0, abs=0.01)                     # tree top over the trunk
    assert tz[25, 27] == pytest.approx(20.0) and tz[25, 30] == 0.0         # inside / outside the crown
