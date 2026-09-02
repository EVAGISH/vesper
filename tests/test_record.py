import numpy as np

from vesper.record import TrajectoryWriter, read_trajectory


def test_round_trip(tmp_path):
    w = TrajectoryWriter(tmp_path)
    for i in range(10):
        w.append(i / 30.0, [i, 0.0, 3.0], [1.0, 0.0, 0.0, 0.0])
    path = w.close()
    traj = read_trajectory(path)
    assert len(traj["t"]) == 10
    np.testing.assert_allclose(traj["px"], np.arange(10, dtype=float))
    np.testing.assert_allclose(traj["pz"], 3.0)
