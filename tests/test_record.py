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


def _tiny_run(tmp_path, **kw):
    """Capture three 32x32 frames on two streams and encode them."""
    import numpy as np
    from vesper.capture.run import RunCapture

    cap = RunCapture("t", root=tmp_path, **kw)
    for i in range(3):
        f = np.full((32, 32, 3), i * 40, dtype=np.uint8)
        cap.add_frame(f, "overview")
        cap.add_frame(f, "fpv")
    cap.finish(fps=3)
    return cap


def test_frames_deleted_after_encoding(tmp_path):
    """The PNG sequence is ffmpeg scratch; keeping it grew runs/ to 68 GB."""
    import json

    cap = _tiny_run(tmp_path)
    assert (cap.dir / "overview.mp4").exists() and (cap.dir / "fpv.mp4").exists()
    assert not (cap.dir / "frames").exists(), "overview PNGs should be gone"
    assert not (cap.dir / "frames_fpv").exists(), "fpv PNGs should be gone"
    meta = json.loads((cap.dir / "manifest.json").read_text())
    assert meta["frames"] == {"overview": 3, "fpv": 3}, "frame counts still recorded"
    assert meta["kept_frames"] is False


def test_keep_frames_opt_in(tmp_path):
    cap = _tiny_run(tmp_path, keep_frames=True)
    assert (cap.dir / "overview.mp4").exists()
    assert len(list((cap.dir / "frames").glob("*.png"))) == 3
    assert len(list((cap.dir / "frames_fpv").glob("*.png"))) == 3
