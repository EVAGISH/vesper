"""View a run's trajectory (and manifest) in rerun on the Mac."""
import json
from pathlib import Path

import numpy as np

from vesper.record import read_trajectory


def view_run(run_dir: str | Path, spawn: bool = True) -> None:
    import rerun as rr

    run_dir = Path(run_dir)
    traj = read_trajectory(run_dir / "trajectory.parquet")
    rr.init(f"vesper/{run_dir.name}", spawn=spawn)

    pts = np.column_stack([traj["px"], traj["py"], traj["pz"]])
    rr.log("world/path", rr.LineStrips3D([pts]), static=True)

    manifest = run_dir / "manifest.json"
    if manifest.exists():
        rr.log("run/manifest", rr.TextDocument(json.dumps(json.loads(manifest.read_text()), indent=2)), static=True)

    for i in range(len(traj["t"])):
        rr.set_time_seconds("sim_time", float(traj["t"][i]))
        rr.log("world/drone", rr.Transform3D(
            translation=pts[i],
            rotation=rr.Quaternion(xyzw=[traj["qx"][i], traj["qy"][i], traj["qz"][i], traj["qw"][i]]),
        ))
        rr.log("world/drone/marker", rr.Points3D([[0, 0, 0]], radii=[0.12]))
        rr.log("plots/altitude", rr.Scalar(float(traj["pz"][i])))
