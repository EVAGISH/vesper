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

    rays_path = run_dir / "rays.parquet"
    if rays_path.exists():
        import math
        r = read_trajectory(rays_path)
        K = sum(1 for c in r if c.startswith("r"))
        for i in range(len(r["t"])):
            rr.set_time_seconds("sim_time", float(r["t"][i]))
            o = np.array([r["px"][i], r["py"][i], r["pz"][i]])
            strips = []
            for k in range(K):
                a = r["yaw"][i] + 2 * math.pi * k / K
                strips.append([o, o + r[f"r{k}"][i] * np.array([math.cos(a), math.sin(a), 0.0])])
            rr.log("world/rays", rr.LineStrips3D(strips, radii=0.01))

    for i in range(len(traj["t"])):
        rr.set_time_seconds("sim_time", float(traj["t"][i]))
        rr.log("world/drone", rr.Transform3D(
            translation=pts[i],
            rotation=rr.Quaternion(xyzw=[traj["qx"][i], traj["qy"][i], traj["qz"][i], traj["qw"][i]]),
        ))
        rr.log("world/drone/marker", rr.Points3D([[0, 0, 0]], radii=[0.12]))
        rr.log("plots/altitude", rr.Scalar(float(traj["pz"][i])))
