"""Trajectory log: one parquet file per run, full pose every logged step.

Schema v1 (fidelity lane; actions and sensor obs columns join in later steps):
  t          float64  sim seconds
  px py pz   float64  world position (m)
  qw qx qy qz float64 world orientation quaternion
"""
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

SCHEMA_VERSION = 1
_COLS = ["t", "px", "py", "pz", "qw", "qx", "qy", "qz"]


class TrajectoryWriter:
    def __init__(self, run_dir: str | Path):
        self.path = Path(run_dir) / "trajectory.parquet"
        self._rows: list[list[float]] = []

    def append(self, t: float, pos, quat) -> None:
        p = np.asarray(pos, dtype=float).reshape(-1)
        q = np.asarray(quat, dtype=float).reshape(-1)
        self._rows.append([float(t), p[0], p[1], p[2], q[0], q[1], q[2], q[3]])

    def __len__(self) -> int:
        return len(self._rows)

    def close(self) -> Path:
        arr = np.asarray(self._rows, dtype=float).reshape(-1, len(_COLS))
        table = pa.table(
            {c: arr[:, i] for i, c in enumerate(_COLS)},
            metadata={b"vesper_schema": str(SCHEMA_VERSION).encode()},
        )
        pq.write_table(table, self.path)
        return self.path


def read_trajectory(path: str | Path) -> dict[str, np.ndarray]:
    table = pq.read_table(path)
    return {c: table[c].to_numpy() for c in table.column_names}
