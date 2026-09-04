"""Torch view of a geo world: the rasters a policy needs but USD cannot give it fast.

scripts/export_world_map.py bakes a world USD into height rasters plus masks;
this loads them onto the GPU and answers the questions the search task asks
every step, batched over thousands of environments:

  ground/canopy height under a point      -> bilinear sample
  can A see B                             -> march the segment against terrain + buildings
  how much foliage is in the way          -> integrate canopy density along the same march
  where can a vehicle spawn / drive       -> road, parking and trunk layers

Optional layers (road, road_yaw, parking, park_yaw, trunks, tree_z) default to
empty when an older export lacks them, so a map built before they existed still
loads -- the vehicles then fall back to plain drivable ground.

Pure numpy/torch: no Isaac, no pxr, unit-testable on a Mac.
Frame: x east, y north, z up, metres, origin at the world centre. Raster row
indexes +y, column indexes +x, both spanning [-half_m, half_m].
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch


class WorldMap:
    FIELDS = ("ground_z", "obstacle_z", "canopy_z", "canopy_d")
    OPTIONAL = ("road", "road_yaw", "parking", "park_yaw", "trunks")

    def __init__(self, npz_path, device="cpu"):
        d = np.load(str(npz_path))
        self.device = device
        for f in self.FIELDS:
            setattr(self, f, torch.as_tensor(np.asarray(d[f], np.float32), device=device))
        self.drivable = torch.as_tensor(np.asarray(d["drivable"], np.float32), device=device)
        self.concealed = torch.as_tensor(np.asarray(d["concealed"], np.float32), device=device)
        for f in self.OPTIONAL:
            arr = np.asarray(d[f], np.float32) if f in d.files else np.zeros_like(d["ground_z"], np.float32)
            setattr(self, f, torch.as_tensor(arr, device=device))
        # hard tree geometry (trunk + crown colliders), absent when the world's
        # trees are visual only; then it is just the ground and changes nothing
        self.tree_z = (torch.as_tensor(np.asarray(d["tree_z"], np.float32), device=device)
                       if "tree_z" in d.files else self.ground_z)
        self.has_tree_solids = "tree_z" in d.files
        self.n = int(self.ground_z.shape[0])
        meta_path = Path(str(npz_path)).with_suffix(".json")
        if "half_m" in d:
            self.half_m, self.cell = float(d["half_m"]), float(d["cell"])
        elif meta_path.exists():
            m = json.loads(meta_path.read_text())
            self.half_m, self.cell = float(m["half_m"]), float(m["cell"])
        else:
            raise ValueError(f"{npz_path} has no half_m/cell and no sidecar json")
        # solid top: what stops a ray or a drone. Buildings always; trees only
        # when the world gave them colliders (tree_z), otherwise a quad can be
        # flown into a crown and the canopy layer models that separately.
        self.solid_z = torch.maximum(torch.maximum(self.ground_z, self.obstacle_z), self.tree_z)

    # ---------------------------------------------------------------- sampling
    def _uv(self, x, y):
        """World xy -> fractional raster (col, row), clamped inside the grid."""
        c = ((x + self.half_m) / self.cell).clamp(0, self.n - 1.001)
        r = ((y + self.half_m) / self.cell).clamp(0, self.n - 1.001)
        return c, r

    def sample(self, field: torch.Tensor, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Bilinear sample of any raster at world xy. Shapes broadcast freely."""
        c, r = self._uv(x, y)
        c0, r0 = c.floor().long(), r.floor().long()
        c1, r1 = (c0 + 1).clamp(max=self.n - 1), (r0 + 1).clamp(max=self.n - 1)
        fc, fr = (c - c0).to(field.dtype), (r - r0).to(field.dtype)
        f00 = field[r0, c0]; f01 = field[r0, c1]
        f10 = field[r1, c0]; f11 = field[r1, c1]
        return (f00 * (1 - fc) * (1 - fr) + f01 * fc * (1 - fr)
                + f10 * (1 - fc) * fr + f11 * fc * fr)

    def ground_at(self, x, y):
        return self.sample(self.ground_z, x, y)

    def solid_at(self, x, y):
        return self.sample(self.solid_z, x, y)

    def canopy_at(self, x, y):
        return self.sample(self.canopy_z, x, y)

    def canopy_density_at(self, x, y):
        return self.sample(self.canopy_d, x, y)

    def nearest_cell(self, x, y):
        c, r = self._uv(x, y)
        return r.round().long().clamp(0, self.n - 1), c.round().long().clamp(0, self.n - 1)

    def is_drivable(self, x, y):
        r, c = self.nearest_cell(x, y)
        return self.drivable[r, c] > 0.5

    def yaw_at(self, field: torch.Tensor, x, y):
        """Nearest-cell read of a heading raster (radians), no interpolation:
        angles wrap, so blending neighbours would invent directions."""
        r, c = self.nearest_cell(x, y)
        return field[r, c]

    # ---------------------------------------------------------------- visibility
    def trace(self, p0: torch.Tensor, p1: torch.Tensor, samples: int = 40):
        """March the segment p0->p1 (both [N,3] world).

        Returns (clear [N] bool, foliage [N] metres of canopy the ray crosses,
        weighted by density). `clear` is False when terrain or a building rises
        above the ray anywhere strictly between the endpoints -- endpoints are
        excluded so a target sitting on the ground is not occluded by the ground
        it sits on.
        """
        t = torch.linspace(0.0, 1.0, samples + 2, device=p0.device)[1:-1]        # [S]
        t = t.view(1, -1, 1)
        p = p0.unsqueeze(1) + (p1 - p0).unsqueeze(1) * t                          # [N,S,3]
        x, y, z = p[..., 0], p[..., 1], p[..., 2]
        blocked = (z < self.sample(self.solid_z, x, y)).any(dim=1)
        canopy_top = self.sample(self.canopy_z, x, y)
        ground = self.sample(self.ground_z, x, y)
        inside = ((z < canopy_top) & (z > ground)).float()
        dens = self.sample(self.canopy_d, x, y)
        seg = (p1 - p0).norm(dim=1) / samples
        foliage = (inside * dens).sum(dim=1) * seg
        return ~blocked, foliage

    # ---------------------------------------------------------------- spawning
    def sample_mask_xy(self, mask: torch.Tensor, n: int, half: float, generator=None,
                       tries: int = 24, centre=(0.0, 0.0)):
        """n random xy inside the square of half-extent `half` where `mask` is set.

        Rejection sampling with a bounded number of rounds: whatever is still
        unplaced after the last round is returned wherever it landed, so this
        never hangs on a mask that is empty in the requested box.
        """
        dev = self.ground_z.device
        cx, cy = centre
        xy = torch.zeros(n, 2, device=dev)
        ok = torch.zeros(n, dtype=torch.bool, device=dev)
        for _ in range(tries):
            k = int((~ok).sum())
            if k == 0:
                break
            cand = (torch.rand(k, 2, device=dev, generator=generator) * 2 - 1) * half
            cand[:, 0] += cx; cand[:, 1] += cy
            r, c = self.nearest_cell(cand[:, 0], cand[:, 1])
            good = mask[r, c] > 0.5
            idx = torch.nonzero(~ok).flatten()
            xy[idx] = cand
            ok[idx[good]] = True
        return xy, ok
