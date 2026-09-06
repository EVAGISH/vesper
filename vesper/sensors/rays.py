"""Batched ray-casting against prism (AABB) worlds, pure torch.

The policy's range sensor: K horizontal rays fanned around the vehicle's yaw,
slab-tested against every building box. Visibility clips ray length -- the
mechanistic fog failure from the plan lives exactly here.
Spec frame (north, east) maps to world (y, x), same as worlds.usd_stage.
"""
import math

import torch


class PrismRayCaster:
    def __init__(self, buildings: list[dict], num_rays: int = 16, max_range: float = 50.0, device="cpu"):
        self.num_rays, self.max_range, self.device = num_rays, max_range, device
        if buildings:
            lo, hi = [], []
            for b in buildings:
                (n, e), (dn, de), h = b["center"], b["size"], b["height"]
                lo.append([e - de / 2, n - dn / 2, 0.0])
                hi.append([e + de / 2, n + dn / 2, h])
            self.box_lo = torch.tensor(lo, device=device)  # [B,3] world frame
            self.box_hi = torch.tensor(hi, device=device)
        else:
            self.box_lo = torch.zeros(0, 3, device=device)
            self.box_hi = torch.zeros(0, 3, device=device)
        ang = torch.arange(num_rays, device=device) * (2 * math.pi / num_rays)
        self.ray_dirs_body = torch.stack([ang.cos(), ang.sin(), torch.zeros_like(ang)], dim=1)  # [K,3]

    def cast(self, pos: torch.Tensor, yaw: torch.Tensor, visibility: torch.Tensor | float = None):
        """pos [N,3] world, yaw [N] -> ranges [N,K], clipped to min(max_range, visibility)."""
        N, K = pos.shape[0], self.num_rays
        cy, sy = yaw.cos(), yaw.sin()
        dirs = torch.empty(N, K, 3, device=pos.device)
        dx, dy = self.ray_dirs_body[:, 0], self.ray_dirs_body[:, 1]
        dirs[:, :, 0] = cy.unsqueeze(1) * dx - sy.unsqueeze(1) * dy
        dirs[:, :, 1] = sy.unsqueeze(1) * dx + cy.unsqueeze(1) * dy
        dirs[:, :, 2] = 0.0

        limit = torch.full((N, 1), float(self.max_range), device=pos.device)
        if visibility is not None:
            vis = torch.as_tensor(visibility, dtype=torch.float32, device=pos.device).reshape(-1, 1)
            limit = torch.minimum(limit, vis.expand(N, 1))

        if len(self.box_lo) == 0:
            return limit.expand(N, K).clone()

        o = pos.unsqueeze(1).unsqueeze(2)                     # [N,1,1,3]
        d = dirs.unsqueeze(2)                                 # [N,K,1,3]
        lo = self.box_lo.view(1, 1, -1, 3)
        hi = self.box_hi.view(1, 1, -1, 3)
        inv = 1.0 / d.where(d.abs() > 1e-9, torch.full_like(d, 1e-9))
        t1, t2 = (lo - o) * inv, (hi - o) * inv
        tmin = torch.minimum(t1, t2)[..., :2].amax(dim=3)     # xy slabs
        tmax = torch.maximum(t1, t2)[..., :2].amin(dim=3)
        z_ok = (pos[:, 2].view(N, 1, 1) >= lo[..., 2]) & (pos[:, 2].view(N, 1, 1) <= hi[..., 2])
        hit = (tmax >= tmin) & (tmax > 0) & z_ok               # [N,K,B]
        t_hit = torch.where(hit, tmin.clamp(min=0.0), torch.full_like(tmin, float("inf")))
        ranges = t_hit.amin(dim=2)                             # [N,K]
        return torch.minimum(ranges, limit.expand(N, K))
