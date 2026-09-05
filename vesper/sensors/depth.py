"""A small stereo-class depth sensor on top of the renderer's perfect depth.

The tiled camera hands back exact distance per pixel out to the far clip. A
RealSense / OAK-class module on a drone does not: it stops at 10-20 m, its
error grows with the square of range (stereo disparity quantisation), it has
holes on sky, thin structures and low-texture surfaces. Training on perfect
depth breaks on the real one, so this is applied to every frame the policy
sees, in torch, before it reaches the network.

Output is normalised range in [0, 1]: 1 = at or beyond max_range (or sky),
0 = no return (a hole). The policy never sees metres.
"""
from __future__ import annotations

import torch


class DepthModel:
    def __init__(self, max_range: float = 20.0, noise_frac: float = 0.02, hole_p: float = 0.02,
                 hole_p_far: float = 0.15, generator=None):
        self.max_range = float(max_range)
        self.noise_frac = float(noise_frac)      # 1-sigma error at max_range, as a fraction of it
        self.hole_p = float(hole_p)              # random dropout everywhere
        self.hole_p_far = float(hole_p_far)      # extra dropout in the last 20% of range
        self.gen = generator

    def __call__(self, depth: torch.Tensor) -> torch.Tensor:
        """depth [N,H,W] or [N,H,W,1] in metres (inf / huge = sky) -> [N,H,W,1] in [0,1]."""
        if depth.dim() == 3:
            depth = depth.unsqueeze(-1)
        d = depth.float()
        d = torch.where(torch.isfinite(d), d, torch.full_like(d, self.max_range * 10))
        r = d / self.max_range
        # stereo error grows with range^2; at max_range it is noise_frac of max_range
        sigma = self.noise_frac * r.clamp(max=1.0) ** 2
        r = r + torch.randn(r.shape, device=r.device, generator=self.gen) * sigma
        u = torch.rand(r.shape, device=r.device, generator=self.gen)
        p_hole = self.hole_p + self.hole_p_far * ((r - 0.8) / 0.2).clamp(0.0, 1.0)
        out = r.clamp(0.0, 1.0)
        return torch.where(u < p_hole, torch.zeros_like(out), out)
