"""Sensor corruption, seeded and batched: range noise + dropout, GPS drift."""
import torch


class RangeNoise:
    def __init__(self, std=0.1, dropout_p=0.0, generator=None):
        self.std, self.dropout_p, self.gen = std, dropout_p, generator

    def apply(self, ranges: torch.Tensor, max_range: float) -> torch.Tensor:
        noisy = ranges + self.std * torch.randn(ranges.shape, device=ranges.device, generator=self.gen)
        if self.dropout_p > 0:
            drop = torch.rand(ranges.shape, device=ranges.device, generator=self.gen) < self.dropout_p
            noisy = torch.where(drop, torch.full_like(noisy, max_range), noisy)
        return noisy.clamp(0.0, max_range)


class GpsNoise:
    """White noise + OU bias drift on position."""

    def __init__(self, num_envs, std=0.3, bias_tau_s=60.0, bias_std=1.0, dt=0.02,
                 device="cpu", generator=None):
        import math
        self.std, self.gen = std, generator
        self.alpha = math.exp(-dt / bias_tau_s)
        self.sigma = bias_std * math.sqrt(1 - self.alpha**2)
        self.bias = torch.zeros(num_envs, 3, device=device)

    def apply(self, pos: torch.Tensor) -> torch.Tensor:
        self.bias = self.alpha * self.bias + self.sigma * torch.randn(
            self.bias.shape, device=pos.device, generator=self.gen)
        return pos + self.bias + self.std * torch.randn(pos.shape, device=pos.device, generator=self.gen)
