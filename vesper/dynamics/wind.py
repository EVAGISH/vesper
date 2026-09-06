"""Gust model: Ornstein-Uhlenbeck colored noise around a mean wind vector.

A Dryden-lite stand-in: correlated gusts, seeded, batched. The full
MIL-HDBK-1797 Dryden filter can replace this behind the same interface.
"""
import math

import torch


class GustField:
    def __init__(self, num_envs, mean_wind, gust_std=0.0, tau_s=2.0, dt=0.02,
                 device="cpu", generator=None):
        self.mean = torch.as_tensor(mean_wind, dtype=torch.float32, device=device)
        if self.mean.dim() == 1:
            self.mean = self.mean.expand(num_envs, 3).clone()
        self.gust_std = gust_std
        self.alpha = math.exp(-dt / tau_s)
        self.sigma = gust_std * math.sqrt(1 - self.alpha**2)
        self.state = torch.zeros(num_envs, 3, device=device)
        self.gen = generator

    def step(self) -> torch.Tensor:
        noise = torch.randn(self.state.shape, device=self.state.device, generator=self.gen)
        self.state = self.alpha * self.state + self.sigma * noise
        return self.mean + self.state
