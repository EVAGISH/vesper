"""The pixels-and-proprio policy network for the search task.

What the actor gets is exactly what SearchEnv hands out under "pixels" and
"policy": a body-fixed camera frame and the airframe's own measurements, no
map, no GPS. Memory of where it has looked has to live in the network, so the
trunk is recurrent. Roughly 3M parameters at the defaults:

  encoder   4 stride-2 convolutions over the frame          -> 256
  proprio   small MLP over the 11 proprio values             -> 64
  memory    GRU over [encoder, proprio], hidden 512, carried across the episode
  actor     GRU state -> action mean (+ learned log-std)
  critic    GRU state ++ privileged vector -> value (asymmetric: the critic may
            see the truth, the actor never does -- STACK.md section 4)
  aux       encoder -> is a vehicle in frame, and where in the image; trained
            from the segmentation labels so the encoder learns to see forklifts
            before the sparse reward can teach it to

This file is the network only. The recurrent PPO / distillation loop that
trains it is deliberately not here yet: the environment side has to be right
first, and this module exists so the env's observation contract has a consumer
that is shape-checked on the Mac.
"""
from __future__ import annotations

import torch
import torch.nn as nn


def conv_encoder(in_ch=3, widths=(32, 64, 128, 128), out_dim=256, res=128):
    layers, c = [], in_ch
    for w in widths:
        layers += [nn.Conv2d(c, w, 3, stride=2, padding=1), nn.ELU()]
        c = w
    layers += [nn.AdaptiveAvgPool2d(4), nn.Flatten(), nn.Linear(c * 16, out_dim), nn.ELU()]
    return nn.Sequential(*layers)


class VisionActorCritic(nn.Module):
    def __init__(self, proprio_dim, act_dim, priv_dim=0, res=128, enc_dim=256,
                 hidden=512, init_std=0.5):
        super().__init__()
        self.enc = conv_encoder(out_dim=enc_dim, res=res)
        self.prop = nn.Sequential(nn.Linear(proprio_dim, 64), nn.ELU())
        self.gru = nn.GRUCell(enc_dim + 64, hidden)
        self.actor = nn.Sequential(nn.Linear(hidden, 256), nn.ELU(), nn.Linear(256, act_dim))
        self.critic = nn.Sequential(nn.Linear(hidden + priv_dim, 256), nn.ELU(), nn.Linear(256, 1))
        self.aux = nn.Sequential(nn.Linear(enc_dim, 64), nn.ELU(), nn.Linear(64, 3))   # p(in frame), u, v
        self.log_std = nn.Parameter(torch.full((act_dim,), float(torch.log(torch.tensor(init_std)))))
        self.hidden = hidden

    def initial_state(self, n, device=None):
        return torch.zeros(n, self.hidden, device=device)

    @staticmethod
    def prep(pixels):
        """uint8 [N,H,W,3] from the env -> float [N,3,H,W] in [-1, 1]."""
        return pixels.permute(0, 3, 1, 2).float() / 127.5 - 1.0

    def forward(self, pixels, proprio, h, priv=None, done=None):
        """One step. Returns (action mean, value, next h, aux logits).

        `done` [N] bool zeroes the memory of environments that just restarted,
        which is how the recurrent state stays inside one episode.
        """
        if done is not None:
            h = h * (~done).float().unsqueeze(1)
        z = self.enc(self.prep(pixels))
        x = torch.cat([z, self.prop(proprio)], dim=1)
        h = self.gru(x, h)
        mean = self.actor(h)
        cin = h if priv is None else torch.cat([h, priv], dim=1)
        value = self.critic(cin).squeeze(-1)
        return mean, value, h, self.aux(z)

    def dist(self, mean):
        return torch.distributions.Normal(mean, self.log_std.exp())

    def n_params(self):
        return sum(p.numel() for p in self.parameters())
