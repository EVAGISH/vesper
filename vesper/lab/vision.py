"""The end-to-end vision policy: camera in, velocity command out.

Inputs are exactly what ChaseEnv hands the actor: the body-fixed camera as
RGB + depth (four channels at 96 px), and the airframe's own instruments
(vesper.lab.frames.proprio, 11 values). Nothing else reaches the actor.

Why this shape, and why a GRU (see the design notes in STEPS.md, Step 11):

  * a single frame says where a forklift is *now*; the task needs it to be
    remembered through the seconds it is out of frame (a bank, a tree
    crown, the final dive under the nose), needs range from how fast it grows
    across frames, and needs to know where it has already looked before the
    first sighting. That is state the network has to carry, so the trunk is
    recurrent. A GRU is the smallest thing that does it; frame stacking
    covers a fraction of a second and a transformer is the wrong size.
  * a 256-unit state holds a relative target vector, a range rate and a
    coarse heading history with room to spare.

Sized for the airframe: ~1.4M parameters, ~17M multiply-adds per frame.

  encoder   conv 4->32->48->64->96, stride 2 each, 96 px -> 6x6 -> 256
  proprio   11 -> 32
  memory    GRU(288 -> 256), carried across the episode, zeroed on reset
  actor     256 -> 128 -> 3 (mean) + learned log-std
  belief    256 -> 3: the true relative target vector (scaled), supervised
            whenever a forklift is in frame. Gives the recurrent state a
            reason to hold the target long before the touch reward can.
  critic    [256, privileged] -> 256 -> 1 (asymmetric: the critic may see
            the truth, the actor never does -- STACK.md section 4)

The pixel pre-processing (uint8 RGB and [0,1] depth -> four float channels
about zero) is part of the module so the deployed network and the trained
network see identical numbers.
"""
from __future__ import annotations

import torch
import torch.nn as nn


def conv_encoder(in_ch=4, widths=(32, 48, 64, 96), res=96, out_dim=256):
    layers, c = [], in_ch
    for w in widths:
        layers += [nn.Conv2d(c, w, 3, stride=2, padding=1), nn.ELU()]
        c = w
    side = res
    for _ in widths:
        side = (side + 1) // 2
    layers += [nn.Flatten(), nn.Linear(c * side * side, out_dim), nn.ELU()]
    return nn.Sequential(*layers)


class VisionActorCritic(nn.Module):
    def __init__(self, proprio_dim=11, act_dim=3, priv_dim=0, res=96, enc_dim=256,
                 hidden=256, init_std=0.5):
        super().__init__()
        self.res, self.hidden, self.priv_dim = int(res), int(hidden), int(priv_dim)
        self.enc = conv_encoder(res=res, out_dim=enc_dim)
        self.prop = nn.Sequential(nn.Linear(proprio_dim, 32), nn.ELU())
        self.gru = nn.GRUCell(enc_dim + 32, hidden)
        self.actor = nn.Sequential(nn.Linear(hidden, 128), nn.ELU(), nn.Linear(128, act_dim))
        self.belief = nn.Linear(hidden, 3)
        self.critic = nn.Sequential(nn.Linear(hidden + priv_dim, 256), nn.ELU(), nn.Linear(256, 1))
        self.log_std = nn.Parameter(torch.full((act_dim,), float(torch.log(torch.tensor(init_std)))))

    def initial_state(self, n, device=None):
        return torch.zeros(n, self.hidden, device=device)

    @staticmethod
    def prep(pixels, depth):
        """uint8 RGB [N,H,W,3] + depth [N,H,W,1] in [0,1] -> float [N,4,H,W] about zero."""
        rgb = pixels.permute(0, 3, 1, 2).float() / 127.5 - 1.0
        d = depth.permute(0, 3, 1, 2).float() * 2.0 - 1.0
        return torch.cat([rgb, d], dim=1)

    def features(self, pixels, depth, proprio):
        return torch.cat([self.enc(self.prep(pixels, depth)), self.prop(proprio)], dim=1)

    def forward(self, pixels, depth, proprio, h, priv=None, done=None):
        """One step. Returns (action mean, value, next h, belief).

        `done` [N] bool zeroes the memory of environments that just restarted,
        which is how the recurrent state stays inside one episode.
        """
        if done is not None:
            h = h * (~done).float().unsqueeze(1)
        h = self.gru(self.features(pixels, depth, proprio), h)
        return self.heads(h, priv) + (h,)

    def heads(self, h, priv=None):
        """(action mean, value, belief). On the drone there is no privileged
        vector and no critic to run: the value comes back as zeros rather than
        failing, since nothing downstream of the airframe reads it."""
        mean = self.actor(h)
        if priv is None and self.priv_dim:
            value = torch.zeros(h.shape[0], device=h.device, dtype=h.dtype)
        else:
            cin = h if priv is None else torch.cat([h, priv], dim=1)
            value = self.critic(cin).squeeze(-1)
        return mean, value, self.belief(h)

    @torch.no_grad()
    def act(self, pixels, depth, proprio, h, deterministic=True):
        """What runs on the drone: encoder, memory, actor. No critic, no belief."""
        h = self.gru(self.features(pixels, depth, proprio), h)
        mean = self.actor(h)
        return (mean if deterministic else self.dist(mean).sample()), h

    def dist(self, mean):
        return torch.distributions.Normal(mean, self.log_std.exp())

    def n_params(self, deployed=False):
        mods = [self.enc, self.prop, self.gru, self.actor] if deployed else [self]
        return sum(p.numel() for m in mods for p in m.parameters())

    def macs_per_frame(self):
        """Multiply-adds for one actor step, from the layer shapes (no critic)."""
        macs, side, c_in = 0, self.res, 4
        for m in self.enc:
            if isinstance(m, nn.Conv2d):
                side = (side + 1) // 2
                macs += side * side * m.out_channels * c_in * 9
                c_in = m.out_channels
            elif isinstance(m, nn.Linear):
                macs += m.in_features * m.out_features
        macs += self.prop[0].in_features * self.prop[0].out_features
        macs += 3 * (self.gru.input_size + self.gru.hidden_size) * self.gru.hidden_size
        for m in self.actor:
            if isinstance(m, nn.Linear):
                macs += m.in_features * m.out_features
        return macs
