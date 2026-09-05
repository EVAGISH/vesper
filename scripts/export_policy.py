"""Export a trained policy to ONNX for the drone's onboard hardware.

    .venv/bin/python scripts/export_policy.py runs/<id>/search.pt

Writes <checkpoint>.onnx: a single graph that takes the RAW observation vector and
returns the action -- the running-mean/var normalizer is folded in, so nothing on
the drone needs to know about it. The policy is a small MLP, so it runs in
microseconds on a Jetson (Orin Nano/NX). For max perf on Jetson, compile the ONNX
to a TensorRT engine on-device:

    trtexec --onnx=search.onnx --saveEngine=search.plan --fp16

This is the sim -> real-hardware step: the same weights trained in the reconstructed
world run on the aircraft's companion computer via ONNX Runtime / TensorRT.
"""
import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from vesper.lab.ppo import ActorCritic, RunningNorm

ap = argparse.ArgumentParser()
ap.add_argument("checkpoint")
ap.add_argument("--out", default=None)
a = ap.parse_args()

ck = torch.load(a.checkpoint, map_location="cpu")
obs_dim, act_dim = ck["obs_dim"], ck["act_dim"]
ac = ActorCritic(obs_dim, act_dim)
ac.load_state_dict(ck["ac"])
ac.eval()
norm = RunningNorm(obs_dim)
norm.load_state_dict(ck["norm"])
norm.eval()


class Deployable(nn.Module):
    """raw observation -> normalize -> actor -> action. Everything the drone runs."""
    def __init__(self, norm, actor):
        super().__init__()
        self.norm = norm
        self.actor = actor

    @torch.no_grad()
    def forward(self, obs):
        return self.actor(self.norm(obs))


model = Deployable(norm, ac.actor).eval()
out = Path(a.out or Path(a.checkpoint).with_suffix(".onnx"))
dummy = torch.zeros(1, obs_dim)

torch.onnx.export(
    model, dummy, str(out),
    input_names=["observation"], output_names=["action"],
    dynamic_axes={"observation": {0: "batch"}, "action": {0: "batch"}},
    opset_version=17, dynamo=False,
)

# verify torch vs onnxruntime agree
import onnxruntime as ort  # noqa: E402
x = torch.randn(4, obs_dim)
torch_out = model(x).numpy()
sess = ort.InferenceSession(str(out), providers=["CPUExecutionProvider"])
onnx_out = sess.run(["action"], {"observation": x.numpy()})[0]
err = float(np.abs(torch_out - onnx_out).max())

print(f"exported {out}  ({out.stat().st_size/1024:.0f} KB)")
print(f"  obs_dim={obs_dim} act_dim={act_dim}  torch↔onnx max diff {err:.2e}")
print(f"  runs via ONNX Runtime anywhere; on Jetson: trtexec --onnx={out.name} --saveEngine={out.stem}.plan --fp16")
