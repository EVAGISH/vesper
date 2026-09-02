"""Instantiate a spec's buildings as static collidable prisms (container only).

Spec frame is (north, east); Isaac world is ENU: north -> +y, east -> +x.
"""
import numpy as np


def build_world(world, spec) -> None:
    from isaacsim.core.api.objects import FixedCuboid

    rng = np.random.default_rng(spec.seed)
    for i, b in enumerate(spec.buildings):
        (n, e), (dn, de), h = b["center"], b["size"], b["height"]
        shade = rng.uniform(0.35, 0.75)
        world.scene.add(FixedCuboid(
            prim_path=f"/World/buildings/b{i:03d}", name=f"building_{i}",
            position=np.array([e, n, h / 2.0]),   # east -> x, north -> y
            scale=np.array([de, dn, h]),
            color=np.array([shade, shade * 0.95, shade * 0.9]),
        ))
