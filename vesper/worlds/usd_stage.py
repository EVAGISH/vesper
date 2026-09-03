"""Instantiate a spec's buildings as static collidable prisms (container only).

Spec frame is (north, east); Isaac world is ENU: north -> +y, east -> +x.
"""
import numpy as np


def add_terrain(spec, prim_path: str = "/World/terrain") -> None:
    """Reference spec.terrain["usd"] into the stage with its placement xform.
    The USD is expected to carry its own collision (see scripts/convert_asset.py)."""
    if not spec.terrain:
        return
    from pathlib import Path

    import omni.usd
    from isaacsim.core.utils.stage import add_reference_to_stage
    from pxr import Gf, UsdGeom

    t = spec.terrain
    usd_path = Path(t["usd"])
    if not usd_path.is_absolute():
        usd_path = Path(__file__).resolve().parents[2] / usd_path
    add_reference_to_stage(str(usd_path), prim_path)
    prim = omni.usd.get_context().get_stage().GetPrimAtPath(prim_path)
    xf = UsdGeom.XformCommonAPI(prim)
    tr = t.get("translation", [0, 0, 0]); rot = t.get("rotation_xyz_deg", [0, 0, 0]); sc = t.get("scale", 1.0)
    xf.SetTranslate(Gf.Vec3d(*tr))
    xf.SetRotate(Gf.Vec3f(*rot))
    xf.SetScale(Gf.Vec3f(sc, sc, sc))


def build_world(world, spec) -> None:
    from isaacsim.core.api.objects import FixedCuboid

    add_terrain(spec)
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
