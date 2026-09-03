"""Generate a ground-vehicle USD: a rigid body that drives on terrain.

Written with plain pxr so it builds and is testable on the Mac, no Isaac needed.
The result is one PhysX rigid body (hull collider + mass) carrying the visual
shapes of a tracked AFV, so it collides with terrain and obstacles, is pushed
around by contacts, and can be driven by a velocity controller in the env.

    python3 -c "from vesper.worlds.vehicle import write_tank_usd; \
                write_tank_usd('assets/vehicles/tank.usd')"
"""
from pathlib import Path

from pxr import Gf, Usd, UsdGeom, UsdPhysics, UsdShade


# rough T-72/M1 class proportions, metres
HULL = (7.0, 3.4, 1.15)          # length, width, height
TURRET = (3.6, 2.6, 0.85)
BARREL = (4.6, 0.13)             # length, radius
TRACK = (7.2, 0.62, 0.75)
MASS_KG = 42000.0


def _material(stage, path, rgb, rough=0.85):
    mat = UsdShade.Material.Define(stage, path)
    sh = UsdShade.Shader.Define(stage, f"{path}/Shader")
    sh.CreateIdAttr("UsdPreviewSurface")
    sh.CreateInput("diffuseColor", Sdf_Color3f()).Set(Gf.Vec3f(*rgb))
    sh.CreateInput("roughness", Sdf_Float()).Set(rough)
    sh.CreateInput("metallic", Sdf_Float()).Set(0.0)
    mat.CreateSurfaceOutput().ConnectToSource(sh.ConnectableAPI(), "surface")
    return mat


def Sdf_Color3f():
    from pxr import Sdf
    return Sdf.ValueTypeNames.Color3f


def Sdf_Float():
    from pxr import Sdf
    return Sdf.ValueTypeNames.Float


def _box(stage, path, size, translate, mat):
    c = UsdGeom.Cube.Define(stage, path)
    c.CreateSizeAttr(2.0)                      # unit cube of edge 2; scale to size
    x = UsdGeom.Xformable(c)
    x.AddTranslateOp().Set(Gf.Vec3d(*translate))
    x.AddScaleOp().Set(Gf.Vec3f(size[0] / 2, size[1] / 2, size[2] / 2))
    UsdShade.MaterialBindingAPI.Apply(c.GetPrim()).Bind(mat)
    return c


def _cyl(stage, path, radius, height, translate, mat, axis="X"):
    c = UsdGeom.Cylinder.Define(stage, path)
    c.CreateRadiusAttr(radius); c.CreateHeightAttr(height); c.CreateAxisAttr(axis)
    UsdGeom.Xformable(c).AddTranslateOp().Set(Gf.Vec3d(*translate))
    UsdShade.MaterialBindingAPI.Apply(c.GetPrim()).Bind(mat)
    return c


def write_tank_usd(out_path, olive=(0.26, 0.29, 0.18)) -> Path:
    out = Path(out_path); out.parent.mkdir(parents=True, exist_ok=True)
    stage = Usd.Stage.CreateNew(str(out)) if not out.exists() else Usd.Stage.Open(str(out))
    stage.RemovePrim("/Tank")
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)

    root = UsdGeom.Xform.Define(stage, "/Tank")
    stage.SetDefaultPrim(root.GetPrim())
    # one rigid body for the whole vehicle
    UsdPhysics.RigidBodyAPI.Apply(root.GetPrim())
    mass = UsdPhysics.MassAPI.Apply(root.GetPrim())
    mass.CreateMassAttr(MASS_KG)
    mass.CreateCenterOfMassAttr(Gf.Vec3f(0.0, 0.0, 0.35))

    dark = _material(stage, "/Tank/Looks/dark", (v * 0.72 for v in olive)) if False else \
        _material(stage, "/Tank/Looks/dark", tuple(v * 0.72 for v in olive))
    body = _material(stage, "/Tank/Looks/body", olive)
    rubber = _material(stage, "/Tank/Looks/track", (0.07, 0.07, 0.075), rough=0.95)

    # hull carries the collider; everything else is visual
    hull = _box(stage, "/Tank/hull", HULL, (0.0, 0.0, HULL[2] / 2 + 0.42), body)
    UsdPhysics.CollisionAPI.Apply(hull.GetPrim())
    _box(stage, "/Tank/turret", TURRET, (-0.25, 0.0, HULL[2] + 0.42 + TURRET[2] / 2), dark)
    _cyl(stage, "/Tank/barrel", BARREL[1], BARREL[0],
         (BARREL[0] / 2 + 1.0, 0.0, HULL[2] + 0.42 + TURRET[2] * 0.6), dark, axis="X")
    for sy, tag in ((TRACK[1] / 2 + HULL[1] / 2 - 0.30, "l"), (-(TRACK[1] / 2 + HULL[1] / 2 - 0.30), "r")):
        t = _box(stage, f"/Tank/track_{tag}", TRACK, (0.0, sy, TRACK[2] / 2), rubber)
        UsdPhysics.CollisionAPI.Apply(t.GetPrim())
    stage.GetRootLayer().Save()
    return out


if __name__ == "__main__":
    import sys
    print(write_tank_usd(sys.argv[1] if len(sys.argv) > 1 else "assets/vehicles/tank.usd"))
