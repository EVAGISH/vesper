"""Generate a ground-vehicle USD: a rigid body that drives on terrain.

This is the offline fallback target for the pursuit task. The preferred target is
NVIDIA's stock forklift prop (see vesper.lab.pursuit_env.VEHICLE_SPECS), which is
a real modelled vehicle; this proxy exists so the task still builds and runs when
the Isaac asset server is unreachable, and so it stays cheap enough to clone
across thousands of environments -- it carries one box collider, not a mesh.

Written with plain pxr so it builds and is testable on the Mac, no Isaac needed.
The result is one PhysX rigid body (chassis collider + mass) carrying the visual
shapes of a small flatbed utility vehicle, so it collides with terrain and
obstacles, is pushed around by contacts, and can be driven by a velocity
controller in the env. Forward is +X.

    python3 -c "from vesper.worlds.vehicle import write_vehicle_usd; \
                write_vehicle_usd('assets/vehicles/utility_cart.usd')"
"""
from pathlib import Path

from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics, UsdShade


# flatbed utility vehicle (Kubota RTV / campus cart class), metres
CHASSIS = (3.40, 1.60, 0.34)     # length, width, height -- carries the collider
DECK = (1.70, 1.50, 0.12)        # rear flatbed
HOOD = (0.95, 1.35, 0.42)
ROOF = (1.45, 1.55, 0.07)
POST = (0.07, 0.07, 1.05)        # roll-cage uprights
WHEEL_R, WHEEL_W = 0.34, 0.26
WHEELBASE, TRACK_W = 2.10, 1.52
CHASSIS_Z = WHEEL_R + CHASSIS[2] / 2 - 0.06
MASS_KG = 1400.0


def _material(stage, path, rgb, rough=0.85):
    mat = UsdShade.Material.Define(stage, path)
    sh = UsdShade.Shader.Define(stage, f"{path}/Shader")
    sh.CreateIdAttr("UsdPreviewSurface")
    sh.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*rgb))
    sh.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(rough)
    sh.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(0.0)
    mat.CreateSurfaceOutput().ConnectToSource(sh.ConnectableAPI(), "surface")
    return mat


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


def write_vehicle_usd(out_path, paint=(0.82, 0.45, 0.10)) -> Path:
    """Write a drivable utility-vehicle rigid body. Default paint is safety orange."""
    out = Path(out_path); out.parent.mkdir(parents=True, exist_ok=True)
    stage = Usd.Stage.CreateNew(str(out)) if not out.exists() else Usd.Stage.Open(str(out))
    stage.RemovePrim("/Vehicle")
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)

    root = UsdGeom.Xform.Define(stage, "/Vehicle")
    stage.SetDefaultPrim(root.GetPrim())
    # one rigid body for the whole vehicle
    UsdPhysics.RigidBodyAPI.Apply(root.GetPrim())
    mass = UsdPhysics.MassAPI.Apply(root.GetPrim())
    mass.CreateMassAttr(MASS_KG)
    mass.CreateCenterOfMassAttr(Gf.Vec3f(0.0, 0.0, WHEEL_R))

    body = _material(stage, "/Vehicle/Looks/body", paint)
    dark = _material(stage, "/Vehicle/Looks/dark", (0.16, 0.17, 0.19))
    rubber = _material(stage, "/Vehicle/Looks/rubber", (0.07, 0.07, 0.075), rough=0.95)

    # Everything drawn is visual. The single collider is an invisible box that
    # spans wheel-bottom to chassis-top: putting it on the chassis instead left
    # the collider floor 0.28 m above the model origin, so the body rested with
    # its origin at -0.28 and the wheels buried in the terrain.
    _box(stage, "/Vehicle/chassis", CHASSIS, (0.0, 0.0, CHASSIS_Z), body)
    col_h = CHASSIS_Z + CHASSIS[2] / 2
    col = _box(stage, "/Vehicle/collision", (CHASSIS[0], TRACK_W, col_h),
               (0.0, 0.0, col_h / 2), body)
    UsdPhysics.CollisionAPI.Apply(col.GetPrim())
    UsdGeom.Imageable(col).CreateVisibilityAttr(UsdGeom.Tokens.invisible)

    deck_z = CHASSIS_Z + CHASSIS[2] / 2 + DECK[2] / 2
    _box(stage, "/Vehicle/deck", DECK, (-0.72, 0.0, deck_z), dark)
    _box(stage, "/Vehicle/hood", HOOD, (1.18, 0.0, CHASSIS_Z + CHASSIS[2] / 2 + HOOD[2] / 2), body)

    post_z = CHASSIS_Z + CHASSIS[2] / 2 + POST[2] / 2
    for px in (0.62, -0.34):
        for py in (TRACK_W / 2 - 0.14, -(TRACK_W / 2 - 0.14)):
            _box(stage, f"/Vehicle/post_{'f' if px > 0 else 'r'}{'l' if py > 0 else 'r'}",
                 POST, (px, py, post_z), dark)
    _box(stage, "/Vehicle/roof", ROOF, (0.14, 0.0, post_z + POST[2] / 2 + ROOF[2] / 2), body)

    for wx, tagx in ((WHEELBASE / 2, "f"), (-WHEELBASE / 2, "r")):
        for wy, tagy in ((TRACK_W / 2, "l"), (-TRACK_W / 2, "r")):
            _cyl(stage, f"/Vehicle/wheel_{tagx}{tagy}", WHEEL_R, WHEEL_W,
                 (wx, wy, WHEEL_R), rubber, axis="Y")
    stage.GetRootLayer().Save()
    return out


if __name__ == "__main__":
    import sys
    print(write_vehicle_usd(sys.argv[1] if len(sys.argv) > 1 else "assets/vehicles/utility_cart.usd"))
