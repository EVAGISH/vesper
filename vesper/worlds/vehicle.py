"""Generate Vesper's tank USD: a rigid body that drives on terrain.

The model is authored here so the simulation never depends on a remote asset
server. It stays cheap enough to clone across thousands of environments: the
tracks, road wheels, hull, turret and barrel are visual geometry while one box
collider represents the complete vehicle in PhysX.

Written with plain pxr so it builds and is testable on the Mac, no Isaac needed.
The result is one PhysX rigid body (chassis collider + mass) carrying the visual
shapes of a tracked tank, so it collides with terrain and
obstacles, is pushed around by contacts, and can be driven by a velocity
controller in the env. Forward is +X.

    python3 -c "from vesper.worlds.vehicle import write_tank_usd; \
                write_tank_usd('assets/vehicles/tank.usd')"
"""
from pathlib import Path

from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics, UsdShade


# Compact modern tank, metres. Forward is +X.
HULL = (6.20, 3.35, 0.82)
UPPER_HULL = (4.65, 2.85, 0.58)
TRACK = (5.55, 0.46, 0.72)
TURRET = (2.45, 2.25, 0.62)
BARREL_LENGTH, BARREL_R = 3.45, 0.105
ROAD_WHEEL_R, ROAD_WHEEL_W = 0.34, 0.12
TRACK_CENTRE_Y = 1.48
HULL_Z = 0.78
MASS_KG = 38_000.0


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


def write_tank_usd(out_path, paint=(0.24, 0.31, 0.18)) -> Path:
    """Write the custom drivable tank rigid body used by the training tasks."""
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
    mass.CreateCenterOfMassAttr(Gf.Vec3f(0.0, 0.0, HULL_Z))

    body = _material(stage, "/Vehicle/Looks/body", paint, rough=0.9)
    dark = _material(stage, "/Vehicle/Looks/dark", (0.09, 0.105, 0.075), rough=0.94)
    metal = _material(stage, "/Vehicle/Looks/metal", (0.18, 0.20, 0.16), rough=0.72)
    rubber = _material(stage, "/Vehicle/Looks/rubber", (0.035, 0.04, 0.03), rough=0.98)

    # Everything drawn is visual. One invisible collider spans the tracks and
    # hull, keeping contact simulation cheap while putting the track bottoms at
    # z=0 so placement on sampled terrain is predictable.
    _box(stage, "/Vehicle/lower_hull", HULL, (0.0, 0.0, HULL_Z), body)
    col_h = HULL_Z + HULL[2] / 2
    col = _box(stage, "/Vehicle/collision", (HULL[0], HULL[1], col_h),
               (0.0, 0.0, col_h / 2), body)
    UsdPhysics.CollisionAPI.Apply(col.GetPrim())
    UsdGeom.Imageable(col).CreateVisibilityAttr(UsdGeom.Tokens.invisible)

    upper_z = HULL_Z + HULL[2] / 2 + UPPER_HULL[2] / 2
    _box(stage, "/Vehicle/upper_hull", UPPER_HULL, (-0.18, 0.0, upper_z), body)

    for side, y in (("left", TRACK_CENTRE_Y), ("right", -TRACK_CENTRE_Y)):
        _box(stage, f"/Vehicle/track_{side}", TRACK, (0.0, y, TRACK[2] / 2), dark)
        for i, x in enumerate((-2.15, -1.30, -0.43, 0.43, 1.30, 2.15)):
            _cyl(stage, f"/Vehicle/road_wheel_{side}_{i}", ROAD_WHEEL_R, ROAD_WHEEL_W,
                 (x, y, ROAD_WHEEL_R + 0.04), rubber, axis="Y")
            _cyl(stage, f"/Vehicle/hub_{side}_{i}", ROAD_WHEEL_R * 0.42, ROAD_WHEEL_W + 0.03,
                 (x, y, ROAD_WHEEL_R + 0.04), metal, axis="Y")

    turret_z = upper_z + UPPER_HULL[2] / 2 + TURRET[2] / 2
    _box(stage, "/Vehicle/turret", TURRET, (0.25, 0.0, turret_z), body)
    _cyl(stage, "/Vehicle/commander_hatch", 0.38, 0.16,
         (0.08, 0.45, turret_z + TURRET[2] / 2 + 0.08), dark, axis="Z")
    barrel_x = 0.25 + TURRET[0] / 2 + BARREL_LENGTH / 2
    barrel_z = turret_z + 0.08
    _cyl(stage, "/Vehicle/barrel", BARREL_R, BARREL_LENGTH,
         (barrel_x, 0.0, barrel_z), metal, axis="X")
    _cyl(stage, "/Vehicle/muzzle", BARREL_R * 1.45, 0.28,
         (barrel_x + BARREL_LENGTH / 2 - 0.06, 0.0, barrel_z), dark, axis="X")
    stage.GetRootLayer().Save()
    return out


def write_vehicle_usd(out_path, paint=(0.24, 0.31, 0.18)) -> Path:
    """Compatibility wrapper for callers of the former proxy generator."""
    return write_tank_usd(out_path, paint=paint)


if __name__ == "__main__":
    import sys
    print(write_tank_usd(sys.argv[1] if len(sys.argv) > 1 else "assets/vehicles/tank.usd"))
