"""CPU test: synthetic DEM + hand-written OSM elements -> USD site with terrain, buildings,
water, trees, spawn and a canopy-safe loop. Uses one real NVIDIA tree asset if present,
otherwise a stand-in cube tree, so the test runs without the 190 MB vegetation download."""
import json
from pathlib import Path

import numpy as np
import pytest
from pxr import Usd, UsdGeom, UsdPhysics

from vesper.worlds import geo
from vesper.worlds.geo import GeoSite, build_site
from vesper.worlds.prepare import ground_height

LAT, LON = 48.845, 37.635


def _ll(x, y):
    return {"lat": LAT + y / 110574.0, "lon": LON + x / (111320.0 * np.cos(np.radians(LAT)))}


def _ring(pts):
    return [_ll(x, y) for x, y in pts] + [_ll(*pts[0])]


def _make_inputs(d: Path):
    # 30 m DEM: gentle slope rising to the east, 100 m base
    cols, rows = 20, 20
    a = 30 / (111320.0 * np.cos(np.radians(LAT))); e = -30 / 110574.0
    c = LON - a * cols / 2; f = LAT - e * rows / 2
    col = np.arange(cols)[None, :]; dem = (100 + 0.2 * (col - cols / 2) * 30 * np.ones((rows, 1))).astype(np.float32)
    np.save(d / "dem.npy", dem)
    (d / "dem_meta.json").write_text(json.dumps({"transform": [a, 0, c, 0, e, f]}))
    els = [
        {"type": "way", "id": 1, "tags": {"building": "house", "building:levels": "2"}, "geometry": _ring([(-60, 40), (-50, 40), (-50, 52), (-60, 52)])},
        {"type": "way", "id": 2, "tags": {"building": "industrial"}, "geometry": _ring([(-120, -120), (-80, -120), (-80, -90), (-120, -90)])},
        {"type": "way", "id": 3, "tags": {"highway": "residential"}, "geometry": [_ll(-200, 0), _ll(200, 0)]},
        {"type": "way", "id": 4, "tags": {"natural": "wood"}, "geometry": _ring([(60, 60), (200, 60), (200, 200), (60, 200)])},
        {"type": "way", "id": 5, "tags": {"natural": "water"}, "geometry": _ring([(-200, -200), (-150, -200), (-150, -160), (-200, -160)])},
        {"type": "way", "id": 6, "tags": {"landuse": "residential"}, "geometry": _ring([(-150, 20), (-20, 20), (-20, 120), (-150, 120)])},
        {"type": "way", "id": 7, "tags": {"natural": "tree_row"}, "geometry": [_ll(-100, -50), _ll(0, -50)]},
    ]
    (d / "osm.json").write_text(json.dumps({"elements": els}))


def _veg_dir(tmp_path: Path) -> Path:
    real = Path(__file__).resolve().parents[1] / "assets" / "vegetation"
    if (real / "Trees" / "Yellow_Pine.usd").exists():
        return real
    veg = tmp_path / "vegetation"; (veg / "Trees").mkdir(parents=True)
    for name, _, _, _ in geo.SPECIES:
        st = Usd.Stage.CreateNew(str(veg / "Trees" / f"{name}.usd"))
        UsdGeom.SetStageMetersPerUnit(st, 0.01); UsdGeom.SetStageUpAxis(st, UsdGeom.Tokens.z)
        root = UsdGeom.Xform.Define(st, "/Root"); st.SetDefaultPrim(root.GetPrim())
        m = UsdGeom.Mesh.Define(st, "/Root/trunk")
        m.CreatePointsAttr([(-50, -50, 0), (50, -50, 0), (50, 50, 0), (-50, 50, 0), (0, 0, 1000)])
        m.CreateFaceVertexCountsAttr([4, 3]); m.CreateFaceVertexIndicesAttr([0, 1, 2, 3, 0, 1, 4])
        st.GetRootLayer().Save()
    return veg


def test_build_site(tmp_path):
    data = tmp_path / "site"; data.mkdir(); _make_inputs(data)
    site = GeoSite(LAT, LON, half_m=250.0, res_m=10.0, tex_px=512, seed=1)
    rep = build_site(site, data, _veg_dir(tmp_path), data / "site.usd")
    assert rep.terrain_verts == 51 * 51 and rep.buildings == 2 and rep.water == 1 and rep.trees > 100
    st = Usd.Stage.Open(rep.usd)
    terr = st.GetPrimAtPath("/World/terrain"); bld = st.GetPrimAtPath("/World/buildings")
    assert terr.HasAPI(UsdPhysics.CollisionAPI) and bld.HasAPI(UsdPhysics.CollisionAPI)
    assert st.GetPrimAtPath("/World/trees").IsA(UsdGeom.PointInstancer)
    assert (data / "ground.png").exists() and (data / "facade_0.png").exists()
    # terrain: origin is z=0 by construction, and it rises to the east
    assert abs(ground_height(rep.usd, 0.0, 0.0)) < 0.5
    assert ground_height(rep.usd, 200.0, 0.0) > ground_height(rep.usd, -200.0, 0.0) + 10
    # spawn is open ground: not on the road, not inside the wood, near it
    sx, sy = rep.spawn_xy
    assert abs(sy) > 10 and not (60 < sx < 200 and 60 < sy < 200)
    # loop altitudes clear the 22 m canopy over the wood with margin, all >= min alt
    assert all(w[2] >= 25.0 for w in rep.waypoints) and rep.takeoff_alt_m >= 25.0
    # scenario round-trip
    from vesper.scenario.spec import imported_scenario
    spec = imported_scenario(rep.usd, tuple(rep.spawn_xy), rep.spawn_ground_z)
    assert spec.terrain["translation"][2] == -rep.spawn_ground_z
