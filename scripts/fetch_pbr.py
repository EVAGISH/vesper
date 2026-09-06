"""Download CC0 scanned PBR material sets from ambientCG into textures/pbr/<name>/.

    uv run python scripts/fetch_pbr.py

Each set is 2K JPEG: Color, NormalGL, Roughness, AmbientOcclusion (when present),
renamed to color.jpg / normal.jpg / rough.jpg / ao.jpg. `vesper.worlds.buildings`
maps them onto walls and roofs as full UsdPreviewSurface materials, which Unreal's USD
importer turns into real PBR materials.
"""
import io
import zipfile
from pathlib import Path

import requests

OUT = Path(__file__).resolve().parents[1] / "textures" / "pbr"
# our material name -> ambientCG asset id
SETS = {
    "roof_flat": "Asphalt033",             # dark charcoal, reads as rolled bitumen
    "roof_slate": "CorrugatedSteel009",    # dark grey corrugated: asbestos-slate look
    "roof_metal": "CorrugatedSteel006A",   # green paint with rust
    "brick": "Bricks075A",                 # beige/yellow silicate brick (Donbas houses)
    "brick_red": "Bricks085",              # red factory brick (garages, sheds)
    "plaster": "PaintedPlaster017",        # white painted plaster
    "concrete": "Concrete036",             # old dark grey concrete (panel joints detail)
    "industrial_wall": "CorrugatedSteel007A",
    "asphalt": "Road012A",
    "grass": "Grass004",
    "dirt": "Ground110",
}
NAMES = {"Color": "color", "NormalGL": "normal", "Roughness": "rough", "AmbientOcclusion": "ao"}


def fetch(name: str, asset: str) -> None:
    d = OUT / name
    if (d / "color.jpg").exists():
        print(f"{name}: cached"); return
    d.mkdir(parents=True, exist_ok=True)
    url = f"https://ambientcg.com/get?file={asset}_2K-JPG.zip"
    r = requests.get(url, timeout=300, headers={"User-Agent": "dig-twin/0.2"}); r.raise_for_status()
    with zipfile.ZipFile(io.BytesIO(r.content)) as z:
        for member in z.namelist():
            for key, short in NAMES.items():
                if member.endswith(f"_{key}.jpg"):
                    (d / f"{short}.jpg").write_bytes(z.read(member))
    (d / "SOURCE.txt").write_text(f"{asset} from ambientCG.com, CC0\n")
    print(f"{name}: {asset} -> {sorted(p.name for p in d.glob('*.jpg'))}")


if __name__ == "__main__":
    for name, asset in SETS.items():
        try:
            fetch(name, asset)
        except Exception as exc:  # noqa: BLE001
            print(f"{name}: FAILED {exc}")
