"""Download NVIDIA's public vegetation USDs (plus every USD, MDL and texture they pull in)
into assets/vegetation/Trees, for vesper.worlds.geo's tree prototypes.

    python3 scripts/fetch_vegetation.py            # the species listed in geo.SPECIES
    python3 scripts/fetch_vegetation.py Red_Oak    # extra species

Dependencies come from two places: USD composition (UsdUtils) and texture paths inside
the MDL files, which USD cannot see -- both are crawled.
"""
import os
import re
import sys
import urllib.request
from pathlib import Path

from pxr import UsdUtils

from vesper.worlds.geo import SPECIES

BUCKET = "https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Vegetation/Trees/"
ROOT = Path(__file__).resolve().parents[1] / "assets" / "vegetation" / "Trees"
TEX_RE = re.compile(r'"(\./)?([^"]+\.(?:png|jpg|jpeg|dds|exr|tga))"')


def fetch(rel: str) -> bool:
    dst = ROOT / rel
    if dst.exists():
        return False
    dst.parent.mkdir(parents=True, exist_ok=True)
    try:
        urllib.request.urlretrieve(BUCKET + rel, dst)
        return True
    except Exception as e:  # noqa: BLE001
        print(f"  miss {rel}: {e}")
        return False


def crawl(rel: str, seen: set) -> None:
    if rel in seen:
        return
    seen.add(rel)
    fetch(rel)
    path = ROOT / rel
    if not path.exists():
        return
    if rel.endswith((".usd", ".usda", ".usdc")):
        layers, refs, unresolved = UsdUtils.ComputeAllDependencies(str(path))
        base = path.parent
        for d in [str(l.identifier) for l in layers] + [str(r) for r in refs] + [str(u) for u in unresolved]:
            p = Path(d)
            sub = os.path.normpath(str(base.relative_to(ROOT) / d)) if not p.is_absolute() else \
                os.path.relpath(p, ROOT)
            if not sub.startswith(".."):
                crawl(sub, seen)
    elif rel.endswith(".mdl"):
        for _, tex in TEX_RE.findall(path.read_text(errors="ignore")):
            crawl(os.path.normpath(str(path.parent.relative_to(ROOT) / tex)), seen)


if __name__ == "__main__":
    os.environ.setdefault("PXR_USDC_EMIT_DEPRECATION_WARNINGS", "0")
    species = sys.argv[1:] or [s[0] for s in SPECIES]
    seen: set = set()
    for sp in species:
        before = len(seen)
        crawl(f"{sp}.usd", seen)
        print(f"{sp}: {len(seen) - before} files")
    total = sum(f.stat().st_size for f in ROOT.rglob("*") if f.is_file())
    print(f"{ROOT}: {sum(1 for f in ROOT.rglob('*') if f.is_file())} files, {total / 1e6:.0f} MB")
