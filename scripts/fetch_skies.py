"""Download NVIDIA's public sky HDRs into assets/skies.

    python3 scripts/fetch_skies.py              # every outdoor sky (~1.5 GB)
    python3 scripts/fetch_skies.py --group Clear Cloudy

The synthetic-detection dataset randomises the dome light per frame, and one
HDR (noon_grass) means one sun angle, one colour temperature and one sky
gradient across every image -- a detector trained on that learns the lighting as
much as the vehicle. Indoor/Studio domes are skipped: they light a vehicle from
inside a room and look nothing like anything the drone will fly under.
"""
import argparse
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path

BUCKET = "https://omniverse-content-production.s3-us-west-2.amazonaws.com"
PREFIX = "Assets/Skies/2022_1/Skies/"
OUTDOOR = ("Clear", "Cloudy", "Evening", "Night", "Storm")
ROOT = Path(__file__).resolve().parents[1] / "assets" / "skies"


def listing():
    url = f"{BUCKET}/?list-type=2&prefix={PREFIX}&max-keys=1000"
    with urllib.request.urlopen(url, timeout=60) as r:
        tree = ET.fromstring(r.read())
    ns = {"s3": "http://s3.amazonaws.com/doc/2006-03-01/"}
    return [k.text for k in tree.findall(".//s3:Contents/s3:Key", ns) if k.text.endswith(".hdr")]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--group", nargs="*", default=list(OUTDOOR),
                    help=f"sky groups to fetch (default: {' '.join(OUTDOOR)})")
    args = ap.parse_args()

    ROOT.mkdir(parents=True, exist_ok=True)
    keys = [k for k in listing()
            if k[len(PREFIX):].split("/")[0] in args.group and "/materials/" not in k]
    print(f"{len(keys)} skies in {', '.join(args.group)}")
    for key in keys:
        group, name = key[len(PREFIX):].split("/")[0], Path(key).name
        out = ROOT / f"{group.lower()}_{name}"
        if out.exists():
            print(f"  have {out.name}")
            continue
        print(f"  fetching {out.name}", flush=True)
        urllib.request.urlretrieve(f"{BUCKET}/{key}", out)
    total = sum(p.stat().st_size for p in ROOT.glob("*.hdr"))
    print(f"{len(list(ROOT.glob('*.hdr')))} skies in {ROOT} ({total / 1e6:.0f} MB)")


if __name__ == "__main__":
    main()
