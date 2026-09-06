"""Write a lighter variant of a built world by deactivating one tree species.

    python3 scripts/thin_world_trees.py assets/vuhledar/vuhledar.usd \
        --species Norway_Spruce --out assets/vuhledar/vuhledar_light.usd

Why this exists: Isaac's RTX loader dies with cudaErrorIllegalAddress on worlds
that hand it too many instanced prims, and the count is dominated by a single
species. Norway_Spruce's prototype is 994 prims per tree against roughly nine
for every other species, so on a big site its instances are ~98% of the scene
graph -- Vuhledar's 8,814 spruces are 8.76M of its 8.92M prims.

This deactivates those instances in a **new sublayer**. The original world USD is
opened read-only and never modified, so the change is reversible by deleting one
file, and a site can be re-thinned differently without a rebuild. The real fix
is to drop or flatten the species in vesper.worlds.geo SPECIES and rebuild; this
is the version that does not require refetching a world.
"""
import argparse
import os
from pathlib import Path

from pxr import Sdf, Usd

ap = argparse.ArgumentParser()
ap.add_argument("world", help="path to <site>.usd")
ap.add_argument("--species", nargs="+", default=["Norway_Spruce"],
                help="species whose instances get deactivated")
ap.add_argument("--out", default=None, help="default: <site>_light.usd beside the source")
args = ap.parse_args()

src_path = Path(args.world).resolve()
out_path = Path(args.out).resolve() if args.out else src_path.with_name(f"{src_path.stem}_light.usd")

src = Sdf.Layer.FindOrOpen(str(src_path))
if src is None:
    raise SystemExit(f"cannot open {src_path}")


def species_of(spec) -> str | None:
    """The asset a prim spec references, by filename stem."""
    refs = spec.referenceList
    for items in (refs.prependedItems, refs.explicitItems, refs.appendedItems, refs.addedItems):
        for ref in items:
            if ref.assetPath:
                return Path(ref.assetPath).stem
    return None


targets, counts = [], {}
def walk(spec):
    for child in spec.nameChildren:
        name = species_of(child)
        if name:
            counts[name] = counts.get(name, 0) + 1
            if name in args.species:
                targets.append(child.path)
        else:
            walk(child)


walk(src.pseudoRoot)
print(f"{src_path.name}: " + ", ".join(f"{k}={v}" for k, v in sorted(counts.items())))
if not targets:
    raise SystemExit(f"no instances of {args.species} found -- nothing to thin")

if out_path.exists():
    out_path.unlink()
out = Sdf.Layer.CreateNew(str(out_path))
# A sublayer, not a reference: it keeps every prim path identical to the source,
# so exported maps, zones and prim-path assumptions elsewhere still line up.
# Relative, computed from wherever --out puts us: the light layer is usually
# written into its own site directory so scripts that discover worlds by the
# assets/<name>/<name>.usd convention pick it up as a site of its own.
out.subLayerPaths.append(os.path.relpath(src_path, out_path.parent))
# defaultPrim is root-layer-only metadata and is NOT inherited from a sublayer.
# Without copying it, anything that references this file by asset path alone
# resolves <defaultPrim> to nothing and composes an empty site, silently.
out.defaultPrim = src.defaultPrim
stage = Usd.Stage.Open(out)
for path in targets:
    stage.GetPrimAtPath(path).SetActive(False)
stage.GetRootLayer().Save()

kept = sum(v for k, v in counts.items() if k not in args.species)
print(f"deactivated {len(targets)} {'/'.join(args.species)} instances, kept {kept} trees")
print(f"wrote {out_path}")
