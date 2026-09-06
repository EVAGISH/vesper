"""Write assets/<site>/ground.jpg: a 2048-px JPEG of the ground albedo for the web
map. The full ground.png (30-80 MB) stays on the build machine; the UI only needs this.

    python scripts/ground_preview.py assets/<site>
"""
import sys
from pathlib import Path

from PIL import Image

Image.MAX_IMAGE_PIXELS = None
d = Path(sys.argv[1])
src = d / "ground.png"
if not src.exists():
    raise SystemExit(f"no {src}")
img = Image.open(src).convert("RGB")
img.thumbnail((2048, 2048), Image.BILINEAR)
img.save(d / "ground.jpg", "JPEG", quality=82)
print(f"wrote {d / 'ground.jpg'} ({img.size[0]}px)")
