"""Small deterministic explosion compositor for simulation capture streams."""
from __future__ import annotations

import math

import numpy as np
from PIL import Image, ImageDraw, ImageFilter


def explosion_frame(rgb: np.ndarray, center: tuple[float, float] | None, phase: float,
                    radius_px: float = 110.0) -> np.ndarray:
    """Overlay one frame of a fireball, shock ring, sparks and smoke.

    ``phase`` runs from 0 to 1. The effect is deterministic, which keeps capture
    replays stable and avoids touching simulation RNG state.
    """
    base = Image.fromarray(np.asarray(rgb, dtype=np.uint8)[..., :3]).convert("RGBA")
    w, h = base.size
    cx, cy = center if center is not None else (w / 2, h / 2)
    phase = min(max(float(phase), 0.0), 1.0)

    # A hot flash expands quickly, then gives way to a darker smoke ball.
    grow = 1.0 - (1.0 - min(phase / 0.58, 1.0)) ** 2
    fire_r = radius_px * (0.10 + 0.90 * grow)
    fade = max(0.0, 1.0 - phase)
    glow = Image.new("RGBA", base.size, (0, 0, 0, 0))
    gd = ImageDraw.Draw(glow, "RGBA")
    for scale, color in (
        (1.45, (255, 75, 8, int(100 * fade))),
        (1.00, (255, 145, 18, int(210 * fade))),
        (0.58, (255, 225, 105, int(245 * fade))),
        (0.22, (255, 255, 235, int(255 * fade))),
    ):
        r = fire_r * scale
        gd.ellipse((cx - r, cy - r, cx + r, cy + r), fill=color)
    glow = glow.filter(ImageFilter.GaussianBlur(max(2.0, fire_r * 0.13)))
    base = Image.alpha_composite(base, glow)

    detail = Image.new("RGBA", base.size, (0, 0, 0, 0))
    dd = ImageDraw.Draw(detail, "RGBA")
    # Expanding shock ring is strongest at the beginning of the blast.
    ring_r = radius_px * (0.25 + 1.35 * phase)
    ring_alpha = int(230 * max(0.0, 1.0 - phase / 0.72))
    if ring_alpha:
        dd.ellipse((cx - ring_r, cy - ring_r, cx + ring_r, cy + ring_r),
                   outline=(255, 225, 150, ring_alpha), width=max(2, int(radius_px * 0.045)))

    # Fixed-angle debris reads clearly even in low-resolution policy footage.
    spark_alpha = int(255 * max(0.0, 1.0 - phase / 0.82))
    for i in range(18):
        a = i * 2.399963 + 0.37
        speed = 0.65 + 0.35 * ((i * 7) % 11) / 10.0
        d0 = fire_r * 0.35
        d1 = radius_px * (0.45 + 1.6 * phase) * speed
        x0, y0 = cx + math.cos(a) * d0, cy + math.sin(a) * d0
        x1, y1 = cx + math.cos(a) * d1, cy + math.sin(a) * d1 + radius_px * phase * phase * 0.35
        dd.line((x0, y0, x1, y1), fill=(255, 180 + (i % 3) * 25, 70, spark_alpha),
                width=max(1, int(radius_px * 0.018)))

    if phase > 0.35:
        smoke = (phase - 0.35) / 0.65
        smoke_r = radius_px * (0.45 + smoke * 0.85)
        for i in range(7):
            a = i * 2.1
            ox = math.cos(a) * smoke_r * 0.28
            oy = math.sin(a) * smoke_r * 0.18 - smoke * radius_px * 0.35
            rr = smoke_r * (0.28 + 0.05 * (i % 3))
            dd.ellipse((cx + ox - rr, cy + oy - rr, cx + ox + rr, cy + oy + rr),
                       fill=(35, 38, 34, int(145 * smoke * (1.0 - 0.45 * phase))))

    return np.asarray(Image.alpha_composite(base, detail).convert("RGB"))
