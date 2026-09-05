"""replay.json -> cinematic tactical mp4, the impressive per-run playback.

Renders the same story the live tactical console tells -- satellite basemap,
coverage sweeping open behind the swarm, the lead's sensor footprint, targets
flashing to DETECTED and then NEUTRALIZED, a mission HUD -- from the after-action
log a run carries (vesper.native.replay). Server-side PIL/numpy into system
ffmpeg, so it runs on the Mac with no browser and no extra deps, and drops
<run>/tactical.mp4 which the Runs tab already renders.

    .venv/bin/python scripts/render_replay.py runs/<id> [--world kramatorsk]

Photoreal (Isaac RTX) replay is a separate track that reads the same log.
"""
import argparse
import json
import subprocess
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

Image.MAX_IMAGE_PIXELS = None
ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / "assets"

ACCENT = (12, 163, 12)
RED = (208, 59, 59)
ORANGE = (232, 140, 34)
CW, CH = 1280, 720          # output resolution
FPS = 24
STRIKE = FPS                # strike animation length: ~1 s of output frames
LINGER = int(FPS * 2.5)     # camera holds on the impact point this long
OW = 1200                   # working ortho resolution
WMx = 360.0                 # half-window in metres (x); y scales by aspect


def load_font(sz, bold=False):
    for p in ("/System/Library/Fonts/SFNSMono.ttf",
              "/System/Library/Fonts/Menlo.ttc",
              "/Library/Fonts/Arial.ttf"):
        try:
            return ImageFont.truetype(p, sz)
        except OSError:
            continue
    return ImageFont.load_default()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("--world", default=None, help="override world (default: from replay.json)")
    ap.add_argument("--stride", type=int, default=2, help="use every Nth logged frame")
    ap.add_argument("--caption", default=None,
                    help="extra HUD line under the title (e.g. 'ITER 0200 / 0800')")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    run = Path(args.run_dir)
    rep = json.loads((run / "replay.json").read_text())
    world = args.world or rep["world"]
    half = float(rep["half_m"])
    frames = rep["frames"][:: max(1, args.stride)]
    K = rep["targets"]

    # the kill often ends the episode within a few log steps; hold the final
    # frame long enough for the strike animation + callout to play out
    prev, last_edge = [0] * K, None
    for si, f in enumerate(frames):
        for i, t in enumerate(f["tg"]):
            if t[3] and not prev[i]:
                last_edge = si
            prev[i] = t[3]
    if last_edge is not None:
        hold = max(0, STRIKE + FPS - (len(frames) - 1 - last_edge))
        frames = frames + [frames[-1]] * hold

    # --- basemap: the ground ortho, downscaled once, darkened for the veil
    gp = ASSETS / world / "ground.png"
    if gp.exists():
        base = Image.open(gp).convert("RGB").resize((OW, OW), Image.BILINEAR)
        ortho = np.asarray(base, np.float32) / 255.0
    else:
        ortho = np.full((OW, OW, 3), 0.18, np.float32)
    # cool tactical grade
    tint = np.array([0.82, 0.9, 1.0], np.float32)
    ortho = np.clip(ortho * 0.62 * tint, 0, 1)
    dark = ortho * 0.42
    bright = np.clip(ortho * 1.05, 0, 1)
    green = np.array([0.05, 0.55, 0.16], np.float32)

    sc = OW / (2 * half)                      # ortho px per metre
    def op(x, y):                             # world -> ortho px
        return (x + half) * sc, (half - y) * sc

    cov = np.zeros((OW, OW), np.float32)      # coverage alpha, grows over the run
    yy, xx = np.mgrid[0:OW, 0:OW]

    font = load_font(15)
    fontS = load_font(12)
    fontB = load_font(19)
    fontHUD = load_font(30)

    WMy = WMx * CH / CW
    # eased camera + detection/strike state
    cx = frames[0]["d"][0][0]
    cy = frames[0]["d"][0][1]
    detect_t = {}                             # target idx -> first-seen frame index
    strike_t = {}                             # target idx -> frame index reached flipped true
    strike_focus = None                       # (x, y, frame) the camera lingers on

    ff = subprocess.Popen(
        ["ffmpeg", "-y", "-f", "rawvideo", "-pixel_format", "rgb24",
         "-video_size", f"{CW}x{CH}", "-framerate", str(FPS), "-i", "-",
         "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20",
         "-preset", "medium", "-movflags", "+faststart",
         str(Path(args.out) if args.out else run / "tactical.mp4")],
        stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    def w2c(x, y):                            # world -> canvas px (follow window)
        return ((x - (cx - WMx)) / (2 * WMx) * CW,
                ((cy + WMy) - y) / (2 * WMy) * CH)

    for fi, fr in enumerate(frames):
        drones = fr["d"]
        hdgs = fr.get("hdg", [0.0] * len(drones))
        lead = drones[0]
        agl = fr["agl"]
        # rising edges first, so the camera can react this same frame
        for i in range(K):
            tx, ty, known, reached = fr["tg"][i]
            if known and i not in detect_t:
                detect_t[i] = fi
            if reached and i not in strike_t:
                strike_t[i] = fi
                strike_focus = (tx, ty, fi)
        # ease camera toward the lead -- unless a strike just landed, then
        # linger on the impact point (same focus-pull the detect ring gets)
        if strike_focus and fi - strike_focus[2] < LINGER:
            cx += (strike_focus[0] - cx) * 0.14
            cy += (strike_focus[1] - cy) * 0.14
        else:
            cx += (lead[0] - cx) * 0.10
            cy += (lead[1] - cy) * 0.10

        # stamp coverage: a disc ahead-and-below each drone's camera
        for d, h in zip(drones, hdgs):
            r_m = float(np.clip(d[2] * 0.8, 18, 120))
            fx = d[0] + np.cos(h) * d[2] * 0.6
            fy = d[1] + np.sin(h) * d[2] * 0.6
            ox, oy = op(fx, fy)
            rp = r_m * sc
            x0, x1 = int(max(0, ox - rp)), int(min(OW, ox + rp + 1))
            y0, y1 = int(max(0, oy - rp)), int(min(OW, oy + rp + 1))
            if x0 < x1 and y0 < y1:
                sub = (xx[y0:y1, x0:x1] - ox) ** 2 + (yy[y0:y1, x0:x1] - oy) ** 2
                cov[y0:y1, x0:x1] = np.maximum(cov[y0:y1, x0:x1],
                                               np.clip(1 - sub / (rp * rp), 0, 1))

        # composite display ortho: dark veil, brightened + green where swept
        a = np.clip(cov, 0, 1)[..., None]
        disp = dark * (1 - a) + bright * a + green * (a * 0.30)
        disp = np.clip(disp * 255, 0, 255).astype(np.uint8)

        # crop the follow window and scale to the canvas
        l, t = op(cx - WMx, cy + WMy)
        r, b = op(cx + WMx, cy - WMy)
        win = Image.fromarray(disp).crop((int(l), int(t), int(r), int(b))).resize((CW, CH), Image.BILINEAR)
        img = win.convert("RGB")
        g = ImageDraw.Draw(img, "RGBA")

        # sensor beam + footprint from the lead
        h = hdgs[0]
        r_m = float(np.clip(agl * 0.8, 18, 120))
        fx = lead[0] + np.cos(h) * agl * 0.6
        fy = lead[1] + np.sin(h) * agl * 0.6
        lx, ly = w2c(lead[0], lead[1])
        fcx, fcy = w2c(fx, fy)
        rpx = r_m / (2 * WMx) * CW
        g.polygon([(lx, ly), (fcx - rpx * 0.5, fcy - rpx * 0.5),
                   (fcx + rpx * 0.5, fcy - rpx * 0.5)], fill=(12, 163, 12, 30))
        g.ellipse([fcx - rpx, fcy - rpx, fcx + rpx, fcy + rpx], outline=(12, 163, 12, 150), width=2)

        # swarm markers
        for d in drones[1:]:
            dx, dy = w2c(d[0], d[1])
            g.ellipse([dx - 3, dy - 3, dx + 3, dy + 3], fill=(150, 210, 200, 210))

        # targets: hidden until detected, then labelled
        for i in range(K):
            tx, ty, known, reached = fr["tg"][i]
            sx, sy = w2c(tx, ty)
            if reached:
                age = fi - strike_t[i]
                _wreck(g, sx, sy, fi)
                if age < STRIKE:
                    _strike(g, sx, sy, age, i)
                _label(g, sx, sy, f"TGT-{i+1:02d} · TANK", "NEUTRALIZED", ORANGE, fontS,
                       diamond=True)
            elif known:
                col = ACCENT
                pulse = max(0, 18 - (fi - detect_t[i]))       # expanding ring on first sight
                if pulse:
                    g.ellipse([sx - 8 - pulse, sy - 8 - pulse, sx + 8 + pulse, sy + 8 + pulse],
                              outline=col + (180,), width=2)
                g.polygon([(sx, sy - 7), (sx + 7, sy), (sx, sy + 7), (sx - 7, sy)],
                          outline=col + (255,), width=2)
                _label(g, sx, sy, f"TGT-{i+1:02d} · TANK", "DETECTED", col, fontS)

        # lead chevron, rotated to heading (screen: north up)
        _chevron(img, lx, ly, -h)

        # HUD
        t_s = fr["t"]
        found = sum(1 for i in range(K) if fr["tg"][i][2])
        neut = sum(1 for i in range(K) if fr["tg"][i][3])
        g.text((26, 22), "VESPER · TACTICAL", font=fontB, fill=(230, 240, 230, 255))
        g.text((26, 46), f"{world.upper()} · AO {int(2*half)} M", font=fontS, fill=(150, 165, 150, 220))
        if args.caption:
            g.text((26, 62), args.caption, font=fontS, fill=ORANGE + (235,))
        clk = f"T+{int(t_s)//60:02d}:{int(t_s)%60:02d}"
        g.text((CW - 150, 22), clk, font=fontHUD, fill=(230, 240, 230, 255))
        g.text((26, CH - 40), f"DETECTED {found}/{K}    NEUTRALIZED {neut}/{K}    ASSETS {len(drones)}",
               font=font, fill=ACCENT + (255,))
        # corner ticks
        for (ox, oy, dx, dy) in [(14, 14, 1, 1), (CW - 14, 14, -1, 1),
                                 (14, CH - 14, 1, -1), (CW - 14, CH - 14, -1, -1)]:
            g.line([ox, oy, ox + 18 * dx, oy], fill=(120, 140, 120, 200), width=2)
            g.line([ox, oy, ox, oy + 18 * dy], fill=(120, 140, 120, 200), width=2)

        ff.stdin.write(np.asarray(img, np.uint8).tobytes())

    ff.stdin.close()
    ff.wait()
    out = Path(args.out) if args.out else run / "tactical.mp4"
    print(f"wrote {out} ({len(frames)} frames, {len(frames)/FPS:.0f}s)")


def _strike(g, sx, sy, age, i):
    """~1 s kill animation at the impact point: flash, shockwave, debris.

    Rendered on output frames, so its wall-clock length is stride-independent.
    Deterministic per target (seeded by i) so a re-render is identical.
    """
    p = age / STRIKE
    if age < 2:                                            # impact flash
        g.rectangle([0, 0, CW, CH], fill=(255, 244, 214, 46 if age == 0 else 20))
        fr = 26 - 8 * age
        g.ellipse([sx - fr, sy - fr, sx + fr, sy + fr], fill=(255, 236, 180, 235))
    ring = 6 + p * 48                                      # expanding shockwave
    a = int(220 * (1 - p))
    g.ellipse([sx - ring, sy - ring, sx + ring, sy + ring], outline=ORANGE + (a,), width=3)
    r2 = ring * 0.55
    g.ellipse([sx - r2, sy - r2, sx + r2, sy + r2], outline=(255, 210, 90, int(a * 0.7)), width=2)
    rng = np.random.default_rng(1000 + i)                  # debris specks
    for ang, spd, sz in zip(rng.uniform(0, 2 * np.pi, 7), rng.uniform(0.5, 1.0, 7),
                            rng.integers(1, 3, 7)):
        d = 8 + p * 42 * spd
        px, py = sx + np.cos(ang) * d, sy + np.sin(ang) * d
        g.ellipse([px - sz, py - sz, px + sz, py + sz], fill=(255, 190, 110, a))


def _wreck(g, sx, sy, fi):
    """Settled kill marker: dark hull with an ember X, and a lazy smoke drift."""
    g.ellipse([sx - 9, sy - 9, sx + 9, sy + 9], fill=(28, 24, 22, 235), outline=(110, 92, 70, 255), width=2)
    g.line([sx - 5, sy - 5, sx + 5, sy + 5], fill=(224, 100, 42, 255), width=2)
    g.line([sx - 5, sy + 5, sx + 5, sy - 5], fill=(224, 100, 42, 255), width=2)
    for k in range(3):                                     # smoke: 3 looping puffs
        ph = (fi * 0.045 + k / 3.0) % 1.0
        px = sx + np.sin(fi * 0.11 + k * 2.1) * 4
        py = sy - 11 - ph * 26
        r = 3 + ph * 5
        g.ellipse([px - r, py - r, px + r, py + r], fill=(128, 128, 126, int(110 * (1 - ph))))


def _label(g, sx, sy, name, status, col, fontS, diamond=False):
    ind = 11 if diamond else 0                # SF Mono lacks ◆; draw it ourselves
    tw = int(max(g.textlength(name, font=fontS), g.textlength(status, font=fontS))) + ind
    bx, by = sx + 12, sy + 10
    g.rectangle([bx - 4, by - 3, bx + tw + 6, by + 26], fill=(6, 12, 6, 210), outline=col + (220,))
    g.line([sx, sy, bx - 2, by + 2], fill=col + (180,), width=1)
    if diamond:
        dx, dy = bx + 3, by + 5
        g.polygon([(dx, dy - 4), (dx + 4, dy), (dx, dy + 4), (dx - 4, dy)], fill=col + (255,))
    g.text((bx + ind, by - 1), name, font=fontS, fill=(235, 245, 235, 255))
    g.text((bx + ind, by + 12), status, font=fontS, fill=col + (255,))


def _chevron(img, cx, cy, ang):
    pts = np.array([[11, 0], [-7, -7], [-3, 0], [-7, 7]], np.float32)
    ca, sa = np.cos(ang), np.sin(ang)
    R = np.array([[ca, -sa], [sa, ca]], np.float32)
    p = (pts @ R.T) + [cx, cy]
    g = ImageDraw.Draw(img, "RGBA")
    g.polygon([tuple(v) for v in p], fill=ACCENT + (255,), outline=(223, 255, 223, 255))


if __name__ == "__main__":
    main()
