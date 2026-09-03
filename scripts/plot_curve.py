"""Render a training curve.jsonl as a standalone SVG (no matplotlib).

    python3 scripts/plot_curve.py runs/<id>/curve.jsonl

Writes curve.svg beside it: episode return on its own axis, and the fractions
(found / cleared / swept) on a shared 0-1 axis, so the shape of the run is
readable at a glance instead of by scrolling a log.
"""
import json
import sys
from pathlib import Path

SERIES = [
    ("cleared", "#3ddc84", "vehicles reached"),
    ("found", "#4db8ff", "vehicles found"),
    ("coverage", "#c9a0ff", "ground swept"),
    ("intercept_rate", "#ffd166", "all three cleared"),
]


def smooth(xs, ys, k=9):
    out = []
    for i in range(len(ys)):
        w = [y for y in ys[max(0, i - k):i + k + 1] if y == y]
        out.append(sum(w) / len(w) if w else float("nan"))
    return xs, out


def main(path):
    p = Path(path)
    rows = [json.loads(l) for l in p.read_text().splitlines() if l.strip()]
    rows = [r for r in rows if r.get("episodes", 0) > 0]
    if not rows:
        raise SystemExit("no rows with completed episodes")
    W, H, PADL, PADR, PADT, PADB = 980, 460, 62, 62, 28, 46
    x0, x1 = rows[0]["iter"], rows[-1]["iter"]
    px = lambda i: PADL + (i - x0) / max(1, x1 - x0) * (W - PADL - PADR)
    py = lambda v: H - PADB - max(0.0, min(1.0, v)) * (H - PADT - PADB)

    rets = [r["ep_return"] for r in rows if r["ep_return"] == r["ep_return"]]
    lo, hi = (min(rets), max(rets)) if rets else (0, 1)
    span = (hi - lo) or 1.0
    pyr = lambda v: H - PADB - (v - lo) / span * (H - PADT - PADB)

    s = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}"'
         ' font-family="ui-sans-serif,system-ui,sans-serif" font-size="12">',
         f'<rect width="{W}" height="{H}" fill="#12141a"/>']
    for f in (0, 0.25, 0.5, 0.75, 1.0):
        y = py(f)
        s.append(f'<line x1="{PADL}" y1="{y:.1f}" x2="{W-PADR}" y2="{y:.1f}" stroke="#2a2e39"/>')
        s.append(f'<text x="{PADL-8}" y="{y+4:.1f}" fill="#7d8595" text-anchor="end">{f:.2f}</text>')
    s.append(f'<text x="{PADL-8}" y="{PADT-10}" fill="#7d8595" text-anchor="end">fraction</text>')
    s.append(f'<text x="{W-PADR+8}" y="{PADT-10}" fill="#7d8595">return</text>')
    for v in (lo, (lo + hi) / 2, hi):
        s.append(f'<text x="{W-PADR+8}" y="{pyr(v)+4:.1f}" fill="#7d8595">{v:.0f}</text>')

    xs = [r["iter"] for r in rows]
    if rets:
        _, ys = smooth(xs, [r["ep_return"] for r in rows])
        pts = " ".join(f"{px(i):.1f},{pyr(v):.1f}" for i, v in zip(xs, ys) if v == v)
        s.append(f'<polyline points="{pts}" fill="none" stroke="#8b93a5" stroke-width="1.6"'
                 ' stroke-dasharray="5 4"/>')
    for key, col, _ in SERIES:
        if key not in rows[0]:
            continue
        _, ys = smooth(xs, [r.get(key, float("nan")) for r in rows])
        pts = " ".join(f"{px(i):.1f},{py(v):.1f}" for i, v in zip(xs, ys) if v == v)
        if pts:
            s.append(f'<polyline points="{pts}" fill="none" stroke="{col}" stroke-width="2.4"/>')
    s.append(f'<line x1="{PADL}" y1="{H-PADB}" x2="{W-PADR}" y2="{H-PADB}" stroke="#454b5a"/>')
    for i in (x0, (x0 + x1) // 2, x1):
        s.append(f'<text x="{px(i):.1f}" y="{H-PADB+18}" fill="#7d8595" text-anchor="middle">{i}</text>')
    s.append(f'<text x="{W/2}" y="{H-8}" fill="#7d8595" text-anchor="middle">PPO iteration</text>')
    lx = PADL + 8
    for key, col, label in SERIES + [("ep_return", "#8b93a5", "episode return (right axis)")]:
        if key != "ep_return" and key not in rows[0]:
            continue
        s.append(f'<rect x="{lx}" y="{PADT-16}" width="10" height="10" fill="{col}"/>')
        s.append(f'<text x="{lx+15}" y="{PADT-7}" fill="#c3c9d6">{label}</text>')
        lx += 22 + 7.0 * len(label)
    s.append("</svg>")
    out = p.with_name("curve.svg")
    out.write_text("\n".join(s))
    print(f"wrote {out}")
    last = rows[-1]
    print({k: round(last[k], 3) for k in ("iter", "ep_return", "found", "cleared", "coverage",
                                          "intercept_rate") if k in last})


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "curve.jsonl")
