"""Sweep aggregation: success binned per condition dimension + auto-findings."""


def _bin_edges(dim):
    return {
        "wind_speed_ms": [0, 2, 4, 6, 8.01],
        "visibility_m": [0, 100, 250, 600, 1e9],
        "range_noise_std": [0, 0.1, 0.25, 0.5],
        "spawn_east": [-1.5, -0.5, 0.5, 1.51],
    }[dim]


def bin_success(results: list[dict], dim: str) -> list[dict]:
    edges = _bin_edges(dim)
    rows = []
    for lo, hi in zip(edges, edges[1:]):
        sel = [r for r in results if lo <= r[dim] < hi]
        if sel:
            rows.append({
                "bin": f"{lo:g}-{hi:g}", "n": len(sel),
                "success": round(sum(r["success"] for r in sel) / len(sel), 3),
                "collisions": sum(1 for r in sel if r.get("failure") == "collision"),
                "timeouts": sum(1 for r in sel if r.get("failure") == "timeout"),
            })
    return rows


def findings(results: list[dict], dims=("wind_speed_ms", "visibility_m", "range_noise_std")) -> list[str]:
    overall = sum(r["success"] for r in results) / max(len(results), 1)
    out = [f"Overall success: {overall:.0%} over {len(results)} variants."]
    for dim in dims:
        rows = bin_success(results, dim)
        if len(rows) < 2:
            continue
        worst, best = min(rows, key=lambda r: r["success"]), max(rows, key=lambda r: r["success"])
        drop = best["success"] - worst["success"]
        if drop >= 0.15:
            out.append(f"Success drops {drop:.0%} ({best['success']:.0%} -> {worst['success']:.0%}) "
                       f"when {dim} is in [{worst['bin']}] (n={worst['n']}).")
    return out
