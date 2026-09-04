"use client";

import { useEffect, useRef, useState } from "react";
import { fetchJSON, type Trajectory } from "@/lib/vesper";

// Top-down flight path. Equal-aspect (it's a map), meters, start ring / end dot,
// single accent hue running light→dark with time, hover = nearest sample.
export function TrajectoryPlot({ runId }: { runId: string }) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [traj, setTraj] = useState<Trajectory | null>(null);
  const [failed, setFailed] = useState(false);
  const [tip, setTip] = useState<{ x: number; y: number; t: number; px: number; py: number; pz: number } | null>(null);

  useEffect(() => {
    let alive = true;
    fetchJSON<Trajectory>(`/api/runs/${runId}/trajectory`).then((d) => {
      if (!alive) return;
      if (d && d.t.length > 1) setTraj(d);
      else setFailed(true);
    });
    return () => {
      alive = false;
    };
  }, [runId]);

  useEffect(() => {
    const cv = canvasRef.current;
    if (!cv || !traj) return;

    const draw = () => {
      const dpr = window.devicePixelRatio || 1;
      const W = cv.clientWidth;
      const H = cv.clientHeight;
      if (!W || !H) return;
      cv.width = W * dpr;
      cv.height = H * dpr;
      const g = cv.getContext("2d")!;
      g.scale(dpr, dpr);
      g.clearRect(0, 0, W, H);

      const { px: xs, py: ys } = traj;
      const n = xs.length;
      const pad = 28;
      const x0 = Math.min(...xs), x1 = Math.max(...xs);
      const y0 = Math.min(...ys), y1 = Math.max(...ys);
      const span = Math.max(x1 - x0, y1 - y0, 1);
      const s = (Math.min(W, H) - 2 * pad) / span;
      const X = (v: number) => pad + (v - x0) * s + (W - 2 * pad - (x1 - x0) * s) / 2;
      const Y = (v: number) => H - pad - (v - y0) * s - (H - 2 * pad - (y1 - y0) * s) / 2;

      g.strokeStyle = "#2c2c2a";
      g.lineWidth = 1;
      for (let i = 0; i <= 4; i++) {
        const gx = pad + ((W - 2 * pad) * i) / 4;
        const gy = pad + ((H - 2 * pad) * i) / 4;
        g.beginPath(); g.moveTo(gx, pad); g.lineTo(gx, H - pad); g.stroke();
        g.beginPath(); g.moveTo(pad, gy); g.lineTo(W - pad, gy); g.stroke();
      }

      // sequential ramp on the accent hue: light early → dark late
      g.lineWidth = 2;
      g.lineJoin = "round";
      for (let i = 1; i < n; i++) {
        g.strokeStyle = `oklch(${78 - 30 * (i / n)}% 0.13 255)`;
        g.beginPath();
        g.moveTo(X(xs[i - 1]), Y(ys[i - 1]));
        g.lineTo(X(xs[i]), Y(ys[i]));
        g.stroke();
      }

      g.strokeStyle = "#c3c2b7";
      g.lineWidth = 2;
      g.beginPath(); g.arc(X(xs[0]), Y(ys[0]), 5, 0, 7); g.stroke();
      g.fillStyle = "#3987e5";
      g.beginPath(); g.arc(X(xs[n - 1]), Y(ys[n - 1]), 5, 0, 7); g.fill();

      g.fillStyle = "#8a897f";
      g.font = "11px system-ui, sans-serif";
      g.fillText("start", X(xs[0]) + 8, Y(ys[0]) + 4);
      g.fillText("end", X(xs[n - 1]) + 8, Y(ys[n - 1]) + 4);
      g.fillText(`${Math.round(span)} m across · +x east, +y north`, pad, H - 8);
    };

    draw();
    const ro = new ResizeObserver(draw);
    ro.observe(cv);
    return () => ro.disconnect();
  }, [traj]);

  const onMove = (e: React.MouseEvent) => {
    const cv = canvasRef.current;
    if (!cv || !traj) return;
    const r = cv.getBoundingClientRect();
    const W = cv.clientWidth, H = cv.clientHeight;
    const { px: xs, py: ys, pz: zs, t: ts } = traj;
    const n = xs.length;
    const pad = 28;
    const x0 = Math.min(...xs), x1 = Math.max(...xs);
    const y0 = Math.min(...ys), y1 = Math.max(...ys);
    const span = Math.max(x1 - x0, y1 - y0, 1);
    const s = (Math.min(W, H) - 2 * pad) / span;
    const X = (v: number) => pad + (v - x0) * s + (W - 2 * pad - (x1 - x0) * s) / 2;
    const Y = (v: number) => H - pad - (v - y0) * s - (H - 2 * pad - (y1 - y0) * s) / 2;
    const mx = e.clientX - r.left, my = e.clientY - r.top;
    let best = 0, bd = Infinity;
    for (let i = 0; i < n; i++) {
      const dd = (X(xs[i]) - mx) ** 2 + (Y(ys[i]) - my) ** 2;
      if (dd < bd) { bd = dd; best = i; }
    }
    if (bd > 30 ** 2) { setTip(null); return; }
    setTip({ x: mx, y: my, t: ts[best], px: xs[best], py: ys[best], pz: zs[best] });
  };

  if (failed)
    return <div className="p-6 text-sm text-muted-foreground">trajectory unavailable</div>;

  return (
    <div className="relative">
      <canvas
        ref={canvasRef}
        className="block h-[340px] w-full"
        onMouseMove={onMove}
        onMouseLeave={() => setTip(null)}
      />
      {tip && (
        <div
          className="pointer-events-none absolute z-10 rounded-md border border-border bg-popover px-2.5 py-1.5 text-xs shadow-lg"
          style={{ left: tip.x + 14, top: tip.y + 14 }}
        >
          <span className="text-muted-foreground">t=</span>{tip.t.toFixed(1)} s<br />
          <span className="text-muted-foreground">xy </span>{tip.px.toFixed(0)}, {tip.py.toFixed(0)} m<br />
          <span className="text-muted-foreground">alt </span>{tip.pz.toFixed(0)} m
        </div>
      )}
    </div>
  );
}
