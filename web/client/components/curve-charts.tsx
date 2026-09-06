"use client";

import { useEffect, useMemo, useRef, useState } from "react";
import { media, parseJSONL } from "@/lib/vesper";

// Training curves from curve.jsonl (one JSON object per iteration).
// One small-multiple line chart per numeric metric, single accent hue,
// crosshair + tooltip on hover. Never two y-scales on one chart.

const ITER_KEYS = ["iter", "iteration", "step", "epoch", "i"];

type Series = { name: string; x: number[]; y: number[] };

export function CurveCharts({ runId }: { runId: string }) {
  const [rows, setRows] = useState<Record<string, unknown>[] | null>(null);

  useEffect(() => {
    let alive = true;
    fetch(media(runId, "curve.jsonl"))
      .then((r) => (r.ok ? r.text() : ""))
      .then((t) => alive && setRows(parseJSONL(t)))
      .catch(() => alive && setRows([]));
    return () => {
      alive = false;
    };
  }, [runId]);

  const series = useMemo<Series[]>(() => {
    if (!rows || !rows.length) return [];
    const keys = Object.keys(rows[0]);
    const iterKey = ITER_KEYS.find((k) => keys.includes(k));
    const metricKeys = keys.filter(
      (k) => k !== iterKey && rows.some((r) => typeof r[k] === "number"),
    );
    return metricKeys.map((name) => {
      const x: number[] = [];
      const y: number[] = [];
      rows.forEach((r, i) => {
        const v = r[name];
        if (typeof v !== "number" || !isFinite(v)) return;
        const xv = iterKey && typeof r[iterKey] === "number" ? (r[iterKey] as number) : i;
        x.push(xv);
        y.push(v);
      });
      return { name, x, y };
    }).filter((s) => s.y.length > 1);
  }, [rows]);

  if (rows === null)
    return <div className="p-6 text-sm text-muted-foreground">loading training progress…</div>;
  if (!series.length)
    return <div className="p-6 text-sm text-muted-foreground">no training metrics recorded</div>;

  return (
    <div className="grid gap-3 p-3 sm:grid-cols-2">
      {series.map((s) => (
        <LineChart key={s.name} s={s} />
      ))}
    </div>
  );
}

function LineChart({ s }: { s: Series }) {
  const W = 320, H = 160, padL = 40, padR = 10, padT = 12, padB = 22;
  const ref = useRef<SVGSVGElement>(null);
  const [hover, setHover] = useState<number | null>(null);

  const x0 = s.x[0], x1 = s.x[s.x.length - 1] || 1;
  const yMin = Math.min(...s.y), yMax = Math.max(...s.y);
  const ySpan = yMax - yMin || 1;
  const X = (v: number) => padL + ((v - x0) / (x1 - x0 || 1)) * (W - padL - padR);
  const Y = (v: number) => H - padB - ((v - yMin) / ySpan) * (H - padT - padB);
  const pts = s.x.map((xv, i) => `${X(xv).toFixed(1)},${Y(s.y[i]).toFixed(1)}`).join(" ");

  const fmt = (v: number) =>
    Math.abs(v) >= 1000 ? v.toFixed(0) : Math.abs(v) >= 10 ? v.toFixed(1) : v.toFixed(3);

  const onMove = (e: React.MouseEvent) => {
    const svg = ref.current;
    if (!svg) return;
    const r = svg.getBoundingClientRect();
    const mx = ((e.clientX - r.left) / r.width) * W;
    let best = 0, bd = Infinity;
    for (let i = 0; i < s.x.length; i++) {
      const d = Math.abs(X(s.x[i]) - mx);
      if (d < bd) { bd = d; best = i; }
    }
    setHover(best);
  };

  return (
    <div className="rounded-md border border-border bg-background">
      <div className="flex items-baseline justify-between px-3 pt-2 text-xs">
        <span className="font-medium text-secondary-foreground">{s.name}</span>
        <span className="tabular-nums text-muted-foreground">
          {hover !== null
            ? `${s.name} ${fmt(s.y[hover])} @ iter ${s.x[hover]}`
            : `final ${fmt(s.y[s.y.length - 1])}`}
        </span>
      </div>
      <svg
        ref={ref}
        viewBox={`0 0 ${W} ${H}`}
        className="block w-full"
        onMouseMove={onMove}
        onMouseLeave={() => setHover(null)}
      >
        {[0, 0.5, 1].map((f) => {
          const gy = padT + f * (H - padT - padB);
          const val = yMax - f * ySpan;
          return (
            <g key={f}>
              <line x1={padL} x2={W - padR} y1={gy} y2={gy} stroke="#2c2c2a" strokeWidth="1" />
              <text x={padL - 5} y={gy + 3} textAnchor="end" fontSize="9" fill="#8a897f">
                {fmt(val)}
              </text>
            </g>
          );
        })}
        <text x={padL} y={H - 6} fontSize="9" fill="#8a897f">{x0}</text>
        <text x={W - padR} y={H - 6} textAnchor="end" fontSize="9" fill="#8a897f">
          {x1} iters
        </text>
        <polyline fill="none" stroke="#3987e5" strokeWidth="2" strokeLinejoin="round" points={pts} />
        {hover !== null && (
          <g>
            <line
              x1={X(s.x[hover])} x2={X(s.x[hover])} y1={padT} y2={H - padB}
              stroke="#8a897f" strokeWidth="1" strokeDasharray="3 3"
            />
            <circle cx={X(s.x[hover])} cy={Y(s.y[hover])} r="3.5" fill="#3987e5" stroke="#1a1a19" strokeWidth="2" />
          </g>
        )}
      </svg>
    </div>
  );
}
