"use client";

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { useVesper } from "@/components/vesper-provider";
import {
  fetchJSON, media, parseEvents, type RunEvent, type Site, type Trajectory,
} from "@/lib/vesper";

// Interactive AO map: the site's own ground ortho (world frame, ±half_m,
// +x east / +y north) rendered to canvas with pan (drag), zoom (wheel or
// buttons), a live coordinate readout, and clickable markers. Layers: last
// sortie's flight track, mission waypoints, and target events. When live
// telemetry exists this becomes the live position layer.

type ScenarioSpec = { waypoints?: [number, number, number][] };
type Pick = { sx: number; sy: number; lines: string[] };

const MAX_ZOOM = 14;

export function SiteMap() {
  const { runs } = useVesper();
  const [site, setSite] = useState<Site | null | undefined>(undefined);
  const [traj, setTraj] = useState<Trajectory | null>(null);
  const [events, setEvents] = useState<RunEvent[]>([]);
  const [waypoints, setWaypoints] = useState<[number, number, number][]>([]);
  const [imgReady, setImgReady] = useState(false);
  const [picked, setPicked] = useState<Pick | null>(null);

  const canvasRef = useRef<HTMLCanvasElement>(null);
  const imgRef = useRef<HTMLImageElement | null>(null);
  const view = useRef({ cx: 0, cy: 0, zoom: 1 }); // zoom 1 = whole site fits
  const drag = useRef({ on: false, x: 0, y: 0, moved: 0 });
  const hover = useRef<{ sx: number; sy: number } | null>(null);

  // one source run for every overlay, so the layers agree with each other
  const sortie = useMemo(
    () =>
      runs?.find((r) =>
        ["trajectory.parquet", "events.json", "scenario.json"].some((f) => r.files.includes(f)),
      ) ?? null,
    [runs],
  );

  useEffect(() => {
    let alive = true;
    fetchJSON<Site[]>("/api/site").then((d) => alive && setSite(d?.[0] ?? null));
    return () => {
      alive = false;
    };
  }, []);

  useEffect(() => {
    if (!site) return;
    const img = new window.Image();
    img.onload = () => {
      imgRef.current = img;
      setImgReady(true);
    };
    img.src = site.ground;
  }, [site]);

  useEffect(() => {
    if (!sortie) return;
    let alive = true;
    if (sortie.files.includes("trajectory.parquet"))
      fetchJSON<Trajectory>(`/api/runs/${sortie.id}/trajectory?max_points=800`).then(
        (d) => alive && setTraj(d),
      );
    if (sortie.files.includes("events.json"))
      fetchJSON<unknown>(media(sortie.id, "events.json")).then(
        (d) => alive && setEvents(parseEvents(d)),
      );
    if (sortie.files.includes("scenario.json"))
      fetchJSON<ScenarioSpec>(media(sortie.id, "scenario.json")).then(
        (d) => alive && d?.waypoints && setWaypoints(d.waypoints),
      );
    return () => {
      alive = false;
    };
  }, [sortie]);

  // world↔screen for the current viewport; also used by the event handlers
  const mapping = useCallback(() => {
    const cv = canvasRef.current!;
    const W = cv.clientWidth, H = cv.clientHeight;
    const half = site?.half_m ?? 1000;
    const fit = Math.min(W, H) / (2 * half);
    const ppm = fit * view.current.zoom;
    // clamp the center so the viewport never leaves the site
    const spanX = W / (2 * ppm), spanY = H / (2 * ppm);
    view.current.cx = spanX >= half ? 0 : Math.max(-half + spanX, Math.min(half - spanX, view.current.cx));
    view.current.cy = spanY >= half ? 0 : Math.max(-half + spanY, Math.min(half - spanY, view.current.cy));
    const { cx, cy } = view.current;
    return {
      W, H, half, ppm,
      toX: (wx: number) => W / 2 + (wx - cx) * ppm,
      toY: (wy: number) => H / 2 + (cy - wy) * ppm,
      toWorld: (sx: number, sy: number): [number, number] => [
        cx + (sx - W / 2) / ppm,
        cy - (sy - H / 2) / ppm,
      ],
    };
  }, [site]);

  const draw = useCallback(() => {
    const cv = canvasRef.current;
    const img = imgRef.current;
    if (!cv || !site || !img) return;
    const dpr = window.devicePixelRatio || 1;
    const m = mapping();
    cv.width = m.W * dpr;
    cv.height = m.H * dpr;
    const g = cv.getContext("2d")!;
    g.scale(dpr, dpr);
    g.fillStyle = "#0d0d0d";
    g.fillRect(0, 0, m.W, m.H);
    const size = 2 * m.half * m.ppm;
    g.drawImage(img, m.toX(-m.half), m.toY(m.half), size, size);

    // mission waypoints: dashed route + diamonds
    if (waypoints.length) {
      g.strokeStyle = "rgba(255,255,255,0.6)";
      g.lineWidth = 1;
      g.setLineDash([4, 4]);
      g.beginPath();
      g.moveTo(m.toX(waypoints[0][0]), m.toY(waypoints[0][1]));
      for (const [x, y] of waypoints.slice(1)) g.lineTo(m.toX(x), m.toY(y));
      g.stroke();
      g.setLineDash([]);
      for (const [x, y] of waypoints) {
        g.save();
        g.translate(m.toX(x), m.toY(y));
        g.rotate(Math.PI / 4);
        g.fillStyle = "rgba(0,0,0,0.6)";
        g.fillRect(-4.5, -4.5, 9, 9);
        g.strokeStyle = "#ffffff";
        g.lineWidth = 1.2;
        g.strokeRect(-3.5, -3.5, 7, 7);
        g.restore();
      }
    }

    // flight track: dark halo + accent, end dot
    if (traj && traj.t.length > 1) {
      const { px, py } = traj;
      for (const pass of [
        { style: "rgba(0,0,0,0.55)", width: 4.5 },
        { style: "#3987e5", width: 2 },
      ]) {
        g.strokeStyle = pass.style;
        g.lineWidth = pass.width;
        g.lineJoin = "round";
        g.beginPath();
        g.moveTo(m.toX(px[0]), m.toY(py[0]));
        for (let i = 1; i < px.length; i++) g.lineTo(m.toX(px[i]), m.toY(py[i]));
        g.stroke();
      }
      const n = px.length - 1;
      g.fillStyle = "#3987e5";
      g.strokeStyle = "#ffffff";
      g.lineWidth = 1.5;
      g.beginPath();
      g.arc(m.toX(px[n]), m.toY(py[n]), 5, 0, 7);
      g.fill();
      g.stroke();
    }

    // targets: sighted = amber triangle, reached = green dot
    for (const e of events) {
      if (!e.xy) continue;
      const sx = m.toX(e.xy[0]), sy = m.toY(e.xy[1]);
      g.lineWidth = 2;
      if (/reach/i.test(e.label)) {
        g.fillStyle = "#0ca30c";
        g.strokeStyle = "rgba(0,0,0,0.7)";
        g.beginPath();
        g.arc(sx, sy, 6, 0, 7);
        g.fill();
        g.stroke();
      } else {
        g.strokeStyle = "#c98500";
        g.fillStyle = "rgba(0,0,0,0.5)";
        g.beginPath();
        g.moveTo(sx, sy - 7);
        g.lineTo(sx - 6, sy + 5);
        g.lineTo(sx + 6, sy + 5);
        g.closePath();
        g.fill();
        g.stroke();
      }
    }

    // hover reticle: hairline crosshair with the E/N readout on its own axes
    if (hover.current && !drag.current.on) {
      const { sx, sy } = hover.current;
      g.strokeStyle = "rgba(255,255,255,0.28)";
      g.lineWidth = 1;
      g.beginPath();
      g.moveTo(sx, 0); g.lineTo(sx, m.H);
      g.moveTo(0, sy); g.lineTo(m.W, sy);
      g.stroke();
      g.strokeStyle = "rgba(255,255,255,0.85)";
      g.beginPath();
      g.moveTo(sx - 7, sy); g.lineTo(sx - 2, sy);
      g.moveTo(sx + 2, sy); g.lineTo(sx + 7, sy);
      g.moveTo(sx, sy - 7); g.lineTo(sx, sy - 2);
      g.moveTo(sx, sy + 2); g.lineTo(sx, sy + 7);
      g.stroke();
      const [wx, wy] = m.toWorld(sx, sy);
      g.font = "9px ui-monospace, monospace";
      const eTxt = `${wx.toFixed(0)} E`, nTxt = `${wy.toFixed(0)} N`;
      const eW = g.measureText(eTxt).width, nW = g.measureText(nTxt).width;
      g.fillStyle = "rgba(0,0,0,0.65)";
      g.fillRect(Math.min(sx + 4, m.W - eW - 8), 2, eW + 6, 12);
      g.fillRect(2, Math.max(sy - 14, 2), nW + 6, 12);
      g.fillStyle = "#ffffff";
      g.fillText(eTxt, Math.min(sx + 7, m.W - eW - 5), 11);
      g.fillText(nTxt, 5, Math.max(sy - 5, 11));
    }

    // scale bar: a round number of meters near 90 screen px
    const raw = 90 / m.ppm;
    const step = 10 ** Math.floor(Math.log10(raw));
    const nice = [1, 2, 5, 10].map((k) => k * step).reduce((a, b) =>
      Math.abs(b - raw) < Math.abs(a - raw) ? b : a);
    const barPx = nice * m.ppm;
    g.strokeStyle = "#ffffff";
    g.fillStyle = "#ffffff";
    g.lineWidth = 2;
    g.beginPath();
    g.moveTo(m.W - 14 - barPx, m.H - 16);
    g.lineTo(m.W - 14, m.H - 16);
    g.stroke();
    g.font = "9px ui-monospace, monospace";
    g.textAlign = "right";
    g.fillText(`${nice >= 1000 ? `${nice / 1000} km` : `${nice} m`}`, m.W - 14, m.H - 22);
    g.textAlign = "left";
  }, [site, traj, events, waypoints, mapping]);

  useEffect(() => {
    draw();
    const cv = canvasRef.current;
    if (!cv) return;
    const ro = new ResizeObserver(draw);
    ro.observe(cv);
    return () => ro.disconnect();
  }, [draw, imgReady]);

  // wheel zoom (non-passive so the page doesn't scroll)
  useEffect(() => {
    const cv = canvasRef.current;
    if (!cv) return;
    const onWheel = (e: WheelEvent) => {
      e.preventDefault();
      const r = cv.getBoundingClientRect();
      const sx = e.clientX - r.left, sy = e.clientY - r.top;
      const m = mapping();
      const [wx, wy] = m.toWorld(sx, sy);
      const v = view.current;
      const nz = Math.max(1, Math.min(MAX_ZOOM, v.zoom * Math.exp(-e.deltaY * 0.0015)));
      if (nz === v.zoom) return;
      // keep the point under the cursor fixed while zooming
      const scale = v.zoom / nz;
      v.cx = wx - (wx - v.cx) * scale;
      v.cy = wy - (wy - v.cy) * scale;
      v.zoom = nz;
      setPicked(null);
      draw();
    };
    cv.addEventListener("wheel", onWheel, { passive: false });
    return () => cv.removeEventListener("wheel", onWheel);
  }, [draw, mapping]);

  const hitTest = (sx: number, sy: number): Pick | null => {
    const m = mapping();
    // keep the tooltip inside the map: clamp its anchor while we know the size
    const clamp = (p: Pick): Pick => ({
      ...p,
      sx: Math.min(p.sx + 12, m.W - 150),
      sy: Math.min(p.sy + 12, m.H - 44),
    });
    const near = (x: number, y: number, r = 11) =>
      (m.toX(x) - sx) ** 2 + (m.toY(y) - sy) ** 2 < r * r;
    for (const e of events) {
      if (e.xy && near(e.xy[0], e.xy[1]))
        return clamp({
          sx: m.toX(e.xy[0]), sy: m.toY(e.xy[1]),
          lines: [
            `${e.label}${e.vehicle !== undefined ? ` — target ${e.vehicle}` : ""}`,
            `t=${e.t.toFixed(1)} s · ${e.xy[0].toFixed(0)} E, ${e.xy[1].toFixed(0)} N`,
          ],
        });
    }
    for (let i = 0; i < waypoints.length; i++) {
      const [x, y, z] = waypoints[i];
      if (near(x, y))
        return clamp({
          sx: m.toX(x), sy: m.toY(y),
          lines: [`waypoint ${i + 1} of ${waypoints.length}`, `${x.toFixed(0)} E, ${y.toFixed(0)} N · alt ${z.toFixed(0)} m`],
        });
    }
    if (traj && traj.t.length) {
      const n = traj.t.length - 1;
      if (near(traj.px[n], traj.py[n]))
        return clamp({
          sx: m.toX(traj.px[n]), sy: m.toY(traj.py[n]),
          lines: ["last position", `t=${traj.t[n].toFixed(0)} s · alt ${traj.pz[n].toFixed(0)} m`],
        });
    }
    return null;
  };

  const zoomBy = (f: number) => {
    const v = view.current;
    v.zoom = Math.max(1, Math.min(MAX_ZOOM, v.zoom * f));
    setPicked(null);
    draw();
  };

  if (site === undefined)
    return <div className="p-4 text-xs text-muted-foreground">loading map…</div>;
  if (site === null)
    return (
      <div className="p-4 text-xs text-muted-foreground">
        no site map available for this world
      </div>
    );

  return (
    <div>
      <div className="relative aspect-square w-full overflow-hidden bg-black">
        <canvas
          ref={canvasRef}
          className="absolute inset-0 h-full w-full cursor-crosshair touch-none"
          onPointerDown={(e) => {
            drag.current = { on: true, x: e.clientX, y: e.clientY, moved: 0 };
            (e.target as HTMLElement).setPointerCapture(e.pointerId);
          }}
          onPointerMove={(e) => {
            const cv = canvasRef.current!;
            const r = cv.getBoundingClientRect();
            const sx = e.clientX - r.left, sy = e.clientY - r.top;
            if (drag.current.on) {
              const m = mapping();
              const dx = e.clientX - drag.current.x, dy = e.clientY - drag.current.y;
              drag.current.moved += Math.abs(dx) + Math.abs(dy);
              view.current.cx -= dx / m.ppm;
              view.current.cy += dy / m.ppm;
              drag.current.x = e.clientX;
              drag.current.y = e.clientY;
              cv.style.cursor = "grabbing";
            } else {
              // pointer over a clickable marker beats the crosshair
              cv.style.cursor = hitTest(sx, sy) ? "pointer" : "crosshair";
            }
            hover.current = { sx, sy };
            draw();
          }}
          onPointerUp={(e) => {
            const was = drag.current;
            drag.current = { on: false, x: 0, y: 0, moved: 0 };
            const cv = canvasRef.current!;
            cv.style.cursor = "crosshair";
            if (was.moved < 5) {
              const r = cv.getBoundingClientRect();
              setPicked(hitTest(e.clientX - r.left, e.clientY - r.top));
            }
            draw();
          }}
          onPointerLeave={() => {
            drag.current.on = false;
            hover.current = null;
            draw();
          }}
        />
        {picked && (
          <div
            className="pointer-events-none absolute z-10 border border-border bg-popover px-2 py-1 font-mono text-[10px] leading-relaxed shadow-lg"
            style={{ left: picked.sx, top: picked.sy }}
          >
            {picked.lines.map((l, i) => (
              <div key={i} className={i ? "text-muted-foreground" : "text-foreground"}>{l}</div>
            ))}
          </div>
        )}
        <div className="absolute left-2 top-2 bg-black/60 px-1.5 py-0.5 font-mono text-[9px] uppercase tracking-[0.12em] text-secondary-foreground">
          {site.world} · {site.half_m * 2} m × {site.half_m * 2} m · N up
        </div>
        <div className="absolute right-2 top-2 flex flex-col overflow-hidden rounded-sm border border-border">
          <button
            className="bg-black/70 px-2 py-0.5 font-mono text-xs text-secondary-foreground hover:text-foreground"
            onClick={() => zoomBy(1.5)}
            aria-label="zoom in"
          >
            +
          </button>
          <button
            className="border-t border-border bg-black/70 px-2 py-0.5 font-mono text-xs text-secondary-foreground hover:text-foreground"
            onClick={() => zoomBy(1 / 1.5)}
            aria-label="zoom out"
          >
            −
          </button>
          <button
            className="border-t border-border bg-black/70 px-2 py-0.5 font-mono text-[9px] text-secondary-foreground hover:text-foreground"
            onClick={() => {
              view.current = { cx: 0, cy: 0, zoom: 1 };
              setPicked(null);
              draw();
            }}
            aria-label="reset view"
          >
            ⌂
          </button>
        </div>
        {sortie && (
          <span className="absolute bottom-2 left-2 bg-black/60 px-1.5 py-0.5 font-mono text-[9px] text-muted-foreground">
            last sortie: {sortie.id}
          </span>
        )}
      </div>
      <div className="flex flex-wrap gap-x-4 gap-y-1 border-t border-border px-3 py-1.5 font-mono text-[9px] uppercase tracking-[0.1em] text-muted-foreground">
        <span><span className="text-[#3987e5]">—</span> track</span>
        <span><span className="text-white">◇</span> waypoints</span>
        <span><span className="text-[#c98500]">△</span> sighted</span>
        <span><span className="text-[#0ca30c]">●</span> reached</span>
        <span className="ml-auto normal-case">drag to pan · scroll to zoom · click a marker</span>
      </div>
    </div>
  );
}
