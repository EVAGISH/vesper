"use client";

import { useEffect, useRef, useState } from "react";
import { CurveCharts } from "@/components/curve-charts";
import { EventsTimeline } from "@/components/events-timeline";
import { SweepTable } from "@/components/sweep-table";
import { TrajectoryPlot } from "@/components/trajectory-plot";
import {
  artifactLabel, fetchJSON, fmtDur, fmtTime, KIND_COLOR, media, runKind, type Run,
} from "@/lib/vesper";

function Panel({
  title, right, children, wide,
}: {
  title: string;
  right?: React.ReactNode;
  children: React.ReactNode;
  wide?: boolean;
}) {
  return (
    <section
      className={`hud-corners rounded-lg border border-border bg-card ${wide ? "col-span-full" : ""}`}
    >
      <h3 className="flex items-center border-b border-border px-3 py-1.5 font-mono text-[10px] font-semibold uppercase tracking-[0.14em] text-secondary-foreground">
        <span className="mr-1.5 text-muted-foreground">▮</span>
        {title}
        {right && (
          <span className="ml-auto font-normal normal-case tracking-normal text-muted-foreground">
            {right}
          </span>
        )}
      </h3>
      {children}
    </section>
  );
}

function KV({ rows }: { rows: [string, React.ReactNode][] }) {
  return (
    <table className="w-full border-collapse text-xs tabular-nums">
      <tbody>
        {rows.map(([k, v]) => (
          <tr key={k}>
            <td className="w-24 whitespace-nowrap py-[3px] pr-2.5 align-top text-muted-foreground">
              {k}
            </td>
            <td className="py-[3px] align-top text-secondary-foreground">{v}</td>
          </tr>
        ))}
      </tbody>
    </table>
  );
}

type ScenarioSpec = {
  world?: string;
  waypoints?: unknown[];
  wind_speed_ms?: number;
  visibility_m?: number;
  cruise_ms?: number;
  max_sim_s?: number;
};

export function RunDetail({ run }: { run: Run }) {
  const m = run.manifest;
  const kind = runKind(run);
  const containerRef = useRef<HTMLDivElement>(null);
  const [scenario, setScenario] = useState<ScenarioSpec | null>(null);

  const vids = run.files.filter((f) => f.endsWith(".mp4"));
  const pngs = run.files.filter((f) => f.endsWith(".png"));
  const hasTraj = run.files.includes("trajectory.parquet");

  // RunDetail is keyed by run.id, so a run switch remounts and resets state.
  useEffect(() => {
    let alive = true;
    if (run.files.includes("scenario.json"))
      fetchJSON<ScenarioSpec>(media(run.id, "scenario.json")).then(
        (d) => alive && setScenario(d),
      );
    return () => {
      alive = false;
    };
  }, [run.id, run.files]);

  const seekVideos = (t: number) => {
    containerRef.current?.querySelectorAll("video").forEach((v) => {
      v.currentTime = t;
      v.play().catch(() => {});
    });
  };

  const frames = Object.values(m.frames ?? {})[0];
  const simDuration = frames && m.fps ? frames / m.fps : undefined;

  return (
    <div ref={containerRef} className="flex-1 overflow-y-auto p-4">
      <div className="mb-3.5 flex flex-wrap items-baseline gap-3">
        <h2 className="font-mono text-base font-bold">{run.id}</h2>
        <span className="border border-border px-2 py-0.5 font-mono text-[10px] uppercase tracking-[0.12em] text-secondary-foreground">
          <i className="mr-1.5 not-italic" style={{ color: KIND_COLOR[kind] }}>
            ●
          </i>
          {kind}
        </span>
        <span className="text-xs tabular-nums text-muted-foreground">
          {fmtTime(m.started)}
          {fmtDur(m) && ` · ${fmtDur(m)}`}
          {m.scene && ` · ${m.scene}`}
        </span>
      </div>

      <div className="grid grid-cols-1 gap-3 lg:grid-cols-2">
        {vids.map((v) => {
          const stream = v.replace(/\.mp4$/, "");
          return (
            <Panel
              key={v}
              title={artifactLabel(v)}
              right={
                m.streams?.[stream]
                  ? `${m.streams[stream]}${m.frames?.[stream] ? ` · ${m.frames[stream]} frames` : ""}`
                  : undefined
              }
            >
              <video
                controls
                preload="metadata"
                src={media(run.id, v)}
                className="block w-full bg-black"
              />
            </Panel>
          );
        })}

        {pngs.map((p) => (
          <Panel key={p} title={artifactLabel(p)}>
            {/* eslint-disable-next-line @next/next/no-img-element -- streamed from /media, never optimized/copied */}
            <img src={media(run.id, p)} alt={p} className="block w-full bg-black" />
          </Panel>
        ))}

        {run.files.includes("events.json") && (
          <Panel title="Events" right="click to jump video" wide>
            <EventsTimeline runId={run.id} simDuration={simDuration} onSeek={seekVideos} />
          </Panel>
        )}

        {hasTraj && (
          <Panel title="Trajectory — top-down" right="meters · equal aspect">
            <TrajectoryPlot runId={run.id} />
          </Panel>
        )}

        {run.files.includes("curve.jsonl") && (
          <Panel title="Training progress" right="per iteration" wide>
            <CurveCharts runId={run.id} />
          </Panel>
        )}

        {(run.files.includes("report.json") || run.files.includes("results.jsonl")) && (
          <Panel title="Sweep report" right="failures link to their runs" wide>
            <SweepTable runId={run.id} files={run.files} />
          </Panel>
        )}

        <Panel title="Run">
          <div className="p-3">
            <KV
              rows={[
                ["id", <span key="id" className="font-mono">{run.id}</span>],
                ["scene", m.scene ?? "—"],
                ["started", fmtTime(m.started)],
                [
                  "duration",
                  fmtDur(m)
                    ? `${fmtDur(m)}${simDuration ? ` · ${simDuration.toFixed(1)} s sim` : ""}`
                    : "—",
                ],
                [
                  "streams",
                  Object.entries(m.streams ?? {})
                    .map(([k, v]) => `${k} (${v})`)
                    .join(", ") || "—",
                ],
                [
                  "frames",
                  frames !== undefined
                    ? `${frames} @ ${m.fps ?? "?"} fps${m.resolution ? ` · ${m.resolution.join("×")}` : ""}`
                    : "—",
                ],
              ]}
            />
          </div>
          {scenario && (
            <>
              <h3 className="border-y border-border px-3 py-1.5 font-mono text-[10px] font-semibold uppercase tracking-[0.14em] text-secondary-foreground">
                <span className="mr-1.5 text-muted-foreground">▮</span>Scenario
              </h3>
              <div className="p-3">
                <KV
                  rows={[
                    ["world", scenario.world ?? "—"],
                    ["waypoints", scenario.waypoints?.length ?? "—"],
                    ["wind", `${scenario.wind_speed_ms ?? 0} m/s`],
                    [
                      "visibility",
                      scenario.visibility_m != null ? `${scenario.visibility_m} m` : "∞",
                    ],
                    ["cruise", scenario.cruise_ms != null ? `${scenario.cruise_ms} m/s` : "—"],
                    ["sim cap", scenario.max_sim_s != null ? `${scenario.max_sim_s} s` : "—"],
                  ]}
                />
              </div>
            </>
          )}
        </Panel>
      </div>
    </div>
  );
}
