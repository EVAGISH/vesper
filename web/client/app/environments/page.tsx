"use client";

import { JobButton } from "@/components/job-controls";
import { useVesper } from "@/components/vesper-provider";
import type { Scenario } from "@/lib/vesper";

function ScenarioCard({ s }: { s: Scenario }) {
  return (
    <section className="hud-corners rounded-lg border border-border bg-card">
      <h3 className="flex items-center border-b border-border px-3 py-1.5 font-mono text-[10px] font-semibold uppercase tracking-[0.14em] text-secondary-foreground">
        <span className="mr-1.5 text-muted-foreground">▮</span>
        {s.file.replace(/\.json$/, "").replace(/[_-]/g, " ")}
        {s.world && (
          <span className="ml-auto font-normal normal-case tracking-normal text-muted-foreground">
            {s.world} site
          </span>
        )}
      </h3>
      <div className="p-3">
        <table className="w-full border-collapse text-xs tabular-nums">
          <tbody>
            {(
              [
                ["terrain", s.terrain_usd ? "site terrain" : "flat"],
                ["waypoints", s.waypoints],
                ["wind", `${s.wind_ms ?? 0} m/s`],
                ["visibility", s.visibility_m != null ? `${s.visibility_m} m` : "∞"],
                ["cruise", s.cruise_ms != null ? `${s.cruise_ms} m/s` : "—"],
                ["sim cap", s.max_sim_s != null ? `${s.max_sim_s} s` : "—"],
              ] as [string, React.ReactNode][]
            ).map(([k, v]) => (
              <tr key={k}>
                <td className="w-24 whitespace-nowrap py-[3px] pr-2.5 text-muted-foreground">{k}</td>
                <td className="py-[3px] text-secondary-foreground">{v}</td>
              </tr>
            ))}
          </tbody>
        </table>
        <div className="mt-3">
          <JobButton label="▶ LAUNCH MISSION" body={{ kind: "mission", scenario: s.file }} />
          <div className="mt-1 text-[11px] text-muted-foreground">
            Flies the waypoint mission on the box; the filmed run lands in Runs.
          </div>
        </div>
      </div>
    </section>
  );
}

export default function Environments() {
  const { scenarios } = useVesper();
  return (
    <main className="min-h-0 flex-1 overflow-y-auto p-4">
      <div className="mb-3.5 flex items-baseline gap-3">
        <h2 className="text-base font-bold">Environments</h2>
        <span className="text-xs text-muted-foreground">
          mission scenarios for this site — launch straight to the GPU box
        </span>
      </div>
      {scenarios === null ? (
        <div className="p-10 text-center text-muted-foreground">Loading scenarios…</div>
      ) : scenarios.length === 0 ? (
        <div className="p-10 text-center text-muted-foreground">
          No scenario specs found at the repo root
        </div>
      ) : (
        <div className="grid grid-cols-[repeat(auto-fill,minmax(380px,1fr))] gap-3">
          {scenarios.map((s) => (
            <ScenarioCard key={s.file} s={s} />
          ))}
        </div>
      )}
    </main>
  );
}
