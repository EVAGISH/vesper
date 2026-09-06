"use client";

import { useMemo, useState } from "react";
import { Badge } from "@/components/ui/badge";
import {
  artifactLabel, fmtDur, fmtTime, KIND_COLOR, runKind, runLane,
  type Run, type RunLane,
} from "@/lib/vesper";
import { cn } from "@/lib/utils";

function dayLabel(ts?: number) {
  return ts
    ? new Date(ts * 1000).toLocaleDateString(undefined, {
        weekday: "short", month: "short", day: "numeric",
      })
    : "undated";
}

const LANES: { id: RunLane | "all"; label: string }[] = [
  { id: "all", label: "All" },
  { id: "operations", label: "Sorties" },
  { id: "lab", label: "Lab" },
];

export function RunRail({
  runs,
  activeId,
  onSelect,
}: {
  runs: Run[];
  activeId: string | null;
  onSelect: (id: string) => void;
}) {
  const [lane, setLane] = useState<RunLane | "all">("all");
  const counts = useMemo(() => {
    const c = { operations: 0, lab: 0 };
    for (const r of runs) c[runLane(r)]++;
    return c;
  }, [runs]);
  const shown = useMemo(
    () => (lane === "all" ? runs : runs.filter((r) => runLane(r) === lane)),
    [runs, lane],
  );

  return (
    <aside className="flex w-[292px] shrink-0 flex-col overflow-y-auto border-r border-border bg-card">
      <div className="sticky top-0 z-10 flex gap-0.5 border-b border-border bg-card p-1.5">
        {LANES.map((l) => {
          const n = l.id === "all" ? runs.length : counts[l.id];
          return (
            <button
              key={l.id}
              onClick={() => setLane(l.id)}
              className={cn(
                "flex-1 cursor-pointer rounded px-2 py-1 font-mono text-[10px] uppercase tracking-[0.1em]",
                lane === l.id
                  ? "bg-secondary text-foreground"
                  : "text-muted-foreground hover:text-foreground",
              )}
            >
              {l.label} <span className="opacity-60">{n}</span>
            </button>
          );
        })}
      </div>
      {shown.length === 0 && (
        <div className="p-8 text-center text-muted-foreground">
          {runs.length === 0
            ? "No runs yet — pull artifacts from the box (Jobs panel)"
            : `No ${lane === "operations" ? "sorties" : "lab runs"} yet`}
        </div>
      )}
      {shown.map((r, i) => {
        const day = dayLabel(r.manifest.started);
        const head = i === 0 || day !== dayLabel(shown[i - 1].manifest.started);
        const kind = runKind(r);
        return (
          <div key={r.id}>
            {head && (
              <div className="px-3.5 pb-1 pt-2 font-mono text-[10px] uppercase tracking-[0.14em] text-muted-foreground">
                {day}
              </div>
            )}
            <button
              onClick={() => onSelect(r.id)}
              className={cn(
                "block w-full cursor-pointer border-b border-l-[3px] border-b-black/20 border-l-transparent px-3.5 py-2 text-left hover:bg-secondary",
                r.id === activeId && "border-l-[#3987e5] bg-secondary",
              )}
            >
              <div className="flex items-baseline text-xs font-semibold">
                <span className="truncate">{r.manifest.name || r.id}</span>
                <span
                  className="ml-auto shrink-0 pl-2 font-mono text-[9.5px] font-normal uppercase tracking-[0.08em]"
                  style={{ color: KIND_COLOR[kind] }}
                >
                  ● {kind}
                </span>
              </div>
              <div className="mt-px text-[11px] tabular-nums text-muted-foreground">
                {fmtTime(r.manifest.started)}
                {fmtDur(r.manifest) && ` · ${fmtDur(r.manifest)}`}
              </div>
              <div className="mt-1.5 flex flex-wrap gap-1">
                {r.files
                  .filter((f) => f !== "manifest.json")
                  .map((f) => (
                    <Badge
                      key={f}
                      variant="outline"
                      className="rounded-none bg-background px-1.5 py-0 font-mono text-[9px] font-normal text-secondary-foreground"
                    >
                      {artifactLabel(f)}
                    </Badge>
                  ))}
              </div>
            </button>
          </div>
        );
      })}
    </aside>
  );
}
