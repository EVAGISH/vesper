"use client";

import { Badge } from "@/components/ui/badge";
import { artifactLabel, fmtDur, fmtTime, KIND_COLOR, runKind, type Run } from "@/lib/vesper";
import { cn } from "@/lib/utils";

function dayLabel(ts?: number) {
  return ts
    ? new Date(ts * 1000).toLocaleDateString(undefined, {
        weekday: "short", month: "short", day: "numeric",
      })
    : "undated";
}

export function RunRail({
  runs,
  activeId,
  onSelect,
}: {
  runs: Run[];
  activeId: string | null;
  onSelect: (id: string) => void;
}) {
  return (
    <aside className="flex w-[292px] shrink-0 flex-col overflow-y-auto border-r border-border bg-card">
      {runs.length === 0 && (
        <div className="p-8 text-center text-muted-foreground">
          No runs yet — pull artifacts from the box (Jobs panel)
        </div>
      )}
      {runs.map((r, i) => {
        const day = dayLabel(r.manifest.started);
        const head = i === 0 || day !== dayLabel(runs[i - 1].manifest.started);
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
