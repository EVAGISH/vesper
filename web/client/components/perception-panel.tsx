"use client";

import { useCallback, useEffect, useState } from "react";
import { Button } from "@/components/ui/button";
import {
  fetchJSON, fmtBytes, fmtTime, GEOMETRIC, postJSON,
  type Detector, type DetectorStatus,
} from "@/lib/vesper";
import { cn } from "@/lib/utils";

// Perception: what decides that the drone has SEEN a vehicle during training.
// The built-in sensor model is the default and always available; a detector is
// a checkpoint that has to be running on the GPU box before a training run can
// send it frames, so deploying is a real action with a real wait.

const POLL_MS = 8000;

export function useDetectors() {
  const [detectors, setDetectors] = useState<Detector[] | null>(null);
  const [status, setStatus] = useState<DetectorStatus | null>(null);

  const refresh = useCallback(async () => {
    const list = await fetchJSON<Detector[]>("/api/detectors");
    if (list) setDetectors(list);
    const s = await fetchJSON<DetectorStatus>("/api/detectors/status");
    if (s) setStatus(s);
    return s;
  }, []);

  useEffect(() => {
    let alive = true;
    let timer: ReturnType<typeof setTimeout>;
    const tick = async () => {
      const s = await refresh();
      if (!alive) return;
      // polling status costs an ssh round trip to the box: only keep it up
      // while something is actually deployed
      timer = setTimeout(tick, s?.id ? POLL_MS : POLL_MS * 4);
    };
    tick();
    return () => {
      alive = false;
      clearTimeout(timer);
    };
  }, [refresh]);

  return { detectors, status, refresh };
}

function Metrics({ d }: { d: Detector }) {
  const m = d.metrics ?? {};
  const bits: string[] = [];
  if (m.precision !== undefined && m.recall !== undefined)
    bits.push(`P ${m.precision.toFixed(2)} · R ${m.recall.toFixed(2)}${m.split ? ` (${m.split})` : ""}`);
  if (m.images) bits.push(`${m.images} synthetic frames`);
  if (m.epochs) bits.push(`${m.epochs} epochs`);
  if (!bits.length) return null;
  return <span className="tabular-nums">{bits.join(" · ")}</span>;
}

export function PerceptionPanel({
  detectors, status, selected, onSelect, onRefresh,
}: {
  detectors: Detector[] | null;
  status: DetectorStatus | null;
  selected: string;
  onSelect: (id: string) => void;
  onRefresh: () => void;
}) {
  const [busy, setBusy] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);

  const deploy = async (id: string) => {
    setBusy(id);
    setError(null);
    try {
      await postJSON("/api/detectors/deploy", { id });
      onSelect(id);
      onRefresh();
    } catch (e) {
      setError(e instanceof Error ? e.message : "deploy failed");
    } finally {
      setBusy(null);
    }
  };

  const stop = async () => {
    setBusy("stop");
    setError(null);
    try {
      await postJSON("/api/detectors/stop");
      onSelect(GEOMETRIC);
      onRefresh();
    } catch (e) {
      setError(e instanceof Error ? e.message : "stop failed");
    } finally {
      setBusy(null);
    }
  };

  return (
    <section className="hud-corners rounded-lg border border-border bg-card">
      <h3 className="flex items-center border-b border-border px-3 py-1.5 font-mono text-[10px] font-semibold uppercase tracking-[0.14em] text-secondary-foreground">
        <span className="mr-1.5 text-muted-foreground">▮</span>Perception
        <span className="ml-auto font-mono font-normal normal-case tracking-normal text-muted-foreground">
          what counts as a sighting
        </span>
      </h3>

      {detectors === null ? (
        <div className="p-3 text-xs text-muted-foreground">loading…</div>
      ) : (
        <div>
          {detectors.map((d) => {
            const active = selected === d.id;
            const serving = status?.id === d.id;
            return (
              <div
                key={d.id}
                onClick={() => onSelect(d.id)}
                className={cn(
                  "cursor-pointer border-b border-border/50 px-3 py-2 last:border-0 hover:bg-secondary",
                  active && "bg-secondary shadow-[inset_2px_0_0_#3987e5]",
                )}
              >
                <div className="flex items-center gap-2 text-xs">
                  <span
                    className={cn(
                      "inline-block size-[9px] shrink-0 rounded-full border",
                      active ? "border-[#3987e5] bg-[#3987e5]" : "border-muted-foreground",
                    )}
                  />
                  <span className="truncate font-mono font-semibold">{d.name}</span>
                  {d.kind === "builtin" && (
                    <span className="shrink-0 rounded-sm border border-border px-1 font-mono text-[9px] uppercase tracking-[0.08em] text-muted-foreground">
                      default
                    </span>
                  )}
                  {serving && (
                    <span
                      className={cn(
                        "shrink-0 rounded-sm px-1 font-mono text-[9px] uppercase tracking-[0.08em]",
                        status?.ready ? "text-[#0ca30c]" : "text-[#c98500]",
                      )}
                    >
                      {status?.ready ? "● serving" : `● ${status?.status ?? "loading"}`}
                    </span>
                  )}
                  <span className="ml-auto shrink-0 font-mono text-[10px] tabular-nums text-muted-foreground">
                    {d.bytes ? fmtBytes(d.bytes) : ""}
                    {d.mtime ? ` · ${fmtTime(d.mtime)}` : ""}
                  </span>
                </div>
                <div className="mt-1 pl-[17px] text-[11px] leading-relaxed text-muted-foreground">
                  {d.detail}
                </div>
                <div className="mt-0.5 flex flex-wrap items-center gap-2 pl-[17px] text-[11px] text-secondary-foreground">
                  <Metrics d={d} />
                  {d.deployable && (
                    serving ? (
                      <Button
                        size="sm" variant="secondary" disabled={busy !== null}
                        onClick={(e) => { e.stopPropagation(); stop(); }}
                        className="h-6 cursor-pointer px-2 font-mono text-[10px] tracking-[0.08em]"
                      >
                        ■ STOP
                      </Button>
                    ) : (
                      <Button
                        size="sm" variant="secondary" disabled={busy !== null}
                        onClick={(e) => { e.stopPropagation(); deploy(d.id); }}
                        className="h-6 cursor-pointer px-2 font-mono text-[10px] tracking-[0.08em]"
                      >
                        {busy === d.id ? "DEPLOYING…" : "▲ DEPLOY TO BOX"}
                      </Button>
                    )
                  )}
                </div>
              </div>
            );
          })}
        </div>
      )}

      {error && <div className="px-3 py-2 text-[11px] text-[#d03b3b]">{error}</div>}
      {status?.error && (
        <div className="border-t border-border/60 px-3 py-2 font-mono text-[11px] text-[#d03b3b]">
          {status.error}
        </div>
      )}
      <div className="border-t border-border/60 px-3 py-2 text-[11px] leading-relaxed text-muted-foreground">
        A deployed detector runs beside the simulator on the GPU box and reads the
        drone&apos;s rendered camera. Training against one replaces the built-in
        sensor: the policy only knows a vehicle is there when the network puts a
        box on it.
      </div>
    </section>
  );
}
