"use client";

import { useEffect, useState } from "react";
import { Button } from "@/components/ui/button";
import { fetchJSON, postJSON, type Job } from "@/lib/vesper";
import { cn } from "@/lib/utils";

// Point-and-click job controls. No commands anywhere — buttons post to the
// jobs API, which runs whitelisted work on the GPU box.

export function JobButton({
  label, body, variant = "default", onLaunched, className,
}: {
  label: string;
  body: { kind: Job["kind"]; policy?: string; scenario?: string; world?: string; map?: string };
  variant?: "default" | "secondary";
  onLaunched?: (id: string) => void;
  className?: string;
}) {
  const [state, setState] = useState<"idle" | "busy" | "sent">("idle");
  const [error, setError] = useState<string | null>(null);

  const launch = async () => {
    setState("busy");
    setError(null);
    try {
      const r = await postJSON<{ id: string }>("/api/jobs", body);
      setState("sent");
      onLaunched?.(r.id);
      setTimeout(() => setState("idle"), 2500);
    } catch (e) {
      setError(e instanceof Error ? e.message : "launch failed");
      setState("idle");
    }
  };

  return (
    <span className={className}>
      <Button
        variant={variant}
        size="sm"
        disabled={state !== "idle"}
        onClick={launch}
        className="h-7 cursor-pointer px-3 font-mono text-[11px] tracking-[0.08em]"
      >
        {state === "busy" ? "LAUNCHING…" : state === "sent" ? "✓ LAUNCHED" : label}
      </Button>
      {error && <span className="ml-2 text-[11px] text-[#d03b3b]">{error}</span>}
    </span>
  );
}

export function SyncButton({ onSynced }: { onSynced?: () => void }) {
  const [busy, setBusy] = useState(false);
  const [note, setNote] = useState<string | null>(null);
  return (
    <span>
      <Button
        variant="secondary"
        size="sm"
        disabled={busy}
        className="h-7 cursor-pointer px-3 font-mono text-[11px] tracking-[0.08em]"
        onClick={async () => {
          setBusy(true);
          setNote(null);
          try {
            await postJSON("/api/sync");
            setNote("synced");
            onSynced?.();
          } catch (e) {
            setNote(e instanceof Error ? e.message : "sync failed");
          } finally {
            setBusy(false);
            setTimeout(() => setNote(null), 3000);
          }
        }}
      >
        {busy ? "PULLING…" : "⇣ PULL ARTIFACTS"}
      </Button>
      {note && <span className="ml-2 text-[11px] text-muted-foreground">{note}</span>}
    </span>
  );
}

const KIND_LABEL: Record<Job["kind"], string> = {
  train: "training", fly: "sortie", eval: "evaluation", mission: "mission",
  live: "live session", warm: "warm session",
};

export function JobsPanel({ pollMs = 6000, className }: { pollMs?: number; className?: string }) {
  const [jobs, setJobs] = useState<Job[] | null>(null);
  const [nowS, setNowS] = useState<number | null>(null); // clock frozen per poll, pure render

  useEffect(() => {
    let alive = true;
    const poll = () =>
      fetchJSON<Job[]>("/api/jobs").then((d) => {
        if (!alive || !d) return;
        setJobs(d);
        setNowS(Date.now() / 1000);
      });
    poll();
    const id = setInterval(poll, pollMs);
    return () => {
      alive = false;
      clearInterval(id);
    };
  }, [pollMs]);

  const elapsed = (j: Job) => {
    const end = j.finished ?? nowS ?? j.started;
    const s = Math.max(0, Math.round(end - j.started));
    return s >= 90 ? `${Math.round(s / 60)} min` : `${s} s`;
  };

  return (
    <section
      className={cn(
        "hud-corners flex flex-col rounded-lg border border-border bg-card",
        className,
      )}
    >
      <h3 className="flex shrink-0 items-center border-b border-border px-3 py-1.5 font-mono text-[10px] font-semibold uppercase tracking-[0.14em] text-secondary-foreground">
        <span className="mr-1.5 text-muted-foreground">▮</span>Jobs on the box
        <span className="ml-auto"><SyncButton /></span>
      </h3>
      {jobs === null ? (
        <div className="p-3 text-xs text-muted-foreground">checking…</div>
      ) : jobs.length === 0 ? (
        <div className="p-3 text-xs text-muted-foreground">
          nothing launched yet — deploy a model or start a mission
        </div>
      ) : (
        <div className="min-h-0 overflow-y-auto">
          {jobs.map((j) => (
            <div key={j.id} className="border-b border-border/50 px-3 py-2 last:border-0">
              <div className="flex items-center gap-2 text-xs">
                <span
                  className={cn(
                    "inline-block size-[7px] rounded-full",
                    j.status === "running" ? "animate-pulse bg-[#0ca30c]"
                      : j.status === "done" ? "bg-[#3987e5]"
                        : j.status === "failed" ? "bg-[#d03b3b]" : "bg-[#8a897f]",
                  )}
                />
                <span className="font-mono font-semibold">{KIND_LABEL[j.kind]}</span>
                {j.policy && (
                  <span className="truncate font-mono text-[10px] text-muted-foreground">
                    {j.policy.replace(/^runs\//, "").replace(/\.pt$/, "").replace("/", " · ")}
                  </span>
                )}
                <span className="ml-auto shrink-0 font-mono text-[10px] tabular-nums text-muted-foreground">
                  {j.status} · {elapsed(j)}
                </span>
                {j.status === "running" && (
                  <Button
                    variant="secondary"
                    size="sm"
                    className="h-5 cursor-pointer px-2 font-mono text-[9px]"
                    onClick={() => postJSON(`/api/jobs/${j.id}/stop`).catch(() => {})}
                  >
                    STOP
                  </Button>
                )}
              </div>
              {(j.status === "running" || j.status === "failed") && j.log && (
                <pre className="mt-1.5 max-h-20 overflow-y-auto whitespace-pre-wrap break-all rounded-sm bg-background p-2 font-mono text-[10px] leading-relaxed text-muted-foreground">
                  {j.log.trimEnd().split("\n").slice(-4).join("\n")}
                </pre>
              )}
            </div>
          ))}
        </div>
      )}
    </section>
  );
}
