"use client";

import Link from "next/link";
import { useEffect, useState } from "react";
import { JobButton, JobsPanel } from "@/components/job-controls";
import { Button } from "@/components/ui/button";
import { fetchJSON, fmtBytes, fmtTime, modelLabel, postJSON, type Model } from "@/lib/vesper";
import { cn } from "@/lib/utils";

function ExportToHardware({ model }: { model: Model }) {
  const [state, setState] = useState<"idle" | "busy" | "done" | "error">(
    model.onnx ? "done" : "idle",
  );
  const [url, setUrl] = useState<string | null>(
    model.onnx ? `/download/${model.run}/${model.onnx.split("/").pop()}` : null,
  );
  const [note, setNote] = useState<string | null>(null);

  const run = async () => {
    setState("busy"); setNote(null);
    try {
      const r = await postJSON<{ url: string; bytes: number }>(
        "/api/models/export", { policy: model.path });
      setUrl(r.url); setState("done");
      setNote(`${Math.round(r.bytes / 1024)} KB ONNX — runs on a Jetson via TensorRT`);
    } catch (e) {
      setState("error"); setNote(e instanceof Error ? e.message : "export failed");
    }
  };

  return (
    <div>
      <div className="flex flex-wrap items-center gap-2">
        <Button
          size="sm" variant="secondary" disabled={state === "busy"} onClick={run}
          className="h-7 cursor-pointer px-3 font-mono text-[11px] tracking-[0.08em]"
        >
          {state === "busy" ? "EXPORTING…" : "⬇ EXPORT TO HARDWARE (ONNX)"}
        </Button>
        {url && state === "done" && (
          <a href={url} download
            className="font-mono text-[11px] text-[#3987e5] underline-offset-2 hover:underline">
            download .onnx
          </a>
        )}
      </div>
      <div className="mt-1 text-[11px] text-muted-foreground">
        {note ?? "Folds in the normalizer and exports the policy to ONNX — the sim→drone step."}
      </div>
    </div>
  );
}

// The train → deploy loop, point and click. Checkpoints are what training
// writes into runs/<id>/ (*.pt); deploying one flies or scores it on the box.

const POLL_MS = 15000;

export default function Models() {
  const [models, setModels] = useState<Model[] | null>(null);
  const [selected, setSelected] = useState<string | null>(null);

  useEffect(() => {
    let alive = true;
    const poll = () =>
      fetchJSON<Model[]>("/api/models").then((d) => alive && d && setModels(d));
    poll();
    const id = setInterval(poll, POLL_MS);
    return () => {
      alive = false;
      clearInterval(id);
    };
  }, []);

  const active = models?.find((m) => m.path === selected) ?? models?.[0] ?? null;

  return (
    <main className="min-h-0 flex-1 overflow-y-auto p-4">
      <div className="mb-3.5 flex items-baseline gap-3">
        <h2 className="text-base font-bold">Models</h2>
        <span className="text-xs text-muted-foreground">
          trained flight models — pick one, deploy it
        </span>
      </div>

      <div className="grid grid-cols-1 gap-3 lg:grid-cols-[minmax(0,1fr)_400px]">
        <section className="hud-corners rounded-lg border border-border bg-card">
          <h3 className="flex items-center border-b border-border px-3 py-1.5 font-mono text-[10px] font-semibold uppercase tracking-[0.14em] text-secondary-foreground">
            <span className="mr-1.5 text-muted-foreground">▮</span>Model library
          </h3>
          {models === null ? (
            <div className="p-6 text-sm text-muted-foreground">loading…</div>
          ) : models.length === 0 ? (
            <div className="p-6 text-sm text-muted-foreground">
              No models in the library yet — start a training job, then pull
              artifacts (Jobs panel).
            </div>
          ) : (
            <table className="w-full border-collapse text-xs">
              <thead>
                <tr className="border-b border-border font-mono text-[10px] uppercase tracking-[0.1em] text-muted-foreground">
                  <th className="px-3 py-1.5 text-left font-normal">model</th>
                  <th className="px-3 py-1.5 text-left font-normal">trained by</th>
                  <th className="px-3 py-1.5 text-right font-normal">final metrics</th>
                  <th className="px-3 py-1.5 text-right font-normal">size</th>
                  <th className="px-3 py-1.5 text-right font-normal">trained</th>
                </tr>
              </thead>
              <tbody>
                {models.map((m) => (
                  <tr
                    key={m.path}
                    onClick={() => setSelected(m.path)}
                    className={cn(
                      "cursor-pointer border-b border-border/50 hover:bg-secondary",
                      active?.path === m.path && "bg-secondary shadow-[inset_2px_0_0_#3987e5]",
                    )}
                  >
                    <td className="px-3 py-2 font-mono">{modelLabel(m)}</td>
                    <td className="px-3 py-2">
                      <Link
                        href={`/runs?run=${encodeURIComponent(m.run)}`}
                        className="font-mono text-[#3987e5] underline-offset-2 hover:underline"
                        onClick={(e) => e.stopPropagation()}
                      >
                        {m.run}
                      </Link>
                    </td>
                    <td className="px-3 py-2 text-right tabular-nums text-secondary-foreground">
                      {Object.entries(m.metrics)
                        .filter(([k]) => k !== "iter" && k !== "iteration" && k !== "step")
                        .slice(0, 3)
                        .map(([k, v]) => `${k} ${Math.abs(v) >= 10 ? v.toFixed(1) : v.toFixed(3)}`)
                        .join(" · ") || "—"}
                    </td>
                    <td className="px-3 py-2 text-right tabular-nums text-secondary-foreground">
                      {fmtBytes(m.bytes)}
                    </td>
                    <td className="px-3 py-2 text-right tabular-nums text-muted-foreground">
                      {fmtTime(m.mtime)}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          )}
        </section>

        <div className="flex flex-col gap-3">
          <section className="hud-corners rounded-lg border border-border bg-card">
            <h3 className="flex items-center border-b border-border px-3 py-1.5 font-mono text-[10px] font-semibold uppercase tracking-[0.14em] text-secondary-foreground">
              <span className="mr-1.5 text-muted-foreground">▮</span>Deploy
              {active && (
                <span className="ml-auto truncate font-mono font-normal normal-case tracking-normal text-muted-foreground">
                  {modelLabel(active)} · {active.run}
                </span>
              )}
            </h3>
            {active ? (
              <div className="flex flex-col gap-3 p-3">
                <div>
                  <JobButton
                    label="▶ FLY SORTIE"
                    body={{ kind: "fly", policy: active.path }}
                  />
                  <div className="mt-1 text-[11px] text-muted-foreground">
                    Filmed search-and-reach flight — fpv, chase, track and events
                    land in Runs when it finishes.
                  </div>
                </div>
                <div>
                  <JobButton
                    label="◎ SCORE POLICY"
                    variant="secondary"
                    body={{ kind: "eval", policy: active.path }}
                  />
                  <div className="mt-1 text-[11px] text-muted-foreground">
                    Full funnel over 400 episodes — swept, found, reached, and how
                    each episode ended.
                  </div>
                </div>
                <div className="border-t border-border/60 pt-3">
                  <ExportToHardware key={active.path} model={active} />
                </div>
              </div>
            ) : (
              <div className="p-3 text-xs text-muted-foreground">select a model</div>
            )}
          </section>

          <section className="hud-corners rounded-lg border border-border bg-card">
            <h3 className="flex items-center border-b border-border px-3 py-1.5 font-mono text-[10px] font-semibold uppercase tracking-[0.14em] text-secondary-foreground">
              <span className="mr-1.5 text-muted-foreground">▮</span>Train a new model
            </h3>
            <div className="p-3">
              <JobButton label="▲ START TRAINING" body={{ kind: "train" }} />
              <div className="mt-1 text-[11px] text-muted-foreground">
                Search-and-reach on the Cornell world, 1500 iterations. Progress
                appears in Runs as a training curve; the model lands here.
              </div>
            </div>
          </section>

          <JobsPanel />
        </div>
      </div>
    </main>
  );
}
