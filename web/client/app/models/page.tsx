"use client";

import Link from "next/link";
import { useEffect, useState } from "react";
import { CommandBar } from "@/components/command-bar";
import { fetchJSON, fmtBytes, fmtTime, SIM, type Model } from "@/lib/vesper";
import { cn } from "@/lib/utils";

// The train → deploy loop. Checkpoints are what train_* writes into runs/<id>/
// (*.pt); deploying one means flying or scoring it with that checkpoint path.

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
          policy checkpoints written by training runs — pick one to deploy
        </span>
      </div>

      <div className="grid grid-cols-1 gap-3 lg:grid-cols-[minmax(0,1fr)_420px]">
        <section className="hud-corners rounded-lg border border-border bg-card">
          <h3 className="flex items-center border-b border-border px-3 py-1.5 font-mono text-[10px] font-semibold uppercase tracking-[0.14em] text-secondary-foreground">
            <span className="mr-1.5 text-muted-foreground">▮</span>Checkpoints
          </h3>
          {models === null ? (
            <div className="p-6 text-sm text-muted-foreground">loading…</div>
          ) : models.length === 0 ? (
            <div className="p-6 text-sm text-muted-foreground">
              No checkpoints under runs/ yet — train one (panel on the right), then
              pull it with scripts/capture_pull.sh.
            </div>
          ) : (
            <table className="w-full border-collapse text-xs">
              <thead>
                <tr className="border-b border-border font-mono text-[10px] uppercase tracking-[0.1em] text-muted-foreground">
                  <th className="px-3 py-1.5 text-left font-normal">checkpoint</th>
                  <th className="px-3 py-1.5 text-left font-normal">training run</th>
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
                    <td className="px-3 py-2 font-mono">{m.file}</td>
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
                <span className="ml-auto font-mono font-normal normal-case tracking-normal text-muted-foreground">
                  {active.path}
                </span>
              )}
            </h3>
            {active ? (
              <div className="p-3">
                <div className="font-mono text-[10px] uppercase tracking-[0.12em] text-muted-foreground">
                  Fly it — filmed sortie (fpv + chase + track + events)
                </div>
                <CommandBar
                  command={`${SIM} scripts/fly_search.py --policy ${active.path} --seconds 90 --headless --enable_cameras`}
                />
                <div className="mt-3 font-mono text-[10px] uppercase tracking-[0.12em] text-muted-foreground">
                  Score it — full funnel over 400 episodes
                </div>
                <CommandBar
                  command={`${SIM} scripts/eval_search.py --policy ${active.path} --num_envs 256 --episodes 400 --headless`}
                />
                <div className="mt-3 text-[11px] text-muted-foreground">
                  Run on the GPU box; the sortie lands in Runs after
                  scripts/capture_pull.sh. One-click launch from here is the next
                  backend step (run-trigger endpoint).
                </div>
              </div>
            ) : (
              <div className="p-3 text-xs text-muted-foreground">select a checkpoint</div>
            )}
          </section>

          <section className="hud-corners rounded-lg border border-border bg-card">
            <h3 className="flex items-center border-b border-border px-3 py-1.5 font-mono text-[10px] font-semibold uppercase tracking-[0.14em] text-secondary-foreground">
              <span className="mr-1.5 text-muted-foreground">▮</span>Train a new model
            </h3>
            <div className="p-3">
              <div className="font-mono text-[10px] uppercase tracking-[0.12em] text-muted-foreground">
                Search-and-reach on the Cornell world (writes search.pt + curve.jsonl)
              </div>
              <CommandBar
                command={`${SIM} scripts/train_search.py --num_envs 1024 --iters 1500 --headless`}
              />
              <div className="mt-2 text-[11px] text-muted-foreground">
                Training progress shows up in Runs as the curve.jsonl chart once pulled.
              </div>
            </div>
          </section>
        </div>
      </div>
    </main>
  );
}
