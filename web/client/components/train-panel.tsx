"use client";

import { useCallback, useEffect, useMemo, useState } from "react";
import { Button } from "@/components/ui/button";
import { fetchJSON, postJSON } from "@/lib/vesper";

// "Train a new model" — the one launcher for the fast native lane
// (POST /api/train/*): pick a trainable world, set iters, watch live
// iter/metrics from the box, stop or wait for the run to land in Runs with
// its three.js progression reel. Mounted on the Models page and the
// Environments page (where the per-world TRAIN HERE buttons call startTrain).

export type TrainStatus = {
  running: boolean;
  status?: "launching" | "running" | "pulling" | "rendering" | "done" | "failed" | "stopped";
  world?: string;
  iters?: number;
  num_envs?: number;
  iter?: number | null;
  metrics?: Record<string, number>;
  run_id?: string | null;
  error?: string | null;
  started?: number;
  log?: string;
};

type TrainableWorld = { name: string; map: string | null };

const POLL_MS = 3000;
const WORLDS_MS = 15000;

export function useTrainControl() {
  const [status, setStatus] = useState<TrainStatus | null>(null);
  const [worlds, setWorlds] = useState<TrainableWorld[] | null>(null);
  const [active, setActive] = useState<string | null>(null);
  const [world, setWorld] = useState("");
  const [iters, setIters] = useState("150");
  const [numEnvs, setNumEnvs] = useState("4096");
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState<string | null>(null);

  useEffect(() => {
    let alive = true;
    const tick = () => fetchJSON<TrainStatus>("/api/train/status").then((s) => alive && s && setStatus(s));
    tick();
    const id = setInterval(tick, POLL_MS);
    return () => { alive = false; clearInterval(id); };
  }, []);

  useEffect(() => {
    let alive = true;
    const load = () => {
      // /api/train/worlds is a cheap glob (no USD reads), so the dropdown
      // populates instantly instead of blocking ~6 s on /api/environments
      fetchJSON<TrainableWorld[]>("/api/train/worlds").then((d) => alive && d && setWorlds(d));
      fetchJSON<{ name: string } | null>("/api/active").then((d) => alive && setActive(d?.name ?? null));
    };
    load();
    const id = setInterval(load, WORLDS_MS);
    return () => { alive = false; clearInterval(id); };
  }, []);

  const trainable = useMemo(() => (worlds ?? []).filter((w) => w.map), [worlds]);
  const loadingWorlds = worlds === null;

  // default the picker to the active AO (when trainable), else the last-trained
  // world, else the first trainable — never a hardcoded name
  useEffect(() => {
    if (world && trainable.some((w) => w.name === world)) return;
    const pick = trainable.find((w) => w.name === active)?.name
      ?? trainable.find((w) => w.name === status?.world)?.name
      ?? trainable[0]?.name;
    if (pick) setWorld(pick);
  }, [trainable, active, status, world]);

  const startTrain = useCallback(async (w?: string) => {
    const target = w ?? world;
    if (!target) return;
    setBusy(true); setErr(null);
    try {
      await postJSON("/api/train/start", {
        world: target,
        iters: parseInt(iters, 10) || 150,
        num_envs: parseInt(numEnvs, 10) || 4096,
      });
      setWorld(target);
      const s = await fetchJSON<TrainStatus>("/api/train/status");
      if (s) setStatus(s);
    } catch (e) {
      setErr(e instanceof Error ? e.message : "training failed to start");
    } finally {
      setBusy(false);
    }
  }, [world, iters, numEnvs]);

  const stopTrain = useCallback(async () => {
    setBusy(true); setErr(null);
    try {
      await postJSON("/api/train/stop", {});
      const s = await fetchJSON<TrainStatus>("/api/train/status");
      if (s) setStatus(s);
    } catch (e) {
      setErr(e instanceof Error ? e.message : "stop failed");
    } finally {
      setBusy(false);
    }
  }, []);

  return { status, trainable, loadingWorlds, world, setWorld, iters, setIters, numEnvs, setNumEnvs,
           busy, err, startTrain, stopTrain,
           blocked: !!status?.running || busy };
}

export type TrainControl = ReturnType<typeof useTrainControl>;

const TRAIN_PHASE: Record<string, string> = {
  launching: "launching on the GPU box…",
  running: "training",
  pulling: "training done · pulling the run home…",
  rendering: "rendering the progression reel…",
};

const inputCls = "rounded border border-border bg-background px-2 py-1.5 font-mono text-xs text-foreground outline-none focus:border-[#3987e5] disabled:opacity-50";
const labelCls = "font-mono text-[10px] uppercase tracking-widest text-muted-foreground";
const btnCls = "h-7 cursor-pointer px-2.5 font-mono text-[10px] tracking-[0.08em]";

export function TrainPanel({ ctl }: { ctl: TrainControl }) {
  const { status, trainable, loadingWorlds, world, setWorld, iters, setIters, numEnvs, setNumEnvs,
          busy, err, startTrain, stopTrain } = ctl;
  const active = !!status?.running;
  const m = status?.metrics ?? {};
  const total = status?.iters ?? 0;
  const it = status?.iter ?? null;
  const pct = active && total && it != null ? Math.min(100, Math.round(((it + 1) / total) * 100)) : null;
  const fmtM = (k: string) => (m[k] != null ? m[k].toFixed(2) : "—");
  return (
    <section className="hud-corners rounded-lg border border-border bg-card">
      <h3 className="flex items-center border-b border-border px-3 py-1.5 font-mono text-[10px] font-semibold uppercase tracking-[0.14em] text-secondary-foreground">
        <span className="mr-2 rounded bg-secondary px-1.5 text-muted-foreground">◉</span>Train a new model
        <span className="ml-auto font-normal normal-case tracking-normal text-muted-foreground">native lane · GPU box</span>
      </h3>
      <div className="flex flex-col gap-2.5 p-3">
        <label className="flex flex-col gap-1">
          <span className={labelCls}>environment</span>
          <select value={world} disabled={active || trainable.length === 0}
            onChange={(e) => setWorld(e.target.value)} className={inputCls}>
            {trainable.length === 0 && (
              <option value="">{loadingWorlds ? "loading worlds…" : "no trainable worlds"}</option>
            )}
            {trainable.map((w) => (
              <option key={w.name} value={w.name}>{w.name.replace(/_/g, " ")}</option>
            ))}
          </select>
        </label>
        <div className="flex gap-2">
          <label className="flex flex-1 flex-col gap-1">
            <span className={labelCls}>iterations</span>
            <input inputMode="numeric" value={iters} disabled={active}
              onChange={(e) => setIters(e.target.value.replace(/[^0-9]/g, ""))} className={inputCls} />
          </label>
          <label className="flex flex-1 flex-col gap-1">
            <span className={labelCls}>parallel envs</span>
            <input inputMode="numeric" value={numEnvs} disabled={active}
              onChange={(e) => setNumEnvs(e.target.value.replace(/[^0-9]/g, ""))} className={inputCls} />
          </label>
        </div>
        {active ? (
          <div className="flex flex-col gap-2">
            <div className="rounded border border-border/60 bg-background/60 px-2.5 py-2 font-mono text-[10px] leading-relaxed">
              <div className="text-[#eda100]">
                {status?.world?.replace(/_/g, " ")} · {TRAIN_PHASE[status?.status ?? ""] ?? status?.status}
              </div>
              {status?.status === "running" && (
                <>
                  <div className="text-foreground">
                    iter {it != null ? it + 1 : "…"} / {total}
                    {pct != null && ` · ${pct}%`}
                  </div>
                  <div className="text-muted-foreground">
                    found {fmtM("found")} · cleared {fmtM("cleared")} · swept {fmtM("coverage")}
                  </div>
                  {pct != null && (
                    <div className="mt-1.5 h-1 overflow-hidden rounded bg-border">
                      <div className="h-full bg-[#eda100] transition-all" style={{ width: `${pct}%` }} />
                    </div>
                  )}
                </>
              )}
            </div>
            {(status?.status === "launching" || status?.status === "running") && (
              <Button size="sm" variant="secondary" onClick={stopTrain} disabled={busy} className={btnCls}>
                ■ STOP TRAINING
              </Button>
            )}
          </div>
        ) : (
          <Button size="sm" onClick={() => startTrain()} disabled={busy || !world} className={btnCls}>
            {busy ? "STARTING…" : "◉ TRAIN"}
          </Button>
        )}
        {!active && status?.status === "done" && status.run_id && (
          <div className="text-[11px] text-[#3ddc97]">
            ✓ trained → <span className="font-mono">{status.run_id}</span> in the Runs tab (progression reel + search.pt)
          </div>
        )}
        {!active && status?.status === "failed" && (
          <div className="text-[11px] text-[#d03b3b]">training failed: {status.error ?? "see server log"}</div>
        )}
        {err && <div className="text-[11px] text-[#d03b3b]">{err}</div>}
        {!active && (
          <div className="text-[11px] text-muted-foreground">
            trains scripts/train_search_native.py on the box&apos;s CUDA; the finished policy lands in Runs with a three.js progression reel (~10 min at the defaults)
          </div>
        )}
      </div>
    </section>
  );
}
