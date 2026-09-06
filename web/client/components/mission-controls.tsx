"use client";

import { useEffect, useState } from "react";
import { Button } from "@/components/ui/button";
import { postJSON } from "@/lib/vesper";

// Mission controls for the NATIVE warm session — the demo's live lane.
// Start/Stop post to the runs server's /api/session endpoints, which launch or
// kill scripts/warm_session_native.py as a local subprocess (no Isaac, no
// droplet). Record posts {"kind":"record"} straight to the session's /command
// queue on 8180: the mission buffered since the last reset lands in Runs as
// replay.json + trajectory.parquet, with tactical.mp4 rendering behind it.

const STARTUP_POLL_MS = 2000;
const STARTUP_BAIL_MS = 90_000;

export function StartMissionButton({
  live,
  onChanged,
  compact,
}: {
  live: boolean;
  onChanged: () => void;
  compact?: boolean; // text-style button for panel headers
}) {
  const [starting, setStarting] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // after a successful start, re-poll /api/live fast until the session answers
  useEffect(() => {
    if (!starting) return;
    if (live) {
      setStarting(false);
      return;
    }
    const id = setInterval(onChanged, STARTUP_POLL_MS);
    const bail = setTimeout(() => {
      setStarting(false);
      setError("session did not come up — check .vesper_session.log");
    }, STARTUP_BAIL_MS);
    return () => {
      clearInterval(id);
      clearTimeout(bail);
    };
  }, [starting, live, onChanged]);

  if (live) return null;
  const start = async () => {
    setError(null);
    setStarting(true);
    try {
      await postJSON("/api/session/start", {});
    } catch (e) {
      setError(e instanceof Error ? e.message : "start failed");
      setStarting(false);
    }
  };
  const text = starting ? "STARTING…" : "▶ START MISSION";
  return (
    <span className="flex items-center gap-2">
      {compact ? (
        <button
          onClick={start}
          disabled={starting}
          className="cursor-pointer font-mono text-[10px] font-semibold normal-case tracking-[0.08em] text-[#0ca30c] hover:text-[#2ec52e] disabled:cursor-default"
          title="launch the native warm session on this machine"
        >
          {text.toLowerCase()}
        </button>
      ) : (
        <Button
          size="sm"
          disabled={starting}
          onClick={start}
          className="h-7 cursor-pointer px-3 font-mono text-[11px] tracking-[0.08em]"
        >
          {text}
        </Button>
      )}
      {error && <span className="text-[11px] text-[#d03b3b]">{error}</span>}
    </span>
  );
}

// Renderer FLAG for a recorded sortie — which replay lane(s) the keeper kicks
// off. three.js is the default: real-time on this machine, same scene as the
// live 3D view. Isaac remains the photoreal hero-shot lane on the GPU box.
const RENDER_CHOICES: { value: string; label: string; renderers: string }[] = [
  { value: "three", label: "3D (fast, on-device)", renderers: "three,tactical" },
  { value: "isaac", label: "photoreal (isaac, box, slow)", renderers: "isaac,tactical" },
  { value: "tactical", label: "tactical only", renderers: "tactical" },
];

/** Compact header controls while the native session is live: record + stop. */
export function LiveMissionControls({
  ip,
  onChanged,
}: {
  ip: string;
  onChanged: () => void;
}) {
  const [rec, setRec] = useState<"idle" | "busy" | "saved" | "failed">("idle");
  const [savedRun, setSavedRun] = useState<string | null>(null);
  const [stopping, setStopping] = useState(false);
  const [renderer, setRenderer] = useState("three");

  const record = async () => {
    setRec("busy");
    setSavedRun(null);
    try {
      const choice = RENDER_CHOICES.find((c) => c.value === renderer) ?? RENDER_CHOICES[0];
      const r = await fetch(`http://${ip}:8180/command`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ kind: "record", renderers: choice.renderers }),
      });
      if (!r.ok) throw new Error();
      // the dump is async in the session; grab the run id once it lands
      await new Promise((res) => setTimeout(res, 1500));
      const st = await fetch(`http://${ip}:8180/state`, { cache: "no-store" })
        .then((x) => (x.ok ? x.json() : null))
        .catch(() => null);
      setSavedRun(st?.last_record?.run ?? null);
      setRec("saved");
    } catch {
      setRec("failed");
    }
    setTimeout(() => setRec("idle"), 5000);
  };

  const stop = async () => {
    setStopping(true);
    try {
      await postJSON("/api/session/stop");
    } catch {}
    onChanged();
    setStopping(false);
  };

  return (
    <span className="flex items-center gap-3">
      <label className="flex items-center gap-1 font-mono text-[10px] normal-case tracking-[0.08em] text-muted-foreground">
        renderer
        <select
          value={renderer}
          onChange={(e) => setRenderer(e.target.value)}
          className="cursor-pointer border border-border/60 bg-background/60 px-1 py-0.5 font-mono text-[10px] text-secondary-foreground"
          title="which replay renderer a recorded sortie kicks off"
        >
          {RENDER_CHOICES.map((c) => (
            <option key={c.value} value={c.value}>{c.label}</option>
          ))}
        </select>
      </label>
      <button
        onClick={record}
        disabled={rec !== "idle"}
        className="cursor-pointer font-mono text-[10px] font-semibold normal-case tracking-[0.08em] text-[#d03b3b] hover:text-[#ff6b5b] disabled:cursor-default"
        title="save the mission so far to Runs (replay + the selected render lane)"
      >
        {rec === "busy"
          ? "◉ RECORDING…"
          : rec === "saved"
            ? `✓ saved to Runs${savedRun ? ` · ${savedRun}` : ""}`
            : rec === "failed"
              ? "record failed"
              : "◉ RECORD SORTIE"}
      </button>
      <button
        onClick={stop}
        disabled={stopping}
        className="cursor-pointer font-mono text-[10px] normal-case tracking-[0.08em] text-muted-foreground hover:text-foreground disabled:cursor-default"
        title="kill the native warm session"
      >
        {stopping ? "STOPPING…" : "■ STOP MISSION"}
      </button>
    </span>
  );
}
