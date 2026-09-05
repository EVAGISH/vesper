"use client";

import Link from "next/link";
import { useEffect, useState } from "react";
import { DroneFeeds } from "@/components/drone-feeds";
import { LiveViewport } from "@/components/live-viewport";
import { MissionPanel } from "@/components/mission-panel";
import { TacticalView } from "@/components/tactical-view";
import { Teleop } from "@/components/teleop";
import { useVesper } from "@/components/vesper-provider";
import { fetchJSON, fmtTime, KIND_COLOR, runKind } from "@/lib/vesper";

// Operator home, in watch → situate → control → reference order: the drone
// feed is the hero (left, 60%), the AO map sits beside it over the jobs rail
// (right, 40%), the latest sortie is a compact jump-off to /runs, and the
// Isaac editor viewport is a tool behind a toggle — its own secondary panel
// below the feed, never competing with it.

const LIVE_POLL_MS = 15000;

function Panel({ title, right, children, className }: {
  title: string; right?: React.ReactNode; children: React.ReactNode; className?: string;
}) {
  return (
    <section className={`hud-corners rounded-lg border border-border bg-card ${className ?? ""}`}>
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

export default function Live() {
  const { runs } = useVesper();
  const [ip, setIp] = useState<string | null | undefined>(undefined);
  // /?viewport=1 opens the Isaac editor viewport immediately (bookmarkable)
  const [viewport, setViewport] = useState(
    () => typeof window !== "undefined" &&
      !!new URLSearchParams(window.location.search).get("viewport"),
  );

  useEffect(() => {
    let alive = true;
    const poll = () =>
      fetchJSON<{ ip: string | null }>("/api/live").then(
        (d) => alive && setIp(d ? d.ip : null),
      );
    poll();
    const id = setInterval(poll, LIVE_POLL_MS);
    return () => {
      alive = false;
      clearInterval(id);
    };
  }, []);

  const latest = runs?.[0];

  return (
    <main className="flex min-h-0 flex-1 overflow-y-auto p-4 lg:overflow-hidden">
      <div className="grid min-h-0 flex-1 grid-cols-1 items-start gap-3 lg:grid-cols-[minmax(0,3fr)_minmax(0,2fr)] lg:items-stretch">
        {/* ── left column: watch, then reference. Fits the fold; only an
              opened world inspector makes it scroll. ─────────────── */}
        <div className="flex min-w-0 flex-col gap-3 lg:min-h-0 lg:overflow-y-auto">
          <Panel
            title="Live downlink"
            right={
              ip === undefined ? "checking…" : !ip ? (
                <span className="font-mono text-[#d03b3b]">no session</span>
              ) : ip === "localhost" ? (
                <span className="font-mono text-[#0ca30c]">native sim @ localhost</span>
              ) : (
                <span className="flex items-center gap-3">
                  {!viewport && (
                    <button
                      onClick={() => setViewport(true)}
                      className="cursor-pointer font-mono text-[10px] normal-case tracking-normal text-muted-foreground hover:text-foreground"
                      title="free-fly editor camera for inspecting the world (not the drone)"
                    >
                      inspect world ⧉
                    </button>
                  )}
                  <span className="font-mono text-[#0ca30c]">vesper-dev @ {ip}</span>
                </span>
              )
            }
          >
            {ip ? (
              <>
                {/* Native session → the tactical operator console (live, on-device).
                    Isaac box session → its rendered MJPEG feeds. */}
                {ip === "localhost" ? <TacticalView ip={ip} /> : <DroneFeeds ip={ip} />}
                <div className="border-t border-border">
                  <Teleop ip={ip} />
                </div>
              </>
            ) : (
              <div className="flex aspect-video flex-col items-center justify-center gap-2 bg-black text-center">
                <div className="font-mono text-sm tracking-[0.3em] text-muted-foreground">
                  STANDBY
                </div>
                <div className="max-w-sm px-6 text-xs text-muted-foreground">
                  No session is up. Launch a warm session to fly.
                </div>
              </div>
            )}
          </Panel>

          {viewport && ip && (
            <Panel
              title="World inspector"
              right={
                <span className="flex items-center gap-3">
                  <span>free-fly editor camera — not the drone</span>
                  <button
                    onClick={() => setViewport(false)}
                    className="cursor-pointer font-mono text-[10px] text-muted-foreground hover:text-foreground"
                  >
                    close ✕
                  </button>
                </span>
              }
            >
              <div className="relative aspect-video bg-black">
                <LiveViewport ip={ip} />
              </div>
            </Panel>
          )}

          <Panel title="Latest sortie">
            {latest ? (
              <Link
                href={`/runs?run=${encodeURIComponent(latest.id)}`}
                className="flex items-baseline gap-3 px-3 py-2 hover:bg-secondary"
              >
                <span className="truncate font-mono text-xs font-semibold">
                  {latest.manifest.name || latest.id}
                </span>
                <span
                  className="shrink-0 font-mono text-[9.5px] uppercase"
                  style={{ color: KIND_COLOR[runKind(latest)] }}
                >
                  ● {runKind(latest)}
                </span>
                <span className="truncate text-[11px] tabular-nums text-muted-foreground">
                  {fmtTime(latest.manifest.started)}
                  {latest.manifest.scene && ` · ${latest.manifest.scene}`}
                </span>
                <span className="ml-auto shrink-0 text-[11px] text-[#3987e5]">
                  open in runs →
                </span>
              </Link>
            ) : (
              <div className="px-3 py-2 text-xs text-muted-foreground">no runs yet</div>
            )}
          </Panel>
        </div>

        {/* ── right rail: the live mission picture the map can't say in glyphs —
              target roster, detection log, telemetry from the session's /state. ── */}
        <div className="flex min-w-0 flex-col gap-3 lg:h-full lg:min-h-0">
          <Panel
            title="Mission"
            right={ip === "localhost" ? "live · kramatorsk AO" : "no active mission"}
            className="lg:flex lg:min-h-0 lg:flex-1 lg:flex-col"
          >
            <MissionPanel ip={ip === undefined ? null : ip} />
          </Panel>
        </div>
      </div>
    </main>
  );
}
