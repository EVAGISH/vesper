"use client";

import Link from "next/link";
import { useEffect, useState } from "react";
import { JobButton, JobsPanel } from "@/components/job-controls";
import { LiveViewport } from "@/components/live-viewport";
import { SiteMap } from "@/components/site-map";
import { useVesper } from "@/components/vesper-provider";
import { fetchJSON, fmtTime, KIND_COLOR, runKind } from "@/lib/vesper";

// Operator home: downlink + AO map. Live sessions start with one click; the
// in-page WebRTC viewport connects once a live session is up on the box.

const LIVE_POLL_MS = 15000;

function Panel({ title, right, children }: {
  title: string; right?: React.ReactNode; children: React.ReactNode;
}) {
  return (
    <section className="hud-corners rounded-lg border border-border bg-card">
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
  const [viewport, setViewport] = useState(false);

  // /?viewport=1 opens the stream immediately (bookmarkable operator view)
  useEffect(() => {
    if (new URLSearchParams(window.location.search).get("viewport")) setViewport(true);
  }, []);

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
    <main className="min-h-0 flex-1 overflow-y-auto p-4">
      <div className="grid grid-cols-1 items-start gap-3 lg:grid-cols-[minmax(0,7fr)_minmax(0,5fr)]">
        <Panel
          title="Live downlink"
          right={
            ip === undefined ? "checking…" : ip ? (
              <span className="font-mono text-[#0ca30c]">vesper-dev @ {ip}</span>
            ) : (
              <span className="font-mono text-[#d03b3b]">gpu box offline</span>
            )
          }
        >
          <div className="relative flex aspect-video items-center justify-center bg-black">
            {viewport && ip ? (
              <div className="absolute inset-0">
                <LiveViewport ip={ip} />
              </div>
            ) : (
            <div className="text-center">
              <div className="font-mono text-sm tracking-[0.3em] text-muted-foreground">
                {ip ? "AWAITING STREAM" : "STANDBY"}
              </div>
              <div className="mx-auto mt-2 max-w-sm px-6 text-xs text-muted-foreground">
                {ip
                  ? "Start a live session on the box, give it ~2 min to load the world, then connect the viewport."
                  : "No GPU box is up — bring one up to fly live."}
              </div>
              {ip && (
                <div className="mt-4 flex items-center justify-center gap-2">
                  <JobButton label="▶ START LIVE SESSION" body={{ kind: "live" }} />
                  <button
                    onClick={() => setViewport(true)}
                    className="rounded border border-border bg-secondary px-3 py-1.5 font-mono text-[11px] text-secondary-foreground hover:text-foreground"
                  >
                    ⧉ CONNECT VIEWPORT
                  </button>
                </div>
              )}
            </div>
            )}
            <div className="pointer-events-none absolute left-3 top-3 h-4 w-4 border-l border-t border-[#4a4a44]" />
            <div className="pointer-events-none absolute right-3 top-3 h-4 w-4 border-r border-t border-[#4a4a44]" />
            <div className="pointer-events-none absolute bottom-3 left-3 h-4 w-4 border-b border-l border-[#4a4a44]" />
            <div className="pointer-events-none absolute bottom-3 right-3 h-4 w-4 border-b border-r border-[#4a4a44]" />
          </div>
        </Panel>

        <div className="row-span-2 flex min-w-0 flex-col gap-3">
          <Panel title="AO map" right="site frame · N up">
            <SiteMap />
          </Panel>
          <Panel title="Latest sortie">
            {latest ? (
              <Link
                href={`/runs?run=${encodeURIComponent(latest.id)}`}
                className="block p-3 hover:bg-secondary"
              >
                <div className="flex items-baseline text-xs font-semibold">
                  <span className="truncate font-mono">{latest.manifest.name || latest.id}</span>
                  <span
                    className="ml-auto shrink-0 pl-2 font-mono text-[9.5px] font-normal uppercase"
                    style={{ color: KIND_COLOR[runKind(latest)] }}
                  >
                    ● {runKind(latest)}
                  </span>
                </div>
                <div className="mt-0.5 text-[11px] tabular-nums text-muted-foreground">
                  {fmtTime(latest.manifest.started)}
                  {latest.manifest.scene && ` · ${latest.manifest.scene}`}
                </div>
                <div className="mt-1.5 text-[11px] text-[#3987e5]">open in runs →</div>
              </Link>
            ) : (
              <div className="p-3 text-xs text-muted-foreground">no runs yet</div>
            )}
          </Panel>
        </div>

        <JobsPanel />
      </div>
    </main>
  );
}
