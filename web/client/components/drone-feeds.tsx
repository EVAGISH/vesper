"use client";

import { useEffect, useState } from "react";
import { JobButton } from "@/components/job-controls";

// The live downlink: the drone's own camera feeds from the run currently in the
// air on the GPU box. Polls the box's frame server (vesper.capture.live, 8180);
// each camera is an MJPEG <img>. This is what the drone sees — not the Isaac
// editor viewport. One camera is shown large; the rest form a thumbnail strip.

const POLL_MS = 4000;
// which camera earns the big frame, best first
const PRIMARY_ORDER = ["fpv", "chase", "overview"];

function pickPrimary(streams: string[], chosen: string | null): string {
  if (chosen && streams.includes(chosen)) return chosen;
  for (const s of PRIMARY_ORDER) if (streams.includes(s)) return s;
  return streams[0];
}

export function DroneFeeds({ ip }: { ip: string }) {
  const [info, setInfo] = useState<{ run: string; streams: string[] } | null>(null);
  const [chosen, setChosen] = useState<string | null>(null);
  const [checked, setChecked] = useState(false);

  useEffect(() => {
    let alive = true;
    const poll = () =>
      fetch(`http://${ip}:8180/streams`, { cache: "no-store" })
        .then((r) => (r.ok ? r.json() : null))
        .then((d) => {
          if (!alive) return;
          setInfo(d && d.streams?.length ? d : null);
          setChecked(true);
        })
        .catch(() => {
          if (!alive) return;
          setInfo(null);
          setChecked(true);
        });
    poll();
    const id = setInterval(poll, POLL_MS);
    return () => {
      alive = false;
      clearInterval(id);
    };
  }, [ip]);

  // nothing flying: standby with a one-click mission so the operator is self-serve
  if (!info) {
    return (
      <div className="flex aspect-video flex-col items-center justify-center gap-3 bg-black text-center">
        <div className="font-mono text-sm tracking-[0.3em] text-muted-foreground">
          {checked ? "NO DRONE IN FLIGHT" : "CHECKING FEED…"}
        </div>
        <div className="max-w-sm px-6 text-xs text-muted-foreground">
          The feed is live only while a drone is flying. Launch a mission — cameras
          appear here as soon as it takes off (~2 min to load the world).
        </div>
        <JobButton
          label="▶ FLY MISSION"
          body={{ kind: "mission", scenario: "cornell_core.json" }}
        />
      </div>
    );
  }

  const primary = pickPrimary(info.streams, chosen);
  const others = info.streams.filter((s) => s !== primary);

  return (
    <div className="bg-black">
      <figure className="relative aspect-video">
        {/* eslint-disable-next-line @next/next/no-img-element */}
        <img
          key={primary}
          src={`http://${ip}:8180/${primary}.mjpeg`}
          alt={`${primary} feed`}
          className="h-full w-full object-contain"
        />
        <figcaption className="absolute left-3 top-3 bg-black/60 px-1.5 py-0.5 font-mono text-[10px] uppercase tracking-[0.14em] text-[#0ca30c]">
          ● {primary}
        </figcaption>
        <span className="absolute right-3 top-3 bg-black/60 px-1.5 py-0.5 font-mono text-[10px] text-muted-foreground">
          {info.run}
        </span>
      </figure>
      {others.length > 0 && (
        <div className="flex gap-px">
          {others.map((s) => (
            <button
              key={s}
              onClick={() => setChosen(s)}
              className="relative flex-1 cursor-pointer"
              title={`show ${s} large`}
            >
              {/* eslint-disable-next-line @next/next/no-img-element */}
              <img
                src={`http://${ip}:8180/${s}.mjpeg`}
                alt={`${s} feed`}
                className="block aspect-video w-full object-cover opacity-80 hover:opacity-100"
              />
              <span className="absolute left-1.5 top-1 font-mono text-[9px] uppercase tracking-widest text-white/80">
                {s}
              </span>
            </button>
          ))}
        </div>
      )}
    </div>
  );
}
