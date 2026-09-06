"use client";

import { useEffect, useState } from "react";

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
  // a session that answers /streams but publishes no cameras (the native warm
  // session: telemetry only) — the map is live, there is just no video downlink
  const [session, setSession] = useState<string | null>(null);

  useEffect(() => {
    let alive = true;
    const poll = () =>
      fetch(`http://${ip}:8180/streams`, { cache: "no-store" })
        .then((r) => (r.ok ? r.json() : null))
        .then((d) => {
          if (!alive) return;
          setInfo(d && d.streams?.length ? d : null);
          setSession(d?.run ?? null);
          setChecked(true);
        })
        .catch(() => {
          if (!alive) return;
          setInfo(null);
          setSession(null);
          setChecked(true);
        });
    poll();
    const id = setInterval(poll, POLL_MS);
    return () => {
      alive = false;
      clearInterval(id);
    };
  }, [ip]);

  // telemetry-only session (native sim): the map is live, video needs Isaac
  if (!info && session) {
    return (
      <div className="flex aspect-video flex-col items-center justify-center gap-3 bg-black text-center">
        <div className="font-mono text-sm tracking-[0.3em] text-[#0ca30c]">
          TELEMETRY LIVE — NO VIDEO DOWNLINK
        </div>
        <div className="max-w-sm px-6 text-xs text-muted-foreground">
          A native session ({session}) is flying and the map is live. Camera
          feeds render on the GPU box only — start the Isaac warm session there
          for video.
        </div>
      </div>
    );
  }

  // nothing flying on the box. The one-click Isaac launch buttons that lived
  // here are gone on purpose: the demo lane is the NATIVE session (▶ START
  // MISSION on the Live tab), and Isaac jobs on the droplet stay an explicit
  // choice from the Models / Environments tabs.
  if (!info) {
    return (
      <div className="flex aspect-video flex-col items-center justify-center gap-3 bg-black text-center">
        <div className="font-mono text-sm tracking-[0.3em] text-muted-foreground">
          {checked ? "NO DRONE IN FLIGHT" : "CHECKING FEED…"}
        </div>
        <div className="max-w-sm px-6 text-xs text-muted-foreground">
          The GPU box is up but nothing is flying. Launch Isaac work explicitly
          from the Models or Environments tabs — or stop the box and use
          ▶ START MISSION for the native local session.
        </div>
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
              className="relative basis-1/4 cursor-pointer"
              title={`show ${s} large`}
            >
              {/* eslint-disable-next-line @next/next/no-img-element */}
              {/* object-contain, not cover: the fpv stream is square (640×640),
                  and cover would crop away its top/bottom third in a 16:9 tile */}
              <img
                src={`http://${ip}:8180/${s}.mjpeg`}
                alt={`${s} feed`}
                className="block aspect-video w-full bg-black object-contain opacity-80 hover:opacity-100"
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
