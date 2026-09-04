"use client";

import { useEffect, useMemo, useState } from "react";
import { fetchJSON, media, parseEvents, type RunEvent } from "@/lib/vesper";

// Search-run event timeline from events.json (first-sighting / reached per
// vehicle). Clicking a marker seeks every video in the run detail to its t.
export function EventsTimeline({
  runId,
  simDuration,
  onSeek,
}: {
  runId: string;
  simDuration?: number;
  onSeek: (t: number) => void;
}) {
  const [events, setEvents] = useState<RunEvent[] | null>(null);

  useEffect(() => {
    let alive = true;
    fetchJSON<unknown>(media(runId, "events.json")).then(
      (d) => alive && setEvents(parseEvents(d)),
    );
    return () => {
      alive = false;
    };
  }, [runId]);

  const tMax = useMemo(() => {
    const last = events?.length ? events[events.length - 1].t : 0;
    return Math.max(simDuration ?? 0, last) * 1.02 || 1;
  }, [events, simDuration]);

  if (events === null)
    return <div className="p-4 text-sm text-muted-foreground">loading events.json…</div>;
  if (!events.length)
    return <div className="p-4 text-sm text-muted-foreground">no events recorded</div>;

  const isReach = (e: RunEvent) => /reach/i.test(e.label);

  return (
    <div className="px-4 pb-2 pt-3">
      {/* markers live in an inset track so edge labels hang into the margins
          instead of clipping at the panel border */}
      <div className="relative mx-14 h-12">
        <div className="absolute inset-x-0 top-[15px] h-0.5 bg-secondary" />
        {events.map((e, i) => (
          <button
            key={i}
            className="group absolute top-1 -translate-x-1/2 cursor-pointer text-center"
            style={{ left: `${(e.t / tMax) * 100}%` }}
            onClick={() => onSeek(e.t)}
            title={`jump video to t=${e.t.toFixed(1)} s`}
          >
            {isReach(e) ? (
              <span className="mx-auto mt-[3px] block size-[11px] rounded-full bg-[#3987e5] group-hover:ring-2 group-hover:ring-[#3987e5]/40" />
            ) : (
              <span className="mx-auto block size-0 border-x-[6px] border-b-[10px] border-x-transparent border-b-[#d95926] group-hover:opacity-80" />
            )}
            <span className="mt-1 block whitespace-nowrap text-[10px] tabular-nums text-muted-foreground group-hover:text-secondary-foreground">
              {e.label}
              {e.vehicle !== undefined ? ` ${e.vehicle}` : ""} · {e.t.toFixed(1)} s
            </span>
          </button>
        ))}
      </div>
      <div className="mx-14 flex justify-between text-[10px] tabular-nums text-muted-foreground">
        <span>0 s</span>
        <span>{tMax.toFixed(0)} s</span>
      </div>
      <div className="mt-1.5 text-[10px] text-muted-foreground">
        ▲ sighting · ● reached — click to jump video
      </div>
    </div>
  );
}
