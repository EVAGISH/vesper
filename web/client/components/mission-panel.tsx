"use client";

import { useEffect, useRef, useState } from "react";

// The operator's mission panel: what the tactical map can't say in glyphs.
// Polls the live session's /state and shows the target roster with status, a
// timestamped detection log built from status transitions, and lead telemetry.
// Replaces the old jobs list on the operator home — this is what's actually
// happening in the AO right now, not what containers are running on a box.

const ACCENT = "#0ca30c";
const POLL_MS = 400;

type Vehicle = { x: number; y: number; found: boolean; reached: boolean };
// strike-approval queue: the drone raises a request when it detects a target;
// it reaches the operator only once the drone has an RF link (PENDING_UPLINK
// until then), and the strike is held until the operator APPROVEs
type StrikeReq = {
  target: number;
  status: "PENDING_UPLINK" | "AWAITING_APPROVAL" | "APPROVED" | "DENIED";
  x: number; y: number; detected_t: number; uplink?: boolean;
};
type LiveState = {
  t: number;
  world?: string;
  policy?: string;
  found: number;
  reached: number;
  targets: number;
  drone0?: { speed: number; vz: number; agl: number };
  drones?: { x: number; y: number; z: number }[];
  vehicles?: Vehicle[];
  requests?: StrikeReq[];
  strikes?: { pending: number; awaiting: number; approved: number; denied: number };
};

type Ev = { t: number; kind: "detect" | "neutralize" | "request" | "approve" | "deny"; id: number };

function fmtT(t: number) {
  const s = Math.max(0, Math.floor(t));
  return `${String(Math.floor(s / 60)).padStart(2, "0")}:${String(s % 60).padStart(2, "0")}`;
}
const tid = (i: number) => `TGT-${String(i + 1).padStart(2, "0")}`;

export function MissionPanel({ ip }: { ip: string | null }) {
  const [st, setSt] = useState<LiveState | null>(null);
  const [live, setLive] = useState(false);
  const [events, setEvents] = useState<Ev[]>([]);
  const prev = useRef<{ found: boolean[]; reached: boolean[]; req: Record<number, string> }>(
    { found: [], reached: [], req: {} },
  );

  const decide = (target: number, kind: "approve" | "deny") =>
    fetch(`http://${ip}:8180/command`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ kind, target }),
    }).catch(() => {});

  useEffect(() => {
    if (!ip) { setLive(false); setSt(null); return; }
    let alive = true;
    const poll = () =>
      fetch(`http://${ip}:8180/state`, { cache: "no-store" })
        .then((r) => (r.ok ? r.json() : null))
        .then((d: LiveState | null) => {
          if (!alive) return;
          setLive(!!d);
          if (!d) return;
          setSt(d);
          // log the rising edge of each target's found / reached
          const vs = d.vehicles ?? [];
          const pf = prev.current.found, pr = prev.current.reached;
          const add: Ev[] = [];
          vs.forEach((v, i) => {
            if (v.found && !pf[i]) add.push({ t: d.t, kind: "detect", id: i });
            if (v.reached && !pr[i]) add.push({ t: d.t, kind: "neutralize", id: i });
          });
          // strike-request transitions (edge-triggered like found/reached)
          const preq = prev.current.req;
          const nreq: Record<number, string> = {};
          (d.requests ?? []).forEach((r) => {
            nreq[r.target] = r.status;
            const was = preq[r.target];
            if (r.status !== was) {
              if (r.status === "AWAITING_APPROVAL") add.push({ t: d.t, kind: "request", id: r.target });
              if (r.status === "APPROVED") add.push({ t: d.t, kind: "approve", id: r.target });
              if (r.status === "DENIED") add.push({ t: d.t, kind: "deny", id: r.target });
            }
          });
          if (add.length) setEvents((e) => [...add.reverse(), ...e].slice(0, 40));
          prev.current = {
            found: vs.map((v) => v.found),
            reached: vs.map((v) => v.reached),
            req: nreq,
          };
        })
        .catch(() => alive && setLive(false));
    poll();
    const h = setInterval(poll, POLL_MS);
    return () => { alive = false; clearInterval(h); };
  }, [ip]);

  // a fresh session (clock reset) clears the log so it tracks this mission only
  const tRef = useRef(0);
  useEffect(() => {
    if (st && st.t < tRef.current - 2) { setEvents([]); prev.current = { found: [], reached: [], req: {} }; }
    if (st) tRef.current = st.t;
  }, [st]);

  if (!ip || !live || !st) {
    return (
      <div className="flex h-full min-h-[220px] items-center justify-center">
        <span className="font-mono text-xs tracking-[0.25em] text-muted-foreground">
          {ip ? "AWAITING TELEMETRY" : "NO ACTIVE MISSION"}
        </span>
      </div>
    );
  }

  const vs = st.vehicles ?? [];
  const assets = st.drones?.length ?? 0;
  const complete = st.reached >= st.targets && st.targets > 0;
  const reqs = st.requests ?? [];
  const reqBy = new Map(reqs.map((r) => [r.target, r]));
  const awaiting = reqs.filter((r) => r.status === "AWAITING_APPROVAL");
  const pendingUplink = reqs.filter((r) => r.status === "PENDING_UPLINK");

  return (
    <div className="flex h-full min-h-0 flex-col">
      {/* mission header */}
      <div className="grid grid-cols-4 gap-px border-b border-border bg-border">
        {[
          ["CLOCK", `T+${fmtT(st.t)}`, "text-foreground"],
          ["DETECTED", `${st.found}/${st.targets}`, "text-[#0ca30c]"],
          ["NEUTRAL.", `${st.reached}/${st.targets}`, st.reached ? "text-[#d03b3b]" : "text-muted-foreground"],
          ["ASSETS", String(assets), "text-foreground"],
        ].map(([k, v, c]) => (
          <div key={k} className="bg-card px-2 py-2">
            <div className="font-mono text-[9px] uppercase tracking-widest text-muted-foreground">{k}</div>
            <div className={`font-mono text-lg tabular-nums ${c}`}>{v}</div>
          </div>
        ))}
      </div>

      {/* strike approval queue: the human is the release authority */}
      {(awaiting.length > 0 || pendingUplink.length > 0) && (
        <div className="border-b border-border px-3 py-2">
          <div className="mb-1.5 font-mono text-[9px] uppercase tracking-widest text-muted-foreground">
            Strike authority
          </div>
          <div className="flex flex-col gap-1.5">
            {awaiting.map((r) => (
              <div
                key={r.target}
                className="border border-[#e8b430]/60 bg-[#e8b430]/10 px-2 py-1.5"
              >
                <div className="flex items-center gap-2 font-mono text-[11px]">
                  <span className="inline-block h-1.5 w-1.5 animate-pulse rounded-full bg-[#e8b430]" />
                  <span className="font-semibold text-[#e8b430]">STRIKE REQUEST</span>
                  <span className="text-foreground">{tid(r.target)}</span>
                  <span className="text-muted-foreground">TANK · T+{fmtT(r.detected_t)}</span>
                </div>
                <div className="mt-1.5 flex gap-1.5">
                  <button
                    onClick={() => decide(r.target, "approve")}
                    className="cursor-pointer border border-[#0ca30c] bg-[#0ca30c]/15 px-2.5 py-0.5 font-mono text-[10px] font-semibold tracking-[0.12em] text-[#0ca30c] hover:bg-[#0ca30c]/30"
                  >
                    ✓ APPROVE
                  </button>
                  <button
                    onClick={() => decide(r.target, "deny")}
                    className="cursor-pointer border border-[#d03b3b] bg-[#d03b3b]/10 px-2.5 py-0.5 font-mono text-[10px] font-semibold tracking-[0.12em] text-[#d03b3b] hover:bg-[#d03b3b]/25"
                  >
                    ✕ DENY
                  </button>
                </div>
              </div>
            ))}
            {pendingUplink.map((r) => (
              <div key={r.target} className="flex items-center gap-2 border border-border bg-card px-2 py-1.5 font-mono text-[10.5px]">
                <span className="inline-block h-1.5 w-1.5 animate-pulse rounded-full bg-amber-500" />
                <span className="text-amber-500">UPLINK PENDING</span>
                <span className="text-muted-foreground">contact in dead zone · drone egressing to relay</span>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* target roster */}
      <div className="border-b border-border px-3 py-2">
        <div className="mb-1.5 font-mono text-[9px] uppercase tracking-widest text-muted-foreground">
          Target roster
        </div>
        <div className="flex flex-col gap-1">
          {vs.map((v, i) => {
            const rq = reqBy.get(i);
            let status = v.reached ? "NEUTRALIZED" : v.found ? "DETECTED" : "UNLOCATED";
            let color = v.reached ? "#d03b3b" : v.found ? ACCENT : "#6b7280";
            if (!v.reached && rq) {
              if (rq.status === "PENDING_UPLINK") { status = "UPLINK PENDING"; color = "#f59e0b"; }
              else if (rq.status === "AWAITING_APPROVAL") { status = "AWAITING APPROVAL"; color = "#e8b430"; }
              else if (rq.status === "APPROVED") { status = "CLEARED HOT"; color = "#0ca30c"; }
              else if (rq.status === "DENIED") { status = "HELD"; color = "#8a949c"; }
            }
            return (
              <div key={i} className="flex items-center gap-2 font-mono text-[11px]">
                <span className="tabular-nums text-foreground">{tid(i)}</span>
                <span className="text-muted-foreground">TANK</span>
                <span className="ml-auto flex items-center gap-1.5" style={{ color }}>
                  <span className="inline-block h-1.5 w-1.5 rounded-full" style={{ background: color }} />
                  {status}
                </span>
              </div>
            );
          })}
        </div>
      </div>

      {/* live event log */}
      <div className="min-h-0 flex-1 overflow-y-auto px-3 py-2">
        <div className="mb-1.5 font-mono text-[9px] uppercase tracking-widest text-muted-foreground">
          Detection log
        </div>
        {events.length === 0 ? (
          <div className="font-mono text-[11px] text-muted-foreground">
            {complete ? "AO cleared — all targets serviced." : "Searching… no contacts yet."}
          </div>
        ) : (
          <div className="flex flex-col gap-1">
            {events.map((e, i) => {
              const glyph = { detect: "◈", neutralize: "◆", request: "▲", approve: "✓", deny: "✕" }[e.kind];
              const color = {
                detect: ACCENT, neutralize: "#d03b3b", request: "#e8b430",
                approve: "#0ca30c", deny: "#d03b3b",
              }[e.kind];
              const text = {
                detect: "acquired by EO/IR",
                neutralize: "neutralized",
                request: "strike request uplinked — awaiting approval",
                approve: "strike APPROVED — cleared hot",
                deny: "strike DENIED — weapons hold",
              }[e.kind];
              return (
                <div key={i} className="flex items-baseline gap-2 font-mono text-[11px]">
                  <span className="tabular-nums text-muted-foreground">T+{fmtT(e.t)}</span>
                  <span style={{ color }}>{glyph}</span>
                  <span className="text-foreground">{tid(e.id)}</span>
                  <span className="text-muted-foreground">{text}</span>
                </div>
              );
            })}
          </div>
        )}
      </div>

      {/* lead telemetry footer */}
      {st.drone0 && (
        <div className="grid grid-cols-3 gap-px border-t border-border bg-border">
          {[
            ["LEAD SPD", `${st.drone0.speed.toFixed(1)} m/s`],
            ["AGL", `${st.drone0.agl.toFixed(0)} m`],
            ["POLICY", st.policy ?? "—"],
          ].map(([k, v]) => (
            <div key={k} className="truncate bg-card px-2 py-1.5">
              <div className="font-mono text-[9px] uppercase tracking-widest text-muted-foreground">{k}</div>
              <div className="truncate font-mono text-xs tabular-nums text-foreground">{v}</div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
