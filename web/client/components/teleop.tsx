"use client";

import { useEffect, useRef, useState } from "react";
import { Button } from "@/components/ui/button";
import type { LiveState } from "@/lib/vesper";

// Manual control of drone 0 in the warm session. Keys map to the same
// body-frame velocity command the policy emits (forward / left / up), so the
// operator flies exactly the action the network will learn. The page re-sends
// the stick every 100 ms; the sim hovers if it hears nothing for 0.7 s.

const KEYS: Record<string, [number, number, number]> = {
  KeyW: [1, 0, 0], KeyS: [-1, 0, 0],
  KeyA: [0, 1, 0], KeyD: [0, -1, 0],
  KeyR: [0, 0, 1], KeyF: [0, 0, -1],
  ArrowUp: [0, 0, 1], ArrowDown: [0, 0, -1],
};
const SEND_MS = 100;

function isTyping(e: KeyboardEvent) {
  const t = e.target as HTMLElement | null;
  return !!t && (t.tagName === "INPUT" || t.tagName === "TEXTAREA" || t.isContentEditable);
}

const STATE_POLL_MS = 1000;

export function Teleop({ ip }: { ip: string }) {
  const [live, setLive] = useState<LiveState | null>(null);
  const [armed, setArmed] = useState(false);
  const [axes, setAxes] = useState<[number, number, number]>([0, 0, 0]);
  const [note, setNote] = useState<string | null>(null);
  const pressed = useRef<Set<string>>(new Set());
  const armedRef = useRef(false);
  useEffect(() => { armedRef.current = armed; }, [armed]);

  const post = (body: unknown) =>
    fetch(`http://${ip}:8180/command`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    }).catch(() => setNote("box unreachable"));

  const setManual = async (on: boolean) => {
    setArmed(on);
    pressed.current.clear();
    setAxes([0, 0, 0]);
    await post({ kind: "manual", on });
  };

  // key state -> axes; ESC hands the drone back to the policy
  useEffect(() => {
    const sum = (): [number, number, number] => {
      const a: [number, number, number] = [0, 0, 0];
      for (const k of pressed.current) {
        const v = KEYS[k];
        if (v) { a[0] += v[0]; a[1] += v[1]; a[2] += v[2]; }
      }
      return a.map((x) => Math.max(-1, Math.min(1, x))) as [number, number, number];
    };
    const down = (e: KeyboardEvent) => {
      if (!armedRef.current || isTyping(e)) return;
      if (e.code === "Escape") { setManual(false); return; }
      if (!(e.code in KEYS)) return;
      e.preventDefault();
      pressed.current.add(e.code);
      setAxes(sum());
    };
    const up = (e: KeyboardEvent) => {
      if (!(e.code in KEYS)) return;
      pressed.current.delete(e.code);
      setAxes(sum());
    };
    const blur = () => { pressed.current.clear(); setAxes([0, 0, 0]); };
    window.addEventListener("keydown", down);
    window.addEventListener("keyup", up);
    window.addEventListener("blur", blur);
    return () => {
      window.removeEventListener("keydown", down);
      window.removeEventListener("keyup", up);
      window.removeEventListener("blur", blur);
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [ip]);

  // while armed, stream the stick at 10 Hz (the dead-man on the box needs it)
  useEffect(() => {
    if (!armed) return;
    const id = setInterval(() => post({ kind: "teleop", axes }), SEND_MS);
    return () => clearInterval(id);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [armed, axes, ip]);

  // a light poll of /state for the manual flag and drone 0's speed and height
  useEffect(() => {
    let alive = true;
    const poll = () =>
      fetch(`http://${ip}:8180/state`, { cache: "no-store" })
        .then((r) => (r.ok ? r.json() : null))
        .then((d: LiveState | null) => alive && setLive(d && d.drones ? d : null))
        .catch(() => alive && setLive(null));
    poll();
    const id = setInterval(poll, STATE_POLL_MS);
    return () => { alive = false; clearInterval(id); };
  }, [ip]);

  // disarm on unmount so the sim does not sit in manual with nobody flying
  useEffect(() => () => { if (armedRef.current) post({ kind: "manual", on: false }); }, [ip]); // eslint-disable-line react-hooks/exhaustive-deps

  const key = (label: string, on: boolean) => (
    <kbd
      className={`inline-flex h-6 min-w-6 items-center justify-center rounded border px-1 font-mono text-[10px] ${
        on ? "border-[#0ca30c] bg-[#0ca30c]/20 text-[#0ca30c]" : "border-border text-muted-foreground"
      }`}
    >
      {label}
    </kbd>
  );
  const d0 = live?.drone0;
  const simManual = live?.manual ?? false;

  return (
    <div className="flex flex-wrap items-center gap-3 px-3 py-2">
      <Button
        variant={armed ? "default" : "secondary"}
        size="sm"
        className="h-7 cursor-pointer px-3 font-mono text-[11px] tracking-[0.08em]"
        onClick={() => setManual(!armed)}
        title="take drone 0 by hand (ESC to release)"
      >
        {armed ? "■ RELEASE (ESC)" : "◎ TAKE CONTROL"}
      </Button>
      <span className="flex items-center gap-1">
        {key("W", axes[0] > 0)}{key("S", axes[0] < 0)}
        <span className="mx-1 text-[10px] text-muted-foreground">fwd/back</span>
        {key("A", axes[1] > 0)}{key("D", axes[1] < 0)}
        <span className="mx-1 text-[10px] text-muted-foreground">left/right</span>
        {key("R", axes[2] > 0)}{key("F", axes[2] < 0)}
        <span className="mx-1 text-[10px] text-muted-foreground">up/down</span>
      </span>
      <span className="ml-auto font-mono text-[10px] text-muted-foreground">
        {armed && !simManual && "arming…"}
        {simManual && (
          <span className="text-[#0ca30c]">
            MANUAL{d0 && ` · ${d0.speed} m/s · vz ${d0.vz} · AGL ${d0.agl} m`}
          </span>
        )}
        {!armed && !simManual && (live?.policy ? `policy ${live.policy}` : "policy flying")}
        {note && <span className="ml-2 text-[#d03b3b]">{note}</span>}
      </span>
    </div>
  );
}
