"use client";

import Link from "next/link";
import { useEffect, useState } from "react";
import { usePathname } from "next/navigation";
import { useVesper } from "@/components/vesper-provider";
import { cn } from "@/lib/utils";

const TABS = [
  { href: "/", label: "LIVE" },
  { href: "/runs", label: "RUNS" },
  { href: "/models", label: "MODELS" },
  { href: "/environments", label: "ENVIRONMENTS" },
];

function UtcClock() {
  const [now, setNow] = useState<Date | null>(null);
  useEffect(() => {
    const tick = () => setNow(new Date());
    const t0 = setTimeout(tick, 0); // first paint after hydration, avoids SSR mismatch
    const id = setInterval(tick, 1000);
    return () => {
      clearTimeout(t0);
      clearInterval(id);
    };
  }, []);
  if (!now) return <span className="font-mono text-[11px] text-muted-foreground">--:--:--Z</span>;
  const p = (n: number) => String(n).padStart(2, "0");
  return (
    <span className="font-mono text-[11px] tabular-nums text-secondary-foreground">
      {p(now.getUTCHours())}:{p(now.getUTCMinutes())}:{p(now.getUTCSeconds())}
      <span className="text-muted-foreground">Z</span>
    </span>
  );
}

export function TopBar() {
  const path = usePathname();
  const { runs, scenarios, online } = useVesper();

  return (
    <header className="flex h-11 shrink-0 items-center gap-4 border-b border-border bg-card px-4">
      <h1 className="text-sm font-bold tracking-[0.24em]">
        VESPER<span className="ml-2 align-[2px] font-mono text-[9px] font-normal tracking-[0.14em] text-muted-foreground">SIM OPS</span>
      </h1>
      <div className="flex items-center gap-4 font-mono text-[10px] uppercase tracking-[0.08em] text-muted-foreground">
        <span>
          <span
            className={cn(
              "mr-1.5 inline-block size-[7px] rounded-full",
              online ? "bg-[#0ca30c]" : "bg-[#d03b3b]",
            )}
          />
          {online ? (
            <span className="text-[#0ca30c]">link nominal</span>
          ) : (
            <span className="text-[#d03b3b]">link down</span>
          )}
        </span>
        <span className="tabular-nums">
          <b className="font-semibold text-secondary-foreground">{runs?.length ?? "—"}</b> runs
        </span>
        <span className="tabular-nums">
          <b className="font-semibold text-secondary-foreground">{scenarios?.length ?? "—"}</b>{" "}
          scenarios
        </span>
        <UtcClock />
      </div>
      <nav className="ml-auto flex gap-0.5">
        {TABS.map((t) => (
          <Link
            key={t.href}
            href={t.href}
            className={cn(
              "border border-transparent px-3 py-1 font-mono text-[11px] tracking-[0.1em]",
              path === t.href
                ? "border-border bg-secondary text-foreground"
                : "text-secondary-foreground hover:text-foreground",
            )}
          >
            {t.label}
          </Link>
        ))}
      </nav>
    </header>
  );
}
