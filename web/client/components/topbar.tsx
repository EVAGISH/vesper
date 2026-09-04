"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { useVesper } from "@/components/vesper-provider";
import { cn } from "@/lib/utils";

const TABS = [
  { href: "/", label: "RUNS" },
  { href: "/environments", label: "ENVIRONMENTS" },
];

export function TopBar() {
  const path = usePathname();
  const { runs, scenarios, online } = useVesper();

  return (
    <header className="flex h-11 shrink-0 items-center gap-4 border-b border-border bg-card px-4">
      <h1 className="text-sm font-bold tracking-[0.18em]">VESPER</h1>
      <div className="flex gap-4 text-[11px] text-muted-foreground">
        <span>
          <span
            className={cn(
              "mr-1.5 inline-block size-[7px] rounded-full",
              online ? "bg-[#0ca30c]" : "bg-[#d03b3b]",
            )}
          />
          {online ? "runs server" : "server unreachable"}
        </span>
        <span className="tabular-nums">
          <b className="font-semibold text-secondary-foreground">{runs?.length ?? "—"}</b> runs
        </span>
        <span className="tabular-nums">
          <b className="font-semibold text-secondary-foreground">{scenarios?.length ?? "—"}</b>{" "}
          scenarios
        </span>
      </div>
      <nav className="ml-auto flex gap-0.5">
        {TABS.map((t) => (
          <Link
            key={t.href}
            href={t.href}
            className={cn(
              "rounded-md border border-transparent px-3 py-1 text-xs tracking-[0.06em]",
              path === t.href
                ? "border-border bg-secondary text-foreground"
                : "text-secondary-foreground hover:text-foreground",
            )}
          >
            {t.label}
          </Link>
        ))}
        <span className="cursor-default rounded-md px-3 py-1 text-xs tracking-[0.06em] text-muted-foreground">
          LIVE
          <i className="ml-1.5 rounded-sm border border-border px-1 align-[1px] text-[9px] not-italic">
            V2
          </i>
        </span>
      </nav>
    </header>
  );
}
