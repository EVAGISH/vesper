"use client";

import { Suspense, useEffect } from "react";
import { useRouter, useSearchParams } from "next/navigation";
import { RunDetail } from "@/components/run-detail";
import { RunRail } from "@/components/run-rail";
import { useVesper } from "@/components/vesper-provider";

// Selection lives in ?run= so sweep failure rows (and people) can deep-link a run.
function RunsBrowser() {
  const { runs } = useVesper();
  const router = useRouter();
  const params = useSearchParams();
  const activeId = params.get("run");

  // Default to the newest run once the list arrives.
  useEffect(() => {
    if (!activeId && runs?.length)
      router.replace(`/?run=${encodeURIComponent(runs[0].id)}`, { scroll: false });
  }, [activeId, runs, router]);

  if (runs === null)
    return <div className="flex-1 p-10 text-center text-muted-foreground">Loading runs…</div>;

  const active = runs.find((r) => r.id === activeId) ?? null;

  return (
    <>
      <RunRail
        runs={runs}
        activeId={activeId}
        onSelect={(id) => router.replace(`/?run=${encodeURIComponent(id)}`, { scroll: false })}
      />
      {active ? (
        <RunDetail key={active.id} run={active} />
      ) : (
        <div className="flex-1 p-10 text-center text-muted-foreground">
          {activeId ? `No run named ${activeId}` : "Select a run"}
        </div>
      )}
    </>
  );
}

export default function Home() {
  return (
    <main className="flex min-h-0 flex-1">
      <Suspense>
        <RunsBrowser />
      </Suspense>
    </main>
  );
}
