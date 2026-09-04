"use client";

import { createContext, useContext, useEffect, useState } from "react";
import { fetchJSON, type Run, type Scenario } from "@/lib/vesper";

// Filesystem-is-the-database: new runs appear when files appear, so poll cheaply.
const RUNS_POLL_MS = 5000;
const SCENARIOS_POLL_MS = 30000;

type VesperData = {
  runs: Run[] | null; // null = first load in flight
  scenarios: Scenario[] | null;
  online: boolean;
};

const Ctx = createContext<VesperData>({ runs: null, scenarios: null, online: true });

export const useVesper = () => useContext(Ctx);

export function VesperProvider({ children }: { children: React.ReactNode }) {
  const [runs, setRuns] = useState<Run[] | null>(null);
  const [scenarios, setScenarios] = useState<Scenario[] | null>(null);
  const [online, setOnline] = useState(true);

  useEffect(() => {
    let alive = true;
    const pollRuns = async () => {
      const d = await fetchJSON<Run[]>("/api/runs");
      if (!alive) return;
      setOnline(d !== null);
      if (d !== null) setRuns(d);
    };
    const pollScenarios = async () => {
      const d = await fetchJSON<Scenario[]>("/api/scenarios");
      if (alive && d !== null) setScenarios(d);
    };
    pollRuns();
    pollScenarios();
    const a = setInterval(pollRuns, RUNS_POLL_MS);
    const b = setInterval(pollScenarios, SCENARIOS_POLL_MS);
    return () => {
      alive = false;
      clearInterval(a);
      clearInterval(b);
    };
  }, []);

  return <Ctx.Provider value={{ runs, scenarios, online }}>{children}</Ctx.Provider>;
}
