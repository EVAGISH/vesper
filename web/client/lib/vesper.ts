// Data contract with web/server/app.py (FastAPI on :8777, proxied via /api and /media).

export type Manifest = {
  name?: string;
  scene?: string;
  started?: number;
  finished?: number;
  streams?: Record<string, string>;
  frames?: Record<string, number>;
  fps?: number;
  resolution?: [number, number];
};

export type Run = { id: string; manifest: Manifest; files: string[] };

export type Scenario = {
  file: string;
  world?: string | null;
  terrain_usd?: string | null;
  waypoints: number;
  wind_ms?: number | null;
  visibility_m?: number | null;
  cruise_ms?: number | null;
  max_sim_s?: number | null;
  command: string;
};

export type Trajectory = { t: number[]; px: number[]; py: number[]; pz: number[] };

export type Model = {
  run: string;
  file: string;
  path: string;
  bytes: number;
  mtime: number;
  metrics: Record<string, number>;
};

export const fmtBytes = (b: number) =>
  b >= 1 << 20 ? `${(b / (1 << 20)).toFixed(1)} MB` : `${Math.round(b / 1024)} KB`;

// The docker lane every launch command runs through on the GPU box.
export const SIM = "docker compose run --rm sim /isaac-sim/python.sh";

export type RunKind = "flight" | "search" | "training" | "sweep" | "view";

// Fixed categorical hue per kind (dark-surface steps of the reference palette).
// Identity colors — never cycled, never reassigned.
export const KIND_COLOR: Record<RunKind, string> = {
  flight: "#3987e5",
  search: "#d95926",
  training: "#199e70",
  sweep: "#c98500",
  view: "#d55181",
};

export function runKind(run: Run): RunKind {
  const f = run.files;
  if (f.includes("report.json") || f.includes("results.jsonl")) return "sweep";
  if (f.includes("curve.jsonl")) return "training";
  if (f.includes("events.json") || f.includes("track.png") || f.includes("chase.mp4"))
    return "search";
  if (f.some((n) => n.startsWith("view_") && n.endsWith(".png"))) return "view";
  return "flight";
}

export const media = (runId: string, file: string) => `/media/${runId}/${file}`;

export const fmtTime = (ts?: number) =>
  ts
    ? new Date(ts * 1000).toLocaleString(undefined, {
        month: "short", day: "numeric", hour: "2-digit", minute: "2-digit", second: "2-digit",
      })
    : "—";

export const fmtDur = (m: Manifest) =>
  m.finished && m.started ? `${Math.round(m.finished - m.started)} s wall` : "";

// ── artifact fetch/parse helpers ──────────────────────────────────────────

export async function fetchJSON<T>(url: string): Promise<T | null> {
  try {
    const r = await fetch(url);
    return r.ok ? ((await r.json()) as T) : null;
  } catch {
    return null;
  }
}

/** Parse a .jsonl body into objects, skipping blank/corrupt lines. */
export function parseJSONL(text: string): Record<string, unknown>[] {
  const out: Record<string, unknown>[] = [];
  for (const line of text.split("\n")) {
    const s = line.trim();
    if (!s) continue;
    try {
      const v = JSON.parse(s);
      if (v && typeof v === "object" && !Array.isArray(v)) out.push(v);
    } catch {
      /* tolerate a torn tail line while a run is still writing */
    }
  }
  return out;
}

export type RunEvent = { t: number; label: string; vehicle?: string };

/** Normalize events.json (array of events, or {events: [...]}) into {t, label}. */
export function parseEvents(raw: unknown): RunEvent[] {
  const arr = Array.isArray(raw)
    ? raw
    : raw && typeof raw === "object" && Array.isArray((raw as { events?: unknown }).events)
      ? ((raw as { events: unknown[] }).events)
      : [];
  const out: RunEvent[] = [];
  for (const e of arr) {
    if (!e || typeof e !== "object") continue;
    const o = e as Record<string, unknown>;
    const t = [o.t, o.time, o.t_s, o.sim_t].find((v) => typeof v === "number") as
      | number
      | undefined;
    if (t === undefined) continue;
    const label =
      [o.event, o.type, o.kind, o.name].find((v) => typeof v === "string") as string | undefined;
    const vehicle = [o.vehicle, o.target, o.id].find(
      (v) => typeof v === "string" || typeof v === "number",
    );
    out.push({ t, label: label ?? "event", vehicle: vehicle !== undefined ? String(vehicle) : undefined });
  }
  return out.sort((a, b) => a.t - b.t);
}
