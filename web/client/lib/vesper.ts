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

export type Job = {
  id: string;
  kind: "train" | "fly" | "eval" | "mission" | "live" | "warm";
  policy?: string | null;
  started: number;
  finished?: number | null;
  status: "running" | "done" | "stopped" | "failed";
  log: string;
};

export type Site = { world: string; half_m: number; ground: string };

/** Operator-drawn zones on a site, in site metres. */
export type SiteZones = {
  world: string;
  launch: [number, number][] | null;
  safe: [number, number][][];
  source: string | null;
};

/** Live world snapshot published by the warm session's /state endpoint. */
export type LiveState = {
  t: number;
  policy?: string;
  manual?: boolean;
  teleop_age_s?: number | null;
  drone0?: { speed: number; vz: number; agl: number };
  found: number;
  reached: number;
  targets: number;
  drones: { x: number; y: number; z: number }[];
  vehicles: { x: number; y: number; found: boolean; reached: boolean }[];
};

export async function postJSON<T>(url: string, body?: unknown): Promise<T> {
  const r = await fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: body === undefined ? undefined : JSON.stringify(body),
  });
  if (!r.ok) {
    let detail = r.statusText;
    try {
      detail = ((await r.json()) as { detail?: string }).detail ?? detail;
    } catch {}
    throw new Error(detail);
  }
  return (await r.json()) as T;
}

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

// Operator-facing names for run artifacts — never show raw file names in the UI.
const ARTIFACT_LABELS: Record<string, string> = {
  "overview.mp4": "chase cam",
  "chase.mp4": "chase cam",
  "fpv.mp4": "fpv",
  "track.png": "track map",
  "trajectory.parquet": "track",
  "scenario.json": "mission",
  "curve.jsonl": "training",
  "events.json": "events",
  "report.json": "sweep",
  "results.jsonl": "variants",
};

export function artifactLabel(file: string): string {
  if (file.startsWith("view_")) return `still ${file.replace(/\D/g, "")}`;
  return ARTIFACT_LABELS[file] ?? file.replace(/\.[^.]+$/, "");
}

/** "runs/<id>/search.pt" → "search — <id>" for operator-facing model names. */
export function modelLabel(m: { run: string; file: string }): string {
  return m.file.replace(/\.pt$/, "");
}

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

export type RunEvent = { t: number; label: string; vehicle?: string; xy?: [number, number] };

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
    const xy =
      Array.isArray(o.xy) && o.xy.length >= 2 &&
      typeof o.xy[0] === "number" && typeof o.xy[1] === "number"
        ? ([o.xy[0], o.xy[1]] as [number, number])
        : undefined;
    out.push({
      t,
      label: label ?? "event",
      vehicle: vehicle !== undefined ? String(vehicle) : undefined,
      xy,
    });
  }
  return out.sort((a, b) => a.t - b.t);
}
