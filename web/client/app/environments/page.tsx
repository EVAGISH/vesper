"use client";

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import {
  AreaMap, LAUNCH_R_M, siteXY,
  type Area, type Center, type Footprint, type Tool, type ZoneOverlay,
} from "@/components/area-map";
import { JobButton } from "@/components/job-controls";
import { TrainPanel, useTrainControl } from "@/components/train-panel";
import { Button } from "@/components/ui/button";
import { fetchJSON, postJSON } from "@/lib/vesper";

// Environments: frame an area of the earth on the satellite map, lock it, drop
// the launch pin and draw friendly zones, build. The launch pin becomes the
// spawn (the builder keeps trees clear of it) and both land in
// assets/<name>/zones.json (vesper.worlds.zones) for the tasks to read.

type DemoMedia = { file: string; url: string; kind: "image" | "video" };
type Environment = {
  name: string;
  center?: [number, number] | null;     // [lat, lon] of the built world
  half_m?: number | null;
  usd: string | null;
  scenario: string | null;
  map: string | null;
  mb?: number | null;
  build_status?: string;
  where?: string | null;
  log?: string | null;
  demo?: DemoMedia[];
};
type Hit = { name: string; lat: number; lon: number; half_km: number; type: string };
type ZonesDoc = {
  world: string;
  launch_point: { lat: number; lon: number; x: number; y: number; r_m: number } | null;
  safe_geo: [number, number][][] | null;
  source: string | null;
};

const POLL_MS = 4000;
const MIN_KM = 1;
const MAX_KM = 8;
const NAME_RE = /^[a-z][a-z0-9_]{1,31}$/;

const clampKm = (v: number) => Math.min(MAX_KM, Math.max(MIN_KM, Number.isFinite(v) ? v : MIN_KM));
const fmt = (c: Center) => `${c.lat.toFixed(5)}, ${c.lon.toFixed(5)}`;

/** point inside the area square (with an inset so a 5 m pad fits)? */
function insideArea(p: Center, a: Area, insetM = 0) {
  const { x, y } = siteXY(p, a);
  const half = (a.widthKm * 1000) / 2 - insetM;
  return Math.abs(x) <= half && Math.abs(y) <= half;
}

const inputCls = "rounded border border-border bg-background px-2 py-1.5 font-mono text-xs text-foreground outline-none focus:border-[#3987e5] disabled:opacity-50";
const labelCls = "font-mono text-[10px] uppercase tracking-widest text-muted-foreground";
const btnCls = "h-7 cursor-pointer px-2.5 font-mono text-[10px] tracking-[0.08em]";

function Head({ n, title, right }: { n: string; title: string; right?: string }) {
  return (
    <h3 className="flex items-center border-b border-border px-3 py-1.5 font-mono text-[10px] font-semibold uppercase tracking-[0.14em] text-secondary-foreground">
      <span className="mr-2 rounded bg-secondary px-1.5 text-muted-foreground">{n}</span>{title}
      {right && <span className="ml-auto font-normal normal-case tracking-normal text-muted-foreground">{right}</span>}
    </h3>
  );
}

// ------------------------------------------------------------------ step 1: frame
function FrameStep({
  center, zoom, widthKm, setWidthKm, name, setName, locked, onLock, onUnlock, onFlyTo, onInteract, disabled,
}: {
  center: Center | null; zoom: number;
  widthKm: number; setWidthKm: (v: number) => void;
  name: string; setName: (s: string) => void;
  locked: Area | null; onLock: () => void; onUnlock: () => void;
  onFlyTo: (c: Center) => void; onInteract: () => void; disabled: boolean;
}) {
  const [widthText, setWidthText] = useState(String(widthKm));
  const [err, setErr] = useState<string | null>(null);
  const [query, setQuery] = useState("");
  const [hits, setHits] = useState<Hit[] | null>(null);
  const [searching, setSearching] = useState(false);
  const [seenWidth, setSeenWidth] = useState(widthKm);
  if (seenWidth !== widthKm) { setSeenWidth(widthKm); setWidthText(String(widthKm)); }

  const commitWidth = (raw: string) => {
    const v = clampKm(Math.round(parseFloat(raw) * 4) / 4);   // 0.25 km steps
    setWidthKm(v); setWidthText(String(v));
  };

  // live suggestions: geocode as you type (debounced; Nominatim allows ~1 req/s)
  const seq = useRef(0);
  const suppress = useRef(false);
  useEffect(() => {
    if (suppress.current) { suppress.current = false; return; }
    const q = query.trim();
    const mine = ++seq.current;
    const t = setTimeout(async () => {
      if (q.length < 3) { setHits(null); return; }
      setSearching(true);
      try {
        const r = await fetchJSON<Hit[]>(`/api/geocode?q=${encodeURIComponent(q)}`);
        if (mine === seq.current) setHits(r || []);
      } catch {
        if (mine === seq.current) setErr("place search failed");
      } finally {
        if (mine === seq.current) setSearching(false);
      }
    }, q.length < 3 ? 0 : 350);
    return () => clearTimeout(t);
  }, [query]);

  const pick = (h: Hit) => {
    onFlyTo({ lat: h.lat, lon: h.lon });
    if (!name.trim()) {
      const slug = h.name.split(",")[0].toLowerCase().replace(/[^a-z0-9]+/g, "_")
        .replace(/^_+|_+$/g, "").slice(0, 24) || "area";
      setName(/^[a-z]/.test(slug) ? slug : "area_" + slug);
    }
    seq.current++; suppress.current = true;
    setHits(null); setQuery(h.name.split(",")[0]);
  };
  const go = async () => {
    const q = query.trim();
    if (q.length < 2) return;
    let top = hits?.[0];
    if (!top) {
      setSearching(true);
      try { top = (await fetchJSON<Hit[]>(`/api/geocode?q=${encodeURIComponent(q)}`))?.[0]; }
      catch { /* fall through */ } finally { setSearching(false); }
    }
    if (top) pick(top); else setErr("no match for that place");
  };

  const zoomedIn = zoom >= 9;
  return (
    <section onPointerDownCapture={onInteract} onFocusCapture={onInteract}
      className={`hud-corners rounded-lg border bg-card ${locked ? "border-border/60" : "border-border"}`}>
      <Head n="1" title="Frame the area" right={`${MIN_KM}–${MAX_KM} km square`} />
      <div className="relative border-b border-border/60 p-3">
        <label className="flex flex-col gap-1">
          <span className={labelCls}>find a place</span>
          <div className="flex gap-2">
            <input value={query} placeholder="Kramatorsk · Cornell Arts Quad · Kyiv" disabled={disabled}
              onChange={(e) => setQuery(e.target.value)} onKeyDown={(e) => e.key === "Enter" && go()}
              className={`${inputCls} flex-1 font-sans`} />
            <Button size="sm" variant="secondary" disabled={searching || disabled} onClick={go}
              className="h-8 cursor-pointer px-3 font-mono text-[11px]">{searching ? "…" : "GO"}</Button>
          </div>
        </label>
        {hits && (
          <div className="absolute left-3 right-3 z-10 mt-1 max-h-56 overflow-y-auto rounded border border-border bg-card shadow-xl">
            {hits.length === 0 && <div className="px-3 py-2 text-xs text-muted-foreground">no matches</div>}
            {hits.map((h, i) => (
              <button key={i} onClick={() => pick(h)}
                className="block w-full cursor-pointer border-b border-border/50 px-3 py-2 text-left last:border-0 hover:bg-secondary">
                <div className="truncate text-xs">{h.name}</div>
                <div className="font-mono text-[10px] text-muted-foreground">{h.lat.toFixed(4)}, {h.lon.toFixed(4)} · {h.type}</div>
              </button>
            ))}
          </div>
        )}
      </div>
      <div className="flex flex-col gap-3 p-3">
        <label className="flex flex-col gap-1">
          <span className={labelCls}>name</span>
          <input value={name} placeholder="kramatorsk" disabled={disabled}
            onChange={(e) => setName(e.target.value.toLowerCase().replace(/[^a-z0-9_]+/g, "_").replace(/^[^a-z]+/, "").slice(0, 32))}
            className={inputCls} />
        </label>
        <label className="flex flex-col gap-1">
          <span className={labelCls}>width · km</span>
          <div className="flex items-center gap-3">
            <input type="range" min={MIN_KM} max={MAX_KM} step={0.25} value={widthKm} disabled={!!locked || disabled}
              onChange={(e) => setWidthKm(parseFloat(e.target.value))} className="flex-1 accent-[#3987e5] disabled:opacity-50" />
            <input inputMode="decimal" value={widthText} disabled={!!locked || disabled}
              onChange={(e) => setWidthText(e.target.value)} onBlur={(e) => commitWidth(e.target.value)}
              onKeyDown={(e) => e.key === "Enter" && commitWidth((e.target as HTMLInputElement).value)}
              className={`${inputCls} w-16 text-right`} />
          </div>
          <span className="text-[10px] text-muted-foreground">{widthKm} × {widthKm} km · {(widthKm * widthKm).toFixed(2)} km²</span>
        </label>
        <div className="rounded border border-border/60 bg-background/60 px-2.5 py-2 font-mono text-[10px] leading-relaxed text-muted-foreground">
          <div className="uppercase tracking-widest">{locked ? "locked centre" : "centre"}</div>
          <div className="text-foreground">{locked ? fmt(locked) : center ? fmt(center) : "—"}</div>
        </div>
        {locked ? (
          <Button size="sm" variant="secondary" onClick={onUnlock} disabled={disabled} className={btnCls}>◇ UNLOCK · REFRAME</Button>
        ) : (
          <Button size="sm" onClick={onLock} disabled={!zoomedIn || !center || disabled} className={btnCls}>◆ LOCK AREA</Button>
        )}
        {!locked && !zoomedIn && <div className="-mt-1 text-[11px] text-[#eda100]">zoom the map in to the area you want</div>}
        {err && <div className="text-[11px] text-[#d03b3b]">{err}</div>}
      </div>
    </section>
  );
}

// ------------------------------------------------------------------ step 2: place
function PlaceStep({
  tool, setTool, launch, zones, draft, onFinish, onUndo, onClear, note,
}: {
  tool: Tool; setTool: (t: Tool) => void;
  launch: Center | null; zones: Center[][]; draft: Center[];
  onFinish: () => void; onUndo: () => void; onClear: () => void; note: string | null;
}) {
  return (
    <section className="hud-corners rounded-lg border border-border bg-card">
      <Head n="2" title="Launch pin · friendly zones" right={`pad r = ${LAUNCH_R_M} m`} />
      <div className="flex flex-col gap-2.5 p-3">
        <div className="flex flex-wrap gap-2">
          <Button size="sm" variant={tool === "launch" ? "default" : "secondary"} onClick={() => setTool(tool === "launch" ? "none" : "launch")} className={btnCls}>
            ▲ {launch ? "MOVE LAUNCH PIN" : "SET LAUNCH PIN"}
          </Button>
          <Button size="sm" variant={tool === "zone" ? "default" : "secondary"} onClick={() => setTool(tool === "zone" ? "none" : "zone")} className={btnCls}>
            ▦ {tool === "zone" ? "DRAWING ZONE…" : "DRAW FRIENDLY ZONE"}
          </Button>
        </div>
        {tool === "zone" && (
          <div className="flex flex-wrap gap-2">
            <Button size="sm" onClick={onFinish} disabled={draft.length < 3} className={btnCls}>✓ FINISH ZONE ({draft.length} pts)</Button>
            <Button size="sm" variant="secondary" onClick={onUndo} disabled={draft.length === 0} className={btnCls}>↶ UNDO POINT</Button>
          </div>
        )}
        <div className="rounded border border-border/60 bg-background/60 px-2.5 py-2 font-mono text-[10px] leading-relaxed">
          <div className={launch ? "text-[#3ddc97]" : "text-[#eda100]"}>
            launch: {launch ? fmt(launch) : "not set (required)"}
          </div>
          <div className={zones.length ? "text-[#3987e5]" : "text-muted-foreground"}>
            friendly zones: {zones.length}{zones.length ? ` · ${zones.map((z) => z.length).join("+")} pts` : " (optional)"}
          </div>
          {note && <div className="mt-1 text-[#eda100]">{note}</div>}
        </div>
        {(launch || zones.length > 0 || draft.length > 0) && (
          <Button size="sm" variant="secondary" onClick={onClear} className={`${btnCls} self-start`}>✕ CLEAR ALL</Button>
        )}
      </div>
    </section>
  );
}

// ------------------------------------------------------------------ built worlds
function EnvCard({ e, active, selected, editing, trainBlocked, onActivate, onSelect, onEdit, onTrain }: {
  e: Environment; active: boolean; selected: boolean; editing: boolean; trainBlocked: boolean;
  onActivate: (name: string) => void; onSelect: (e: Environment) => void; onEdit: (e: Environment) => void;
  onTrain: (name: string) => void;
}) {
  const building = e.build_status === "running";
  const failed = e.build_status === "failed";
  return (
    <section onClick={() => onSelect(e)}
      className={`hud-corners cursor-pointer rounded-lg border bg-card transition-colors ${
        selected ? "border-[#3987e5] shadow-[0_0_0_1px_#3987e5,0_0_18px_rgba(57,135,229,0.25)]" : "border-border hover:border-muted-foreground/50"}`}>
      <h3 className="flex items-center border-b border-border px-3 py-1.5 font-mono text-[10px] font-semibold uppercase tracking-[0.14em] text-secondary-foreground">
        <span className="mr-1.5 inline-block size-[7px] rounded-full"
          style={{ background: building ? "#eda100" : failed ? "#d03b3b" : "#199e70" }} />
        {e.name.replace(/_/g, " ")}
        {active && <span className="ml-2 rounded bg-[#199e70]/20 px-1.5 font-mono text-[9px] text-[#199e70]">● ACTIVE</span>}
        {building && e.where && <span className="ml-2 font-normal normal-case tracking-normal text-muted-foreground">on the {e.where}</span>}
        {e.mb != null && <span className="ml-auto font-normal normal-case tracking-normal text-muted-foreground">{e.mb} MB{e.map ? " · trainable" : ""}</span>}
        {e.mb == null && e.map && <span className="ml-auto font-normal normal-case tracking-normal text-muted-foreground">trainable</span>}
      </h3>
      {e.demo && e.demo.length > 0 && (
        <div className="grid grid-cols-2 gap-px bg-border">
          {e.demo.slice(0, 2).map((m) =>
            m.kind === "video" ? (
              <video key={m.file} src={m.url} muted loop autoPlay playsInline className="aspect-video w-full bg-black object-cover" />
            ) : (
              // eslint-disable-next-line @next/next/no-img-element
              <img key={m.file} src={m.url} alt={m.file} className="aspect-video w-full bg-black object-cover" />
            ))}
        </div>
      )}
      <div className="p-3">
        {building ? (
          <div className="text-xs text-muted-foreground">
            building…
            {e.log && (
              <pre className="mt-2 max-h-28 overflow-y-auto whitespace-pre-wrap break-all rounded bg-background p-2 font-mono text-[10px] text-muted-foreground">
                {e.log.trimEnd().split("\n").slice(-5).join("\n")}
              </pre>
            )}
          </div>
        ) : failed ? (
          <div className="text-xs text-[#d03b3b]">
            build failed
            {e.log && <pre className="mt-2 max-h-28 overflow-y-auto whitespace-pre-wrap break-all rounded bg-background p-2 font-mono text-[10px] text-muted-foreground">{e.log.trimEnd().split("\n").slice(-4).join("\n")}</pre>}
          </div>
        ) : (
          <div className="flex flex-wrap gap-2" onClick={(ev) => ev.stopPropagation()}>
            {!active && (
              <Button size="sm" variant="secondary" onClick={() => onActivate(e.name)} className={btnCls}>◉ USE THIS</Button>
            )}
            {e.scenario && <JobButton label="▶ FLY MISSION" body={{ kind: "mission", scenario: e.scenario }} />}
            {e.map && (
              <Button size="sm" variant="secondary" disabled={trainBlocked}
                onClick={() => onTrain(e.name)} className={btnCls}>◉ TRAIN HERE</Button>
            )}
            {selected && !editing && (
              <Button size="sm" variant="secondary" onClick={() => onEdit(e)} className={btnCls}>▲ EDIT LAUNCH · ZONES</Button>
            )}
          </div>
        )}
      </div>
    </section>
  );
}

// ------------------------------------------------------------------ page
export default function Environments() {
  const [envs, setEnvs] = useState<Environment[] | null>(null);
  const [active, setActive] = useState<string | null>(null);
  const [widthKm, setWidthKmRaw] = useState(2);
  const [name, setName] = useState("");
  const [center, setCenter] = useState<Center | null>(null);
  const [zoom, setZoom] = useState(3);
  const [flyTo, setFlyTo] = useState<Center | null>(null);
  const [selected, setSelected] = useState<string | null>(null);
  // the plan: locked area + operator marks
  const [locked, setLocked] = useState<Area | null>(null);
  const [tool, setTool] = useState<Tool>("none");
  const [launch, setLaunch] = useState<Center | null>(null);
  const [zones, setZones] = useState<Center[][]>([]);
  const [draft, setDraft] = useState<Center[]>([]);
  const [note, setNote] = useState<string | null>(null);
  const [editing, setEditing] = useState<string | null>(null);      // world whose zones are being edited
  const [saved, setSaved] = useState<ZonesDoc | null>(null);        // selected world's saved zones
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState<string | null>(null);
  const train = useTrainControl();          // fast-lane training, shared with Models

  const setWidthKm = useCallback((v: number) => setWidthKmRaw(clampKm(v)), []);
  const onCenter = useCallback((c: Center, z: number) => { setCenter(c); setZoom(z); }, []);

  const load = useCallback(() => {
    fetchJSON<Environment[]>("/api/environments").then((d) => d && setEnvs(d));
    fetchJSON<{ name: string } | null>("/api/active").then((d) => setActive(d?.name ?? null));
  }, []);
  useEffect(() => {
    load();
    const id = setInterval(load, POLL_MS);
    return () => clearInterval(id);
  }, [load]);

  // saved zones of the selected world (consumers check saved.world === selected,
  // so a stale doc from the previous selection is never drawn)
  useEffect(() => {
    if (!selected) return;
    let alive = true;
    fetchJSON<ZonesDoc>(`/api/zones/${selected}`).then((z) => alive && z && setSaved(z));
    return () => { alive = false; };
  }, [selected]);
  const savedZones = saved && saved.world === selected ? saved : null;

  const activate = useCallback((name: string) => {
    postJSON("/api/active", { name }).then(() => { setActive(name); }).catch(() => {});
  }, []);

  const resetPlan = useCallback(() => {
    setLocked(null); setTool("none"); setLaunch(null); setZones([]); setDraft([]); setNote(null); setEditing(null);
  }, []);

  const selectEnv = useCallback((e: Environment) => {
    if (editing) return;                                            // finish or cancel the edit first
    if (selected === e.name) { setSelected(null); return; }         // click again: back to framing
    setSelected(e.name);
    resetPlan();
    if (e.center) {
      setFlyTo({ lat: e.center[0], lon: e.center[1], widthKm: e.half_m ? (2 * e.half_m) / 1000 : undefined });
    }
  }, [selected, editing, resetPlan]);
  const deselect = useCallback(() => { if (!editing) setSelected(null); }, [editing]);

  // an existing world: edit its launch pad / zones in place
  const editEnv = useCallback((e: Environment) => {
    if (!e.center || !e.half_m) { setErr("world has no centre metadata"); return; }
    setEditing(e.name);
    setLocked({ lat: e.center[0], lon: e.center[1], widthKm: (2 * e.half_m) / 1000 });
    setLaunch(savedZones?.launch_point ? { lat: savedZones.launch_point.lat, lon: savedZones.launch_point.lon } : null);
    setZones((savedZones?.safe_geo ?? []).map((poly) => poly.map(([lat, lon]) => ({ lat, lon }))));
    setDraft([]); setTool("none"); setNote(null);
  }, [savedZones]);

  const lock = useCallback(() => {
    if (!center) return;
    setLocked({ lat: center.lat, lon: center.lon, widthKm });
    setSelected(null); setLaunch(null); setZones([]); setDraft([]); setNote(null);
  }, [center, widthKm]);

  const onMapClick = useCallback((c: Center) => {
    if (!locked) return;
    if (tool === "launch") {
      if (!insideArea(c, locked, LAUNCH_R_M)) { setNote("the launch pad must sit inside the area"); return; }
      setLaunch(c); setNote(null); setTool("none");
    } else if (tool === "zone") {
      if (!insideArea(c, locked)) { setNote("zone points must be inside the area"); return; }
      setDraft((d) => [...d, c]); setNote(null);
    }
  }, [locked, tool]);

  const finishZone = useCallback(() => {
    // a double-click adds two near-identical points before the dblclick fires; drop them
    const pts = [...draft];
    while (pts.length > 1) {
      const a = pts[pts.length - 1], b = pts[pts.length - 2];
      const { x, y } = siteXY(a, b);
      if (Math.hypot(x, y) < 1.0) pts.pop(); else break;
    }
    if (pts.length >= 3) setZones((z) => [...z, pts]);
    setDraft([]); setTool("none");
  }, [draft]);
  const undoPoint = useCallback(() => setDraft((d) => d.slice(0, -1)), []);
  const clearAll = useCallback(() => { setLaunch(null); setZones([]); setDraft([]); setTool("none"); setNote(null); }, []);

  const build = async () => {
    if (!locked || !launch) return;
    setBusy(true); setErr(null);
    try {
      await postJSON("/api/environments/build", {
        name: name.trim(), lat: locked.lat, lon: locked.lon, half_km: locked.widthKm / 2,
        launch: [launch.lat, launch.lon],
        safe: zones.map((z) => z.map((p) => [p.lat, p.lon])),
      });
      setName(""); resetPlan(); load();
    } catch (e) {
      setErr(e instanceof Error ? e.message : "build failed to start");
    } finally {
      setBusy(false);
    }
  };
  const saveZones = async () => {
    if (!editing) return;
    setBusy(true); setErr(null);
    try {
      const r = await fetch(`/api/zones/${editing}`, {
        method: "PUT", headers: { "content-type": "application/json" },
        body: JSON.stringify({ launch: launch ? [launch.lat, launch.lon] : null,
          safe: zones.map((z) => z.map((p) => [p.lat, p.lon])) }),
      });
      if (!r.ok) throw new Error((await r.json()).detail || "save failed");
      setSaved(await r.json());
      const was = editing; resetPlan(); setSelected(was);
    } catch (e) {
      setErr(e instanceof Error ? e.message : "save failed");
    } finally {
      setBusy(false);
    }
  };

  const footprints: Footprint[] = useMemo(() => (envs ?? []).flatMap((e) =>
    e.center && e.half_m ? [{ name: e.name, center: e.center, half_m: e.half_m }] : []), [envs]);

  const overlay: ZoneOverlay | null = locked
    ? { launch, zones, draft, editable: true }
    : savedZones
      ? { launch: savedZones.launch_point ? { lat: savedZones.launch_point.lat, lon: savedZones.launch_point.lon } : null,
          zones: (savedZones.safe_geo ?? []).map((poly) => poly.map(([lat, lon]) => ({ lat, lon }))),
          draft: [], editable: false }
      : null;

  const nameOk = NAME_RE.test(name.trim());
  const blocker = editing ? null
    : !locked ? "lock the area first"
    : !nameOk ? (name.trim().length < 2 ? "enter a name (2+ chars, a-z 0-9 _)" : "name: letters, digits, underscore; start with a letter")
    : !launch ? "set the launch pin"
    : tool === "zone" && draft.length ? "finish or undo the zone you are drawing" : null;

  return (
    <main className="flex min-h-0 flex-1">
      <aside className="flex w-[360px] shrink-0 flex-col gap-3 overflow-y-auto border-r border-border p-3">
        <div className="flex items-baseline gap-2 px-0.5">
          <h2 className="text-base font-bold">Environments</h2>
          <span className="text-xs text-muted-foreground">frame · lock · pin · build</span>
        </div>
        {!editing && (
          <FrameStep center={center} zoom={zoom} widthKm={widthKm} setWidthKm={setWidthKm}
            name={name} setName={setName} locked={locked} onLock={lock} onUnlock={resetPlan}
            onFlyTo={setFlyTo} onInteract={deselect} disabled={busy} />
        )}
        {locked && (
          <PlaceStep tool={tool} setTool={setTool} launch={launch} zones={zones} draft={draft}
            onFinish={finishZone} onUndo={undoPoint} onClear={clearAll} note={note} />
        )}
        {(locked || editing) && (
          <section className="hud-corners rounded-lg border border-border bg-card">
            <Head n="3" title={editing ? `Save · ${editing.replace(/_/g, " ")}` : "Build"} />
            <div className="flex flex-col gap-2 p-3">
              {editing ? (
                <div className="flex gap-2">
                  <Button size="sm" disabled={busy || (tool === "zone" && draft.length > 0)} onClick={saveZones} className={btnCls}>{busy ? "SAVING…" : "✓ SAVE LAUNCH · ZONES"}</Button>
                  <Button size="sm" variant="secondary" disabled={busy} onClick={resetPlan} className={btnCls}>CANCEL</Button>
                </div>
              ) : (
                <Button size="sm" disabled={!!blocker || busy} onClick={build} className="h-8 cursor-pointer px-3 font-mono text-[11px] tracking-[0.08em]">
                  {busy ? "STARTING…" : "▶ BUILD WORLD"}
                </Button>
              )}
              {blocker && <div className="text-[11px] text-[#eda100]">{blocker}</div>}
              {err && <div className="text-[11px] text-[#d03b3b]">{err}</div>}
              {!editing && <div className="text-[11px] text-muted-foreground">the pin becomes the spawn; trees are kept clear of it and both land in the world&apos;s zones file</div>}
            </div>
          </section>
        )}
        {!locked && !editing && (
          <TrainPanel ctl={train} />
        )}
        <div className="mt-1 px-0.5 font-mono text-[10px] uppercase tracking-[0.14em] text-muted-foreground">
          Built worlds {envs ? `· ${envs.length}` : ""}
        </div>
        {envs === null ? (
          <div className="p-6 text-center text-xs text-muted-foreground">Loading…</div>
        ) : envs.length === 0 ? (
          <div className="p-6 text-center text-xs text-muted-foreground">No worlds yet.</div>
        ) : (
          envs.map((e) => (
            <EnvCard key={e.name} e={e} active={e.name === active} selected={e.name === selected}
              editing={editing === e.name} trainBlocked={train.blocked}
              onActivate={activate} onSelect={selectEnv} onEdit={editEnv}
              onTrain={(name) => train.startTrain(name)} />
          ))
        )}
      </aside>
      <div className="relative min-w-0 flex-1">
        <AreaMap widthKm={widthKm} flyTo={flyTo} onCenter={onCenter}
          footprints={footprints} selected={selected} onUserPan={deselect}
          locked={locked} tool={tool} overlay={overlay} onMapClick={onMapClick} onMapDblClick={finishZone} />
      </div>
    </main>
  );
}
