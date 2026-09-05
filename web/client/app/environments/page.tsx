"use client";

import { useCallback, useEffect, useState } from "react";
import { JobButton } from "@/components/job-controls";
import { Button } from "@/components/ui/button";
import { fetchJSON, postJSON } from "@/lib/vesper";

// Add any place on earth as a world: enter a name + coordinates, the server
// builds it (Copernicus terrain + Esri imagery + OSM, no keys) and syncs it to
// the GPU box. Built worlds are listed here and launch straight into Isaac.

type Environment = {
  name: string;
  usd: string | null;
  scenario: string | null;
  map: string | null;
  mb?: number;
  build_status?: string;
  log?: string;
};

const POLL_MS = 4000;

type Hit = { name: string; lat: number; lon: number; half_km: number; type: string };

function AddEnvironment({ onSubmitted }: { onSubmitted: () => void }) {
  const [name, setName] = useState("");
  const [lat, setLat] = useState("");
  const [lon, setLon] = useState("");
  const [halfKm, setHalfKm] = useState("1.0");
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState<string | null>(null);

  const [query, setQuery] = useState("");
  const [hits, setHits] = useState<Hit[] | null>(null);
  const [searching, setSearching] = useState(false);

  const search = async () => {
    if (query.trim().length < 2) return;
    setSearching(true); setHits(null); setErr(null);
    try {
      const r = await fetchJSON<Hit[]>(`/api/geocode?q=${encodeURIComponent(query.trim())}`);
      setHits(r || []);
    } catch {
      setErr("place search failed");
    } finally {
      setSearching(false);
    }
  };

  const pick = (h: Hit) => {
    setLat(h.lat.toFixed(5));
    setLon(h.lon.toFixed(5));
    setHalfKm(String(h.half_km));
    if (!name.trim()) {
      const slug = h.name.split(",")[0].toLowerCase().replace(/[^a-z0-9]+/g, "_")
        .replace(/^_+|_+$/g, "").slice(0, 24) || "site";
      setName(/^[a-z]/.test(slug) ? slug : "site_" + slug);
    }
    setHits(null);
  };

  const submit = async () => {
    setBusy(true); setErr(null);
    try {
      await postJSON("/api/environments/build", {
        name: name.trim(), lat: parseFloat(lat), lon: parseFloat(lon),
        half_km: parseFloat(halfKm),
      });
      setName(""); setLat(""); setLon("");
      onSubmitted();
    } catch (e) {
      setErr(e instanceof Error ? e.message : "build failed to start");
    } finally {
      setBusy(false);
    }
  };

  const ready = /^[a-z][a-z0-9_]{1,31}$/.test(name.trim()) && lat && lon;

  return (
    <section className="hud-corners rounded-lg border border-border bg-card">
      <h3 className="flex items-center border-b border-border px-3 py-1.5 font-mono text-[10px] font-semibold uppercase tracking-[0.14em] text-secondary-foreground">
        <span className="mr-1.5 text-muted-foreground">▮</span>Add an environment
        <span className="ml-auto font-normal normal-case tracking-normal text-muted-foreground">
          Copernicus + Esri + OSM · any coordinates
        </span>
      </h3>
      <div className="relative border-b border-border/60 p-3">
        <label className="flex flex-col gap-1">
          <span className="font-mono text-[10px] uppercase tracking-widest text-muted-foreground">
            search a place
          </span>
          <div className="flex gap-2">
            <input
              value={query}
              placeholder="Kramatorsk, Ukraine   ·   Cornell Arts Quad   ·   Kyiv"
              onChange={(e) => setQuery(e.target.value)}
              onKeyDown={(e) => e.key === "Enter" && search()}
              className="flex-1 rounded border border-border bg-background px-2 py-1.5 text-xs text-foreground outline-none focus:border-[#3987e5]"
            />
            <Button
              size="sm" variant="secondary" disabled={searching} onClick={search}
              className="h-8 cursor-pointer px-3 font-mono text-[11px]"
            >
              {searching ? "…" : "SEARCH"}
            </Button>
          </div>
        </label>
        {hits && (
          <div className="absolute left-3 right-3 z-10 mt-1 max-h-56 overflow-y-auto rounded border border-border bg-card shadow-xl">
            {hits.length === 0 && (
              <div className="px-3 py-2 text-xs text-muted-foreground">no matches</div>
            )}
            {hits.map((h, i) => (
              <button
                key={i}
                onClick={() => pick(h)}
                className="block w-full cursor-pointer border-b border-border/50 px-3 py-2 text-left last:border-0 hover:bg-secondary"
              >
                <div className="truncate text-xs">{h.name}</div>
                <div className="font-mono text-[10px] text-muted-foreground">
                  {h.lat.toFixed(4)}, {h.lon.toFixed(4)} · {h.half_km} km · {h.type}
                </div>
              </button>
            ))}
          </div>
        )}
      </div>
      <div className="flex flex-wrap items-end gap-3 p-3">
        {[
          ["name", name, setName, "kramatorsk", "flex-1 min-w-[140px]"],
          ["lat", lat, setLat, "48.7233", "w-28"],
          ["lon", lon, setLon, "37.5562", "w-28"],
          ["radius km", halfKm, setHalfKm, "1.0", "w-24"],
        ].map(([label, val, set, ph, cls]) => (
          <label key={label as string} className={`flex flex-col gap-1 ${cls}`}>
            <span className="font-mono text-[10px] uppercase tracking-widest text-muted-foreground">{label as string}</span>
            <input
              value={val as string}
              placeholder={ph as string}
              onChange={(e) => (set as (s: string) => void)(e.target.value)}
              className="rounded border border-border bg-background px-2 py-1.5 font-mono text-xs text-foreground outline-none focus:border-[#3987e5]"
            />
          </label>
        ))}
        <Button
          size="sm" disabled={!ready || busy} onClick={submit}
          className="h-8 cursor-pointer px-3 font-mono text-[11px] tracking-[0.08em]"
        >
          {busy ? "STARTING…" : "▶ BUILD WORLD"}
        </Button>
      </div>
      {err && <div className="px-3 pb-2 text-[11px] text-[#d03b3b]">{err}</div>}
      <div className="px-3 pb-3 text-[11px] text-muted-foreground">
        Builds in ~2–3 min, then syncs to the box. Tip: a smaller radius = sharper detail.
      </div>
    </section>
  );
}

function ReconstructFromPhotos({ onSubmitted }: { onSubmitted: () => void }) {
  const [name, setName] = useState("");
  const [files, setFiles] = useState<FileList | null>(null);
  const [busy, setBusy] = useState(false);
  const [msg, setMsg] = useState<string | null>(null);

  const submit = async () => {
    if (!files || files.length < 8) { setMsg("select at least ~8 overlapping photos"); return; }
    setBusy(true); setMsg(null);
    try {
      const fd = new FormData();
      fd.append("name", name.trim());
      fd.append("half_km", "0.3");
      Array.from(files).forEach((f) => fd.append("files", f));
      const r = await fetch("/api/environments/reconstruct", { method: "POST", body: fd });
      if (!r.ok) throw new Error((await r.json()).detail || "upload failed");
      setName(""); setFiles(null);
      setMsg("uploaded — reconstructing (needs the GPU box up; ~20–60 min)");
      onSubmitted();
    } catch (e) {
      setMsg(e instanceof Error ? e.message : "upload failed");
    } finally {
      setBusy(false);
    }
  };

  const ready = /^[a-z][a-z0-9_]{1,31}$/.test(name.trim()) && files && files.length >= 8;

  return (
    <section className="hud-corners rounded-lg border border-border bg-card">
      <h3 className="flex items-center border-b border-border px-3 py-1.5 font-mono text-[10px] font-semibold uppercase tracking-[0.14em] text-secondary-foreground">
        <span className="mr-1.5 text-muted-foreground">▮</span>Reconstruct from photos
        <span className="ml-auto font-normal normal-case tracking-normal text-muted-foreground">
          drone footage → OpenDroneMap → world
        </span>
      </h3>
      <div className="flex flex-wrap items-end gap-3 p-3">
        <label className="flex min-w-[140px] flex-1 flex-col gap-1">
          <span className="font-mono text-[10px] uppercase tracking-widest text-muted-foreground">name</span>
          <input
            value={name} placeholder="recon_site"
            onChange={(e) => setName(e.target.value)}
            className="rounded border border-border bg-background px-2 py-1.5 font-mono text-xs text-foreground outline-none focus:border-[#3987e5]"
          />
        </label>
        <label className="flex flex-col gap-1">
          <span className="font-mono text-[10px] uppercase tracking-widest text-muted-foreground">
            photos {files ? `(${files.length})` : ""}
          </span>
          <input
            type="file" multiple accept="image/*"
            onChange={(e) => setFiles(e.target.files)}
            className="text-xs text-muted-foreground file:mr-2 file:cursor-pointer file:rounded file:border file:border-border file:bg-secondary file:px-2 file:py-1 file:text-xs file:text-foreground"
          />
        </label>
        <Button
          size="sm" disabled={!ready || busy} onClick={submit}
          className="h-8 cursor-pointer px-3 font-mono text-[11px] tracking-[0.08em]"
        >
          {busy ? "UPLOADING…" : "▶ RECONSTRUCT"}
        </Button>
      </div>
      {msg && <div className="px-3 pb-2 text-[11px] text-muted-foreground">{msg}</div>}
      <div className="px-3 pb-3 text-[11px] text-muted-foreground">
        Overlapping (~70%) drone shots reconstruct best; nadir gives terrain, oblique/orbit adds facades.
      </div>
    </section>
  );
}

function EnvCard({ e, active, onActivate }: {
  e: Environment; active: boolean; onActivate: (name: string) => void;
}) {
  const building = e.build_status === "running";
  const failed = e.build_status === "failed";
  return (
    <section className="hud-corners rounded-lg border border-border bg-card">
      <h3 className="flex items-center border-b border-border px-3 py-1.5 font-mono text-[10px] font-semibold uppercase tracking-[0.14em] text-secondary-foreground">
        <span
          className="mr-1.5 inline-block size-[7px] rounded-full"
          style={{ background: building ? "#eda100" : failed ? "#d03b3b" : "#199e70" }}
        />
        {e.name.replace(/_/g, " ")}
        {active && (
          <span className="ml-2 rounded bg-[#199e70]/20 px-1.5 font-mono text-[9px] text-[#199e70]">
            ● ACTIVE
          </span>
        )}
        {e.mb != null && (
          <span className="ml-auto font-normal normal-case tracking-normal text-muted-foreground">
            {e.mb} MB{e.map ? " · trainable" : ""}
          </span>
        )}
      </h3>
      <div className="p-3">
        {building ? (
          <div className="text-xs text-muted-foreground">
            building… (~2–3 min)
            {e.log && (
              <pre className="mt-2 max-h-24 overflow-y-auto whitespace-pre-wrap break-all rounded bg-background p-2 font-mono text-[10px] text-muted-foreground">
                {e.log.trimEnd().split("\n").slice(-4).join("\n")}
              </pre>
            )}
          </div>
        ) : failed ? (
          <div className="text-xs text-[#d03b3b]">build failed — check server log</div>
        ) : (
          <div className="flex flex-wrap gap-2">
            {!active && (
              <Button
                size="sm" variant="secondary" onClick={() => onActivate(e.name)}
                className="h-7 cursor-pointer px-3 font-mono text-[11px] tracking-[0.08em]"
              >
                ◉ USE THIS
              </Button>
            )}
            {e.scenario && (
              <JobButton label="▶ FLY MISSION" body={{ kind: "mission", scenario: e.scenario }} />
            )}
            {e.map && e.usd && (
              <JobButton
                label="◉ TRAIN HERE" variant="secondary"
                body={{ kind: "train", world: e.usd, map: e.map }}
              />
            )}
            {!e.map && (
              <span className="self-center text-[11px] text-muted-foreground">
                export a world-map to train (search task needs occluders)
              </span>
            )}
          </div>
        )}
      </div>
    </section>
  );
}

export default function Environments() {
  const [envs, setEnvs] = useState<Environment[] | null>(null);
  const [active, setActive] = useState<string | null>(null);
  const load = useCallback(() => {
    fetchJSON<Environment[]>("/api/environments").then((d) => d && setEnvs(d));
    fetchJSON<{ name: string } | null>("/api/active").then((d) => setActive(d?.name ?? null));
  }, []);
  useEffect(() => {
    load();
    const id = setInterval(load, POLL_MS);
    return () => clearInterval(id);
  }, [load]);

  const activate = useCallback((name: string) => {
    postJSON("/api/active", { name }).then(() => { setActive(name); }).catch(() => {});
  }, []);

  return (
    <main className="min-h-0 flex-1 overflow-y-auto p-4">
      <div className="mb-3.5 flex items-baseline gap-3">
        <h2 className="text-base font-bold">Environments</h2>
        <span className="text-xs text-muted-foreground">
          add any place on earth, then fly or train in it on the box
        </span>
      </div>
      <div className="mb-3 grid grid-cols-1 gap-3 lg:grid-cols-2">
        <AddEnvironment onSubmitted={load} />
        <ReconstructFromPhotos onSubmitted={load} />
      </div>
      {envs === null ? (
        <div className="p-10 text-center text-muted-foreground">Loading…</div>
      ) : envs.length === 0 ? (
        <div className="p-10 text-center text-muted-foreground">
          No worlds yet — add one above.
        </div>
      ) : (
        <div className="grid grid-cols-1 gap-3 md:grid-cols-2 xl:grid-cols-3">
          {envs.map((e) => (
            <EnvCard key={e.name} e={e} active={e.name === active} onActivate={activate} />
          ))}
        </div>
      )}
    </main>
  );
}
