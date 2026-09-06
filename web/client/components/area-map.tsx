"use client";

import "leaflet/dist/leaflet.css";
import type * as L from "leaflet";
import type { LayerGroup, Map as LeafletMap } from "leaflet";
import { useCallback, useEffect, useRef, useState } from "react";

// Satellite map for the environments page. Two modes:
//   framing  -- the red hatched square is pinned to the viewport centre and scaled
//               to widthKm; pan/zoom until the area you want sits under it.
//   locked   -- the square is fixed to the ground (a geo rectangle); the operator
//               then drops a launch pin and draws friendly zones inside it.
// Existing worlds are drawn as green footprints; the selected one carries its
// saved launch pad and zones.

export type Center = { lat: number; lon: number; widthKm?: number };   // widthKm: frame this width
export type Footprint = { name: string; center: [number, number]; half_m: number };
export type Area = { lat: number; lon: number; widthKm: number };
export type Tool = "none" | "launch" | "zone";
export type ZoneOverlay = {
  launch: Center | null;                // the pin (5 m pad drawn around it)
  zones: Center[][];                    // finished friendly zones
  draft: Center[];                      // zone being drawn
  editable: boolean;                    // bright (operator's) vs dim (saved, read-only)
};

export const LAUNCH_R_M = 5;
// Web-Mercator ground resolution at zoom 0 (m/px, 256-px tiles)
const MPP_Z0 = 156543.03392804097;

// zoom at which a box `widthM` wide fills `fracOfView` of the shorter viewport side
export function zoomFor(widthM: number, lat: number, viewMin: number, fracOfView = 0.5) {
  const mppWanted = widthM / (viewMin * fracOfView);
  return Math.log2((MPP_Z0 * Math.cos((lat * Math.PI) / 180)) / mppWanted);
}

/** local ENU metres of (lat, lon) about a centre -- same frame the builder uses */
export function siteXY(p: { lat: number; lon: number }, c: { lat: number; lon: number }) {
  return {
    x: (p.lon - c.lon) * 111320 * Math.cos((c.lat * Math.PI) / 180),
    y: (p.lat - c.lat) * 110574,
  };
}

function bounds(a: Area): [[number, number], [number, number]] {
  const half = (a.widthKm * 1000) / 2;
  const dLat = half / 110574;
  const dLon = half / (111320 * Math.cos((a.lat * Math.PI) / 180));
  return [[a.lat - dLat, a.lon - dLon], [a.lat + dLat, a.lon + dLon]];
}

export function AreaMap({
  widthKm, flyTo, onCenter, footprints, selected, onUserPan,
  locked, tool, overlay, onMapClick, onMapDblClick,
}: {
  widthKm: number;
  flyTo: Center | null;                 // set by place search / card click; consumed once
  onCenter: (c: Center, zoom: number) => void;
  footprints: Footprint[];              // built worlds, drawn as outlines
  selected: string | null;              // a selected world hides the build target
  onUserPan: () => void;                // user dragged the map: back to framing mode
  locked: Area | null;                  // the fixed area once the operator locks it
  tool: Tool;                           // what a map click does
  overlay: ZoneOverlay | null;          // launch pad + zones to draw
  onMapClick: (c: Center) => void;
  onMapDblClick: () => void;            // finishes the zone being drawn
}) {
  const hostRef = useRef<HTMLDivElement>(null);
  const mapRef = useRef<LeafletMap | null>(null);
  const footLayerRef = useRef<LayerGroup | null>(null);
  const areaLayerRef = useRef<LayerGroup | null>(null);
  const zoneLayerRef = useRef<LayerGroup | null>(null);
  const leafletRef = useRef<typeof L | null>(null);
  const widthRef = useRef(widthKm);
  const onUserPanRef = useRef(onUserPan);
  const onMapClickRef = useRef(onMapClick);
  const onMapDblClickRef = useRef(onMapDblClick);
  const toolRef = useRef<Tool>(tool);
  const [ready, setReady] = useState(false);
  const [boxPx, setBoxPx] = useState(0);
  const [scale, setScale] = useState<{ px: number; label: string }>({ px: 0, label: "" });
  useEffect(() => { onUserPanRef.current = onUserPan; }, [onUserPan]);
  useEffect(() => { onMapClickRef.current = onMapClick; }, [onMapClick]);
  useEffect(() => { onMapDblClickRef.current = onMapDblClick; }, [onMapDblClick]);
  useEffect(() => { toolRef.current = tool; }, [tool]);

  // size of the box in screen pixels for the current view (exact: projects the
  // east edge through the map's own transform) + the scale bar
  const measure = useCallback(() => {
    const map = mapRef.current;
    if (!map) return;
    const c = map.getCenter();
    const halfM = (widthRef.current * 1000) / 2;
    const dLon = halfM / (111320 * Math.cos((c.lat * Math.PI) / 180));
    const p0 = map.latLngToContainerPoint(c);
    const p1 = map.latLngToContainerPoint([c.lat, c.lng + dLon]);
    setBoxPx(Math.abs(p1.x - p0.x) * 2);
    // scale bar: a round distance (1/2/5 x 10^n m) that fits in ~140 px
    const mpp = halfM / Math.abs(p1.x - p0.x);
    const target = 140 * mpp;
    const pow = Math.pow(10, Math.floor(Math.log10(target)));
    const nice = [5, 2, 1].map((k) => k * pow).find((d) => d <= target) ?? pow;
    setScale({ px: nice / mpp, label: nice >= 1000 ? `${nice / 1000} km` : `${nice} m` });
    onCenter({ lat: c.lat, lon: c.lng }, map.getZoom());
  }, [onCenter]);

  useEffect(() => {
    let disposed = false;
    (async () => {
      const L = (await import("leaflet")).default;
      if (disposed || !hostRef.current || mapRef.current) return;
      // optional deep link: /environments?lat=48.72&lon=37.56&z=13
      const q = new URLSearchParams(window.location.search);
      const lat = parseFloat(q.get("lat") ?? ""), lon = parseFloat(q.get("lon") ?? "");
      const z = parseFloat(q.get("z") ?? "");
      const hasStart = Number.isFinite(lat) && Number.isFinite(lon);
      const map = L.map(hostRef.current, {
        center: hasStart ? [lat, lon] : [30, 15],
        zoom: hasStart ? (Number.isFinite(z) ? z : 13) : 3,
        minZoom: 2, maxZoom: 19,
        zoomControl: false, zoomSnap: 0, zoomDelta: 1,
        scrollWheelZoom: false,           // replaced by the direct handler below
        doubleClickZoom: false,           // double-click finishes a zone instead
        worldCopyJump: true, attributionControl: true,
      });
      // Direct wheel/trackpad zoom. Leaflet's built-in handler batches deltas,
      // snaps, and drops input during each 250 ms zoom animation, which feels
      // sluggish on a trackpad. A plain non-animated setView is worse: it fires
      // "viewprereset", which discards every loaded tile, so the map blanks
      // until tiles refetch. flyTo avoids both by stepping the internal move
      // (_moveStart / _move / _moveEnd) with fractional zoom -- existing tiles
      // stay on screen and scale, new ones load underneath. Same trick here.
      // Pinch arrives as a wheel event with ctrlKey and much smaller deltas.
      const host = hostRef.current;
      type Internal = LeafletMap & {
        _animatingZoom?: boolean;
        _stop: () => void;
        _moveStart: (zoomChanged: boolean, noMoveStart: boolean) => LeafletMap;
        _move: (center: L.LatLng, zoom: number, data?: object) => LeafletMap;
        _moveEnd: (zoomChanged: boolean) => LeafletMap;
      };
      const m = map as Internal;
      let gesture: ReturnType<typeof setTimeout> | null = null;
      let gestureStart = 0;
      const onWheel = (e: WheelEvent) => {
        e.preventDefault();
        if (m._animatingZoom) return;                   // +/- button animation in flight
        const px = e.deltaMode === 1 ? e.deltaY * 20 : e.deltaMode === 2 ? e.deltaY * 60 : e.deltaY;
        const gain = e.ctrlKey ? 1 / 40 : 1 / 160;     // zoom levels per px of delta
        const from = map.getZoom();
        const to = Math.max(map.getMinZoom(), Math.min(map.getMaxZoom(), from - px * gain));
        if (to === from) return;
        if (!gesture) { m._stop(); m._moveStart(true, false); gestureStart = performance.now(); }
        else clearTimeout(gesture);
        // keep the point under the cursor fixed (same maths as setZoomAround)
        const scale = map.getZoomScale(to, from);
        const half = map.getSize().divideBy(2);
        const offset = map.mouseEventToContainerPoint(e).subtract(half).multiplyBy(1 - 1 / scale);
        const center = map.containerPointToLatLng(half.add(offset));
        // flyTo:true tells the tile layer to only rescale what is on screen
        // (no prune, no abort) -- exactly what flyTo/pinch do mid-animation
        m._move(center, to, { flyTo: true });
        // every ~350 ms of continuous gesture, checkpoint so tiles for the new
        // level start loading underneath instead of waiting for the gesture to end
        if (performance.now() - gestureStart > 350) {
          m._moveEnd(true); m._moveStart(true, false); gestureStart = performance.now();
        }
        gesture = setTimeout(() => { gesture = null; m._moveEnd(true); }, 150);
      };
      host.addEventListener("wheel", onWheel, { passive: false });
      map.once("unload", () => host.removeEventListener("wheel", onWheel));
      L.tileLayer(
        "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
        // Esri has no imagery past z18 in most places (z19 is a white "no data"
        // tile), so upscale z18 for the last level instead of requesting it
        { maxZoom: 19, maxNativeZoom: 18, attribution: "Esri, Maxar, Earthstar Geographics" },
      ).addTo(map);
      L.tileLayer(
        "https://server.arcgisonline.com/ArcGIS/rest/services/Reference/World_Boundaries_and_Places/MapServer/tile/{z}/{y}/{x}",
        { maxZoom: 19, maxNativeZoom: 18, opacity: 0.9 },
      ).addTo(map);
      L.control.zoom({ position: "bottomright" }).addTo(map);
      footLayerRef.current = L.layerGroup().addTo(map);
      areaLayerRef.current = L.layerGroup().addTo(map);
      zoneLayerRef.current = L.layerGroup().addTo(map);
      map.on("dragstart", () => onUserPanRef.current());
      map.on("click", (e: L.LeafletMouseEvent) => {
        if (toolRef.current !== "none") onMapClickRef.current({ lat: e.latlng.lat, lon: e.latlng.lng });
      });
      map.on("dblclick", () => {
        if (toolRef.current === "zone") onMapDblClickRef.current();
      });
      map.attributionControl.setPrefix(false);
      map.on("move zoom zoomend moveend resize", measure);
      mapRef.current = map;
      leafletRef.current = L;
      measure();
      setReady(true);
    })();
    return () => {
      disposed = true;
      mapRef.current?.remove();
      mapRef.current = null;
    };
  }, [measure]);

  // width changed: rescale the box in place (never touch the zoom)
  useEffect(() => {
    widthRef.current = widthKm;
    measure();
  }, [widthKm, measure]);

  // place search picked / card clicked: fly there at a zoom that frames the box
  useEffect(() => {
    const map = mapRef.current;
    if (!map || !flyTo) return;
    const size = map.getSize();
    const frameM = (flyTo.widthKm ?? widthRef.current) * 1000;
    map.flyTo([flyTo.lat, flyTo.lon],
      zoomFor(frameM, flyTo.lat, Math.min(size.x, size.y)), { duration: 1.2 });
  }, [flyTo]);

  // built-world footprints: green outlines, the selected one bright + labelled.
  // Keyed on the *content* of the list, so re-renders from map movement or the
  // 4 s poll don't tear the shapes down and rebuild them (that made them jump).
  const footKey = JSON.stringify(footprints);
  useEffect(() => {
    const layer = footLayerRef.current, L = leafletRef.current;
    if (!ready || !layer || !L) return;
    layer.clearLayers();
    for (const f of JSON.parse(footKey) as Footprint[]) {
      const [lat, lon] = f.center;
      const dLat = f.half_m / 110574;
      const dLon = f.half_m / (111320 * Math.cos((lat * Math.PI) / 180));
      const on = f.name === selected;
      L.rectangle([[lat - dLat, lon - dLon], [lat + dLat, lon + dLon]], {
        color: on ? "#3ddc97" : "#199e70", weight: on ? 2.5 : 1.5,
        fillColor: "#199e70", fillOpacity: on ? 0.18 : 0.06, dashArray: on ? undefined : "6 4",
        interactive: false,
      }).addTo(layer);
      L.marker([lat + dLat, lon], {
        interactive: false,
        icon: L.divIcon({
          className: "",
          html: `<div style="transform:translate(-50%,-115%);white-space:nowrap;font:600 10px ui-monospace,Menlo,monospace;letter-spacing:.12em;text-transform:uppercase;color:${on ? "#3ddc97" : "#199e70"};background:rgba(0,0,0,.65);padding:1px 6px;border-radius:3px">${f.name.replace(/_/g, " ")}</div>`,
          iconSize: [0, 0],
        }),
      }).addTo(layer);
    }
  }, [footKey, selected, ready]);

  // the locked area: a ground-fixed red square (replaces the screen-pinned box)
  const lockKey = JSON.stringify(locked);
  useEffect(() => {
    const layer = areaLayerRef.current, L = leafletRef.current;
    if (!ready || !layer || !L) return;
    layer.clearLayers();
    const a = JSON.parse(lockKey) as Area | null;
    if (!a) return;
    L.rectangle(bounds(a), {
      color: "#ff3b3b", weight: 2, dashArray: "8 6", fillColor: "#ff3b3b", fillOpacity: 0.08,
      interactive: false,
    }).addTo(layer);
  }, [lockKey, ready]);

  // launch pad + friendly zones (operator's draft or a saved world's)
  const zoneKey = JSON.stringify(overlay);
  useEffect(() => {
    const layer = zoneLayerRef.current, L = leafletRef.current;
    if (!ready || !layer || !L) return;
    layer.clearLayers();
    const o = JSON.parse(zoneKey) as ZoneOverlay | null;
    if (!o) return;
    const blue = o.editable ? "#3987e5" : "#5b8fd6";
    const green = o.editable ? "#3ddc97" : "#199e70";
    for (const z of o.zones) {
      L.polygon(z.map((p) => [p.lat, p.lon] as [number, number]), {
        color: blue, weight: 2, fillColor: blue, fillOpacity: o.editable ? 0.22 : 0.14, interactive: false,
      }).addTo(layer);
    }
    if (o.draft.length) {
      const pts = o.draft.map((p) => [p.lat, p.lon] as [number, number]);
      L.polyline(pts, { color: blue, weight: 2, dashArray: "4 4", interactive: false }).addTo(layer);
      if (pts.length >= 3) {
        L.polygon(pts, { stroke: false, fillColor: blue, fillOpacity: 0.12, interactive: false }).addTo(layer);
      }
      for (const p of pts) {
        L.circleMarker(p, { radius: 4, color: "#fff", weight: 1.5, fillColor: blue, fillOpacity: 1, interactive: false }).addTo(layer);
      }
    }
    if (o.launch) {
      const ll: [number, number] = [o.launch.lat, o.launch.lon];
      L.circle(ll, { radius: LAUNCH_R_M, color: green, weight: 2, fillColor: green, fillOpacity: 0.25, interactive: false }).addTo(layer);
      L.marker(ll, {
        interactive: false,
        icon: L.divIcon({
          className: "",
          html: `<div style="transform:translate(-50%,-100%);display:flex;flex-direction:column;align-items:center;gap:1px;pointer-events:none">
                   <div style="font:600 9px ui-monospace,Menlo,monospace;letter-spacing:.12em;color:${green};background:rgba(0,0,0,.65);padding:1px 5px;border-radius:3px;white-space:nowrap">LAUNCH</div>
                   <div style="width:0;height:0;border-left:5px solid transparent;border-right:5px solid transparent;border-top:9px solid ${green}"></div>
                 </div>`,
          iconSize: [0, 0],
        }),
      }).addTo(layer);
    }
  }, [zoneKey, ready]);

  const framing = !locked && !selected;
  return (
    <div className={`relative h-full w-full overflow-hidden bg-[#0d0d0d] ${tool !== "none" ? "[&_.leaflet-container]:cursor-crosshair" : ""}`}>
      <div ref={hostRef} className="absolute inset-0" />
      {/* the area box: pinned to viewport centre, sized to widthKm. Hidden once the
          area is locked (it becomes a ground-fixed rectangle) or a world is selected */}
      <div className={`pointer-events-none absolute inset-0 z-[500] flex items-center justify-center transition-opacity ${framing ? "opacity-100" : "opacity-0"}`}>
        <div
          className="relative shrink-0 transition-opacity"
          style={{
            width: boxPx, height: boxPx,
            opacity: boxPx < 8 ? 0 : 1,                  // zoomed out: only the reticle shows
            border: "2px dashed #ff3b3b",
            background: "repeating-linear-gradient(45deg, rgba(255,59,59,0.26) 0 2px, transparent 2px 12px)",
            boxShadow: "0 0 0 1px rgba(0,0,0,0.6), 0 0 28px rgba(255,59,59,0.25)",
          }}
        >
          {/* corner ticks */}
          {["-top-px -left-px border-t-2 border-l-2", "-top-px -right-px border-t-2 border-r-2",
            "-bottom-px -left-px border-b-2 border-l-2", "-bottom-px -right-px border-b-2 border-r-2"]
            .map((c) => (
              <span key={c} className={`absolute size-3 border-[#ff3b3b] ${c}`} />
            ))}
          <div className="absolute left-1/2 top-full mt-1.5 -translate-x-1/2 whitespace-nowrap rounded bg-black/70 px-1.5 py-0.5 font-mono text-[10px] tracking-widest text-[#ff6b6b]">
            {widthKm} × {widthKm} KM
          </div>
        </div>
        {/* centre reticle */}
        <span className="absolute size-1.5 rounded-full bg-[#ff3b3b] shadow-[0_0_0_1px_rgba(0,0,0,.7)]" />
      </div>
      {/* tool hint */}
      {tool !== "none" && (
        <div className="pointer-events-none absolute left-1/2 top-3 z-[500] -translate-x-1/2 rounded bg-black/75 px-3 py-1 font-mono text-[10px] tracking-widest text-white">
          {tool === "launch" ? "CLICK TO PLACE THE LAUNCH PIN" : "CLICK TO ADD ZONE POINTS · DOUBLE-CLICK OR FINISH TO CLOSE"}
        </div>
      )}
      {/* scale: end-ticked line with the round distance it spans */}
      {scale.px > 0 && (
        <div className="pointer-events-none absolute bottom-6 left-3 z-[500] font-mono text-[10px] text-white drop-shadow-[0_0_2px_rgba(0,0,0,1)]">
          <div className="mb-0.5 text-center tracking-widest">{scale.label}</div>
          <div className="relative h-2" style={{ width: scale.px }}>
            <span className="absolute inset-x-0 bottom-0 h-px bg-white shadow-[0_0_2px_rgba(0,0,0,1)]" />
            <span className="absolute bottom-0 left-0 h-2 w-px bg-white shadow-[0_0_2px_rgba(0,0,0,1)]" />
            <span className="absolute bottom-0 left-1/2 h-1 w-px bg-white shadow-[0_0_2px_rgba(0,0,0,1)]" />
            <span className="absolute bottom-0 right-0 h-2 w-px bg-white shadow-[0_0_2px_rgba(0,0,0,1)]" />
          </div>
        </div>
      )}
    </div>
  );
}
