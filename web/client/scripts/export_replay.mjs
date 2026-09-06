#!/usr/bin/env node
// Headless capture driver for the three.js render lane (Mac, no droplet).
//
// Drives the app's own /render page (app/render/page.tsx — the exact scene the
// live view draws, lib/scene3d.ts) through headless Chrome, seeking replay
// frames deterministically and piping the JPEG captures into ffmpeg:
//
//   node scripts/export_replay.mjs --replay runs/<id>/replay.json \
//     --world-json assets/<world>/web_world.json [--ground <jpg>] \
//     --out-world runs/<id>/three.mp4 --out-fpv runs/<id>/three_fpv.mp4 \
//     [--cams world,fpv] [--fps 24] [--width 1280] [--height 720] [--full]
//
// Reuses a dev server already on :3000 when one answers, else boots its own
// `next dev` on :3199 and tears it down after. Data never touches the network:
// /__render/* and the ground ortho are served from disk via request
// interception, so no FastAPI server is needed either.

import { spawn, spawnSync } from "node:child_process";
import { readFileSync, existsSync } from "node:fs";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import puppeteer from "puppeteer-core";

const CLIENT_DIR = resolve(dirname(fileURLToPath(import.meta.url)), "..");

function arg(name, dflt = null) {
  const i = process.argv.indexOf(`--${name}`);
  if (i < 0) return dflt;
  const v = process.argv[i + 1];
  return v === undefined || v.startsWith("--") ? true : v;
}

const replayPath = arg("replay");
const worldPath = arg("world-json");
const groundPath = arg("ground");
const outWorld = arg("out-world");
const outFpv = arg("out-fpv");
const cams = String(arg("cams", "world,fpv")).split(",").filter(Boolean);
const fps = Number(arg("fps", 24));
const width = Number(arg("width", 1920));
const height = Number(arg("height", 1080));
const full = arg("full", false) === true;
const baseUrl = arg("url");

if (!replayPath || !worldPath) {
  console.error("usage: export_replay.mjs --replay <replay.json> --world-json <web_world.json> ...");
  process.exit(2);
}

const FFMPEG = existsSync("/opt/homebrew/bin/ffmpeg") ? "/opt/homebrew/bin/ffmpeg" : "ffmpeg";
const CHROME = process.env.CHROME_BIN
  ?? "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome";

async function probe(url) {
  try {
    const ac = new AbortController();
    const t = setTimeout(() => ac.abort(), 1500);
    const r = await fetch(`${url}/render`, { signal: ac.signal });
    clearTimeout(t);
    return r.status < 500;
  } catch {
    return false;
  }
}

// NB: a `next dev` server is deliberately NOT reused — React hydration stalls
// on it under headless Chrome (effects never run), so captures hang. The lane
// runs against a production `next start` on :3199, built on first use.
async function ensureServer() {
  for (const u of [baseUrl, "http://127.0.0.1:3199"].filter(Boolean)) {
    if (await probe(u)) return { url: u, child: null };
  }
  if (!existsSync(resolve(CLIENT_DIR, ".next", "BUILD_ID"))) {
    console.log("[export] no production build — running next build (one-time)…");
    const b = spawnSync("npx", ["next", "build"], { cwd: CLIENT_DIR, stdio: "inherit" });
    if (b.status !== 0) throw new Error("next build failed");
  }
  console.log("[export] booting next start on :3199");
  const child = spawn("npx", ["next", "start", "-p", "3199"], {
    cwd: CLIENT_DIR, stdio: "ignore", detached: true,
  });
  const url = "http://127.0.0.1:3199";
  for (let i = 0; i < 60; i++) {
    await new Promise((r) => setTimeout(r, 1000));
    if (await probe(url)) return { url, child };
  }
  throw new Error("next start did not come up on :3199");
}

function ffmpegTo(outPath) {
  const p = spawn(FFMPEG, [
    "-y", "-f", "image2pipe", "-vcodec", "mjpeg", "-framerate", String(fps), "-i", "-",
    "-c:v", "libx264", "-preset", "veryfast", "-crf", "18", "-pix_fmt", "yuv420p",
    "-movflags", "+faststart", outPath,
  ], { stdio: ["pipe", "ignore", "pipe"] });
  let err = "";
  p.stderr.on("data", (d) => { err += d; if (err.length > 8000) err = err.slice(-8000); });
  return { p, errText: () => err };
}

const t0 = Date.now();
const { url, child } = await ensureServer();
console.log(`[export] page server: ${url}`);

const browser = await puppeteer.launch({
  executablePath: CHROME,
  headless: true,
  args: ["--hide-scrollbars", "--mute-audio", "--force-color-profile=srgb",
         `--window-size=${width + 40},${height + 80}`],
});

let exitCode = 0;
try {
  const page = await browser.newPage();
  await page.setViewport({ width: width + 8, height: height + 8, deviceScaleFactor: 1 });
  await page.setRequestInterception(true);
  page.on("request", (req) => {
    const path = new URL(req.url()).pathname;
    try {
      if (path === "/__render/replay.json") {
        return req.respond({ status: 200, contentType: "application/json",
                             body: readFileSync(replayPath) });
      }
      if (path === "/__render/world.json") {
        return req.respond({ status: 200, contentType: "application/json",
                             body: readFileSync(worldPath) });
      }
      if (groundPath && path.startsWith("/site/") && path.endsWith("/ground")) {
        return req.respond({ status: 200, contentType: "image/jpeg",
                             body: readFileSync(groundPath) });
      }
    } catch (e) {
      return req.respond({ status: 500, body: String(e) });
    }
    req.continue();
  });
  page.on("pageerror", (e) => console.error("[export] page error:", e.message));

  for (const cam of cams) {
    const out = cam === "fpv" ? outFpv : outWorld;
    if (!out) continue;
    const camT0 = Date.now();
    await page.goto(
      `${url}/render?cam=${cam}&fps=${fps}&w=${width}&h=${height}${full ? "&full=1" : ""}`,
      { waitUntil: "domcontentloaded", timeout: 120000 });
    await page.waitForFunction("window.__vesper !== undefined",
                               { timeout: 300000, polling: 250 });
    const meta = await page.evaluate(() => window.__vesper.meta);
    const error = await page.evaluate(() => window.__vesper.error);
    if (error) throw new Error(`render page: ${error}`);
    console.log(`[export] ${cam}: ${meta.frames} frames @ ${fps} fps `
      + `(${meta.duration.toFixed(1)}s clip, hero drone ${meta.hero}, `
      + `${meta.kills.length} strike${meta.kills.length === 1 ? "" : "s"}) -> ${out}`);

    // CDP screenshot of the canvas region — ~10x faster than canvas.toDataURL,
    // which was the old capture path (the GPU render itself is ~1 ms/frame)
    const canvas = await page.$("canvas");
    const bb = await canvas.boundingBox();
    const clip = { x: bb.x, y: bb.y, width, height };
    const { p: ff, errText } = ffmpegTo(out);
    for (let i = 0; i < meta.frames; i++) {
      await page.evaluate((n) => window.__vesper.seekOnly(n), i);
      const buf = await page.screenshot({ type: "jpeg", quality: 90, clip,
                                          optimizeForSpeed: true });
      if (!buf || buf.length < 1000) throw new Error(`frame ${i}: capture failed`);
      if (!ff.stdin.write(buf)) await new Promise((r) => ff.stdin.once("drain", r));
      if (i > 0 && i % 200 === 0) {
        const rate = i / ((Date.now() - camT0) / 1000);
        console.log(`[export]   ${i}/${meta.frames} (${rate.toFixed(0)} fps capture)`);
      }
    }
    ff.stdin.end();
    const code = await new Promise((r) => ff.on("close", r));
    if (code !== 0) throw new Error(`ffmpeg failed (${code}): ${errText().slice(-600)}`);
    console.log(`[export] ${cam} done in ${((Date.now() - camT0) / 1000).toFixed(1)}s`);
  }
} catch (e) {
  console.error("[export] FAILED:", e.message ?? e);
  exitCode = 1;
} finally {
  await browser.close().catch(() => {});
  if (child) {
    try { process.kill(-child.pid, "SIGTERM"); } catch { /* already gone */ }
    // stray next start children — best effort
    spawnSync("pkill", ["-f", "next start -p 3199"], { stdio: "ignore" });
  }
}
console.log(`[export] total ${((Date.now() - t0) / 1000).toFixed(1)}s`);
process.exit(exitCode);
