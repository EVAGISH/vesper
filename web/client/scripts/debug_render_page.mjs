// Diagnostic: load /render headless, dump console + errors + status, screenshot.
import puppeteer from "puppeteer-core";
import { readFileSync } from "node:fs";

const [replayPath, worldPath, groundPath, shot] = process.argv.slice(2);
const CHROME = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome";

const browser = await puppeteer.launch({
  executablePath: CHROME, headless: true,
  args: ["--hide-scrollbars", "--mute-audio"],
});
const page = await browser.newPage();
await page.setViewport({ width: 1300, height: 760 });
page.on("console", (m) => console.log("[console]", m.type(), m.text()));
page.on("pageerror", (e) => console.log("[pageerror]", e.message));
page.on("requestfailed", (r) => console.log("[reqfail]", r.url(), r.failure()?.errorText));
await page.setRequestInterception(true);
page.on("request", (req) => {
  let path = "";
  try { path = new URL(req.url()).pathname; } catch { return req.continue(); }
  try {
    if (path === "/__render/replay.json")
      return req.respond({ status: 200, contentType: "application/json", body: readFileSync(replayPath) });
    if (path === "/__render/world.json")
      return req.respond({ status: 200, contentType: "application/json", body: readFileSync(worldPath) });
    if (groundPath && path.startsWith("/site/") && path.endsWith("/ground"))
      return req.respond({ status: 200, contentType: "image/jpeg", body: readFileSync(groundPath) });
  } catch (e) {
    return req.respond({ status: 500, body: String(e) });
  }
  req.continue();
});
await page.goto("http://127.0.0.1:3000/render?cam=world&fps=24&w=1280&h=720",
                { waitUntil: "domcontentloaded", timeout: 120000 });
for (let i = 0; i < 24; i++) {
  await new Promise((r) => setTimeout(r, 2500));
  const v = await page.evaluate(() => ({
    vesper: typeof window.__vesper,
    err: window.__vesper?.error ?? null,
    status: document.body.innerText.slice(0, 200),
  }));
  console.log(`[probe ${i}]`, JSON.stringify(v));
  if (v.vesper !== "undefined") break;
}
if (shot) await page.screenshot({ path: shot });
await browser.close();
