// Shoot trained-model tile thumbnails on the shared stage. scripts/tile_thumbs.py writes one job per tile (the engine's
// recorded scene + one frame of body poses) and runs this. Headless Chromium on localhost, one page, closed at the end.
//
//   PW_DIR=... node scripts/tile_thumbs.mjs JOBS_DIR
import { createRequire } from "node:module";
import { spawn } from "node:child_process";
import { readFileSync, readdirSync, writeFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import { stampPng } from "./stamp.mjs";

const require = createRequire(`${process.env.PW_DIR || "./node_modules"}/`);
const { chromium } = require("playwright");
const root = join(dirname(fileURLToPath(import.meta.url)), "..");
const jobs = process.argv[2];
const port = 8290 + Math.floor(Math.random() * 90);
const server = spawn("python3", ["-I", "-m", "http.server", String(port), "--bind", "127.0.0.1"], { cwd: root, stdio: "ignore" });
let browser;
try {
  await new Promise((r) => setTimeout(r, 800));
  browser = await chromium.launch({ args: ["--use-angle=swiftshader", "--enable-unsafe-swiftshader"] });
  const page = await browser.newPage({ viewport: { width: 480, height: 360 } });
  await page.goto(`http://127.0.0.1:${port}/web/tile-thumb.html`);
  await page.waitForFunction(() => window.ready === true, null, { timeout: 60000 });
  const meshes = {};
  for (const f of readdirSync(jobs).filter((n) => n.startsWith("meshes-"))) meshes[f.slice(7, -5)] = JSON.parse(readFileSync(join(jobs, f)));
  for (const f of readdirSync(jobs).filter((n) => n.endsWith(".json") && !n.startsWith("meshes-"))) {
    const j = JSON.parse(readFileSync(join(jobs, f)));
    try {
      const info = j.cells
        ? await page.evaluate((a) => window.shootCells(a), { pitch: j.pitch, cells: j.cells.map((c) => ({ ...c, meshes: meshes[c.robot] })) })
        : await page.evaluate((a) => window.shoot(a), { scene: j.scene, frame: j.frame, robot: j.robot, meshes: meshes[j.robot] });
      const png = await page.evaluate(() => { window.draw(); return document.getElementById("c").toDataURL("image/png"); });
      writeFileSync(j.out, Buffer.from(png.split(",")[1], "base64"));
      stampPng(j.out);
      process.stdout.write(`${j.id}\n`);
    } catch (e) {
      process.stdout.write(`skip ${j.id}: ${String(e).slice(0, 120)}\n`);
    }
  }
} finally {
  if (browser) await browser.close();
  server.kill();
}
