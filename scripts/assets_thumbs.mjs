// Render a thumbnail for every library model (and user import) that lacks one: assets/library/<id>/thumb.png.
// Headless Chromium on localhost only, one page, closed at the end. The model server is a throwaway python http.server
// on its own PID, serving the repo root.
//
//   PW_DIR=~/helloworld/qloo-inspo/node_modules node scripts/assets_thumbs.mjs [--force] [ids...]
import { createRequire } from "node:module";
import { spawn } from "node:child_process";
import { readFileSync, existsSync, readdirSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

const require = createRequire(`${process.env.PW_DIR || "./node_modules"}/`);
const { chromium } = require("playwright");
const root = join(dirname(fileURLToPath(import.meta.url)), "..");
const args = process.argv.slice(2);
const force = args.includes("--force");
const only = args.filter((a) => !a.startsWith("--"));
const fileArg = args.find((a) => a.startsWith("--file="))?.slice(7); // --file=assets/_work/x.glb --out=/tmp/x.png
const outArg = args.find((a) => a.startsWith("--out="))?.slice(6);
const port = 8190 + Math.floor(Math.random() * 90);

const server = spawn("python3", ["-I", "-m", "http.server", String(port), "--bind", "127.0.0.1"], { cwd: root, stdio: "ignore" });
const dirs = [join(root, "assets/library"), join(root, "assets/user")];
const jobs = [];
if (fileArg) jobs.push({ id: fileArg, dir: null, url: `/${fileArg}`, out: outArg });
for (const d of fileArg ? [] : dirs) {
  if (!existsSync(d)) continue;
  for (const id of readdirSync(d)) {
    const glb = join(d, id, "model.glb");
    if (!existsSync(glb)) continue;
    if (only.length && !only.includes(id)) continue;
    if (!force && existsSync(join(d, id, "thumb.png"))) continue;
    jobs.push({ id, dir: join(d, id), url: `/${d.slice(root.length + 1)}/${id}/model.glb` });
  }
}
let browser;
try {
  await new Promise((r) => setTimeout(r, 800));
  browser = await chromium.launch({ args: ["--use-angle=swiftshader", "--enable-unsafe-swiftshader"] });
  const page = await browser.newPage({ viewport: { width: 480, height: 360 } });
  await page.goto(`http://127.0.0.1:${port}/web/assets-thumb.html`);
  await page.waitForFunction(() => window.ready === true, null, { timeout: 30000 });
  for (const j of jobs) {
    try {
      const info = await page.evaluate((u) => window.shoot(u), j.url);
      await page.locator("canvas").screenshot({ path: j.out || join(j.dir, "thumb.png") });
      process.stdout.write(`${j.id} ${info.size.map((v) => v.toFixed(2)).join("x")}\n`);
    } catch (e) {
      process.stdout.write(`skip ${j.id}: ${String(e).slice(0, 100)}\n`);
    }
  }
} finally {
  if (browser) await browser.close();
  server.kill();
}
