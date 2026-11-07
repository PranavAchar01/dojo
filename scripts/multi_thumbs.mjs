// Shoot the Library thumbnail of every multi-arm card in the browser stage (the Sim tab, the same renderer as every
// other tile): open the card's saved line run, seek to the moment the schedule has the most arms working with all
// their parts in view, and crop the canvas tight around the arms. Each PNG is stamped with the stage tag
// (scripts/stamp.mjs) and written to skills/tiles/<card id>/thumb.png.
//
//   PW_DIR=~/helloworld/qloo-inspo/node_modules node scripts/multi_thumbs.mjs http://127.0.0.1:8111 [card id ...]
import { createRequire } from "node:module";
import { mkdirSync, renameSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import { stampPng } from "./stamp.mjs";

const require = createRequire(`${process.env.PW_DIR || "./node_modules"}/`);
const { chromium } = require("playwright");
const root = join(dirname(fileURLToPath(import.meta.url)), "..");
const [base, ...only] = process.argv.slice(2);
const W = 480, H = 360;

const listed = (await (await fetch(`${base}/api/library`)).json()).tiles.filter((t) => t.origin === "arms");
const cards = only.length ? listed.filter((t) => only.includes(t.id)) : listed;
const browser = await chromium.launch({ args: ["--use-angle=swiftshader", "--enable-unsafe-swiftshader"] });
const page = await browser.newPage({ viewport: { width: 1500, height: 860 } });
await page.goto(`${base}/app`, { waitUntil: "load" });
await page.waitForTimeout(3000);
for (const t of cards) {
  try {
    const at = await page.evaluate(async (c) => {
      const F = window.forge;
      F.selectSkill(c);
      F.setView("sim");
      await window.forgeArms.openLine(c.line, c.name);
      await new Promise((r) => setTimeout(r, 2500));
      const [sec, n] = window.forgeArms.busiest();
      window.forgeArms.seek(sec);
      return { sec, n };
    }, t);
    await page.waitForTimeout(1500);
    await page.evaluate(() => document.getElementById("view-sim").classList.remove("welcoming", "blank", "noplay"));
    const box = await page.evaluate(() => { const r = document.getElementById("three").getBoundingClientRect(); return { x: r.x, y: r.y, width: r.width, height: r.height }; });
    const dir = join(root, "skills", "tiles", t.id);
    mkdirSync(dir, { recursive: true });
    const raw = join(dir, "raw.png");
    await page.screenshot({ path: raw, clip: box, timeout: 120000 });
    process.stdout.write(`${t.id} t=${at.sec}s arms working=${at.n}/${t.arms.length} ${box.width}x${box.height}\n`);
  } catch (e) {
    process.stdout.write(`skip ${t.id}: ${String(e).slice(0, 400)}\n`);
  }
}
await browser.close();
// scripts/multi_thumbs.py crops the raw shots to tile size and stamps them
