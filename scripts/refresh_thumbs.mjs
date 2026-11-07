// Re-shoot every library tile's thumbnail in the browser studio, from the last frame of its first recorded replay,
// so the tiles match the Sim tab. Raw canvas shots go to OUT; scripts/refresh_thumbs.py crops them to 480x360.
//
//   PW_DIR=~/helloworld/qloo-inspo/node_modules node scripts/refresh_thumbs.mjs http://127.0.0.1:8104 OUT_DIR
import { createRequire } from "node:module";
import { mkdirSync } from "node:fs";

const require = createRequire(`${process.env.PW_DIR || "./node_modules"}/`);
const { chromium } = require("playwright");
const [base, out] = process.argv.slice(2);
mkdirSync(out, { recursive: true });
const browser = await chromium.launch({ args: ["--use-angle=swiftshader", "--enable-unsafe-swiftshader"] });
const page = await browser.newPage({ viewport: { width: 1500, height: 860 } });
await page.goto(`${base}/app`, { waitUntil: "load" });
await page.addStyleTag({ content: "body *{visibility:hidden !important} #three{visibility:visible !important}" });
await page.waitForTimeout(3000);
// every tile on disk has a replay; IDS_FILE (one id per line, from ls skills/tiles) reaches the ones the library list leaves out
import { readFileSync } from "node:fs";
const listed = (await (await fetch(`${base}/api/library`)).json()).tiles.filter((t) => t.replay);
const extra = process.env.IDS_FILE ? readFileSync(process.env.IDS_FILE, "utf8").split("\n").filter(Boolean).map((id) => ({ id })) : [];
const tiles = [...new Map([...listed, ...extra].map((t) => [t.id, t])).values()];
import { existsSync } from "node:fs";
for (const t of tiles) {
  if (existsSync(`${out}/${t.id}.png`)) continue;
  let ok = false;
  try {
  ok = await page.evaluate(async (id) => {
    const r = await fetch(`/api/skills/${id}/replay/0`);
    if (!r.ok) return false;
    const p = await r.json();
    const F = window.forge;
    if (!window.__welcomeOff) { window.__welcomeOff = true; try { F.selectSkill({ id, name: id }); } catch {} } // selecting a skill is what stops the welcome loop
    F.setView("sim");
    // the grasp-to-place stretch of the recorded rollout (55%), not the rest pose
    const frame = p.frames[Math.floor(p.frames.length * 0.55)];
    F.showScene({ ...p.scene, frame });
    for (let i = 0; i < 50 && !F.hasWorld(); i++) await new Promise((r) => setTimeout(r, 100));
    await new Promise((r) => setTimeout(r, 800));
    F.showScene({ ...p.scene, frame });
    F.setCamShift(0, 0.5);
    return true;
  }, t.id);
  if (!ok) continue;
  await page.waitForTimeout(2000);
  await page.evaluate(() => document.getElementById("view-sim").classList.remove("welcoming", "blank"));
  const box = await page.evaluate(() => { const r = document.getElementById("three").getBoundingClientRect(); return { x: r.x, y: r.y, width: r.width, height: r.height }; });
  await page.screenshot({ path: `${out}/${t.id}.png`, clip: box, timeout: 120000 });
  } catch (e) {
    process.stdout.write(`skip ${t.id}: ${String(e).slice(0, 600)}\n`);
    continue;
  }
  process.stdout.write(`${t.id}\n`);
}
await browser.close();
