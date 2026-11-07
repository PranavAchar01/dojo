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
const page = await browser.newPage({ viewport: { width: 1600, height: 1000 } });
await page.goto(base, { waitUntil: "load" });
await page.waitForTimeout(3000);
const tiles = (await (await fetch(`${base}/api/library`)).json()).tiles.filter((t) => t.replay);
for (const t of tiles) {
  const ok = await page.evaluate(async (id) => {
    const r = await fetch(`/api/skills/${id}/replay/0`);
    if (!r.ok) return false;
    const p = await r.json();
    const F = window.forge;
    F.setView("sim");
    await F.buildWorld(p.scene);
    F.applyFrame(p.frames[p.frames.length - 1]);
    F.setBadge("", true);
    document.getElementById("vp-badge").classList.add("hidden");
    return true;
  }, t.id);
  if (!ok) continue;
  await page.waitForTimeout(2500);
  await page.locator("#three").screenshot({ path: `${out}/${t.id}.png` });
  process.stdout.write(`${t.id}\n`);
}
await browser.close();
