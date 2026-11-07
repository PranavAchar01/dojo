// Screenshots of the Models panel at 1440 wide, to ~/Pictures/Screenshots/2026-11-07-dojo-assets-*.png.
//   PW_DIR=... node scripts/assets_shots.mjs http://127.0.0.1:8195 grid|detail|import [model_id]
import { createRequire } from "node:module";
import { homedir } from "node:os";
const require = createRequire(`${process.env.PW_DIR || "./node_modules"}/`);
const { chromium } = require("playwright");
const [base, mode, id = "grey_tote"] = process.argv.slice(2);
const out = (n) => `${homedir()}/Pictures/Screenshots/2026-11-07-dojo-assets-${n}.png`;
const browser = await chromium.launch({ args: ["--use-angle=swiftshader", "--enable-unsafe-swiftshader"] });
try {
  const page = await browser.newPage({ viewport: { width: 1440, height: 900 } });
  page.on("pageerror", (e) => console.log("pageerror", String(e).slice(0, 200)));
  await page.goto(`${base}/static/assets.html`, { waitUntil: "networkidle" });
  await page.waitForSelector(".as-card");
  await page.waitForTimeout(800);
  if (mode === "grid") await page.screenshot({ path: out("library-grid") });
  if (mode === "detail") {
    await page.click(`.as-card[data-id="${id}"]`);
    await page.waitForTimeout(2500);
    await page.screenshot({ path: out(`detail-${id}`) });
  }
  if (mode === "import") {
    await page.setInputFiles(".as-file", process.argv[5]);
    await page.waitForSelector(".as-imp");
    await page.fill(".as-name", "Bracket (my import)");
    await page.screenshot({ path: out("import-flow-1") });
    await page.click(".as-go");
    await page.waitForSelector(".as-sheet:not(.as-imp)", { timeout: 120000 });
    await page.waitForTimeout(2500);
    await page.screenshot({ path: out("import-flow-2") });
  }
} finally { await browser.close(); }
