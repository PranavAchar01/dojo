// Visit every tab of a running VLA Forge at desktop (1920x934) and phone (400x860) size, screenshot each one and
// fail on any page error.
//
//   PW_DIR=~/helloworld/qloo-inspo/node_modules node scripts/ui_tabs.mjs http://127.0.0.1:8100 v5
//
// Screenshots go to SHOTS (default ~/Pictures/Screenshots) as <date>-vla-forge-<tag>-<size>-<tab>.png.
import { createRequire } from "node:module";
import { homedir } from "node:os";

const require = createRequire(`${process.env.PW_DIR || "./node_modules"}/`);
const { chromium } = require("playwright");

const [base = "http://127.0.0.1:8100", tag = "ui"] = process.argv.slice(2);
const shots = process.env.SHOTS || `${homedir()}/Pictures/Screenshots`;
const day = new Date().toISOString().slice(0, 10);
const TABS = ["sim", "line", "skills", "rsi"];
const SIZES = { desktop: [1920, 934], phone: [400, 860] };

const browser = await chromium.launch({ args: ["--use-angle=swiftshader", "--enable-unsafe-swiftshader"] });
const errors = [];
for (const [size, [width, height]] of Object.entries(SIZES)) {
  const page = await browser.newPage({ viewport: { width, height } });
  page.on("pageerror", (e) => errors.push(`${size}: ${e}`));
  await page.goto(base, { waitUntil: "load" });
  await page.waitForTimeout(3500);
  for (const tab of TABS) {
    const btn = page.locator(`[data-view="${tab}"]:visible`).first();
    await btn.click();
    // software WebGL can hold the main thread for a while; wait until the view has actually switched
    await page.waitForFunction((t) => document.getElementById("app").dataset.view === t, tab, { timeout: 60000 });
    await page.waitForTimeout(tab === "line" ? 3500 : 2500);
    // let finite entrance animations finish and two frames paint, so the shot is the settled page
    await page.evaluate(async () => {
      const once = document.getAnimations().filter((a) => a.effect?.getTiming().iterations !== Infinity);
      await Promise.race([Promise.allSettled(once.map((a) => a.finished)), new Promise((r) => setTimeout(r, 8000))]);
      await new Promise((r) => requestAnimationFrame(() => requestAnimationFrame(r)));
    });
    const path = `${shots}/${day}-vla-forge-${tag}-${size}-${tab}.png`;
    await page.screenshot({ path });
    process.stdout.write(`${path}\n`);
  }
  await page.close();
}
await browser.close();
if (errors.length) {
  process.stdout.write(`page errors:\n${errors.join("\n")}\n`);
  process.exit(1);
}
process.stdout.write("no page errors\n");
