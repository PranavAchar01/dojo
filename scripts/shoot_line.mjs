// Screenshot the Line tab with a finished line playing, at desktop and phone size.
//   PW_DIR=... node scripts/shoot_line.mjs http://127.0.0.1:8104 TAG [seconds]
import { createRequire } from "node:module";
import { homedir } from "node:os";
const require = createRequire(`${process.env.PW_DIR || "./node_modules"}/`);
const { chromium } = require("playwright");
const [base, tag = "line", wait = "9"] = process.argv.slice(2);
const day = new Date().toISOString().slice(0, 10);
const browser = await chromium.launch({ args: ["--use-angle=swiftshader", "--enable-unsafe-swiftshader"] });
const errors = [];
for (const [size, [w, h]] of Object.entries({ desktop: [1920, 934], phone: [400, 860] })) {
  const page = await browser.newPage({ viewport: { width: w, height: h } });
  page.on("pageerror", (e) => errors.push(`${size}: ${e}`));
  await page.goto(base, { waitUntil: "load" });
  await page.waitForTimeout(2500);
  await page.locator('[data-view="line"]:visible').first().click();
  await page.waitForTimeout(Number(wait) * 1000);
  await page.evaluate(() => document.querySelector(".ln-3d")?.scrollIntoView({ block: "center" }));
  await page.waitForTimeout(1500);
  const path = `${homedir()}/Pictures/Screenshots/${day}-vla-forge-${tag}-${size}-line.png`;
  await page.screenshot({ path });
  console.log(path);
}
console.log(errors.length ? errors.join("\n") : "no page errors");
await browser.close();
