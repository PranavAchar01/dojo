// End-to-end check of the skill library against a running server, in a real browser.
//
//   PW_DIR=~/helloworld/qloo-inspo/node_modules node scripts/library_e2e.mjs train   http://localhost:8105
//   (restart the server)
//   PW_DIR=~/helloworld/qloo-inspo/node_modules node scripts/library_e2e.mjs persist http://localhost:8105
//
// train:   an infeasible prompt is refused with a reason, then a short real skill trains and lands as a tile.
// persist: after a restart the tile is still there; Test runs it on fresh scenes; Remix trains from it.
// Screenshots go to SHOTS (default ~/Pictures/Screenshots) as <date>-vla-forge-library-*.png.
import { createRequire } from "node:module";
import { homedir } from "node:os";

const require = createRequire(`${process.env.PW_DIR || "./node_modules"}/`);
const { chromium } = require("playwright");

const [phase = "train", base = "http://localhost:8105"] = process.argv.slice(2);
const SKILL = process.env.SKILL_PROMPT || "touch the purple block";
const REMIX = process.env.REMIX_PROMPT || "touch the orange block";
const shots = process.env.SHOTS || `${homedir()}/Pictures/Screenshots`;
const day = new Date().toISOString().slice(0, 10);
const log = (...a) => process.stdout.write(`[${new Date().toISOString().slice(11, 19)}] ${a.join(" ")}\n`);

const browser = await chromium.launch({ args: ["--use-angle=swiftshader", "--enable-unsafe-swiftshader"] });
const page = await browser.newPage({ viewport: { width: 1600, height: 1000 }, deviceScaleFactor: 1 });
const errors = [];
page.on("pageerror", (e) => errors.push(String(e)));
const shot = async (name) => {
  const path = `${shots}/${day}-vla-forge-library-${name}.png`;
  await page.screenshot({ path });
  log("shot", path);
};
const view = (v) => page.click(`.seg button[data-view="${v}"]`);
async function prompt(text) {
  await page.waitForSelector("#go:not([disabled])", { timeout: 120_000 });
  await page.fill("#prompt", text);
  await page.click("#go");
}
const tileFor = (text) => page.locator(".tile:not(.is-loading)", { has: page.locator(".tile-name, .tile-ins", { hasText: text }) }).first();

await page.goto(base);
await page.waitForFunction(() => document.querySelector("#ready-text")?.textContent === "agents online", null, { timeout: 180_000 });
log("agents online");

if (phase === "train") {
  await view("skills");
  await page.waitForSelector(".tile:not(.is-loading)");
  await shot("01-library");
  const before = await page.locator(".tile:not(.is-loading)").count();
  log("tiles before", before);

  await view("sim");
  await prompt("lift the red block 60 cm up");
  await page.waitForSelector(".route-card.is-refused", { timeout: 120_000 });
  const why = await page.locator(".route-card.is-refused .rc-why").last().textContent();
  log("refused:", why);
  if (!/out of reach/.test(why)) throw new Error(`refusal without a reach reason: ${why}`);
  await page.waitForSelector(".progress-card.is-failed");
  await shot("02-refused-height");

  await prompt("fold the laundry and put it in the drawer");
  await page.waitForFunction(() => document.querySelectorAll(".route-card.is-refused").length >= 2, null, { timeout: 120_000 });
  log("refused:", await page.locator(".route-card.is-refused .rc-why").last().textContent());
  await shot("03-refused-objects");

  await prompt(SKILL);
  await page.waitForSelector(".route-card.d-train", { timeout: 120_000 });
  log("routed:", await page.locator(".route-card.d-train .rc-why").last().textContent());
  await page.waitForFunction(() => document.querySelector(".progress-card.is-loading .pc-label")?.textContent.startsWith("Training"), null, { timeout: 300_000 });
  await page.waitForTimeout(8000);
  await shot("04-training");
  await view("skills");
  await page.waitForSelector(".tile.is-loading.run");
  await shot("05-skeleton-tile");
  await view("sim");
  await page.waitForSelector(".lib-toast.is-saved", { timeout: 900_000 });
  await page.waitForSelector(".tile.is-saved", { state: "attached", timeout: 3000 });
  await page.waitForTimeout(700);
  await shot("06-saved-toast");
  await view("skills");
  await shot("07-saved-tile");
  const after = await page.locator(".tile:not(.is-loading)").count();
  log("tiles after", after);
  if (after !== before + 1) throw new Error(`expected one new tile, had ${before} then ${after}`);
  const t = tileFor(SKILL);
  if (!(await t.locator("img").count())) throw new Error("new tile has no thumbnail");
  log("new tile:", (await t.locator(".tile-rate").textContent()).trim(), "|", (await t.locator(".tile-meta").textContent()).trim());
}

if (phase === "persist") {
  await view("skills");
  await page.waitForSelector(".tile:not(.is-loading)");
  const t = tileFor(SKILL);
  await t.waitFor();
  if (!(await t.locator("img").count())) throw new Error("tile lost its thumbnail across the restart");
  await page.waitForTimeout(1000);
  await shot("08-after-restart");

  await t.locator("button", { hasText: "Replay" }).click();
  await page.waitForFunction(() => document.querySelector("#round-label")?.textContent.startsWith("Replay"));
  await page.waitForTimeout(2500);
  await shot("09-replay");

  await view("skills");
  await tileFor(SKILL).locator("button", { hasText: /^Test$/ }).click();
  await page.waitForFunction(() => document.querySelector("#round-label")?.textContent.startsWith("Test"), null, { timeout: 60_000 });
  await page.waitForTimeout(1500);
  await shot("10-test-running");
  await page.waitForFunction(() => /^Test · \d+\/\d+ fresh/.test(document.querySelector("#round-label")?.textContent || ""), null, { timeout: 600_000 });
  await page.waitForTimeout(2500);
  log("test:", await page.locator("#round-label").textContent());
  await shot("11-test-done");
  await view("skills");
  const dots = await tileFor(SKILL).locator(".tdot").allTextContents();
  log("tile tests:", dots.join(" "));
  if (!dots.length) throw new Error("test result not on the tile");

  // the same prompt again: reuse if the tile is good enough, otherwise a retrain
  await view("sim");
  await prompt(SKILL);
  await page.waitForFunction(() => document.querySelectorAll(".route-card").length >= 1, null, { timeout: 120_000 });
  const dec = await page.locator(".route-card .rc-h b").last().textContent();
  log("same prompt again ->", dec);
  await shot("12-same-prompt");
  if (dec === "Reuse a tile") {
    await page.waitForSelector(".progress-card.is-saved");
  } else {
    await page.waitForSelector(".lib-toast.is-saved", { timeout: 900_000 });
  }

  await view("skills");
  await tileFor(SKILL).locator("button", { hasText: "Remix" }).click();
  await page.waitForSelector("#lib-mode:not(.hidden)");
  await shot("13-remix-mode");
  await prompt(REMIX);
  await page.waitForFunction(() => [...document.querySelectorAll(".route-card .rc-why")].some((e) => e.textContent.startsWith("remix")), null, { timeout: 120_000 });
  log("remix routed:", await page.locator(".route-card .rc-why").last().textContent());
  await page.waitForSelector(".lib-toast.is-saved", { timeout: 900_000 });
  await page.waitForSelector(".tile.is-saved", { state: "attached", timeout: 3000 });
  await view("sim");
  await page.waitForTimeout(700);
  await shot("14-remix-saved");
  await view("skills");
  await page.evaluate(() => document.querySelector("#view-skills").scrollTo(0, 0));
  const r = tileFor(REMIX);
  await r.waitFor();
  log("remix tile:", (await r.locator(".tile-meta").textContent()).trim());
  if (!/remix/.test(await r.locator(".tile-meta").textContent())) throw new Error("remix tile not marked as a remix");
  await page.click('#lib-filter button[data-f="mine"]');
  await shot("15-filter-remixed");
  await page.click('#lib-filter button[data-f="all"]');
  await page.fill("#lib-q", "purple");
  await shot("16-search");
}

log("page errors:", errors.length ? errors.join(" | ") : "none");
await browser.close();
if (errors.length) process.exit(1);
