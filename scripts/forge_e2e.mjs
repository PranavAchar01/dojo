// End-to-end check of the merged forge (Sim, RSI, Library, and the robot builder inside the prompt window) against a running server, in a real browser.
//
//   PW_DIR=~/helloworld/qloo-inspo/node_modules node scripts/forge_e2e.mjs http://localhost:8100
//
// Robot builder (the chip in the prompt bar): a WidowX fails the finger check with its reason but can still be made active (reach and push fit);
// the router then refuses a pick on it with the robot check's reason. A Koch cannot be made active at all.
// Back on the SO-101: an out-of-reach prompt is refused, a short real skill trains and lands as a tile.
// Screenshots: SHOTS (default ~/Pictures/Screenshots) as <date>-vla-forge-merged-*.png, 1920x934 and 400 wide.
import { createRequire } from "node:module";
import { homedir } from "node:os";

const require = createRequire(`${process.env.PW_DIR || "./node_modules"}/`);
const { chromium } = require("playwright");

const [base = "http://localhost:8100"] = process.argv.slice(2);
const SKILL = process.env.SKILL_PROMPT || "touch the blue block";
const ONLY = process.env.ONLY || "all"; // all | look (no training)
const shots = process.env.SHOTS || `${homedir()}/Pictures/Screenshots`;
const day = new Date().toISOString().slice(0, 10);
const log = (...a) => process.stdout.write(`[${new Date().toISOString().slice(11, 19)}] ${a.join(" ")}\n`);

const browser = await chromium.launch({ args: ["--use-angle=swiftshader", "--enable-unsafe-swiftshader"] });
const errors = [];
async function open(width, height) {
  const page = await browser.newPage({ viewport: { width, height }, deviceScaleFactor: 1 });
  page.on("pageerror", (e) => errors.push(String(e)));
  page.on("console", (m) => m.type() === "error" && errors.push(`console: ${m.text()}`));
  await page.goto(base, { waitUntil: "domcontentloaded", timeout: 120_000 });
  await page.waitForFunction(() => document.querySelector("#ready-text")?.textContent === "agents online", null, { timeout: 180_000 });
  return page;
}
const shot = async (page, name) => {
  const path = `${shots}/${day}-vla-forge-merged-${name}.png`;
  await page.screenshot({ path });
  log("shot", path);
};
const view = (page, v) => page.click(`.seg button[data-view="${v}"]`);
async function prompt(page, text) {
  await page.waitForSelector("#go:not([disabled])", { timeout: 120_000 });
  await page.fill("#prompt", text);
  await page.click("#go");
}
async function openRobot(page) {
  await page.click("#robot-open");
  await page.waitForSelector("#nm-card[data-pane=robot]");
}
async function pickArm(page, id) {
  await page.click(`.part[data-slot="arm"][data-id="${id}"]`);
  await page.waitForFunction(() => /passed$/.test(document.querySelector("#asm-sum")?.textContent || ""), null, { timeout: 120_000 });
  await page.waitForTimeout(1500);
}
async function useRobot(page) {
  await page.waitForSelector("#use-robot:not([disabled])");
  await page.click("#use-robot");
  await page.waitForFunction(() => document.querySelector("#robot-chip")?.textContent.length > 0 && !document.querySelector("#use-robot")?.offsetParent);
}
const refusals = (page) => page.locator(".route-card.is-refused").count();

const page = await open(1920, 934);
log("agents online");
await page.waitForTimeout(2500);
await shot(page, "01-sim");
await view(page, "rsi");
await page.waitForTimeout(1200);
await shot(page, "02-rsi");
await view(page, "skills");
await page.waitForSelector(".tile:not(.is-loading)");
await page.waitForTimeout(1200);
await shot(page, "03-library");
const robots = await page.locator(".tile-robot").allTextContents();
log("tile robots:", [...new Set(robots)].join(" | "));

await openRobot(page);
await page.waitForSelector(".part");
await page.waitForFunction(() => /passed$/.test(document.querySelector("#asm-sum")?.textContent || ""), null, { timeout: 120_000 });
await page.waitForFunction(() => [...document.querySelectorAll(".part-fig img")].every((i) => i.classList.contains("in")), null, { timeout: 180_000 });
await shot(page, "04-robot-so101");

await pickArm(page, "koch");
const kochFail = await page.locator("#asm-fail").textContent();
log("koch:", kochFail);
if (!(await page.locator("#use-robot").isDisabled())) throw new Error("Koch should not be usable");
await shot(page, "05-robot-koch-blocked");

await pickArm(page, "vx300s");
await page.waitForFunction(() => [...document.querySelectorAll(".part-fig img")].every((i) => i.classList.contains("in")), null, { timeout: 180_000 });
const wristBy = await page.locator('.part[data-slot="wrist"] .part-by').first().textContent();
log("wrist tile:", wristBy);
if (!/ViperX/.test(wristBy)) throw new Error(`wrist tile does not follow the arm: ${wristBy}`);
await shot(page, "06-robot-viperx");
await page.locator('.part[data-slot="wrist"]').first().scrollIntoViewIfNeeded();
await page.waitForTimeout(500);
await shot(page, "06b-robot-viperx-wrist");

await pickArm(page, "wx250s");
const wxFail = await page.locator("#asm-fail").textContent();
log("widowx:", wxFail, "|", await page.locator("#use-note").textContent());
if (!/25 mm/.test(wxFail)) throw new Error(`WidowX finger check reason missing: ${wxFail}`);
await shot(page, "07-robot-widowx-check");
await useRobot(page);
await view(page, "sim");
await page.waitForTimeout(3000);
await shot(page, "08-sim-widowx");
const n0 = await refusals(page);
await prompt(page, "pick up the red block");
await page.waitForFunction((n) => document.querySelectorAll(".route-card.is-refused").length > n, n0, { timeout: 120_000 });
const why = await page.locator(".route-card.is-refused .rc-why").last().textContent();
log("robot refusal:", why);
if (!/38 mm/.test(why)) throw new Error(`robot refusal without the finger reason: ${why}`);
await page.waitForTimeout(800);
await shot(page, "09-refused-robot");

await openRobot(page);
await pickArm(page, "so101");
await useRobot(page);
await view(page, "sim");
await page.waitForTimeout(2500);
const n1 = await refusals(page);
await prompt(page, "lift the red block 60 cm up");
await page.waitForFunction((n) => document.querySelectorAll(".route-card.is-refused").length > n, n1, { timeout: 120_000 });
log("prompt refusal:", await page.locator(".route-card.is-refused .rc-why").last().textContent());
await page.waitForTimeout(800);
await shot(page, "10-refused-prompt");

if (ONLY !== "look") {
  await view(page, "skills");
  const before = await page.locator(".tile:not(.is-loading)").count();
  await view(page, "sim");
  await prompt(page, SKILL);
  await page.waitForSelector(".route-card.d-train", { timeout: 120_000 });
  log("routed:", await page.locator(".route-card.d-train .rc-why").last().textContent());
  await page.waitForFunction(() => document.querySelector(".progress-card.is-loading .pc-label")?.textContent.startsWith("Training"), null, { timeout: 300_000 });
  await page.waitForTimeout(6000);
  await shot(page, "11-training");
  await page.waitForSelector(".lib-toast.is-saved", { timeout: 900_000 });
  await page.waitForTimeout(1500);
  await shot(page, "12-saved");
  await view(page, "skills");
  await page.evaluate(() => document.querySelector("#view-skills").scrollTo(0, 0));
  await page.waitForTimeout(1000);
  await shot(page, "13-new-tile");
  const after = await page.locator(".tile:not(.is-loading)").count();
  log("tiles", before, "->", after);
  if (after !== before + 1) throw new Error(`expected one new tile, had ${before} then ${after}`);
}

await page.close();
const phone = await open(400, 860);
await phone.waitForTimeout(2500);
await shot(phone, "14-phone-sim");
await view(phone, "skills");
await phone.waitForSelector(".tile:not(.is-loading)");
await phone.waitForTimeout(1000);
await shot(phone, "15-phone-library");
await openRobot(phone);
await phone.waitForFunction(() => /passed$/.test(document.querySelector("#asm-sum")?.textContent || ""), null, { timeout: 120_000 });
await phone.waitForTimeout(2500);
await shot(phone, "16-phone-robot");
const wide = await phone.evaluate(() => document.documentElement.scrollWidth);
log("phone scrollWidth", wide);

log("page errors:", errors.length ? errors.join(" | ") : "none");
await browser.close();
if (errors.length) process.exit(1);
