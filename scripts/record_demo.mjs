// Live demo recording: a real, visible Chrome window; the page records itself with tab capture +
// MediaRecorder (?capture=1) while every step runs for real. Nothing is frame-stepped.
// While the forge waits on the machine (collecting demos, training, evaluating, a tile's test run), the page
// shows a "4x" badge and this script logs the span, so the edit speeds up exactly those waits and nothing else.
//
//   PPTR_DIR=~/helloworld/hyperframes-trial/node_modules node scripts/record_demo.mjs
//
// The tour (v4, the factory line): the saved line replaying, a new line described in words and planned into
// stations, the VLA stations trained for real (sped up) and the whole line run, its replay and measured
// numbers, the new station skills in the Library. Every segment start is written to <webm>.marks.json under
// "at" for the narration cues. The v3 tour (prompts, Robot tab, Ladder) is in git history.
import { writeFileSync } from "node:fs";
import { createRequire } from "node:module";

const require = createRequire(`${process.env.PPTR_DIR || "./node_modules"}/`);
const puppeteer = require("puppeteer-core");

const LINE =
  process.env.LINE ||
  "Kitting line: yellow parts go to the yellow tote, then green parts to the green tote, and a scripted ViperX at the end of the line catches everything else in the gray tote.";
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

const browser = await puppeteer.launch({
  executablePath: "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
  headless: false,
  defaultViewport: null,
  userDataDir: process.env.REC_PROFILE || "/tmp/vla-forge-rec-profile",
  args: [
    "--window-size=1920,1200",
    "--window-position=0,0",
    "--app=http://127.0.0.1:8100/?capture=1",
    "--auto-accept-this-tab-capture",
    "--autoplay-policy=no-user-gesture-required",
    "--disable-background-timer-throttling",
    "--disable-renderer-backgrounding",
    "--disable-backgrounding-occluded-windows",
    "--no-first-run",
    "--no-default-browser-check",
  ],
});
const [page] = await browser.pages();
await page.waitForFunction(() => document.querySelector("#ready-dot")?.classList.contains("on") && window.startCapture, { timeout: 120000 });
// start on the stock arm, whatever the last session left active
await page.evaluate(() => fetch("/api/robot/use", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ robot: "so101" }) }));
await page.evaluate(() => { window.setView("line"); window.scrollTo(0, 0); });
await sleep(6000); // the saved line loads and starts its replay
await page.evaluate(() => window.startCapture());
const now = () => page.evaluate(() => (performance.now() - window.captureT0) / 1000);
const marks = { speed: [], prompts: [], at: {} };
const mark = async (name) => {
  marks.at[name] = await now();
  console.log(`${name} @ ${marks.at[name].toFixed(1)}s`);
};
const mood = () => page.evaluate(() => document.querySelector("#view-line .ln-head .mascot")?.dataset.mood || "");
const scrollLine = (sel, block = "start") =>
  page.evaluate((s, b) => document.querySelector(s)?.scrollIntoView({ block: b, behavior: "smooth" }), sel, block);
// run `until` while speeding up whatever `isWait` calls a wait (the page shows the 4x badge)
async function sped(until, isWait) {
  let fast = null;
  while (!(await until())) {
    const w = await isWait();
    if (w && fast === null) {
      await page.evaluate(() => window.setSpeed(true));
      fast = await now();
    } else if (!w && fast !== null) {
      await page.evaluate(() => window.setSpeed(false));
      marks.speed.push([fast, await now()]);
      fast = null;
    }
    await sleep(150);
  }
  if (fast !== null) {
    await page.evaluate(() => window.setSpeed(false));
    marks.speed.push([fast, await now()]);
  }
}

// 0. the line built earlier, replaying
await mark("intro");
await sleep(9000);

// 1. a company describes a new line in words
await mark("describe");
await page.evaluate(() => document.querySelector("#view-line").scrollTo({ top: 0, behavior: "smooth" }));
await page.click("#view-line .ln-ask textarea", { clickCount: 3 });
await page.type("#view-line .ln-ask textarea", LINE, { delay: 28 });
await sleep(600);
const lines0 = await page.evaluate(() => document.querySelectorAll("#view-line .ln-pick option").length);
await page.click("#view-line .ln-plan");
await mark("plan");
await page.waitForFunction((n) => document.querySelectorAll("#view-line .ln-pick option").length > n, { timeout: 120000, polling: 100 }, lines0);
await page.waitForFunction(() => document.querySelectorAll("#view-line .ln-card").length > 0);
await sleep(1500);
await mark("cards");
await scrollLine("#view-line .ln-actions", "start");
await sleep(7000);

// 2. train the stations that need a skill, then run the whole line
await mark("build");
await page.click("#view-line .ln-build");
await sleep(1500);
const done = async () => ["ok", "fail"].includes(await mood()) && !(await page.evaluate(() => document.querySelector("#view-line .ln-build").disabled));
const busy = async () => (await mood()) === "think";
let ran = false;
await sped(done, async () => {
  const running = await page.evaluate(() => /^Running/.test(document.querySelector("#view-line .ln-note").textContent));
  if (running && !ran) { ran = true; marks.at.running = await now(); await scrollLine("#view-line .ln-stage", "center"); }
  return busy();
});
await mark("ran");
console.log("note:", await page.evaluate(() => document.querySelector("#view-line .ln-note").textContent));

// 3. the run, replayed, then the measured numbers
await scrollLine("#view-line .ln-stage", "start");
await sleep(1200);
await mark("replay");
await sleep(26000);
await scrollLine("#view-line .ln-cards", "start");
await mark("results");
await sleep(9000);
console.log("result:", await page.evaluate(() => document.querySelector("#view-line .ln-result").innerText.replace(/\s+/g, " ")));

// 4. each station's skill is a tile in the library
await view("skills");
await page.evaluate(() => document.querySelector("#view-skills").scrollTo({ top: 0 }));
await mark("library");
await sleep(7000);

// 5. back to the line
await view("line");
await page.evaluate(() => document.querySelector("#view-line").scrollTo({ top: 0, behavior: "smooth" }));
await mark("close");
await sleep(9000);
marks.end = await now();
const res = await page.evaluate(() => window.stopCapture());
writeFileSync(res.path.replace(/\.webm$/, ".marks.json"), JSON.stringify(marks, null, 1));
console.log(JSON.stringify({ ...res, marks }));
await browser.close();
