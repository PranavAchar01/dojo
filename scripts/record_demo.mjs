// Live demo recording: a real, visible Chrome window; the page records itself with tab capture +
// MediaRecorder (?capture=1) while every step runs for real. Nothing is frame-stepped.
// While the forge waits on the machine (collecting demos, training, evaluating, a tile's test run), the page
// shows a "4x" badge and this script logs the span, so the edit speeds up exactly those waits and nothing else.
//
//   PPTR_DIR=~/helloworld/hyperframes-trial/node_modules node scripts/record_demo.mjs
//
// The tour: a simple prompt trains, a two-step prompt reuses what the library has, the router refuses an
// out-of-reach prompt, the Library (tiles, Remix), the Robot tab (swap arm and wrist, a check failing with
// its reason, a robot the router refuses a pick on), a skill trained on the ViperX tested on fresh scenes,
// and the Ladder. Every segment start is written to <webm>.marks.json under "at" for the narration cues.
import { writeFileSync } from "node:fs";
import { createRequire } from "node:module";

const require = createRequire(`${process.env.PPTR_DIR || "./node_modules"}/`);
const puppeteer = require("puppeteer-core");

const SIMPLE = process.env.SIMPLE || "touch the yellow block";
const CHAIN = process.env.CHAIN || "put the blue block in the blue bowl and then put the green block in the green bowl";
const FAR = "lift the red block 60 cm up";
const PICK = "pick up the red block";
const CUSTOM = process.env.CUSTOM || "push the green block onto the blue square"; // trained on the ViperX
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
// warm the Robot tab's part renders so the recording does not sit on spinners
await page.evaluate(() => window.setView("robot"));
await page.waitForFunction(() => document.querySelector(".part-fig img") && [...document.querySelectorAll(".part-fig img")].every((i) => i.classList.contains("in")), { timeout: 240000 });
for (const id of ["vx300s", "wx250s", "so101"]) {
  await page.click(`.part[data-slot="arm"][data-id="${id}"]`);
  await page.waitForFunction(() => /passed$/.test(document.querySelector("#asm-sum")?.textContent || ""), { timeout: 120000 });
  await page.waitForFunction(() => [...document.querySelectorAll(".part-fig img")].every((i) => i.classList.contains("in")), { timeout: 240000 });
}
await page.evaluate(() => { window.setView("sim"); window.scrollTo(0, 0); });
await sleep(3000);
await page.evaluate(() => window.startCapture());
const now = () => page.evaluate(() => (performance.now() - window.captureT0) / 1000);
const marks = { speed: [], prompts: [], at: {} };
const mark = async (name) => {
  marks.at[name] = await now();
  console.log(`${name} @ ${marks.at[name].toFixed(1)}s`);
};
await mark("intro");
await sleep(4000);

// a wait on the machine = Demos, Train or Eval live on the stage rail
const waiting = () =>
  page.evaluate(() => {
    const li = [...document.querySelectorAll(".rail li")];
    const by = (n) => li.find((l) => l.textContent.startsWith(n));
    return ["Demos", "Train", "Eval"].some((n) => by(n)?.classList.contains("live"));
  });
const finished = () =>
  page.evaluate(() => {
    const li = [...document.querySelectorAll(".rail li")];
    const pub = li.find((l) => l.textContent.startsWith("Publish"));
    return pub.classList.contains("done") || li.some((l) => l.classList.contains("fail") && !l.classList.contains("off"));
  });
// run `until` while speeding up whatever `isWait` calls a wait
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
async function type(text) {
  await page.click("#prompt", { clickCount: 3 });
  await page.type("#prompt", text, { delay: 40 });
  await sleep(500);
}
async function prompt(text) {
  await page.waitForSelector("#go:not([disabled])", { timeout: 120000 });
  await type(text);
  const t0 = Date.now();
  const p = { prompt: text, start: await now() };
  marks.prompts.push(p);
  const cards = await page.evaluate(() => document.querySelectorAll(".route-card").length);
  await page.click("#go");
  // settled only counts once this prompt's own route card is on screen (the rail still shows the last run)
  await page.waitForFunction((n) => document.querySelectorAll(".route-card").length > n, { timeout: 120000, polling: 100 }, cards);
  await sped(finished, waiting);
  p.published = await now();
  p.seconds = (Date.now() - t0) / 1000;
  console.log(`"${text}" settled after ${p.seconds.toFixed(1)} s`);
  return p;
}
const view = (v) => page.evaluate((x) => window.setView(x), v);
async function tileButton(name, label) {
  const ok = await page.evaluate(
    (n, l) => {
      const tile = [...document.querySelectorAll(".tile")].find((t) => [...t.querySelectorAll(".tile-name, .tile-ins")].some((e) => e.textContent.trim() === n));
      const b = tile && [...tile.querySelectorAll("button")].find((x) => x.textContent.trim() === l);
      if (!b) return false;
      tile.scrollIntoView({ block: "center", behavior: "smooth" });
      setTimeout(() => b.click(), 900);
      return true;
    },
    name,
    label,
  );
  if (!ok) throw new Error(`no ${label} button on the "${name}" tile`);
  await sleep(1000);
}
async function pick(slot, id) {
  await page.evaluate((s, i) => {
    const el = document.querySelector(`.part[data-slot="${s}"][data-id="${i}"]`);
    el.scrollIntoView({ block: "center", behavior: "smooth" });
    setTimeout(() => el.click(), 700);
  }, slot, id);
  await sleep(900);
  await page.waitForFunction(() => /passed$/.test(document.querySelector("#asm-sum")?.textContent || ""), { timeout: 120000 });
}

// 1. a simple prompt trains for real
await mark("simple");
await prompt(SIMPLE);
await mark("simple_done");
await sleep(5000);
const ok = await page.$$(".rtab.ok");
if (ok.length > 1) {
  await ok[1].click();
  await sleep(4000);
}

// 2. one rung up: two steps, both in the library now
await mark("chain");
await prompt(CHAIN);
await mark("chain_done");
await sleep(6000);

// 3. the router refuses what the arm cannot reach
await mark("far");
await prompt(FAR);
await sleep(5500);

// 4. the library: tiles, the new one on top, Remix
await mark("library");
await view("skills");
await page.evaluate(() => document.querySelector("#view-skills").scrollTo({ top: 0 }));
await sleep(4500);
await tileButton(SIMPLE, "Remix");
await mark("remix");
await sleep(3500);
await page.click("#lib-mode button"); // back to auto routing before the next prompt
await page.evaluate(() => document.querySelector("#prompt").blur());

// 5. the robot tab: swap the arm and the wrist, then a robot that fails a check
await mark("robot");
await view("robot");
await page.waitForFunction(() => [...document.querySelectorAll(".part-fig img")].every((i) => i.classList.contains("in")), { timeout: 180000 });
await sleep(2500);
await pick("arm", "vx300s");
await sleep(3000);
await pick("wrist", "locked");
await mark("wrist");
await sleep(3000);
await pick("wrist", "native");
await pick("arm", "wx250s");
await mark("widowx");
await page.evaluate(() => document.querySelector("#asm-fail")?.scrollIntoView({ block: "center", behavior: "smooth" }));
await sleep(4500);
await page.click("#use-robot");
await page.waitForFunction(() => document.querySelector("#use-robot")?.textContent.startsWith("Skills train"));
await sleep(2500);
await view("sim");
await sleep(2500);
await mark("refuse_robot");
await prompt(PICK);
await sleep(5500);

// 6. a skill trained on a custom robot (the ViperX), tested on fresh scenes
await page.evaluate(() => fetch("/api/robot/use", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ robot: "so101" }) }));
await view("skills");
await sleep(1500);
await mark("custom");
await page.evaluate((n) => {
  const tile = [...document.querySelectorAll(".tile")].find((t) => /ViperX/.test(t.textContent) && [...t.querySelectorAll(".tile-name, .tile-ins")].some((e) => e.textContent.trim() === n));
  tile.scrollIntoView({ block: "center", behavior: "smooth" });
  setTimeout(() => [...tile.querySelectorAll("button")].find((x) => x.textContent.trim() === "Test").click(), 1200);
}, CUSTOM);
await sleep(2500);
const tested = () => page.evaluate(() => /Test: \d+\/\d+/.test(document.querySelector("#lib-toast")?.textContent || ""));
await sped(tested, async () => !(await tested()));
await mark("custom_done");
console.log("test:", await page.evaluate(() => document.querySelector("#lib-toast").textContent));
await sleep(7000);

// 7. the ladder
await view("ladder");
await mark("ladder");
await sleep(6000);
marks.end = await now();
const res = await page.evaluate(() => window.stopCapture());
writeFileSync(res.path.replace(/\.webm$/, ".marks.json"), JSON.stringify(marks, null, 1));
console.log(JSON.stringify({ ...res, marks }));
await browser.close();
