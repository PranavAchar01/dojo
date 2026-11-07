// Live demo recording: a real, visible Chrome window; the page records itself with tab capture +
// MediaRecorder (?capture=1) while every step runs for real. Nothing is frame-stepped.
// While the forge waits on the machine (collecting demos, training, evaluating, a tile's test run), the page
// shows a "4x" badge and this script logs the span, so the edit speeds up exactly those waits and nothing else.
//
//   PPTR_DIR=~/helloworld/hyperframes-trial/node_modules node scripts/record_demo.mjs
//
// The tour (v5, the factory line in the round-5 Etch interface): the saved line replaying, a new line described in
// words and planned into stations, the VLA stations trained for real (sped up) and the whole line run, its replay
// and measured numbers, the new station skills in the Library, the router refusing a block 60 cm up, and the
// robot builder with its part previews. Every segment start is written to <webm>.marks.json under "at" for the
// narration cues, and the numbers measured in the run go to demo/v5-results.json.
import { writeFileSync, mkdirSync } from "node:fs";
import { createRequire } from "node:module";

const require = createRequire(`${process.env.PPTR_DIR || "./node_modules"}/`);
const puppeteer = require("puppeteer-core");

const LINE =
  process.env.LINE ||
  "Kitting line: red parts go to the blue tote, then yellow parts to the green tote, and a scripted ViperX at the end of the line catches everything else in the gray tote.";
const SAVED = process.env.SAVED_LINE || "140659-973a"; // a finished line, replayed while the intro plays
const FAR = "lift the red block 60 cm up";
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
await page.evaluate((id) => { const p = document.querySelector("#view-line .ln-pick"); p.value = id; p.dispatchEvent(new Event("change", { bubbles: true })); }, SAVED);
await sleep(6000); // the saved line loads and starts its replay
await page.evaluate(() => window.startCapture());
const now = () => page.evaluate(() => (performance.now() - window.captureT0) / 1000);
const marks = { speed: [], prompts: [], at: {} };
const mark = async (name) => {
  marks.at[name] = await now();
  console.log(`${name} @ ${marks.at[name].toFixed(1)}s`);
};
const view = (v) => page.evaluate((x) => window.setView(x), v);
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
await sleep(5000);

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
await sleep(5000);

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
await sleep(17000);
await scrollLine("#view-line .ln-cards", "start");
await mark("results");
await sleep(8000);
console.log("result:", await page.evaluate(() => document.querySelector("#view-line .ln-result").innerText.replace(/\s+/g, " ")));

// measured numbers of this run
const lineId = await page.evaluate(() => document.querySelector("#view-line .ln-pick").value);
const J = (u) => page.evaluate((x) => fetch(x).then((r) => r.json()), u);
const ln = await J(`/api/line/${lineId}`);
const stations = [];
for (const t of ln.training) {
  const run = await J(`/api/runs/${t.run}`);
  const st = ln.stations[t.station];
  stations.push({ name: st.name, part: st.part, bin: st.bin, train_seconds: t.seconds, held_out_success: t.success_rate, n_eval: t.n_eval,
    rounds: run.rounds.map((r) => ({ succeeded: r.eval?.succeeded, n: r.eval?.n })), from: t.from });
}
const r = ln.result;
const results = { recorded: new Date().toISOString(), line_id: lineId, prompt: LINE, stations,
  line: { parts: r.n_parts, completed: r.completed, success_rate: r.success_rate, parts_per_min: r.throughput_per_min, makespan_s: r.makespan_s },
  retry: stations.some((x) => x.rounds.length > 1) };
mkdirSync("demo", { recursive: true });
writeFileSync("demo/v5-results.json", JSON.stringify(results, null, 1));
console.log("results:", JSON.stringify(results));

// 4. each station's skill is a tile in the library
await view("skills");
await page.evaluate(() => document.querySelector("#view-skills").scrollTo({ top: 0 }));
await mark("library");
await sleep(6000);

// 5. the router refuses what the arm cannot reach
await view("sim");
await sleep(1200);
await mark("far");
await page.click("#prompt");
await page.type("#prompt", FAR, { delay: 28 });
await sleep(400);
await page.click("#go");
await page.waitForSelector(".route-card.is-refused", { timeout: 60000 });
await page.evaluate(() => document.querySelector(".route-card.is-refused:last-of-type, .route-card.is-refused").scrollIntoView({ block: "center", behavior: "smooth" }));
await sleep(7000);

// 6. the robot builder: swap the arm, every part previewed
await mark("robot");
await view("robot");
// the part previews render one by one: a wait, so it is sped up like the others
await page.evaluate(() => window.setSpeed(true));
const t0 = await now();
await page.waitForFunction(() => [...document.querySelectorAll(".part-fig img")].every((i) => i.classList.contains("in")), { timeout: 180000 });
await page.evaluate(() => window.setSpeed(false));
marks.speed.push([t0, await now()]);
await sleep(2000);
await page.evaluate(() => {
  const el = document.querySelector('.part[data-slot="arm"][data-id="vx300s"]');
  el.scrollIntoView({ block: "center", behavior: "smooth" });
  setTimeout(() => el.click(), 700);
});
await sleep(900);
await page.waitForFunction(() => /passed$/.test(document.querySelector("#asm-sum")?.textContent || ""), { timeout: 120000 }).catch(() => {});
await sleep(6000);

// 7. back to the line
await view("line");
await page.evaluate(() => document.querySelector("#view-line").scrollTo({ top: 0, behavior: "smooth" }));
await mark("close");
await sleep(5000);
marks.end = await now();
const res = await page.evaluate(() => window.stopCapture());
writeFileSync(res.path.replace(/\.webm$/, ".marks.json"), JSON.stringify(marks, null, 1));
console.log(JSON.stringify({ ...res, marks }));
await browser.close();
