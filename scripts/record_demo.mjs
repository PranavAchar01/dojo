// Live demo recording: a real, visible Chrome window; the page records itself with tab capture +
// MediaRecorder while every step runs for real. Nothing is frame-stepped.
// While the forge waits on the machine (collecting demos, training, evaluating, a tile's test run), the page
// shows a "4x" badge and this script logs the span, so the edit speeds up exactly those waits and nothing else.
//
//   PPTR_DIR=~/helloworld/hyperframes-trial/node_modules node scripts/record_demo.mjs
//
// The tour (v7): the landing page at / scrolled once, then into the editor at /app, which opens on a blank canvas.
// A new kitting skill for the SO-101 is forged live from a typed task (success check, demos, fine-tune, 20
// held-out scenes, sped up); "add another SO-101" grows the scene in place; a two-arm task fills the Arms list and
// runs; then the Optimize step opens the montage of a saved real optimizer run, round by round, and its
// before/after replay.
// The page cannot record across a navigation, so the landing page is its own capture
// (<webm>.marks.json holds its length under "landing_len"); the editor capture's marks file names it under
// "landing". Every segment start is written to the editor's marks under "at" for the narration cues, and the
// numbers shown are written to demo/v7-results.json.
import { writeFileSync, mkdirSync } from "node:fs";
import { createRequire } from "node:module";

const require = createRequire(`${process.env.PPTR_DIR || "./node_modules"}/`);
const puppeteer = require("puppeteer-core");

const BASE = process.env.BASE || "http://127.0.0.1:8100";
const TASK = process.env.TASK || "put the blue block in the back-left slot of the tray";
const TWO = "two arms: one seats the housing in the pallet, the other inserts it into the socket";
const OPT_RUN = process.env.OPT_RUN || "234442-a649"; // a saved real optimizer run in runs/opt-<id>
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
const browser = await puppeteer.launch({
  executablePath: "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
  headless: false,
  defaultViewport: null,
  userDataDir: process.env.REC_PROFILE || "/tmp/vla-forge-rec-profile",
  args: [
    "--window-size=1920,1200",
    "--window-position=0,0",
    `--app=${BASE}/app?capture=1`,
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
const ready = () => page.waitForFunction(() => document.querySelector("#ready-dot")?.classList.contains("on") && window.startCapture, { timeout: 120000 });
await ready();
// start on the stock arm, whatever the last session left active
await page.evaluate(() => fetch("/api/robot/use", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ robot: "so101" }) }));
// warm the Robot tab's part renders so the recording does not sit on spinners
await page.evaluate(() => window.setView("robot"));
await page.waitForFunction(() => document.querySelector(".part-fig img") && [...document.querySelectorAll(".part-fig img")].every((i) => i.classList.contains("in")), { timeout: 600000 });

// 0. the landing page: its own capture, a few seconds, scrolled once, then the click into the editor
await page.goto(`${BASE}/`, { waitUntil: "load" });
await sleep(1500);
await page.evaluate(() => {
  const chunks = [];
  window.landingT0 = null;
  window.startLanding = async () => {
    const stream = await navigator.mediaDevices.getDisplayMedia({ video: { frameRate: 30, width: 1920, height: 1200 }, audio: false, preferCurrentTab: true });
    window.landingRec = new MediaRecorder(stream, { mimeType: "video/webm;codecs=vp9", videoBitsPerSecond: 12_000_000 });
    window.landingRec.ondataavailable = (e) => e.data.size && chunks.push(e.data);
    window.landingRec.start(1000);
    window.landingT0 = performance.now();
  };
  window.stopLanding = () =>
    new Promise((resolve) => {
      window.landingRec.onstop = async () => {
        window.landingRec.stream.getTracks().forEach((t) => t.stop());
        const r = await fetch("/api/capture", { method: "POST", body: new Blob(chunks, { type: "video/webm" }) });
        resolve(await r.json());
      };
      window.landingRec.stop();
    });
  // the editor opens with the capture flag so its own recorder exists
  for (const a of document.querySelectorAll('a[href="/app"]')) a.setAttribute("href", "/app?capture=1");
});
await page.evaluate(() => window.startLanding());
await sleep(2500);
await page.evaluate(() => window.scrollTo({ top: 620, behavior: "smooth" }));
await sleep(3200);
const landingLen = await page.evaluate(() => (performance.now() - window.landingT0) / 1000);
const landing = await page.evaluate(() => window.stopLanding());
await Promise.all([page.waitForNavigation({ waitUntil: "load" }), page.click("#nav a.cta")]);
writeFileSync(landing.path.replace(/\.webm$/, ".marks.json"), JSON.stringify({ landing_len: landingLen }, null, 1));
console.log("landing", landing.path, landingLen.toFixed(1));

// 1. the editor, on the Sim tab
await ready();
await page.evaluate(() => { window.setView("sim"); window.scrollTo(0, 0); });
await sleep(1500);
await page.evaluate(() => window.startCapture());
const now = () => page.evaluate(() => (performance.now() - window.captureT0) / 1000);
const marks = { speed: [], prompts: [], at: {}, landing: landing.path.replace(/^.*\//, "") };
const mark = async (name) => {
  marks.at[name] = await now();
  console.log(`${name} @ ${marks.at[name].toFixed(1)}s`);
};
const view = (v) => page.evaluate((x) => window.setView(x), v);
const scrollLine = (sel, block = "start") =>
  page.evaluate((s, b) => document.querySelector(s)?.scrollIntoView({ block: b, behavior: "smooth" }), sel, block);
// a wait on the machine = Demos, Train or Eval live on the stage rail
const rail = (names, cls) =>
  page.evaluate(
    (ns, c) => {
      const li = [...document.querySelectorAll(".rail li")];
      return ns.some((n) => li.find((l) => l.textContent.startsWith(n))?.classList.contains(c));
    },
    names,
    cls,
  );
const waiting = () => rail(["Demos", "Train", "Eval"], "live");
const finished = async () =>
  (await rail(["Publish"], "done")) || (await page.evaluate(() => [...document.querySelectorAll(".rail li")].some((l) => l.classList.contains("fail") && !l.classList.contains("off"))));
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
const J = (u) => page.evaluate((x) => fetch(x).then((r) => r.json()), u);

await mark("intro");
await sleep(3500);

// 2. type a task: the agents write the success check, collect demos, fine-tune and test on 20 held-out scenes
await mark("describe");
await page.waitForSelector("#go:not([disabled])", { timeout: 120000 });
await page.click("#prompt", { clickCount: 3 });
await page.type("#prompt", TASK, { delay: 40 });
await sleep(500);
const t0 = Date.now();
const p = { prompt: TASK, start: await now() };
marks.prompts.push(p);
const cards = await page.evaluate(() => document.querySelectorAll(".route-card").length);
const runs0 = (await J("/api/state")).runs.length;
await page.click("#go");
await page.waitForFunction((n) => document.querySelectorAll(".route-card").length > n, { timeout: 120000, polling: 100 }, cards);
await sped(finished, waiting);
p.published = await now();
p.seconds = (Date.now() - t0) / 1000;
console.log(`"${TASK}" settled after ${p.seconds.toFixed(1)} s`);
await mark("trained");
const st = await J("/api/state");
const run = await J(`/api/runs/${st.runs[st.runs.length - 1].id ?? st.runs[st.runs.length - 1]}`);
const sk = st.skills.reduce((a, b) => (b.created > a.created ? b : a));
const sim = { instruction: TASK, skill_id: sk.id, parent: sk.parent, train_seconds: sk.seconds, held_out_success: sk.success_rate, n_eval: sk.n_eval,
  rounds: run.rounds.map((r) => ({ succeeded: r.eval?.succeeded, n: r.eval?.n })), new_runs: st.runs.length - runs0 };
console.log("sim:", JSON.stringify(sim));
await sleep(7000);

// 3. another arm: "add another SO-101" grows the scene in place, then a two-arm task fills the Arms list
await view("sim");
await page.evaluate(() => window.scrollTo(0, 0));
await mark("addarm");
await page.click("#prompt");
await page.type("#prompt", "add another SO-101", { delay: 40 });
await sleep(400);
await page.click("#go");
await page.waitForFunction(() => document.querySelectorAll("#arms .arm").length >= 2, { timeout: 60000, polling: 100 });
await sleep(6500);
await mark("twoarm");
await page.click("#prompt");
await page.type("#prompt", TWO, { delay: 22 });
const before = new Set(((x) => (Array.isArray(x) ? x : x.lines))(await J("/api/lines")).map((l) => l.id));
await sleep(400);
await page.click("#go");
await page.waitForFunction(() => document.querySelectorAll("#arms .arm").length === 2 && !document.querySelector("#arms-go").disabled, { timeout: 90000, polling: 100 });
await sleep(4500);
await mark("armsrun");
await page.click("#arms-go");
await sleep(6000);
await sped(() => page.evaluate(() => /of \d+ tries/.test(document.querySelector("#arms-result")?.innerText || "")), async () => true);
await mark("armsresult");
await sleep(9000);
const lines = await J("/api/lines");
const lid = (Array.isArray(lines) ? lines : lines.lines).map((l) => l.id).find((id) => !before.has(id));
const ln = await J(`/api/line/${lid}`);
const arms = ln.stations.map((x, i) => {
  const t = ln.training?.find((y) => y.station === i);
  return { name: x.name, controller: x.controller, task: x.task, instruction: x.spec?.instruction ?? null, held_out_success: t?.success_rate ?? x.success_rate ?? null, n_eval: t?.n_eval ?? x.n_eval ?? null };
});
const armsText = await page.evaluate(() => document.querySelector("#arms-result").innerText.replace(/\s+/g, " "));
console.log("arms:", armsText);

// 4. the Optimize step: the saved real run, round by round
await page.evaluate(() => window.scrollTo(0, 0));
await mark("optimize");
await page.evaluate(() => [...document.querySelectorAll(".rail li")].find((l) => l.textContent.startsWith("Optimize")).click());
await sleep(2500);
await page.evaluate((id) => {
  const s = [...document.querySelectorAll("#optimizer-root select")].find((x) => [...x.options].some((o) => o.value === id));
  s.value = id;
  s.dispatchEvent(new Event("change", { bubbles: true }));
}, OPT_RUN);
await sleep(2500);
await mark("montage");
await page.evaluate(() => [...document.querySelectorAll("#optimizer-root .opt-btn")].find((x) => /montage/i.test(x.textContent)).click());
await sleep(3000);
const chipCount = await page.evaluate(() => document.querySelectorAll(".opt-overlay .opt-chip").length);
await mark("ticker");
for (let i = 0; i < chipCount; i++) {
  await page.evaluate((k) => document.querySelectorAll(".opt-overlay .opt-chip")[k].click(), i);
  await sleep(i === 0 ? 3600 : 2700);
}
await page.evaluate(() => document.querySelectorAll(".opt-overlay .opt-chip")[3].click());
await sleep(2200);
await mark("replay");
await page.evaluate(() => document.querySelector(".opt-overlay").scrollTo({ top: 99999, behavior: "smooth" }));
await sleep(9000);
const orun = await J(`/api/optimizer/runs/${OPT_RUN}`);
const optimizer = {
  run: OPT_RUN, skill: orun.skill.instruction, n_scenes: orun.n_scenes, best_round: orun.best_round,
  rounds: orun.rounds.map((r, i) => ({ round: i, success: r.success, n: r.n, status: r.status ?? "start", ci: r.ci, change: r.change?.text ?? null })),
};
const results = { recorded: new Date().toISOString(), sim_skill: sim, arms: { line_id: lid, prompt: TWO, arms, text: armsText, tries: ln.result?.n_parts, completed: ln.result?.completed, seed: ln.result?.seed }, optimizer, retry: sim.rounds.length > 1 };
mkdirSync("demo", { recursive: true });
writeFileSync("demo/v7-results.json", JSON.stringify(results, null, 1));
console.log("results:", JSON.stringify(results).slice(0, 600));

// 5. close on the montage's before and after
await mark("close");
await sleep(4000);
marks.end = await now();
const res = await page.evaluate(() => window.stopCapture());
writeFileSync(res.path.replace(/\.webm$/, ".marks.json"), JSON.stringify(marks, null, 1));
console.log(JSON.stringify({ ...res, marks }));
await browser.close();
