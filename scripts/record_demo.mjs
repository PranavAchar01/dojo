// Live demo recording: a real, visible Chrome window; the page records itself with tab capture +
// MediaRecorder while every step runs for real. Nothing is frame-stepped.
// While the forge waits on the machine (collecting demos, training, evaluating, a tile's test run), the page
// shows a "4x" badge and this script logs the span, so the edit speeds up exactly those waits and nothing else.
//
//   PPTR_DIR=~/helloworld/hyperframes-trial/node_modules node scripts/record_demo.mjs
//
// The tour (v6, the merged realism look): the landing page at / scrolled once, then into the editor at /app. A
// new kitting skill for the SO-101 is forged live from a typed task (success check, demos, fine-tune, 20 held-out
// scenes, sped up), then the saved Housing Assembly line (two trained SO-101 stations and a scripted ViperX)
// replays with its measured station table, then the Library, the router refusing a block 60 cm up, and the robot
// builder. The page cannot record across a navigation, so the landing page is its own capture
// (<webm>.marks.json holds its length under "landing_len"); the editor capture's marks file names it under
// "landing". Every segment start is written to the editor's marks under "at" for the narration cues, and the
// numbers shown are written to demo/v6-results.json.
import { writeFileSync, mkdirSync } from "node:fs";
import { createRequire } from "node:module";

const require = createRequire(`${process.env.PPTR_DIR || "./node_modules"}/`);
const puppeteer = require("puppeteer-core");

const BASE = process.env.BASE || "http://127.0.0.1:8100";
const TASK = process.env.TASK || "put the green block in the front-right slot of the tray";
const SAVED = process.env.SAVED_LINE || "194528-ab1b"; // Housing Assembly, trained and measured earlier
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

// 3. the saved Housing Assembly line: the replay, then its measured station table
await view("line");
await page.evaluate(() => document.querySelector("#view-line").scrollTo({ top: 0 }));
await sleep(600);
await page.evaluate((id) => { const s = document.querySelector("#view-line .ln-pick"); s.value = id; s.dispatchEvent(new Event("change", { bubbles: true })); }, SAVED);
await mark("line");
await sleep(11000);
await scrollLine("#view-line .ln-cards", "start");
await mark("cards");
await sleep(15500);
await mark("results");
await sleep(7000);
console.log("result:", await page.evaluate(() => document.querySelector("#view-line .ln-result").innerText.replace(/\s+/g, " ")));

// numbers shown on the station cards and the result tiles
const ln = await J(`/api/line/${SAVED}`);
const stations = ln.stations.map((s, i) => {
  const t = ln.training.find((x) => x.station === i);
  const ps = ln.result.per_station[i];
  return { name: s.name, controller: s.controller, robot: s.robot, held_out_success: t?.success_rate ?? null, n_eval: t?.n_eval ?? null,
    train_seconds: t?.seconds ?? null, on_line: { succeeded: ps.succeeded, parts_in: ps.parts, mean_work_s: ps.mean_work_s } };
});
const r = ln.result;
const results = { recorded: new Date().toISOString(), sim_skill: sim, line_id: SAVED, line_name: ln.name, prompt: ln.prompt, stations,
  line: { parts: r.n_parts, completed: r.completed, success_rate: r.success_rate, parts_per_min: r.throughput_per_min, makespan_s: r.makespan_s },
  retry: sim.rounds.length > 1 };
mkdirSync("demo", { recursive: true });
writeFileSync("demo/v6-results.json", JSON.stringify(results, null, 1));
console.log("results:", JSON.stringify(results));

// 4. each skill is a tile in the library
await view("skills");
await page.evaluate(() => document.querySelector("#view-skills").scrollTo({ top: 0 }));
await mark("library");
await sleep(6500);

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
await page.evaluate(() => window.setSpeed(true));
const tr = await now();
await page.waitForFunction(() => [...document.querySelectorAll(".part-fig img")].every((i) => i.classList.contains("in")), { timeout: 180000 });
await page.evaluate(() => window.setSpeed(false));
marks.speed.push([tr, await now()]);
await sleep(2000);
await page.evaluate(() => {
  const el = document.querySelector('.part[data-slot="arm"][data-id="vx300s"]');
  el.scrollIntoView({ block: "center", behavior: "smooth" });
  setTimeout(() => el.click(), 700);
});
await sleep(900);
await page.waitForFunction(() => /passed$/.test(document.querySelector("#asm-sum")?.textContent || ""), { timeout: 120000 }).catch(() => {});
await sleep(6000);

// 7. close on the library, where the skills forged in this tour sit as tiles
await view("skills");
await page.evaluate(() => document.querySelector("#view-skills").scrollTo({ top: 0, behavior: "smooth" }));
await mark("close");
await sleep(5000);
marks.end = await now();
const res = await page.evaluate(() => window.stopCapture());
writeFileSync(res.path.replace(/\.webm$/, ".marks.json"), JSON.stringify(marks, null, 1));
console.log(JSON.stringify({ ...res, marks }));
await browser.close();
