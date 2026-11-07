// The short v9 UI cut: the round 9 interface recorded live (real Chrome, tab capture + MediaRecorder, nothing
// frame-stepped), without waiting for a training run to finish and without any result numbers.
//
//   PPTR_DIR=~/helloworld/hyperframes-trial/node_modules node scripts/record_ui_v9.mjs
//
// The tour: the landing page, where the roaming robot greets and flies to each section as it scrolls in (its own
// capture, length under "landing_len" in its marks); then the editor: a task is typed and sent, the Agents view
// shows the Fetch.ai uAgents messages pulse and fill the log, back on the canvas the robot flies to the stage being
// worked on and points at it, the Thinking panel (T) opens, the dock is asked "What are you doing?" and answers from
// the live stage, then the wait for training is sped up (the page's 4x badge) until the robot reaches the loss
// chart, and the cut ends there. Marks go to <webm>.marks.json with "cut": "ui" for edit_demo.py.
import { writeFileSync } from "node:fs";
import { createRequire } from "node:module";

const require = createRequire(`${process.env.PPTR_DIR || "./node_modules"}/`);
const puppeteer = require("puppeteer-core");

const BASE = process.env.BASE || "http://127.0.0.1:8100";
const TASK = process.env.TASK || "push the blue block onto the orange square";
const SHOT = `${process.env.HOME}/Pictures/Screenshots/2026-11-07-vla-forge-v9-rec-`;
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
const browser = await puppeteer.launch({
  executablePath: "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
  headless: false,
  defaultViewport: null,
  protocolTimeout: 120000,
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
try {
  const [page] = await browser.pages();
  const ready = () => page.waitForFunction(() => document.querySelector("#ready-dot")?.classList.contains("on") && window.startCapture, { timeout: 120000 });
  await ready();
  console.log("editor ready");

  // 0. the landing page, its own capture: the robot says hello, then flies to each section heading as it scrolls in
  await page.goto(`${BASE}/`, { waitUntil: "load" });
  await sleep(200);
  await page.evaluate(() => {
    const chunks = [];
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
    for (const a of document.querySelectorAll('a[href="/app"]')) a.setAttribute("href", "/app?capture=1");
  });
  console.log("landing loaded");
  await page.evaluate(() => window.startLanding());
  console.log("landing capture on");
  await sleep(3600);
  await page.evaluate(() => document.querySelector("#how").scrollIntoView({ block: "start", behavior: "smooth" }));
  await sleep(4600);
  await page.screenshot({ path: `${SHOT}landing.png` });
  await page.evaluate(() => document.querySelector("#agents").scrollIntoView({ block: "start", behavior: "smooth" }));
  await sleep(4600);
  const landingLen = await page.evaluate(() => (performance.now() - window.landingT0) / 1000);
  const landing = await page.evaluate(() => window.stopLanding());
  await Promise.all([page.waitForNavigation({ waitUntil: "load" }), page.click("#nav a.cta")]);
  writeFileSync(landing.path.replace(/\.webm$/, ".marks.json"), JSON.stringify({ landing_len: landingLen }, null, 1));
  console.log("landing", landing.path, landingLen.toFixed(1));

  // 1. the editor; the Agents view is built once first so its pulses draw from the first message
  await ready();
  await page.evaluate(() => window.setView("agents"));
  await page.waitForSelector("#ag-graph svg", { timeout: 30000 });
  await page.evaluate(() => { window.setView("sim"); window.scrollTo(0, 0); });
  await sleep(1500);
  await page.evaluate(() => window.startCapture());
  const now = () => page.evaluate(() => (performance.now() - window.captureT0) / 1000);
  const marks = { cut: "ui", speed: [], prompts: [], at: {}, landing: landing.path.replace(/^.*\//, "") };
  const mark = async (name) => {
    marks.at[name] = await now();
    console.log(`${name} @ ${marks.at[name].toFixed(1)}s`);
  };
  const view = (v) => page.evaluate((x) => window.setView(x), v);
  const stageLive = (k) => page.evaluate((x) => document.querySelector(`#rail li[data-stage="${x}"]`)?.classList.contains("live"), k);

  await mark("intro");
  await sleep(2000);

  // 2. type the task and send it, then watch the uAgents messages on the Agents view
  await mark("describe");
  await page.waitForSelector("#go:not([disabled])", { timeout: 120000 });
  await page.click("#prompt", { clickCount: 3 });
  await page.type("#prompt", TASK, { delay: 40 });
  await sleep(500);
  marks.prompts.push({ prompt: TASK, start: await now() });
  const rows0 = await page.evaluate(() => document.querySelectorAll("#ag-log tbody tr").length);
  await page.click("#go"); // the Agents view hides the chat, so the task goes from the Sim view
  await mark("agents");
  await view("agents");
  await sleep(10000);
  const agentMsgs = (await page.evaluate(() => document.querySelectorAll("#ag-log tbody tr").length)) - rows0;
  await page.screenshot({ path: `${SHOT}agents.png` });
  console.log("agent messages while on the Agents view:", agentMsgs);

  // 3. back on the canvas: the robot flies to the stage being worked on and points at it
  await view("sim");
  await page.evaluate(() => window.scrollTo(0, 0));
  await mark("roam");
  await sleep(3000);
  await page.screenshot({ path: `${SHOT}roam.png` });
  await sleep(2500);

  // 4. Thinking on (T), then ask the dock what it is doing
  await mark("thinking");
  await page.evaluate(() => document.activeElement?.blur());
  await page.keyboard.press("t");
  await sleep(4500);
  await page.screenshot({ path: `${SHOT}thinking.png` });
  await mark("ask");
  await page.evaluate(() => window.forgeDock.prefill(""));
  await sleep(700);
  await page.type(".ag-input", "What are you doing?", { delay: 45 });
  await sleep(300);
  await page.keyboard.press("Enter");
  await sleep(7500);
  await page.screenshot({ path: `${SHOT}ask.png` });
  const answer = await page.evaluate(() => document.querySelector(".ag-panel, [class*='ag-']")?.closest("body") && [...document.querySelectorAll("[class*='answer']")].map((x) => x.innerText).filter(Boolean).at(-1) || "");
  console.log("bot answer:", answer.replace(/\s+/g, " ").slice(0, 300));
  await page.evaluate(() => { document.querySelector(".th-panel .th-x")?.click(); window.forgeDock.expand(false); });

  // 5. the wait for training, sped up (the page shows its 4x badge), until the robot is on the loss chart
  await mark("wait");
  await page.evaluate(() => window.setSpeed(true));
  const fast = await now();
  const until = Date.now() + 240000;
  while (!(await stageLive("trainer")) && Date.now() < until) await sleep(300);
  await sleep(6000);
  await page.evaluate(() => window.setSpeed(false));
  marks.speed.push([fast, await now()]);
  await mark("loss");
  await sleep(3500);
  await page.screenshot({ path: `${SHOT}loss.png` });
  await sleep(2500);
  await mark("close");
  await sleep(5000);
  marks.end = await now();
  marks.agent_msgs_on_view = agentMsgs;
  marks.bot_answer = answer;
  const res = await page.evaluate(() => window.stopCapture());
  writeFileSync(res.path.replace(/\.webm$/, ".marks.json"), JSON.stringify(marks, null, 1));
  console.log(JSON.stringify({ ...res, marks }));
} finally {
  await browser.close();
}
