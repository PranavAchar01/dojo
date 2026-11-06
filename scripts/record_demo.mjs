// Live demo recording: a real, visible Chrome window; the page records itself with tab capture +
// MediaRecorder (?capture=1) while each prompt trains end to end in real time. Nothing is frame-stepped.
// While a skill is collecting demos and training, the page shows a "4x" badge and this script logs the
// span, so the edit can speed up exactly that wait and nothing else.
//
//   PPTR_DIR=~/helloworld/hyperframes-trial/node_modules node scripts/record_demo.mjs "touch the red block" \
//       "sort the blue and green blocks into the bowls of the same color"
import { writeFileSync } from "node:fs";
import { createRequire } from "node:module";

const require = createRequire(`${process.env.PPTR_DIR || "./node_modules"}/`);
const puppeteer = require("puppeteer-core");

const prompts = process.argv.slice(2);
if (!prompts.length) prompts.push("touch the red block");
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
await sleep(2000);
await page.evaluate(() => window.startCapture());
const now = () => page.evaluate(() => (performance.now() - window.captureT0) / 1000);
const marks = { speed: [], prompts: [] };
await sleep(2500);

// "training wait" = a skill is collecting demos or fine-tuning (Demos or Train live on the stage rail)
const training = () =>
  page.evaluate(() => {
    const li = [...document.querySelectorAll(".rail li")];
    const by = (n) => li.find((l) => l.textContent.startsWith(n));
    return ["Demos", "Train"].some((n) => by(n)?.classList.contains("live"));
  });
const finished = () =>
  page.evaluate(() => {
    const li = [...document.querySelectorAll(".rail li")];
    const pub = li.find((l) => l.textContent.startsWith("Publish"));
    return pub.classList.contains("done") || li.some((l) => l.classList.contains("fail") && !l.classList.contains("off"));
  });

for (const [k, prompt] of prompts.entries()) {
  await page.click("#prompt", { clickCount: 3 });
  await page.type("#prompt", prompt, { delay: 45 });
  await sleep(600);
  const t0 = Date.now();
  marks.prompts.push({ prompt, start: await now() });
  await page.click("#go");
  await page.waitForFunction(() => !document.querySelector(".rail li:last-child").classList.contains("done"), { timeout: 60000, polling: 100 });
  let fast = null;
  while (!(await finished())) {
    const tr = await training();
    if (tr && fast === null) {
      await page.evaluate(() => window.setSpeed(true));
      fast = await now();
    } else if (!tr && fast !== null) {
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
  const secs = (Date.now() - t0) / 1000;
  marks.prompts[k].published = await now();
  marks.prompts[k].seconds = secs;
  console.log(`"${prompt}" published after ${secs.toFixed(1)} s`);
  await sleep(8000);
  // a second successful rollout, then a failure if there is one
  const ok = await page.$$(".rtab.ok");
  if (ok.length > 1) {
    await ok[1].click();
    await sleep(k === prompts.length - 1 ? 9000 : 6000);
  }
  if (k === prompts.length - 1) {
    const bad = await page.$(".rtab.bad");
    if (bad) {
      await bad.click();
      await sleep(7000);
    }
    await page.evaluate(() => window.setView("ladder"));
    marks.ladder = await now();
    await sleep(6000);
  }
}
marks.end = await now();
const res = await page.evaluate(() => window.stopCapture());
writeFileSync(res.path.replace(/\.webm$/, ".marks.json"), JSON.stringify(marks, null, 1));
console.log(JSON.stringify({ ...res, marks }));
await browser.close();
