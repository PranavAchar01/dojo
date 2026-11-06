// Live demo recording: a real, visible Chrome window; the page records itself with tab capture +
// MediaRecorder (?capture=1) while a prompt trains end to end in real time. Nothing is frame-stepped.
//
//   NODE_PATH=~/helloworld/hyperframes-trial/node_modules node scripts/record_demo.mjs "put the red block in the white bowl"
import puppeteer from "puppeteer-core";

const prompt = process.argv[2] || "put the red block in the white bowl";
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
await sleep(2500);
await page.evaluate(() => window.startCapture());
await sleep(2500);

await page.click("#prompt", { clickCount: 3 });
await page.type("#prompt", prompt, { delay: 55 });
await sleep(700);
await page.click("#go");
const t0 = Date.now();

// wait for the registry to publish (or fail)
await page.waitForFunction(
  () => {
    const st = [...document.querySelectorAll(".stage")];
    return st.at(-1)?.classList.contains("done") || st.some((s) => s.classList.contains("fail"));
  },
  { timeout: 300000, polling: 500 },
);
console.log(`published after ${((Date.now() - t0) / 1000).toFixed(1)} s`);
await sleep(9000);
// show a second successful rollout, then a failure, from the published round
const chips = await page.$$(".chip.ok");
if (chips.length > 1) {
  await chips[1].click();
  await sleep(8000);
}
const bad = await page.$(".chip.bad");
if (bad) {
  await bad.click();
  await sleep(7000);
}
const res = await page.evaluate(() => window.stopCapture());
console.log(JSON.stringify(res));
await browser.close();
