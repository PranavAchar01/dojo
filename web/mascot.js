// The forge bot. Etch's character rig (eyes.js, spring.js, goo.js, carried over from deploy/etch-app at a0bb241d) in
// Meta blues, wired to what VLA Forge is doing. This file plays the part of Etch's dock: it reads the page (stage
// rail, rollout badge, composer, chat thread, optimizer chips and the SSE stream re-broadcast as "forge:event"), picks
// the bot's state and activity, fires its one-shot emotes, and shows the running step in a bubble that buds off the
// bot and drops back in.
//
// Etch trigger            -> VLA Forge trigger
//   userTalking           -> typing in the task box (level follows keystrokes)
//   heard (nod, blink)    -> the task is sent
//   thinking (+ think)    -> Task / Route / Plan stages live
//   working + pick        -> Scene stage live (building the scene)
//   working + scan        -> Demos stage live (collecting demonstrations)
//   working + render      -> Train stage live
//   working + inspect     -> Eval stage live (measuring on held-out scenes)
//   working + read        -> Critic stage live (reading the failures)
//   working + repair      -> Optimizer stage live (a round is running)
//   working + route       -> Run stage live (a composed plan executing)
//   working + export      -> Publish stage live (saving the skill)
//   working + wait        -> agents warming up
//   success (hop, delight)-> a finished skill or a successful rollout
//   confused (tilt)       -> a failed rollout, a refused or rejected task
//   error (squish)        -> an error event from the pipeline
//   emote wince, nod      -> a step fails, then the retrain starts; a kept optimizer round nods, a reverted one winces
//   emote pleased / speak -> the forge answers in chat
//   emote surprise + hop  -> arriving on a page (Etch's arrived()), clicking the bot
//   emote delight         -> hovering a skill card (Etch's pick())
//   wobble                -> hover, and every bubble dropping back in
//   yawn, glances, blinks -> idle, built into eyes.js
import { createEyes } from "./eyes.js";
import { spring, reducedMotion } from "./spring.js";
import { gooLayer, spawnBlob } from "./goo.js";
import { createRoamer } from "./roam.js";
import { buildActive, buildSite, buildCaption } from "./build.js";
import { tick } from "./spring.js";

const $ = (s) => document.querySelector(s);

// ---------------------------------------------------------------- what each stage looks like
// [state, activity, line while it runs, line when it finishes]
export const STAGE = {
  task: ["thinking", "think", "Reading the task", "Task parsed"],
  router: ["thinking", "think", "Routing the task", "Routed"],
  planner: ["thinking", "think", "Planning the steps", "Plan ready"],
  scene: ["working", "pick", "Building the scene", "Scene ready"],
  data: ["working", "scan", "Collecting demos", "Demos collected"],
  trainer: ["working", "render", "Training the policy", "Policy trained"],
  eval: ["working", "inspect", "Testing on held-out scenes", "Eval"],
  critic: ["working", "read", "Reading the failures", "Critic"],
  optimizer: ["working", "repair", "Optimizing the skill", "Optimizer"],
  executor: ["working", "route", "Running the plan", "Plan ran"],
  registry: ["working", "export", "Saving the skill", "Skill saved"],
};
const ORDER = Object.keys(STAGE);

// ---------------------------------------------------------------- bots on the page
const bots = new Map(); // host -> { eyes, wob }
export function mountBot(host) {
  if (bots.has(host)) return bots.get(host);
  if (!host.isConnected) return null;
  host.querySelector(":scope > svg")?.remove();
  const eyes = createEyes(host);
  // Etch wobbles the pill whenever something changes under the eyes
  const wob = spring({ x: 1, y: 1 }, { preset: "wobbly", precision: 0.0008, onUpdate: (v) => { host.style.scale = v.x === 1 && v.y === 1 ? "" : `${v.x} ${v.y}`; } });
  const bot = { eyes, wob, host, wobble(s = 1) { wob.impulse({ x: 0.9 * s, y: -1.4 * s }); } };
  bots.set(host, bot);
  host.addEventListener("pointerenter", () => bot.wobble(0.6));
  host.addEventListener("click", () => { eyes.react("surprise"); bot.wobble(1); setTimeout(() => eyes.emote("hop"), 520); });
  return bot;
}
// ---------------------------------------------------------------- the roaming robot (roam.js)
// where the bot goes while a stage runs: the part of the screen that stage is working on
export const TARGET = {
  task: "#thread", router: "#rail", planner: "#plan-sec", scene: "#three", data: "#three", trainer: "#loss-sec",
  eval: "#rounds-sec", critic: "#critic-sec", optimizer: "#optimizer-root .opt-grid-card, #optimizer-root", executor: "#three", registry: "#rail",
};
export const roamer = createRoamer({ mountBot, slot: document.querySelector(".ag-pill-eyes") });
let visitSeq = 0;
export function goTo(sel, o = {}) { visitSeq++; return roamer.visit(sel, o); }

// ---------------------------------------------------------------- the loading screen: this run, being built
// While a run is under way the bot builds the run's own arm and task (build.js) in the sim view: large on the bare
// floor before the scene exists, then as a small card in a free corner once the real scene is drawn. Every change
// there follows an event from the run; the caption is the stage's own words and the rail's real metric.
let optAt = 0; // the last optimizer event from the stream: a saved run that is not running must not look live
const optLive = () => (optAt > 0 && performance.now() - optAt < 20000) || !!document.querySelector("#optimizer-root .opt-live, #optimizer-root [data-live='1']");
const shown = (n) => { if (!n || !n.isConnected) return false; const r = n.getBoundingClientRect(); return r.width > 40 && r.height > 40 && !n.closest(".hidden"); };
function workspaceFor() {
  const sim = $("#view-sim");
  if (!buildActive() || !shown(sim) || $("#boot:not(.gone)")) return null;
  const phone = innerWidth < 760, r = sim.getBoundingClientRect();
  if (sim.classList.contains("blank")) return { el: sim, compact: false, inset: phone ? { l: 10, r: 10, t: 40, b: 272 } : { l: 64, r: 104, t: 56, b: 96 } };
  const w = Math.round(Math.min(phone ? 220 : 300, r.width - 70)), h = Math.round(w * 0.66);
  const l = phone ? r.width - w - 10 : 64, t = phone ? 50 : r.height - 160 - h;
  return { el: sim, compact: true, inset: { l, t, r: r.width - l - w, b: r.height - t - h } };
}
function captionFor(k) {
  const m = ($(`#rail li[data-stage="${k}"] .m`)?.textContent || "").trim();
  return STAGE[k] ? STAGE[k][2] + (m ? ` · ${m}` : "") : "";
}
let wsCompact = null;
function updateWork() {
  const k = liveStage();
  const ws = workspaceFor();
  if (!ws) { if (roamer.working) { roamer.stopWork(); if (k && TARGET[k]) goTo(TARGET[k], { dwell: 0 }); } return false; }
  if (roamer.working && (roamer.working !== ws.el || wsCompact !== ws.compact)) roamer.stopWork(true);
  if (!roamer.working) { visitSeq++; dropBubble(0); }
  wsCompact = ws.compact;
  roamer.work(ws.el, { site: buildSite, compact: ws.compact, caption: k ? captionFor(k) : buildCaption(), inset: ws.inset });
  return true;
}
setInterval(updateWork, 900);
export function say(text, ms = 3200) { showBubble(text, { id: "say", done: true }); dropBubble(ms); }
export function goHomeSoon(ms = 1800) { const n = ++visitSeq; setTimeout(() => { if (n === visitSeq && !roamer.working) roamer.home(); }, ms); }

// a still of the idle face as markup, for places that cannot run script
export function mascotSVG() {
  const h = document.createElement("div");
  const e = createEyes(h);
  const html = h.innerHTML;
  e.destroy();
  return html;
}

// Etch's arrived(): a start, then a hop
function arrive(bot, delay = 0) {
  setTimeout(() => { bot.eyes.emote("surprise"); setTimeout(() => bot.eyes.emote("hop"), 520); bot.wobble(0.8); }, delay);
}

const followers = () => {
  const forge = document.querySelectorAll(".msg.forge .who .mascot");
  return [...document.querySelectorAll(".mascot[data-follow]"), ...(forge.length ? [forge[forge.length - 1]] : [])]
    .map((h) => bots.get(h)).filter(Boolean);
};
export const each = (f) => followers().forEach(f);

function decorate(root) {
  const hosts = [];
  if (root.matches?.(".mascot")) hosts.push(root);
  root.querySelectorAll?.(".mascot").forEach((h) => hosts.push(h));
  root.querySelectorAll?.(".msg.forge .who").forEach((w) => { if (!w.querySelector(".mascot")) { const s = document.createElement("span"); s.className = "mascot xs"; w.prepend(s); hosts.push(s); } });
  // every published skill is its own agent, so each card gets its own little bot
  root.querySelectorAll?.(".skill > .lbl").forEach((l) => {
    if (l.querySelector(".mascot")) return;
    const s = document.createElement("span"); s.className = "mascot xs"; l.prepend(s); hosts.push(s);
    const card = l.closest(".skill");
    card.addEventListener("pointerenter", () => bots.get(s)?.eyes.emote("delight"));
  });
  root.querySelectorAll?.(".empty-note").forEach((n) => {
    if (n.querySelector(".mascot")) return;
    const s = document.createElement("span"); s.className = "mascot md"; n.prepend(s); hosts.push(s);
  });
  let i = 0;
  for (const h of hosts) {
    if (bots.has(h)) continue;
    const bot = mountBot(h);
    if (bot && h.hasAttribute("data-follow")) { arrive(bot, 120 + 160 * i++); bot.eyes.setState(want.state); bot.eyes.setActivity(want.act); }
  }
}

// ---------------------------------------------------------------- the state machine
const want = { state: "idle", act: null };
const hold = { state: null, until: 0, timer: 0 }; // short-lived states: heard, success, confused, error
let typingUntil = 0, typingTimer = 0, speakUntil = 0, speakTimer = 0, speakLevel = 0;
const t0 = performance.now();

function holdState(state, ms) {
  clearTimeout(hold.timer);
  hold.state = state;
  hold.until = ms ? performance.now() + ms : Infinity;
  if (ms) hold.timer = setTimeout(() => { hold.state = null; apply(); }, ms);
  apply();
}
function liveStage() {
  let key = null;
  for (const li of document.querySelectorAll("#rail li.live")) { const k = li.dataset.stage; if (k === "optimizer" && !optLive()) continue; if (k && ORDER.indexOf(k) >= ORDER.indexOf(key)) key = k; }
  return key;
}
function compute() {
  const ready = $("#ready-dot") ? $("#ready-dot").classList.contains("on") : true;
  const k = liveStage();
  const now = performance.now();
  if (!ready) return { state: "working", act: "wait", words: "Warming up the agents" };
  if (k) { hold.state === "error" && clearHold(); const [state, act, words] = STAGE[k]; return { state, act, words, key: k }; }
  if (document.querySelector(".tool.live")) return { state: "working", act: "think", words: "Working" };
  if (hold.state && now < hold.until) return { state: hold.state, act: null, words: "" };
  if (speakUntil > now) return { state: "speaking", act: null, words: "" };
  if (typingUntil > now) return { state: "userTalking", act: null, words: "" };
  return { state: "idle", act: null, words: "" };
}
addEventListener("forge:mood", (ev) => {
  const st = ev.detail && ev.detail.state;
  if (st) holdState(st, ev.detail.ms || 0); else { clearHold(); apply(); }
});
function clearHold() { clearTimeout(hold.timer); hold.state = null; }
function apply() {
  const next = compute();
  const changed = next.state !== want.state || next.act !== want.act;
  want.state = next.state; want.act = next.act;
  document.documentElement.dataset.mood = next.state;
  if (changed) each((b) => { b.eyes.setState(next.state); b.eyes.setActivity(next.act); });
  const boot = $("#boot");
  if (boot && !boot.classList.contains("gone") && $("#ready-dot")?.classList.contains("on")) {
    boot.classList.add("gone");
    setTimeout(() => boot.remove(), 600);
  }
}

// ---------------------------------------------------------------- bubbles (Etch's showBubble / dropBubble)
const ICON_DONE = '<svg viewBox="0 0 16 16"><path d="m4.2 8.4 2.5 2.5 5.1-5.6" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"/></svg>';
const ICON_FAIL = '<svg viewBox="0 0 16 16"><path d="m5 5 6 6M11 5l-6 6" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round"/></svg>';
let layer = null, bub = null, bubT = 0;
function anchorBot() {
  const vis = (b) => { if (getComputedStyle(b.host.closest(".rb") || b.host).visibility === "hidden") return false; const r = b.host.getBoundingClientRect(); return r.width > 0 && r.bottom > 0 && r.top < innerHeight; };
  const f = followers().filter(vis);
  return f.find((b) => b.host === roamer.host) || f.find((b) => b.host.matches(".lg, .xl")) || f[0] || null;
}
function ensureLayer() {
  if (layer) return layer;
  const root = document.createElement("div");
  root.className = "mb-bubbles";
  root.setAttribute("aria-hidden", "true");
  const goo = document.createElement("div"); goo.className = "mb-goo";
  const bubs = document.createElement("div"); bubs.className = "mb-bubs";
  root.append(goo, bubs);
  gooLayer(goo, { id: "ag-goo-dock", blur: 9, alpha: 22, shift: -10, after: "drop-shadow(0 10px 22px rgba(22,28,45,.13)) drop-shadow(0 1px 2px rgba(22,28,45,.12))" });
  document.body.append(root);
  return (layer = { root, goo, bubs });
}
function showBubble(text, { id = null, done = false, failed = false } = {}) {
  const a = anchorBot();
  if (!a || !text) return;
  clearTimeout(bubT);
  const L = ensureLayer(), r = a.host.getBoundingClientRect();
  const below = r.top < 110, H = 64, W = 360;
  const left = Math.max(8, Math.min(r.left + r.width / 2 - 30, innerWidth - W - 8));
  L.root.style.left = left + "px";
  L.root.style.top = (below ? r.bottom + 8 : r.top - 22 - H) + "px"; // Dojo: clear the antenna and headband
  if (!bub) {
    const c = document.createElement("div");
    L.bubs.append(c);
    bub = { c, id, blob: null, bot: a };
  }
  bub.id = id; bub.bot = a; bub.base = { x: r.left, y: r.top };
  L.root.style.translate = "0 0";
  const c = bub.c;
  c.className = "ag-bub" + (done ? " done" : "") + (failed ? " failed" : "");
  c.replaceChildren();
  if (done || failed) { const i = document.createElement("span"); i.className = "ag-ic"; i.innerHTML = failed ? ICON_FAIL : ICON_DONE; c.append(i); }
  else { const sp = document.createElement("span"); sp.className = "ag-spin"; c.append(sp); }
  const tx = document.createElement("span"); tx.className = "ag-bub-t"; tx.textContent = text; c.append(tx);
  const w = c.offsetWidth, h = c.offsetHeight;
  const cx = r.left + r.width / 2 - left;
  const to = { x: Math.max(w / 2, Math.min(W - w / 2, cx - 22 + w / 2)), y: below ? 4 + h / 2 : H - 4 - h / 2 };
  if (!bub.blob) bub.blob = spawnBlob(L.goo, { from: { x: cx, y: below ? -r.height * 0.4 : H + r.height * 0.4 }, to, w, h, radius: Math.min(h / 2, 20), content: c, className: "ag-blob-bub" });
  else { bub.blob.resize(w, h); bub.blob.moveTo(to.x, to.y); }
}
function dropBubble(delay = 0) {
  clearTimeout(bubT);
  if (!bub) return;
  const b = bub;
  bubT = setTimeout(() => {
    if (bub !== b) return;
    bub = null;
    const r = b.bot.host.getBoundingClientRect(), L = layer.root.getBoundingClientRect();
    b.blob.absorb({ x: r.left + r.width / 2 - L.left, y: r.top + r.height / 2 - L.top }).then(() => { b.c.remove(); b.bot.wobble(1); b.bot.eyes.react("blink"); });
  }, delay);
}

// the bubble rides along while the bot flies
tick(() => {
  if (bub && layer && bub.base) {
    const r = bub.bot.host.getBoundingClientRect();
    const L0 = parseFloat(layer.root.style.left) || 0;
    const tx = Math.max(8 - L0, Math.min(innerWidth - 368 - L0, r.left - bub.base.x));
    // Dojo: slide left off anything marked keep-clear (the stage rail, badges) instead of covering it
    const c = bub.c.getBoundingClientRect(), x0 = c.left + (bub.kx || 0), x1 = x0 + c.width;
    let kx = 0;
    if (c.width) for (const k of document.querySelectorAll("[data-keepclear]")) {
      const q = k.getBoundingClientRect();
      if (q.width && c.top < q.bottom && c.bottom > q.top && x1 > q.left && x0 < q.right) kx = Math.max(kx, Math.min(x1 - q.left + 8, x0 - 8));
    }
    bub.kx = kx;
    layer.root.style.translate = `${(tx - kx).toFixed(1)}px ${(r.top - bub.base.y).toFixed(1)}px`;
  }
  return true;
});

// ---------------------------------------------------------------- reading the page for edges
const prev = {};      // stage key -> "live" | "done" | "fail" | ""
const metric = {};    // stage key -> last metric text
let booted = performance.now(), failAt = 0, optKept = 0, optBad = 0, forgeMsgs = 0;
const stageName = (li) => li.dataset.stage;
function railState(li) { return li.classList.contains("live") ? "live" : li.classList.contains("fail") ? "fail" : li.classList.contains("done") ? "done" : ""; }

function watchRail() {
  for (const li of document.querySelectorAll("#rail li[data-stage]")) {
    const k = stageName(li), st = railState(li), was = prev[k] ?? "", m = li.querySelector(".m")?.textContent || "";
    metric[k] = m;
    if (st === was) continue;
    prev[k] = st;
    if (!STAGE[k]) continue;
    if (st === "live") {
      // a saved optimizer run reported as running is not live unless the stream says so
      if (k === "optimizer" && !optLive()) continue;
      if (updateWork()) continue;
      showBubble(STAGE[k][2], { id: k });
      if (TARGET[k]) goTo(TARGET[k], { dwell: 0 });
    } else if (st === "done" && was === "live") {
      const line = k === "eval" || k === "critic" || k === "optimizer" ? `${STAGE[k][3]}${m ? ": " + m : ""}` : STAGE[k][3];
      showBubble(line, { id: k, done: true }); dropBubble(650);
      each((b) => b.eyes.react("nod"));
      goHomeSoon(k === "registry" ? 2600 : 2200);
      if (k === "registry") { each((b) => { b.eyes.emote("delight"); b.wobble(1); }); holdState("success", 2500); setTimeout(() => roamer.bow(), 500); setTimeout(rankUp, 1500); }
      if (k === "optimizer" && !optLive()) continue;
      if (k === "critic" && /retrain/i.test(m)) { each((b) => b.eyes.emote("wince")); setTimeout(() => each((b) => b.eyes.emote("nod")), 900); }
    } else if (st === "" && was === "live") {
      // the step stopped being this skill's (a background optimizer on another skill, a tile picked): its bubble goes
      if (bub && bub.id === k) dropBubble(0);
      if (!liveStage()) goHomeSoon(600);
    } else if (st === "fail") {
      failAt = performance.now();
      showBubble(`${li.querySelector("span")?.textContent || k} failed`, { id: k, failed: true }); dropBubble(1400);
      goHomeSoon(2600);
      each((b) => b.eyes.emote("wince"));
      setTimeout(() => { if (want.state === "working") each((b) => b.eyes.emote("nod")); else if (!hold.state) holdState("confused", 4000); }, 900);
    }
  }
}

function watchBadge() {
  const badge = $("#vp-badge");
  if (!badge || badge.classList.contains("hidden") || performance.now() - booted < 2500) return;
  const ok = badge.classList.contains("ok"), bad = badge.classList.contains("bad");
  const sig = ok ? "ok" : bad ? "bad" : "";
  if (sig === watchBadge.sig) return;
  watchBadge.sig = sig;
  if (ok) { each((b) => { b.eyes.emote("delight"); b.wobble(0.8); }); if (!liveStage()) holdState("success", 2500); }
  else if (bad) { if (!liveStage()) holdState("confused", 4000); else each((b) => b.eyes.emote("wince")); }
}

// stepping through the rounds in the montage: the selected chip's verdict gets its reaction
window.addEventListener("forge:optround", (e) => {
  const st = e.detail && e.detail.state;
  goTo("#optimizer-root .opt-grid-card, #optimizer-root", { dwell: 4200 });
  if (st === "kept") { each((b) => { b.eyes.emote("pleased"); b.eyes.react("nod"); }); showBubble("Round kept", { id: "opt", done: true }); dropBubble(900); }
  else if (st === "reverted") { each((b) => b.eyes.emote("wince")); setTimeout(() => each((b) => b.eyes.emote("nod")), 900); showBubble("Round reverted", { id: "opt", failed: true }); dropBubble(1200); }
});

function watchOptimizer() {
  const kept = document.querySelectorAll("#optimizer-root .opt-chip.kept").length;
  const bad = document.querySelectorAll("#optimizer-root .opt-chip.reverted").length;
  if (kept > optKept || bad > optBad) goTo("#optimizer-root .opt-grid-card, #optimizer-root", { dwell: 4200 });
  if (kept > optKept) { each((b) => { b.eyes.emote("pleased"); b.eyes.react("nod"); }); showBubble("Round kept", { id: "opt", done: true }); dropBubble(900); }
  if (bad > optBad) { each((b) => b.eyes.emote("wince")); setTimeout(() => each((b) => b.eyes.emote("nod")), 900); showBubble("Round reverted", { id: "opt", failed: true }); dropBubble(1200); }
  optKept = kept; optBad = bad;
}

function watchThread() {
  const n = document.querySelectorAll(".msg.forge").length;
  if (n > forgeMsgs && performance.now() - booted > 1500) {
    const last = [...document.querySelectorAll(".msg.forge")].pop();
    const words = (last?.textContent || "").trim().split(/\s+/).length;
    each((b) => b.eyes.emote("pleased"));
    // the forge "speaks" its answer: the eyes bounce with a level that rises and falls like a voice
    speakUntil = performance.now() + Math.min(2600, 500 + words * 70);
    clearInterval(speakTimer);
    const s0 = performance.now();
    speakTimer = setInterval(() => {
      const k = performance.now() - s0;
      if (performance.now() > speakUntil) { clearInterval(speakTimer); each((b) => b.eyes.setLevel(0)); apply(); return; }
      each((b) => b.eyes.setLevel(0.25 + 0.3 * Math.abs(Math.sin(k / 95))));
    }, 50);
    apply();
  }
  forgeMsgs = n;
}

// ---------------------------------------------------------------- the composer: typing, sending
const prompt = $("#prompt");
if (prompt) {
  prompt.addEventListener("input", () => {
    typingUntil = performance.now() + 1500;
    each((b) => b.eyes.setLevel(0.3 + Math.random() * 0.45));
    clearTimeout(typingTimer);
    typingTimer = setTimeout(() => { each((b) => b.eyes.setLevel(0)); apply(); }, 1500);
    apply();
    setTimeout(() => each((b) => b.eyes.setLevel(0.12)), 110);
  });
  prompt.addEventListener("blur", () => { typingUntil = 0; each((b) => b.eyes.setLevel(0)); apply(); });
}
$("#prompt-form")?.addEventListener("submit", () => {
  typingUntil = 0; each((b) => b.eyes.setLevel(0));
  holdState("heard", 1100);
  showBubble("Got it", { id: "sent", done: true }); dropBubble(900);
});

// ---------------------------------------------------------------- the SSE stream the app re-broadcasts
addEventListener("forge:event", (ev) => {
  const e = ev.detail || {};
  if (typeof e.kind === "string" && e.kind.startsWith("opt_")) optAt = performance.now();
  if (e.kind === "error" && !e.child) { each((b) => b.eyes.react("squish")); holdState("error", 0); showBubble("Something broke", { id: "err", failed: true }); dropBubble(1600); }
  else if (e.kind === "rejected" || (e.kind === "route" && e.refused)) { each((b) => b.eyes.emote("wince")); holdState("confused", 4000); }
  else if (e.kind === "reused") { each((b) => { b.eyes.emote("pleased"); b.wobble(0.8); }); holdState("success", 2500); }
  else if (e.kind === "published" && !e.child) { each((b) => b.eyes.emote("delight")); }
});

// a finished tile picked in the library: the bot says what that skill really is, saved, with its measured score
addEventListener("dojo:skill", (ev) => {
  const d = ev.detail || {};
  clearHold();
  each((b) => b.eyes.react("nod"));
  showBubble(`${d.name}: saved, ${d.held} held-out`, { id: "skill", done: true });
  dropBubble(2600);
  apply();
});

// ---------------------------------------------------------------- watching the page
let pend = 0;
function schedule() { if (!pend) pend = requestAnimationFrame(() => { pend = 0; watchRail(); watchBadge(); watchOptimizer(); watchThread(); apply(); }); }
new MutationObserver((recs) => {
  for (const r of recs) for (const n of r.addedNodes) if (n.nodeType === 1) decorate(n.parentElement || n);
  schedule();
}).observe(document.body, { subtree: true, childList: true, attributes: true, attributeFilter: ["class"], characterData: false });

// ---------------------------------------------------------------- landing page: no app behind it, so the bot shows its range
function showreel() {
  const seq = [
    ["idle", null, 3200], ["userTalking", null, 2200], ["heard", null, 900],
    ["thinking", "think", 2600], ["working", "pick", 1800], ["working", "scan", 2200], ["working", "render", 2400],
    ["working", "inspect", 2200], ["working", "read", 2000], ["working", "repair", 2000], ["working", "export", 1600],
    ["success", null, 2600], ["idle", null, 2000], ["confused", null, 2600], ["idle", null, 2200], ["error", null, 2200],
  ];
  let i = 0;
  const step = () => {
    const [state, act, ms] = seq[i++ % seq.length];
    each((b) => { b.eyes.setState(state); b.eyes.setActivity(act); if (state === "userTalking") b.eyes.setLevel(0.5); else b.eyes.setLevel(0); });
    if (state === "userTalking") { let n = 0; const iv = setInterval(() => { each((b) => b.eyes.setLevel(0.2 + Math.random() * 0.5)); if (++n > ms / 120) clearInterval(iv); }, 120); }
    setTimeout(step, ms);
  };
  setTimeout(step, 2800);
}

// ---------------------------------------------------------------- rank: a bow when the library passes the next mark
const RANK_AT = [0, 1, 3, 6, 10, 20, 40];
let rankNow = -1;
async function rankUp() {
  if (!$("#rail")) return;
  try {
    const lib = await (await fetch("/api/library")).json();
    const n = (lib.tiles || []).length;
    const rank = RANK_AT.reduce((r, at, i) => (n >= at ? i : r), 0);
    if (rankNow >= 0 && rank > rankNow) { roamer.bow(); say("Level up", 2600); }
    rankNow = rank;
    roamer.root.title = `${n} skill${n === 1 ? "" : "s"} passed`;
  } catch { /* the server is still warming up */ }
}
setTimeout(rankUp, 1200);

// ---------------------------------------------------------------- go
decorate(document);
apply();
watchRail(); watchOptimizer(); forgeMsgs = document.querySelectorAll(".msg.forge").length;
if (!$("#rail") && !reducedMotion()) showreel();
