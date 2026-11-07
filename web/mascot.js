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

const $ = (s) => document.querySelector(s);

// ---------------------------------------------------------------- what each stage looks like
// [state, activity, line while it runs, line when it finishes]
const STAGE = {
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
function mount(host) {
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
const each = (f) => followers().forEach(f);

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
    const bot = mount(h);
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
  for (const li of document.querySelectorAll("#rail li.live")) { const k = li.dataset.stage; if (k && ORDER.indexOf(k) >= ORDER.indexOf(key)) key = k; }
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
  const vis = (b) => { const r = b.host.getBoundingClientRect(); return r.width > 0 && r.bottom > 0 && r.top < innerHeight; };
  const f = followers().filter(vis);
  return f.find((b) => b.host.matches(".lg, .xl")) || f[0] || null;
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
  L.root.style.top = (below ? r.bottom + 8 : r.top - 8 - H) + "px";
  if (!bub) {
    const c = document.createElement("div");
    L.bubs.append(c);
    bub = { c, id, blob: null, bot: a };
  }
  bub.id = id; bub.bot = a;
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
      showBubble(STAGE[k][2], { id: k });
    } else if (st === "done" && was === "live") {
      const line = k === "eval" || k === "critic" || k === "optimizer" ? `${STAGE[k][3]}${m ? ": " + m : ""}` : STAGE[k][3];
      showBubble(line, { id: k, done: true }); dropBubble(650);
      each((b) => b.eyes.react("nod"));
      if (k === "registry") { each((b) => { b.eyes.emote("delight"); b.wobble(1); }); holdState("success", 2500); }
      if (k === "critic" && /retrain/i.test(m)) { each((b) => b.eyes.emote("wince")); setTimeout(() => each((b) => b.eyes.emote("nod")), 900); }
    } else if (st === "fail") {
      failAt = performance.now();
      showBubble(`${li.querySelector("span")?.textContent || k} failed`, { id: k, failed: true }); dropBubble(1400);
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
  if (st === "kept") { each((b) => { b.eyes.emote("pleased"); b.eyes.react("nod"); }); showBubble("Round kept", { id: "opt", done: true }); dropBubble(900); }
  else if (st === "reverted") { each((b) => b.eyes.emote("wince")); setTimeout(() => each((b) => b.eyes.emote("nod")), 900); showBubble("Round reverted", { id: "opt", failed: true }); dropBubble(1200); }
});

function watchOptimizer() {
  const kept = document.querySelectorAll("#optimizer-root .opt-chip.kept").length;
  const bad = document.querySelectorAll("#optimizer-root .opt-chip.reverted").length;
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
  if (e.kind === "error" && !e.child) { each((b) => b.eyes.react("squish")); holdState("error", 0); showBubble("Something broke", { id: "err", failed: true }); dropBubble(1600); }
  else if (e.kind === "rejected" || (e.kind === "route" && e.refused)) { each((b) => b.eyes.emote("wince")); holdState("confused", 4000); }
  else if (e.kind === "reused") { each((b) => { b.eyes.emote("pleased"); b.wobble(0.8); }); holdState("success", 2500); }
  else if (e.kind === "published" && !e.child) { each((b) => b.eyes.emote("delight")); }
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

// ---------------------------------------------------------------- go
decorate(document);
apply();
watchRail(); watchOptimizer(); forgeMsgs = document.querySelectorAll(".msg.forge").length;
if (!$("#rail") && !reducedMotion()) showreel();
