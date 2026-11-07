// Skill library: every finished skill is a tile (thumbnail from its own rollout, measured success, train time,
// parent, robot). Tiles replay, test on fresh scenes, remix, or start a new skill from the base policy.
// Also draws the router's verdict and the progress card for a run (parse, route, demos, train, eval, save).

import { robotLabel } from "./builder.js";
import { loadingNode } from "./loading.js";

const F = () => window.forge;
const $ = (s) => document.querySelector(s);
const el = (tag, cls, text) => {
  const e = document.createElement(tag);
  if (cls) e.className = cls;
  if (text !== undefined) e.textContent = text;
  return e;
};
const fmt = (s) => {
  s = Math.max(0, Math.round(s));
  return `${Math.floor(s / 60)}:${String(s % 60).padStart(2, "0")}`;
};
const pct = (x) => `${Math.round(100 * x)}%`;
const ago = (t) => {
  const d = Date.now() / 1000 - t;
  if (d < 90) return "just now";
  if (d < 3600) return `${Math.round(d / 60)} min ago`;
  if (d < 86400) return `${Math.round(d / 3600)} h ago`;
  return new Date(t * 1000).toLocaleDateString(undefined, { month: "short", day: "numeric" });
};

const DECISION = {
  reuse: "Reuse a tile",
  train: "Train a new model",
  chain: "Chain models",
  too_complex: "Too complex",
  not_feasible: "Not feasible",
};
const ORIGIN = { prompt: "prompt", remix: "remix", scratch: "from scratch", step: "chain step", chain: "chain", station: "arm in a task", arms: "arms together", scripted: "in progress" };

// "2× SO-101 + ViperX 300": every arm of a multi-arm card, same words as a single-arm card
function armsChip(t) {
  const n = new Map();
  for (const a of t.arms) n.set(a.robot, (n.get(a.robot) || 0) + 1);
  return [...n].map(([r, k]) => `${k > 1 ? `${k}× ` : ""}${robotLabel(r)}`).join(" + ");
}

let tiles = [];
let loaded = false;
let fresh = null; // tile id to flash as just saved
const filt = { q: "", f: "all", sort: "new" };
let mode = { mode: "auto", from_skill: "", name: "" };
const tests = {}; // test id -> { skill, n, done, succeeded, successes }
const testing = {}; // skill id -> test id while a test runs

// ---------------------------------------------------------------- progress of the prompt being worked on
const PHASES = [
  ["parse", "Parse"],
  ["route", "Route"],
  ["demos", "Demos"],
  ["train", "Train"],
  ["eval", "Eval"],
  ["save", "Save"],
];
let prog = null; // { run, prompt, phase, frac, label, detail, state, card, kids:Set, train:{t0,s}, round }

function progStart(run, prompt) {
  prog = { run, prompt, phase: "parse", frac: 0, label: "Reading the prompt", detail: "", state: "live", kids: new Set(), round: 0 };
  drawProg();
}
function setPhase(phase, label, frac = 0, detail = "") {
  if (!prog) return;
  Object.assign(prog, { phase, label, frac, detail });
  drawProg();
}
function progEnd(state, label, detail = "") {
  if (!prog) return;
  Object.assign(prog, { state, label, detail, frac: 1 });
  drawProg();
}

function progCard() {
  if (!prog) return null;
  if (prog.card?.isConnected) return prog.card;
  const list = F()?.toolList(prog.run);
  if (!list) return null;
  const c = el("div", "progress-card is-loading");
  const steps = el("ol", "pc-steps");
  for (const [k, name] of PHASES) {
    const li = el("li", "", name);
    li.dataset.k = k;
    steps.append(li);
  }
  const bar = el("div", "pc-bar");
  bar.append(el("i"));
  const row = el("div", "pc-row");
  row.append(el("span", "pc-label"), el("span", "pc-detail mono"));
  c.append(steps, bar, row);
  list.parentElement.append(c);
  prog.card = c;
  return c;
}
function drawProg() {
  const c = progCard();
  if (c) {
    const at = PHASES.findIndex(([k]) => k === prog.phase);
    c.querySelectorAll(".pc-steps li").forEach((li, i) => {
      li.className = i < at || prog.state === "saved" ? "done" : i === at ? prog.state : "";
    });
    c.classList.toggle("is-loading", prog.state === "live");
    c.classList.toggle("is-saved", prog.state === "saved");
    c.classList.toggle("is-failed", prog.state === "fail");
    c.querySelector(".pc-bar i").style.width = `${Math.round(100 * Math.min(1, prog.frac))}%`;
    c.querySelector(".pc-label").textContent = prog.label;
    c.querySelector(".pc-detail").textContent = prog.detail;
    F()?.scrollThread();
  }
  drawSkeleton();
}
// the training bar moves with the clock between loss events
setInterval(() => {
  if (prog?.phase === "train" && prog.train && prog.state === "live") {
    const t = Date.now() / 1000 - prog.train.t0;
    prog.frac = Math.min(0.99, t / prog.train.s);
    prog.label = `Training · round ${prog.round + 1} · ${fmt(t)} of ~${fmt(prog.train.s)}`;
    drawProg();
  }
}, 250);

// ---------------------------------------------------------------- router card
function routeCard(e) {
  const list = F()?.toolList(e.run);
  if (!list) return;
  const c = el("div", `route-card d-${e.decision}${e.refused ? " is-refused" : ""}`);
  const h = el("div", "rc-h");
  h.append(el("span", "lbl", "Router"), el("b", "", DECISION[e.decision] || e.decision));
  if (e.estimate_s > 0) h.append(el("span", "rc-est mono", `~${fmt(e.estimate_s)}`));
  c.append(h, el("div", "rc-why", e.reason));
  const ul = el("ul", "rc-checks");
  for (const k of e.checks) {
    const li = el("li", k.ok ? "ok" : "bad");
    li.append(el("span", "ck", k.ok ? "✓" : "✕"), el("b", "", k.name), document.createTextNode(` ${k.detail}`));
    ul.append(li);
  }
  c.append(ul);
  if (e.expected != null && !e.refused) c.append(el("div", "rc-note", `Expected success going in: ${pct(e.expected)}`));
  const acts = el("div", "rc-acts");
  if (e.refused && e.suggestion) {
    const b = el("button", "", `Try “${e.suggestion}”`);
    b.onclick = () => { $("#prompt").value = e.suggestion; $("#prompt").focus(); };
    acts.append(b);
  }
  if (e.decision === "reuse" && e.skill_id) {
    const t = el("button", "", "Test it");
    t.onclick = () => startTest(e.skill_id);
    const o = el("button", "", "Open tile");
    o.onclick = () => openTile(e.skill_id);
    const a = el("button", "", "Train anyway");
    a.onclick = () => { setMode("train", "", ""); F().submit(prog?.prompt || e.prompt || ""); };
    acts.append(t, o, a);
  }
  if (acts.children.length) c.append(acts);
  const pc = prog?.run === e.run && prog.card?.isConnected ? prog.card : null;
  list.parentElement.insertBefore(c, pc); // the verdict sits above the progress card
  F().scrollThread();
}

// ---------------------------------------------------------------- toast
function toast(text, sub, action) {
  let t = $("#lib-toast");
  if (!t) {
    t = el("div", "lib-toast");
    t.id = "lib-toast";
    $(".canvas").append(t);
  }
  t.innerHTML = "";
  const ic = el("span", "tk");
  ic.innerHTML = '<svg viewBox="0 0 20 20"><path d="m5 10.5 3 3 7-7"/></svg>';
  const tx = el("div", "tt");
  tx.append(el("b", "", text), el("span", "", sub || ""));
  t.append(ic, tx);
  if (action) {
    const b = el("button", "", action[0]);
    b.onclick = action[1];
    t.append(b);
  }
  t.classList.remove("is-saved");
  void t.offsetWidth;
  t.classList.add("is-saved");
  clearTimeout(t._h);
  t._h = setTimeout(() => t.classList.remove("is-saved"), 4200);
}

// ---------------------------------------------------------------- events (every event, before app.js filters)
function onEvent(e, mine) {
  if (e.run?.startsWith("test-")) return onTest(e);
  if (e.kind === "log" && e.text?.startsWith("chat message")) {
    const p = e.text.replace(/^chat message[^:]*:\s*/, "").replace(/^['"“]|['"”]$/g, "");
    progStart(e.run, p);
    return;
  }
  if (!prog) return;
  const kid = prog.kids.has(e.run);
  if (e.run !== prog.run && !kid) return;
  switch (e.kind) {
    case "task":
      setPhase("route", "Routing: reachability, objects, motions, time", 0.5);
      break;
    case "route":
      if (kid) break;
      routeCard(e);
      if (e.decision === "chain") setPhase("demos", `Chain of ${e.steps.length}: training the missing steps`, 0);
      else if (!e.refused && e.decision !== "reuse") setPhase("demos", "Setting the scene", 0.05, e.estimate_s ? `~${fmt(e.estimate_s)}` : "");
      break;
    case "rejected":
      progEnd("fail", DECISION[e.decision] || "Refused", "nothing trained");
      break;
    case "reused":
      setPhase("save", "Already in the library", 1);
      progEnd("saved", `Reused “${e.tile.name}”`, `${pct(e.tile.success_rate)} measured`);
      flash(e.skill);
      if (e.tile.replay) replay(e.skill);
      break;
    case "child":
      prog.kids.add(e.child);
      prog.step = e.step;
      setPhase("demos", `Step ${e.step + 1}: training “${e.plan[e.step].instruction}”`, 0);
      break;
    case "demos":
      setPhase("demos", `${kid ? `Step ${prog.step + 1} · ` : ""}Expert demos ${e.done}/${e.n}`, e.done / e.n, `${e.ok} usable`);
      break;
    case "train_start":
      prog.round = e.round;
      prog.train = { t0: Date.now() / 1000, s: Math.max(5, e.seconds) };
      setPhase("train", `Training · round ${e.round + 1}`, 0, "");
      break;
    case "loss":
      if (prog.phase === "train") { prog.detail = `loss ${e.ema.toFixed(3)} · ${e.step} steps`; drawProg(); }
      break;
    case "eval_progress":
      setPhase("eval", `Held-out eval · round ${e.round + 1}`, e.done / e.n, `${e.succeeded}/${e.n} so far`);
      break;
    case "compose_progress": {
      const finished = e.phases ? e.phases[e.phases.length - 1] : 0;
      setPhase("eval", "Running the chain on held-out scenes", finished / e.n, `t${e.tick}`);
      break;
    }
    case "critic":
      if (e.plan.action === "retrain") setPhase("demos", `Critic: another round`, 0, e.plan.rationale.slice(0, 60));
      break;
    case "published": // a recorded run reaches its end: the card settles, nothing is saved or announced
      if (String(e.run).startsWith("rp:")) progEnd("saved", "Replay finished", `${pct(e.success_rate)} held-out, as recorded`);
      break;
    case "saving":
      setPhase("save", "Saving to the library: rendering the thumbnail", 0.6);
      break;
    case "saved":
      tiles = [e.tile, ...tiles.filter((t) => t.id !== e.tile.id)];
      if (kid) {
        setPhase("demos", `Saved step “${e.tile.name}”`, 1);
        draw();
        break;
      }
      progEnd("saved", "Saved to library", `${pct(e.tile.success_rate)} · ${fmt(e.tile.seconds)}`);
      flash(e.tile.id);
      toast("Saved to library", `${e.tile.name} · ${pct(e.tile.success_rate)} on ${e.tile.n_eval} held-out scenes`, ["Open", () => openTile(e.tile.id)]);
      break;
    case "error":
      if (!kid) progEnd("fail", "Stopped", e.error?.slice(0, 80) || "");
      break;
  }
  void mine;
}

// ---------------------------------------------------------------- tests on fresh scenes
async function startTest(id, n = 5) {
  const t = tiles.find((x) => x.id === id);
  const r = await fetch(`/api/skills/${id}/test`, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ n }) });
  if (!r.ok) return toast("Test did not start", (await r.text()).slice(0, 80));
  const { test, n: nn } = await r.json();
  tests[test] = { skill: id, n: nn, done: 0, succeeded: 0, successes: null, name: t?.name || id };
  testing[id] = test;
  draw();
  F().setView("sim");
  F().setShown({ run: null, round: -1, index: -1 });
  F().setCaption(`Testing “${t?.name || id}” on ${nn} fresh scenes`);
  testChips(test);
}
function testChips(tid) {
  const T = tests[tid];
  const chips = $("#chips");
  chips.innerHTML = "";
  chips.dataset.pending = "";
  for (let i = 0; i < T.n; i++) {
    const ok = T.successes?.[i];
    const c = el(T.successes ? "button" : "span", `rtab ${T.successes ? (ok ? "ok" : "bad") : "pending"}`);
    c.append(el("span", "o"), el("span", "", T.successes ? (ok ? "success" : "failed") : "running"), el("span", "n", String(i + 1)));
    if (T.successes) c.onclick = () => playTest(tid, i);
    chips.append(c);
  }
  $("#round-label").textContent = T.successes ? `Test · ${T.succeeded}/${T.n} fresh` : `Test · ${T.done}/${T.n} done`;
}
async function playTest(tid, i) {
  const r = await fetch(`/api/tests/${tid}/rollout/${i}`);
  if (!r.ok) return;
  const p = await r.json();
  F().startPlayback(p);
  document.querySelectorAll("#chips .rtab").forEach((c, k) => c.classList.toggle("sel", k === i));
  F().setBadge(p.success ? `Test ${i + 1} · success` : `Test ${i + 1} · ${p.failure}`, p.success);
  $("#pb-label").textContent = `test ${i + 1}/${tests[tid].n}`;
}
function onTest(e) {
  const tid = e.run.slice(5);
  const T = tests[tid];
  if (!T) return; // someone else's test
  switch (e.kind) {
    case "test_start":
      if (e.scene) F().showScene(e.scene);
      F().setBadge(`Testing on ${T.n} fresh scenes`, true);
      break;
    case "test_progress":
      T.done = e.done ?? (e.phases ? e.phases[e.phases.length - 1] : T.done);
      T.succeeded = e.succeeded ?? T.succeeded;
      testChips(tid);
      draw();
      break;
    case "saved":
      tiles = [e.tile, ...tiles.filter((t) => t.id !== e.tile.id)];
      break;
    case "test_done": {
      Object.assign(T, { done: T.n, succeeded: e.succeeded, successes: e.successes });
      delete testing[T.skill];
      tiles = tiles.map((t) => (t.id === e.tile.id ? e.tile : t));
      testChips(tid);
      const first = Math.max(0, e.successes.indexOf(true));
      playTest(tid, first);
      draw();
      flash(T.skill);
      toast(`Test: ${e.succeeded}/${e.n} on fresh scenes`, `${T.name} · ${e.seconds}s`, ["Library", () => openTile(T.skill)]);
      break;
    }
    case "test_error":
      delete testing[T.skill];
      draw();
      toast("Test failed", e.error.slice(0, 90));
      break;
  }
}

// ---------------------------------------------------------------- replay a tile's recorded rollouts
async function replay(id, i = 0) {
  const t = tiles.find((x) => x.id === id);
  if (t?.origin === "arms") return replayArms(t);
  performance.mark("dojo-click");
  const want = F().replayData(id, i, t?.replay_source || ""); // the request goes out before any UI work
  F().simLoading(`Loading “${t?.name || "the model"}”`);
  F().setView("sim");
  if (t) F().selectSkill(t);
  const p = await want;
  if (!p) { F().simLoading(null); return toast("No recording yet", "Run Test to record this tile on fresh scenes"); }
  F().setShown({ run: null, round: -1, index: -1 });
  F().startPlayback(p, true);
  F().setCaption(`“${t?.name || id}” · recorded rollout`);
  F().setBadge(p.success ? `Rollout ${i + 1} · success` : `Rollout ${i + 1} · ${p.failure}`, p.success);
  const chips = $("#chips");
  chips.innerHTML = "";
  chips.dataset.pending = "";
  p.all.forEach((x, k) => {
    const c = el("button", `rtab ${x.success ? "ok" : "bad"}${k === i ? " sel" : ""}`);
    c.append(el("span", "o"), el("span", "", x.success ? "success" : "failed"), el("span", "n", String(k + 1)));
    c.onclick = () => replay(id, k);
    chips.append(c);
  });
  $("#round-label").textContent = `Recorded · ${p.source}`;
  $("#pb-label").textContent = `rollout ${i + 1}/${p.all.length}`;
}

// a multi-arm card plays its saved line run in the Sim with every arm moving (arms.js)
async function replayArms(t) {
  F().simLoading(`Loading “${t.name}”`);
  F().setView("sim");
  F().selectSkill(t);
  try {
    await window.forgeArms.openLine(t.line, t.name);
  } catch (err) {
    toast("No recording", err.message);
  }
  F().simLoading(null);
}

// ---------------------------------------------------------------- composer mode (remix / from scratch)
function setMode(m, from, name) {
  mode = { mode: m, from_skill: from, name };
  let chip = $("#lib-mode");
  if (!chip) {
    chip = el("span", "lib-mode hidden");
    chip.id = "lib-mode";
    $(".composer-row").insertBefore(chip, $("#go"));
  }
  chip.innerHTML = "";
  const show = m === "remix" || m === "scratch";
  chip.classList.toggle("hidden", !show);
  if (!show) { $("#prompt").placeholder = "Tell Dojo what the arm should do..."; return; }
  chip.append(el("span", "", m === "remix" ? `Remix · ${name}` : "From scratch"));
  const x = el("button", "", "×");
  x.setAttribute("aria-label", "Clear mode");
  x.onclick = () => setMode("auto", "", "");
  chip.append(x);
  $("#prompt").placeholder = m === "remix" ? `What should “${name}” learn instead?` : "A new model, trained from the base policy...";
  $("#prompt").focus();
}
function takeMode() {
  const m = { mode: mode.mode, from_skill: mode.from_skill };
  setMode("auto", "", "");
  return m;
}

// ---------------------------------------------------------------- the grid
function openTile(id) {
  F().setView("skills");
  fresh = id;
  draw();
  document.querySelector(`.tile[data-id="${id}"]`)?.scrollIntoView({ block: "center", behavior: "smooth" });
}
function flash(id) {
  fresh = id;
  draw();
  setTimeout(() => { if (fresh === id) { fresh = null; document.querySelector(`.tile[data-id="${id}"]`)?.classList.remove("is-saved"); } }, 8000);
}
function visible() {
  const q = filt.q.toLowerCase();
  let out = tiles.filter((t) => {
    if (q && ![t.name, t.instruction, t.prompt].some((s) => s?.toLowerCase().includes(q))) return false;
    if (filt.f === "mine") return t.origin === "remix" || t.origin === "scratch";
    if (filt.f !== "all") return String(t.tier) === filt.f;
    return true;
  });
  const key = { new: (t) => -t.created, rate: (t) => -t.success_rate, fast: (t) => t.seconds || 1e9 }[filt.sort];
  out = out.sort((a, b) => key(a) - key(b));
  return out;
}
function drawSkeleton() {
  const box = $("#skills");
  if (!box) return;
  let sk = box.querySelector(".tile.is-loading.run");
  const live = prog && prog.state === "live" && prog.phase !== "parse" && prog.phase !== "route";
  if (!live) { sk?.remove(); return; }
  if (!sk) {
    sk = el("article", "tile is-loading run");
    const th = el("div", "tile-thumb");
    th.append(el("div", "shimmer"));
    const body = el("div", "tile-body");
    body.append(el("div", "tile-name"), el("div", "sk-phase"), el("div", "pc-bar"));
    body.querySelector(".pc-bar").append(el("i"));
    sk.append(th, body);
    box.prepend(sk);
  }
  sk.querySelector(".tile-name").textContent = prog.prompt;
  sk.querySelector(".sk-phase").textContent = `${prog.label}${prog.detail ? ` · ${prog.detail}` : ""}`;
  sk.querySelector(".pc-bar i").style.width = `${Math.round(100 * prog.frac)}%`;
}
function tileEl(t, byId) {
  const a = el("article", `tile${t.id === fresh ? " is-saved" : ""}${testing[t.id] ? " is-testing" : ""}`);
  a.dataset.id = t.id;
  const th = el("div", "tile-thumb");
  if (t.thumb) {
    const img = el("img");
    img.alt = t.name;
    img.loading = "lazy";
    img.src = `/api/skills/${t.id}/thumb.png?v=${encodeURIComponent(String(t.tests.length) + t.replay_source + "." + t.thumb_v)}`;
    th.append(img);
  } else {
    th.append(el("div", "no-thumb", "No recording yet. Test records one."));
  }
  const multi = t.origin === "arms";
  th.append(el("span", "tile-tier mono", multi ? "ARMS" : t.origin === "scripted" ? "IN PROGRESS" : t.tier === 3 ? "CHAIN" : `T${t.tier}`), el("span", `tile-robot mono${multi || t.robot !== "so101" ? " custom" : ""}`, multi ? armsChip(t) : robotLabel(t.robot)));
  // one click on the tile plays its recorded rollout straight away; the buttons below keep their own jobs
  a.classList.add("tap");
  a.tabIndex = 0;
  a.setAttribute("aria-label", `${t.name}: play its recorded rollout`);
  a.addEventListener("click", (ev) => { if (!ev.target.closest("button, a, input, .tile-name")) replay(t.id); });
  a.addEventListener("keydown", (ev) => { if (ev.key === "Enter" && ev.target === a) replay(t.id); });
  if (t.replay && !multi) a.addEventListener("pointerenter", () => F().replayData(t.id, 0, t.replay_source), { once: true });
  if (testing[t.id]) {
    const T = tests[testing[t.id]];
    const ov = el("div", "tile-testing");
    ov.append(el("span", "spin"), document.createTextNode(`Testing ${T.done}/${T.n}`));
    th.append(ov);
  }
  const body = el("div", "tile-body");
  const name = el("div", "tile-name", t.origin === "scripted" ? t.name.replace(/\s*\(scripted\)$/i, "") : t.name);
  if (!multi) {
    name.title = "Double-click to rename";
    name.ondblclick = () => rename(t, name);
  }
  body.append(name);
  if (t.name !== t.instruction) body.append(el("div", "tile-ins", t.instruction));
  if (multi) {
    const ol = el("ol", "tile-chain tile-arms");
    for (const [i, ar] of t.arms.entries()) {
      const li = el("li", "");
      li.append(el("b", "", `Arm ${i + 1} · ${ar.controller}`), document.createTextNode(` ${ar.instruction}`));
      const facts = [robotLabel(ar.robot)];
      if (ar.held_out) facts.push(`${ar.held_out.k}/${ar.held_out.n} held-out`);
      else if (ar.controller === "trained VLA") facts.push("held-out not measured");
      if (ar.in_run) facts.push(`${ar.in_run.succeeded}/${ar.in_run.parts} in the run`);
      li.append(el("span", "arm-facts", facts.join(" · ")));
      ol.append(li);
    }
    body.append(ol);
  }
  if (t.detail && t.origin !== "scripted") body.append(el("div", "tile-ins", t.detail)); // an in-progress build shows only its bar
  if (t.steps?.length) {
    const ol = el("ol", "tile-chain");
    for (const st of t.steps) ol.append(el("li", "", st.instruction));
    body.append(ol);
  }
  const rate = el("div", "tile-rate");
  const bar = el("div", "rate-bar");
  const fill = el("i");
  fill.style.width = pct(t.success_rate);
  bar.append(fill);
  rate.append(el("b", "mono", pct(t.success_rate)), bar, el("span", "", multi ? `of ${t.n_eval} tries, all arms, in simulation` : t.origin === "scripted" ? `of ${t.n_eval} builds` : `of ${t.n_eval} held-out`));
  body.append(rate);
  const meta = el("div", "tile-meta");
  const parent = multi ? `${t.arms.length} arms` : t.origin === "scripted" ? "training next" : t.origin === "chain" ? `chain of ${t.steps.length}` : t.parent ? `from “${(byId[t.parent]?.name || "a tile").slice(0, 26)}”` : "from base";
  meta.append(
    el("span", "", multi ? `${Math.round(t.measured.makespan_s)}s simulated` : t.train_seconds ? `train ${fmt(t.train_seconds)}` : `run ${fmt(t.seconds)}`),
    el("span", "", parent),
    el("span", "", ORIGIN[t.origin] || t.origin),
    el("span", "", ago(t.created)),
  );
  body.append(meta);
  if (t.tests.length) {
    const tr = el("div", "tile-tests");
    tr.append(el("span", "lbl", "Tests"));
    for (const x of t.tests) {
      const d = el("span", `tdot ${x.succeeded / x.n >= 0.5 ? "ok" : "bad"}`, `${x.succeeded}/${x.n}`);
      d.title = `${x.succeeded} of ${x.n} fresh scenes${x.t ? `, ${ago(x.t)}` : ""}`;
      tr.append(d);
    }
    body.append(tr);
  }
  if (t.origin === "scripted") { a.append(th, body); return a; } // no policy: nothing to Test, Remix or Call
  const acts = el("div", "tile-acts");
  const mk = (label, fn, cls) => {
    const b = el("button", cls || "", label);
    b.onclick = fn;
    acts.append(b);
    return b;
  };
  if (multi) {
    mk("Play", () => replay(t.id));
    a.append(th, body);
    body.append(acts);
    return a;
  }
  const tb = mk(testing[t.id] ? "Testing..." : "Test", () => startTest(t.id));
  tb.disabled = !!testing[t.id];
  mk("Replay", () => window.forgeReplay.open(t), "replay-btn");
  if (t.origin !== "chain") mk("Remix", () => setMode("remix", t.id, t.name));
  const reply = el("div", "tile-reply mono");
  const cb = mk("Call agent", async () => {
    cb.disabled = true;
    reply.textContent = "running in a fresh scene...";
    const r = await fetch(`/api/skills/${t.id}/run`, { method: "POST" });
    reply.textContent = r.ok ? (await r.json()).reply : "no answer";
    cb.disabled = false;
  }, "ghost");
  body.append(acts, reply);
  a.append(th, body);
  return a;
}
async function rename(t, node) {
  const inp = el("input", "tile-rename");
  inp.value = t.name;
  node.replaceWith(inp);
  inp.focus();
  inp.select();
  const done = async (save) => {
    inp.onblur = null;
    if (save && inp.value.trim() && inp.value.trim() !== t.name) {
      const r = await fetch(`/api/skills/${t.id}`, { method: "PATCH", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ name: inp.value }) });
      if (r.ok) {
        const nt = await r.json();
        tiles = tiles.map((x) => (x.id === nt.id ? nt : x));
      }
    }
    draw();
  };
  inp.onkeydown = (ev) => { if (ev.key === "Enter") done(true); if (ev.key === "Escape") done(false); };
  inp.onblur = () => done(true);
}
function draw() {
  const box = $("#skills");
  if (!box) return;
  box.innerHTML = "";
  if (!loaded) {
    box.append(loadingNode("Loading the models...", { layout: "fill col" }));
    return;
  }
  const byId = Object.fromEntries(tiles.map((t) => [t.id, t]));
  const vis = visible();
  for (const t of vis) box.append(tileEl(t, byId));
  drawSkeleton();
  const chains = tiles.filter((t) => t.tier === 3).length;
  const nt = tiles.reduce((s, t) => s + t.tests.length, 0);
  $("#lib-count").textContent = `${tiles.length} trained model${tiles.length === 1 ? "" : "s"} · ${chains} chain${chains === 1 ? "" : "s"} · ${nt} test run${nt === 1 ? "" : "s"}` + (vis.length !== tiles.length ? ` · showing ${vis.length}` : "");
  if (!vis.length) box.append(el("div", "empty-note", tiles.length ? "Nothing matches." : "No trained models yet. Finish a prompt and it lands here."));
}
async function refresh(freshId) {
  const r = await fetch("/api/library");
  if (!r.ok) return;
  const d = await r.json();
  tiles = d.tiles;
  loaded = true;
  if (d.active && !prog && !["published", "rejected", "reused", "error"].includes(d.active.status)) {
    prog = { run: d.active.id, prompt: d.active.prompt, phase: "demos", frac: 0, label: "In progress", detail: "", state: "live", kids: new Set(), round: 0 };
  }
  if (freshId) flash(freshId);
  else draw();
}

// ---------------------------------------------------------------- toolbar
$("#lib-q").addEventListener("input", (e) => { filt.q = e.target.value; draw(); });
$("#lib-sort").addEventListener("change", (e) => { filt.sort = e.target.value; draw(); });
document.querySelectorAll("#lib-filter button").forEach((b) =>
  b.addEventListener("click", () => {
    filt.f = b.dataset.f;
    document.querySelectorAll("#lib-filter button").forEach((x) => x.classList.toggle("on", x === b));
    draw();
  }),
);
$("#lib-new").addEventListener("click", () => { setMode("scratch", "", ""); window.forge.openNew(); });

window.forgeLib = { onEvent, refresh, render: () => draw(), takeMode, startTest, replay, openTile };
draw();
refresh();

// Trained models | 3D models: the second face is the real-model library (web/assets.js), mounted on first use.
{
  let panel = null;
  document.getElementById("lib-face")?.addEventListener("click", async (e) => {
    const b = e.target.closest("button");
    if (!b) return;
    const three = b.dataset.m === "3d";
    document.querySelectorAll("#lib-face button").forEach((x) => x.classList.toggle("on", x === b));
    document.getElementById("lib-trained").hidden = three;
    document.getElementById("lib-3d").hidden = !three;
    const lede = document.querySelector("#view-skills .doc-h p");
    if (lede) {
      lede.dataset.trained ??= lede.textContent;
      lede.textContent = three ? "Real open-source 3D models, each with its source and licence, already converted for MuJoCo. Click one to look at it, or import your own from a file or a link." : lede.dataset.trained;
    }
    if (three && !panel) {
      const { mountAssets } = await import("/static/assets.js");
      panel = mountAssets(document.getElementById("lib-3d"));
    }
  });
}
