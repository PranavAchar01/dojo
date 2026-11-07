// How it works: one prompt to one published model, as nine steps. Each step says in a line what happens, which model or
// tool does it, and carries a small live slot. While a run is active the current step lights up and finished steps
// show a check; the slots fill from the same SSE stream the rest of the page uses (llm_call, task, route, scene, eval,
// critic, published, duel and the data/trainer logs). When nothing is running the slots show the last run's saved numbers
// (/api/state) and the optimizer's saved duels (/api/llm). Nothing is drawn before it happens; no timers while idle.
import { LOGOS } from "/static/wf-logos.js";
import { mascotSVG } from "/static/mascot.js";
const $ = (s, r = document) => r.querySelector(s);
const ROOT = () => $("#view-agents");
const esc = (s) => String(s ?? "").replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
const num = (n) => Number(n || 0).toLocaleString();
const secs = (ms) => (ms >= 1000 ? (ms / 1000).toFixed(1) + " s" : Math.round(ms) + " ms");
const pct = (x) => Math.round(Number(x || 0) * 100) + "%";

const STEPS = [
  ["Prompt", "You type one sentence.", "You"],
  ["Understand", "Muse Spark reads the task while SubQ checks the whole model library for a match.", "Muse Spark + SubQ Snap"],
  ["Build the scene", "A physics world with the real robot parts.", "MuJoCo, SO-101"],
  ["Demonstrations", "An expert performs the task many times in simulation.", "Scripted expert"],
  ["Train", "Fine-tune a vision-language-action policy, starting from the nearest saved model.", "VLA fine-tune"],
  ["Test on new scenes", "The policy runs on layouts it has never seen.", "MuJoCo rollouts"],
  ["Critic", "Is it good enough, or does it need another round?", "Muse Spark + SubQ Snap"],
  ["Optimizer duel", "Two models each propose one change, both versions train, and the winner must beat the parent.", "Muse Wonder vs SubQ Snap"],
  ["Publish", "The model joins the library, ready to reuse.", "Model library"],
];

let D = null; // the state of the run on screen
let models = {};
let saved = { duels: [], tally: null };
let raf = 0;

const fresh = (run = "") => ({ run, prompt: "", stat: STEPS.map(() => "wait"), calls: {}, round: 0, status: "idle", live: false, duels: [] });

// ---------------------------------------------------------------- idle: the last run's saved numbers
function fromRun(r) {
  const d = fresh(r.id);
  d.prompt = r.prompt || "";
  d.parse = r.parse || {};
  d.spec = r.spec;
  d.round = r.rounds?.length || 0;
  const last = r.rounds?.[r.rounds.length - 1] || {};
  const done = (i) => (d.stat[i] = "done");
  done(0);
  if (r.parse?.model) done(1);
  if (last.demos != null) { done(2); d.demos = last.demos; }
  if (last.demos != null) done(3);
  if (last.train) { done(4); d.train = last.train; }
  if (last.eval?.n) { done(5); d.eval = last.eval; }
  if (last.plan) { done(6); d.critic = last.plan; }
  if (r.skill_id) { done(8); d.skill = r.skill_id; d.best = r.best_round; d.seconds = r.elapsed; d.rate = last.eval?.success_rate; d.stat.fill("done", 0, 7); d.stat[7] = "wait"; }
  d.status = r.status === "running" ? "running" : r.skill_id ? "done" : r.error ? "error" : "idle";
  d.error = r.error || "";
  d.live = d.status === "running";
  if (d.live) { const i = d.stat.indexOf("wait"); if (i >= 0) d.stat[i] = "live"; }
  return d;
}

async function load() {
  const root = ROOT();
  if (!root) return;
  const [st, llm] = await Promise.all([
    fetch("/api/state").then((r) => r.json()).catch(() => null),
    fetch("/api/llm").then((r) => r.json()).catch(() => null),
  ]);
  models = llm?.models || {};
  saved = { duels: llm?.duels?.length ? llm.duels : llm?.saved_duels || [], tally: llm?.duels?.length ? llm.tally : llm?.saved_tally || null };
  if (!D?.live) {
    const runs = (st?.runs || []).filter((r) => !r.parent);
    const r = runs.filter((x) => x.status === "running").pop() || runs.filter((x) => x.skill_id || x.rounds?.length).pop();
    D = r ? fromRun(r) : fresh();
  }
  root.dataset.ready = "1";
  schedule();
}

// ---------------------------------------------------------------- slots
const callLine = (lane, c, who) => {
  if (!c) return "";
  if (c.open) return `<div class="wf-l"><i class="wf-sp"></i><b>${who}</b><span>working</span></div>`;
  if (c.error) return `<div class="wf-l bad"><b>${who}</b><span>${esc(c.error)}</span></div>`;
  const tok = lane === "subq" ? `read ${num(c.tin)} tokens` : `${num(c.tout)} tokens out`;
  return `<div class="wf-l"><b>${who}</b><span>${tok}, ${secs(c.ms)}</span>${c.backend && !/^(meta|subq):/.test(c.backend) ? `<em>${esc(c.backend)}</em>` : ""}</div>`;
};

function metaNote() {
  const m = models.meta;
  if (!m || m.available !== false) return "";
  return `<div class="wf-l bad"><b>Meta unavailable</b><span>${esc(m.why || "no reason given")}${m.fallback ? `. Answering: ${esc(m.fallback)}` : ""}</span></div>`;
}

function slot(i) {
  const d = D, s = d.stat[i];
  const none = `<span class="wf-none">${s === "live" ? "Working..." : d.live ? "Waiting" : d.skill || d.prompt ? "Not recorded for this run" : "Runs on your next prompt."}</span>`;
  switch (i) {
    case 0:
      return d.prompt ? `<q>${esc(d.prompt)}</q>` : none;
    case 1: {
      const p = d.calls.parse || {};
      const rt = d.route ? `<div class="wf-l"><b>${esc(String(d.route.decision || "").replace(/_/g, " "))}</b><span>${esc(d.route.reason || "")}</span></div>` : "";
      if (p.meta || p.subq) return callLine("meta", p.meta, "Muse Spark") + callLine("subq", p.subq, "SubQ") + metaNote() + rt;
      if (d.parse?.model) return `<div class="wf-l"><b>${esc(d.parse.model)}</b><span>${Number(d.parse.seconds || 0).toFixed(1)} s</span></div>` + metaNote();
      return metaNote() || none;
    }
    case 2:
      return d.scene ? `<div class="wf-l"><b>${d.scene} block${d.scene === 1 ? "" : "s"}</b><span>on the table with the arm</span></div>` : s === "done" ? `<div class="wf-l"><b>Scene ready</b></div>` : none;
    case 3:
      return d.demos != null ? `<div class="wf-l"><b>${d.demos}${d.demosOf ? " of " + d.demosOf : ""}</b><span>demonstrations kept${d.round > 1 ? `, round ${d.round}` : ""}</span></div>` : s === "live" && d.demosOf ? `<div class="wf-l"><i class="wf-sp"></i><b>${d.demosOf}</b><span>demonstrations</span></div>` : none;
    case 4: {
      const t = d.train;
      return t ? `<div class="wf-l"><b>${num(t.steps)} steps</b><span>loss ${Number(t.loss || 0).toFixed(3)}${d.from ? `, from model ${esc(d.from)}` : ""}</span></div>` : none;
    }
    case 5: {
      const e = d.eval;
      return e ? `<div class="wf-big"><b>${e.succeeded}</b> of ${e.n} succeeded <em>${pct(e.success_rate)}</em></div>` : none;
    }
    case 6: {
      const c = d.critic, p = d.calls.critic || {};
      const verdict = c ? `<div class="wf-l"><b>${esc(c.action === "retrain" ? "Another round" : c.action || "Done")}</b><span>${esc(c.rationale || "")}</span></div>` : "";
      return (c || p.meta || p.subq) ? callLine("meta", p.meta, "Muse Spark") + callLine("subq", p.subq, "SubQ") + verdict : none;
    }
    case 7: {
      const rows = d.duels.length ? d.duels : saved.duels;
      const t = d.duels.length ? d.tally : saved.tally;
      const open = d.calls.opt;
      if (!rows.length && !open) return `<span class="wf-none">Optional, after publishing. Start it from the Optimize step.</span>`;
      const last = rows[rows.length - 1];
      const k = (x, y) => (x == null ? "?" : `${x}/${y?.n_select || "?"}`);
      return `${open ? `<div class="wf-l"><i class="wf-sp"></i><b>Round ${open}</b><span>both models are proposing</span></div>` : ""}
        ${t ? `<div class="wf-tally"><span class="m">Muse Wonder <b>${t.meta || 0}</b></span><span class="q">SubQ <b>${t.subq || 0}</b></span><span>tie <b>${t.tie || 0}</b></span></div>` : ""}
        ${last ? `<div class="wf-l"><b>Round ${esc(last.round)}</b><span>Wonder ${k(last.meta?.k, last)} vs SubQ ${k(last.subq?.k, last)}, winner ${esc(last.winner)}${last.admitted ? ", beat the parent" : ""}</span></div>` : ""}
        ${d.duels.length ? "" : `<div class="wf-note">${rows.length ? "from the saved optimizer history" : ""}</div>`}`;
    }
    case 8: {
      if (d.skill) return `<div class="wf-l"><b>model ${esc(d.skill)}</b><span>${d.rate != null ? pct(d.rate) + " on unseen scenes" : ""}${d.seconds ? `, ${Math.round(d.seconds)} s start to finish` : ""}</span></div>${d.reused ? `<div class="wf-note">Matched a saved model, nothing new trained.</div>` : ""}`;
      return d.error ? `<div class="wf-l bad"><b>Stopped</b><span>${esc(d.error)}</span></div>` : none;
    }
  }
  return "";
}

// ---------------------------------------------------------------- the graph
// Nodes carry a step index (into STEPS) so status and slots stay the single source of truth. The optimizer duel (step 7)
// is drawn as a branch after Publish: Muse Wonder and SubQ Snap in parallel, merging into "winner vs parent".
const NODES = [
  { id: "n0", s: 0, name: "Prompt", tool: "You", logos: ["user"] },
  { id: "n1", s: 1, name: "Understand", logos: ["meta", "subq"] },
  { id: "n2", s: 2, name: "Build the scene", logos: ["mujoco"] },
  { id: "n3", s: 3, name: "Demonstrations", logos: ["hf"] },
  { id: "n4", s: 4, name: "Train", logos: ["torch"] },
  { id: "n5", s: 5, name: "Test on new scenes", logos: ["mujoco"] },
  { id: "n6", s: 6, name: "Critic", logos: ["meta", "subq"] },
  { id: "n8", s: 8, name: "Publish", tool: "Model library", logos: ["dojo"] },
  { id: "w", s: 7, name: "Muse Wonder", tool: "proposes a change", logos: ["meta"], opt: 1 },
  { id: "q", s: 7, name: "SubQ Snap", tool: "proposes a change", logos: ["subq"], opt: 2 },
  { id: "v", s: 7, name: "Winner vs parent", tool: "Optimizer duel", logos: ["versus"], opt: 3 },
];
// [from, to, from side, to side (wide), from side, to side (narrow), kind]
const EDGES = [
  ["n0", "n1", "r", "l", "b", "t"], ["n1", "n2", "r", "l", "b", "t"], ["n2", "n3", "r", "l", "b", "t"], ["n3", "n4", "r", "l", "b", "t"],
  ["n4", "n5", "r", "r", "b", "t"],
  ["n5", "n6", "l", "r", "b", "t"], ["n6", "n8", "l", "r", "b", "t"],
  ["n6", "n4", "t", "b", "r", "r", "retry"],
  ["n8", "w", "l", "r", "b", "t", "opt"], ["n8", "q", "l", "r", "b", "t", "opt"],
  ["w", "v", "l", "r", "b", "t", "opt"], ["q", "v", "l", "r", "b", "t", "opt"],
];
const SIDE = { l: [-1, 0], r: [1, 0], t: [0, -1], b: [0, 1] };
let pop = null, narrow = false, mark = "";
const SVGNS = "http://www.w3.org/2000/svg";

const logo = (k) => {
  if (k === "dojo") return `<span class="wf-lg dojo">${mark}</span>`;
  const l = LOGOS[k];
  return `<span class="wf-lg ${l.wide ? "wide" : ""} ${k}"><svg viewBox="${l.vb}" aria-hidden="true">${l.html}</svg></span>`;
};

const plain = (html) => { const t = document.createElement("template"); t.innerHTML = html.replace(/<\/(div|span|b|em)>/g, "$& "); return (t.content.textContent || "").replace(/\s+/g, " ").trim(); };

// what a node shows under its name: the live slot, flattened to a line
function caption(n) {
  if (n.opt) {
    const t = D.duels.length ? D.tally : saved.tally;
    const rows = D.duels.length ? D.duels : saved.duels;
    const last = rows[rows.length - 1];
    if (n.opt === 1 && t) return `won ${t.meta || 0} round${t.meta === 1 ? "" : "s"}`;
    if (n.opt === 2 && t) return `won ${t.subq || 0} round${t.subq === 1 ? "" : "s"}`;
    if (n.opt === 3 && last) return `round ${last.round}, winner ${last.winner}${last.admitted ? ", beat the parent" : ""}`;
    if (D.calls.opt && D.stat[7] === "live") return "proposing";
    return "Optional";
  }
  return plain(slot(n.s));
}

function nodeHTML(n) {
  const st = D.stat[n.s];
  const tool = n.tool || STEPS[n.s][2];
  return `<button type="button" class="wn ${st}${n.opt ? " opt" : ""}${pop === n.id ? " sel" : ""}" data-n="${n.id}" style="grid-area:${n.id}">
    ${st === "done" ? `<i class="wn-ok"><svg viewBox="0 0 16 16"><path d="m3.5 8.5 3 3 6-7"/></svg></i>` : st === "live" ? `<i class="wn-live"></i>` : ""}
    <span class="wn-top"><span class="wn-lgs">${n.logos.map(logo).join("")}</span><b>${n.name}</b></span>
    <span class="wn-tool">${tool}</span>
    <span class="wn-cap">${esc(caption(n))}</span></button>`;
}

function popHTML() {
  const n = NODES.find((x) => x.id === pop);
  if (!n) return "";
  const [name, line, tool] = STEPS[n.s];
  return `<div class="wf-pop" role="dialog" aria-label="${esc(name)}"><button type="button" class="wf-x" aria-label="Close">×</button>
    <div class="wf-pt"><b>${esc(name)}</b><span class="wf-tool">${esc(n.tool || tool)}</span></div><p>${line}</p><div class="wf-slot">${slot(n.s)}</div></div>`;
}

function render() {
  raf = 0;
  const root = ROOT();
  if (!root || !D) return;
  if (root.classList.contains("hidden")) return;
  if (!mark) mark = mascotSVG();
  narrow = root.clientWidth < 780;
  const head = D.live ? "Running now" : D.skill ? "Last run" : "";
  root.innerHTML = `<div class="wfg"><div class="wf-h"><h2>How it works</h2><p>One sentence in, one tested robot model out. Everything runs in simulation.</p>${head ? `<span class="wf-tag${D.live ? " on" : ""}">${head}</span>` : ""}</div>
    <div class="wf-canvas ${narrow ? "narrow" : "wide"}${D.live ? " running" : ""}"><svg class="wf-edges" aria-hidden="true"><defs>${["", "done", "flow"].map((k) => `<marker id="wf-ar${k}" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path class="wf-ah ${k}" d="M1 1.5 8.5 5 1 8.5z"/></marker>`).join("")}</defs></svg>
    ${NODES.map(nodeHTML).join("")}${popHTML()}</div></div>`;
  drawEdges();
  placePop();
}

// anchor points come from the laid-out nodes, so the edges follow whatever the grid does
function drawEdges() {
  const cv = $(".wf-canvas");
  const svg = $(".wf-edges");
  if (!cv || !svg) return;
  const box = cv.getBoundingClientRect();
  const rect = (id) => { const r = $(`[data-n="${id}"]`, cv).getBoundingClientRect(); return { x: r.left - box.left, y: r.top - box.top, w: r.width, h: r.height }; };
  const pt = (r, s) => (s === "l" ? [r.x, r.y + r.h / 2] : s === "r" ? [r.x + r.w, r.y + r.h / 2] : s === "t" ? [r.x + r.w / 2, r.y] : [r.x + r.w / 2, r.y + r.h]);
  svg.setAttribute("width", box.width);
  svg.setAttribute("height", box.height);
  const nodeOf = (id) => NODES.find((n) => n.id === id);
  let out = "";
  for (const [a, b, sa0, sb0, sa1, sb1, kind] of EDGES) {
    const sa = narrow ? sa1 : sa0, sb = narrow ? sb1 : sb0;
    const [x0, y0] = pt(rect(a), sa), [x1, y1] = pt(rect(b), sb);
    const same = sa === sb;
    const k = same ? 46 : Math.min(90, Math.max(28, Math.hypot(x1 - x0, y1 - y0) * 0.45));
    const [ax, ay] = SIDE[sa], [bx, by] = SIDE[sb];
    const c0 = [x0 + ax * k, y0 + ay * k], c1 = [x1 + bx * k, y1 + by * k];
    const d = `M${x0} ${y0}C${c0[0]} ${c0[1]} ${c1[0]} ${c1[1]} ${x1} ${y1}`;
    const A = D.stat[nodeOf(a).s], B = D.stat[nodeOf(b).s];
    let st = "";
    if (kind === "retry") st = D.live && D.critic?.action === "retrain" ? "flow" : D.round > 1 ? "done" : "";
    else if (B === "live" && (A === "done" || kind === "opt")) st = "flow";
    else if (A === "done" && B === "done") st = "done";
    out += `<path class="wf-e ${kind || ""} ${st}" d="${d}" marker-end="url(#wf-ar${st})"/>`;
    if (kind === "retry") {
      const m = (t) => { const u = 1 - t; return [0, 1].map((i) => u ** 3 * [x0, y0][i] + 3 * u * u * t * c0[i] + 3 * u * t * t * c1[i] + t ** 3 * [x1, y1][i]); };
      const [mx, my] = m(0.5);
      out += narrow
        ? `<text class="wf-el" transform="rotate(90 ${mx + 13} ${my})" x="${mx + 13}" y="${my + 4}" text-anchor="middle">retrain</text>`
        : `<text class="wf-el" x="${mx}" y="${my - 6}" text-anchor="middle">retrain</text>`;
    }
  }
  svg.insertAdjacentHTML("beforeend", out);
}

function placePop() {
  const cv = $(".wf-canvas"), p = $(".wf-pop"), n = pop && $(`[data-n="${pop}"]`, cv || document);
  if (!cv || !p || !n) return;
  const box = cv.getBoundingClientRect(), r = n.getBoundingClientRect();
  const w = Math.min(320, box.width - 12);
  p.style.width = w + "px";
  p.style.left = Math.max(6, Math.min(box.width - w - 6, r.left - box.left + r.width / 2 - w / 2)) + "px";
  const below = r.bottom - box.top + 8;
  p.style.top = (below + p.offsetHeight > box.height + 40 ? Math.max(6, r.top - box.top - p.offsetHeight - 8) : below) + "px";
}

// delegated once: node click opens its popover, the close button, Escape or a click elsewhere closes it
const root0 = ROOT();
if (root0) {
  root0.addEventListener("click", (ev) => {
    const n = ev.target.closest("[data-n]");
    if (n) pop = pop === n.dataset.n ? null : n.dataset.n;
    else if (ev.target.closest(".wf-x") || !ev.target.closest(".wf-pop")) pop = null;
    else return;
    render();
  });
  new ResizeObserver(() => {
    if (root0.classList.contains("hidden") || !$(".wf-canvas", root0)) return;
    if ((root0.clientWidth < 780) !== narrow) schedule();
    else { $(".wf-edges")?.querySelectorAll(".wf-e,.wf-el").forEach((e) => e.remove()); drawEdges(); placePop(); }
  }).observe(root0);
}
addEventListener("keydown", (ev) => { if (ev.key === "Escape" && pop) { pop = null; schedule(); } });

function schedule() { if (!raf) raf = requestAnimationFrame(render); }

// ---------------------------------------------------------------- live events
function begin(prompt) {
  D = fresh();
  D.live = true;
  D.status = "running";
  D.stat[0] = "done";
  D.stat[1] = "live";
  D.prompt = prompt;
}

const setLive = (i) => { D.stat[i] = "live"; };
const fin = (i, next) => { D.stat[i] = "done"; if (next != null && D.stat[next] !== "done") D.stat[next] = "live"; };

function onCall(e) {
  const purpose = String(e.purpose || "");
  const key = /^parse/.test(purpose) ? "parse" : /^critic/.test(purpose) ? "critic" : null;
  if (/proposal/.test(purpose)) {
    D.calls.opt = (purpose.match(/\d+/) || [""])[0];
    if (e.phase === "start") D.stat[7] = "live";
    return;
  }
  if (!key || e.run !== D.run) return;
  const c = (D.calls[key] ||= {});
  c[e.lane] = e.phase === "start" ? { open: true } : { ms: e.ms, tin: e.tokens_in, tout: e.tokens_out, error: e.error, backend: e.backend };
}

function onEvent(e) {
  if (!D) return;
  const mine = e.run === D.run;
  if (e.kind === "log" && String(e.text || "").startsWith("chat message")) {
    D = fresh(e.run);
    begin(String(e.text).replace(/^chat message[^:]*:\s*/, "").replace(/^['"“]|['"”]$/g, ""));
    D.run = e.run;
    fetch("/api/state").then((r) => r.json()).then((st) => { const r = (st.runs || []).find((x) => x.id === D?.run); if (r) { D.prompt = r.prompt; schedule(); } }).catch(() => {});
    return schedule();
  }
  if (e.kind === "llm_call") { onCall(e); return schedule(); }
  if (e.kind === "duel") {
    D.duels.push(e);
    D.tally = { ...(D.tally || {}) };
    D.tally[e.winner] = (D.tally[e.winner] || 0) + 1;
    D.stat[7] = "done";
    D.calls.opt = "";
    return schedule();
  }
  if (!mine || !D.live) return;
  switch (e.kind) {
    case "task": D.parse = e.parse; if (e.steps) D.chain = e.steps.length; break;
    case "route":
      D.route = e;
      fin(1, e.refused ? null : 2);
      if (e.skill_id) D.from = e.skill_id;
      break;
    case "scene": D.scene = e.layout?.blocks?.length ?? e.blocks ?? 0; fin(2, 3); break;
    case "log": {
      const t = String(e.text || "");
      let m;
      if (e.agent === "data" && (m = t.match(/generating (\d+)/))) {
        if (D.stat[3] === "done" || D.stat[6] === "done") { D.round += 1; D.stat.fill("wait", 3, 7); D.eval = D.train = D.critic = null; D.calls.critic = null; } else D.round = 1;
        D.demos = null; D.demosOf = +m[1]; setLive(3);
      } else if (e.agent === "data" && (m = t.match(/(\d+)\/(\d+) demos ok/))) { D.demos = +m[1]; D.demosOf = +m[2]; fin(3, 4); }
      else if (e.agent === "trainer" && (m = t.match(/(\d+) steps, loss ([\d.]+)/))) { D.train = { steps: +m[1], loss: +m[2] }; fin(4, 5); }
      break;
    }
    case "eval": D.eval = e; fin(5, 6); break;
    case "critic":
      D.critic = e.plan;
      D.stat[6] = "done";
      if (e.plan?.action !== "retrain") setLive(8);
      break;
    case "published": D.skill = e.skill; D.rate = e.success_rate; D.seconds = e.seconds; D.stat[8] = "done"; D.stat[6] = "done"; D.live = false; D.status = "done"; break;
    case "reused": D.skill = e.skill; D.reused = true; D.stat.fill("skip", 2, 7); D.stat[8] = "done"; D.live = false; D.status = "done"; break;
    case "rejected": D.error = e.reason || "Not possible"; D.stat.fill("skip", 2, 9); D.live = false; D.status = "error"; break;
    case "error": D.error = e.error || "error"; D.live = false; D.status = "error"; D.stat = D.stat.map((s) => (s === "live" ? "wait" : s)); break;
    default: return;
  }
  schedule();
}

addEventListener("forge:event", (ev) => {
  const e = ev.detail || {};
  if (!ROOT()?.dataset.ready) return;
  onEvent(e);
});
addEventListener("forge:view", (ev) => { if (ev.detail === "agents") load(); });
