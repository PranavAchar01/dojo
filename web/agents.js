// How it works: one prompt to one published skill, as nine steps. Each step says in a line what happens, which model or
// tool does it, and carries a small live slot. While a run is active the current step lights up and finished steps
// show a check; the slots fill from the same SSE stream the rest of the page uses (llm_call, task, route, scene, eval,
// critic, published, duel and the data/trainer logs). When nothing is running the slots show the last run's saved numbers
// (/api/state) and the optimizer's saved duels (/api/llm). Nothing is drawn before it happens; no timers while idle.
const $ = (s, r = document) => r.querySelector(s);
const ROOT = () => $("#view-agents");
const esc = (s) => String(s ?? "").replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
const num = (n) => Number(n || 0).toLocaleString();
const secs = (ms) => (ms >= 1000 ? (ms / 1000).toFixed(1) + " s" : Math.round(ms) + " ms");
const pct = (x) => Math.round(Number(x || 0) * 100) + "%";

const STEPS = [
  ["Prompt", "You type one sentence.", "You"],
  ["Understand", "Muse Spark reads the task while SubQ checks the whole skill library for a match.", "Muse Spark + SubQ Snap"],
  ["Build the scene", "A physics world with the real robot parts.", "MuJoCo, SO-101"],
  ["Demonstrations", "An expert performs the task many times in simulation.", "Scripted expert"],
  ["Train", "Fine-tune a vision-language-action policy, starting from the nearest saved skill.", "VLA fine-tune"],
  ["Test on new scenes", "The policy runs on layouts it has never seen.", "MuJoCo rollouts"],
  ["Critic", "Is it good enough, or does it need another round?", "Muse Spark + SubQ Snap"],
  ["Optimizer duel", "Two models each propose one change, both versions train, and the winner must beat the parent.", "Muse Wonder vs SubQ Snap"],
  ["Publish", "The skill joins the library, ready to reuse.", "Skill library"],
];

let D = null; // the state of the run on screen
let models = {};
let saved = { duels: [], tally: null };
let raf = 0, tick = 0;

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
      return t ? `<div class="wf-l"><b>${num(t.steps)} steps</b><span>loss ${Number(t.loss || 0).toFixed(3)}${d.from ? `, from skill-${esc(d.from)}` : ""}</span></div>` : none;
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
      if (d.skill) return `<div class="wf-l"><b>skill-${esc(d.skill)}</b><span>${d.rate != null ? pct(d.rate) + " on unseen scenes" : ""}${d.seconds ? `, ${Math.round(d.seconds)} s start to finish` : ""}</span></div>${d.reused ? `<div class="wf-note">Matched a saved skill, nothing new trained.</div>` : ""}`;
      return d.error ? `<div class="wf-l bad"><b>Stopped</b><span>${esc(d.error)}</span></div>` : none;
    }
  }
  return "";
}

function render() {
  raf = 0;
  const root = ROOT();
  if (!root || !D) return;
  const hidden = root.classList.contains("hidden");
  if (hidden) return;
  const head = D.live ? "Running now" : D.skill ? "Last run" : "";
  root.innerHTML = `<div class="wf"><div class="wf-h"><h2>How it works</h2><p>One sentence in, one tested robot skill out. Everything runs in simulation.</p>${head ? `<span class="wf-tag${D.live ? " on" : ""}">${head}</span>` : ""}</div>
    <ol class="wf-list">${STEPS.map(([name, line, tool], i) => {
      const s = D.stat[i];
      return `<li class="wf-s ${s}${i === 7 ? " wf-branch" : ""}"><span class="wf-n">${s === "done" ? `<svg viewBox="0 0 16 16"><path d="m3.5 8.5 3 3 6-7"/></svg>` : s === "skip" ? "–" : i + 1}</span>
        <div class="wf-b"><div class="wf-t"><b>${name}</b><span class="wf-tool">${tool}</span></div><p>${line}</p></div>
        <div class="wf-slot">${slot(i)}</div></li>`;
    }).join("")}</ol></div>`;
  clearInterval(tick);
}

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
