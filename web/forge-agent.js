// Glue: the Etch dock (dock.js) and the Thinking panel on top of the forge's own event stream, with the roaming robot.
// Everything shown comes from the real run: "thought" events are the agents' own log lines (critic rationale, parse
// notes, the plan, optimizer rounds), "agent_msg" events are the uAgents messages, and the answer to "What are you
// doing?" is built from the current stage and the last thoughts. No text is invented.
import { createDock } from "./dock.js";
import { roamer, goTo, goHomeSoon, TARGET } from "./mascot.js";

const $ = (s, r = document) => r.querySelector(s);
const esc = (s) => String(s).replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
const STAGES = ["parse", "route", "plan", "scene", "demos", "train", "eval", "critic", "publish"];

const dock = createDock(document.body, {
  eyes: roamer.bot && roamer.bot.eyes,
  onSend: (t) => answer(t),
  onMention: (ref) => fly(String(ref).toLowerCase()),
});
roamer.setSlot($(".ag-pill-eyes", dock.el));
dock.setStatus("");

// ---------------------------------------------------------------- live state
const st = { run: "", thoughts: [], n: 0, open: null, graph: null, last: 0 };
let loading = false, t0 = 0, loadIv = 0;

function agentLabel(agent, llm) { return llm ? `${agent} · uAgent · Llama 4 Maverick` : `${agent} · uAgent`; }

function railNow() {
  const live = $("#rail li.live"), done = [...document.querySelectorAll("#rail li.done")];
  const name = (li) => (li?.dataset.stage || "").replace(/-/g, " ");
  return { live: name(live), done: done.map(name), all: [...document.querySelectorAll("#rail li[data-stage]")].map(name) };
}

// ---------------------------------------------------------------- Thinking panel
const panel = document.createElement("section");
panel.className = "th-panel";
panel.hidden = true;
panel.setAttribute("aria-label", "Thinking");
panel.innerHTML = `<header><b>Thinking</b><span class="th-sub">what each agent is doing, as it happens</span><button type="button" class="th-x" aria-label="Close thinking">&times;</button></header><div class="th-body"></div><footer class="th-foot"></footer>`;
document.body.append(panel);
const body = $(".th-body", panel);
const foot = $(".th-foot", panel);
const btn = document.createElement("button");
btn.type = "button"; btn.className = "btn th-btn"; btn.id = "thinking-toggle"; btn.title = "Thinking mode (T)";
btn.innerHTML = `<svg viewBox="0 0 20 20" aria-hidden="true"><path d="M7 15.5h6M8 18h4M10 2.5a5.2 5.2 0 0 0-3.2 9.3c.5.4.7.9.7 1.5h5c0-.6.2-1.1.7-1.5A5.2 5.2 0 0 0 10 2.5z"/></svg><span>Thinking</span>`;
$(".top-r")?.prepend(btn);
let thinkingOn = false;
function setThinking(v) {
  thinkingOn = v; panel.hidden = !v; btn.classList.toggle("on", v);
  try { localStorage.setItem("forge-thinking", v ? "1" : "0"); } catch { /* storage may be blocked */ }
  if (v) { renderAll(); dispatchEvent(new CustomEvent("forge:mood", { detail: { state: "thinking", ms: 1200 } })); }
}
btn.onclick = () => setThinking(!thinkingOn);
$(".th-x", panel).onclick = () => setThinking(false);
addEventListener("keydown", (e) => {
  if (e.key.toLowerCase() !== "t" || e.metaKey || e.ctrlKey || e.altKey) return;
  const a = document.activeElement;
  if (a && (a.tagName === "INPUT" || a.tagName === "TEXTAREA" || a.isContentEditable)) return;
  setThinking(!thinkingOn);
});

function row(t) {
  const d = document.createElement("div");
  d.className = "th-row" + (t.llm ? " llm" : "");
  d.innerHTML = `<span class="th-ag" data-agent="${esc(t.agent)}">${esc(t.agent)}</span><span class="th-tx">${esc(t.text)}</span><span class="th-lb">${t.llm ? "Llama 4 Maverick" : "code"}</span>`;
  $(".th-ag", d).onclick = () => fly(t.agent);
  return d;
}
function renderAll() {
  body.replaceChildren(...st.thoughts.slice(-80).map(row));
  body.scrollTop = body.scrollHeight;
  paintFoot();
}
function paintFoot() {
  const b = st.graph ? (st.graph.backends || []).join(", ") || "stub" : "";
  const av = st.graph && st.graph.agentverse;
  const where = av && av.registered.length ? "Reps and its optimizer are registered on Agentverse" : "not registered on Agentverse";
  foot.textContent = st.graph ? `Agents run as uAgents in one local Bureau, ${where}. Model backend answering here: ${b}.` : "";
}
function pushThought(t) {
  st.thoughts.push(t);
  if (st.thoughts.length > 400) st.thoughts.shift();
  if (thinkingOn) {
    body.append(row(t));
    while (body.children.length > 80) body.firstChild.remove();
    body.scrollTop = body.scrollHeight;
  }
}

function fly(agent) {
  const sel = TARGET[agent];
  if (sel) goTo(sel, { dwell: 3600, force: true });
}

// ---------------------------------------------------------------- events to dock steps
function closeOpen(status = "done") {
  if (st.open != null) { dock.step({ id: st.open, status }); st.open = null; }
}
function onThought(e) {
  if (e.run !== st.run) { st.run = e.run; }
  const t = { agent: e.agent || "forge", text: String(e.text || ""), llm: !!e.llm && !/^chat message from/i.test(String(e.text || "")), t: Date.now() };
  if (!t.text) return;
  pushThought(t);
  closeOpen();
  const id = "th" + ++st.n;
  st.open = id;
  dock.step({ id, kind: t.llm ? "reasoning" : "step", title: t.text.length > 90 ? t.text.slice(0, 88) + "…" : t.text, detail: agentLabel(t.agent, t.llm), status: "running" });
  if (!loading) { loading = true; t0 = performance.now(); clearInterval(loadIv); loadIv = setInterval(() => dock.loader(true, `Working · ${((performance.now() - t0) / 1000).toFixed(1)} s`), 200); }
  dock.setStatus(`${t.agent}: ${t.text.slice(0, 40)}`);
  if (thinkingOn) fly(t.agent);
}
addEventListener("forge:event", (ev) => {
  const e = ev.detail || {};
  if (e.kind === "thought") onThought(e);
  else if (["published", "reused", "rejected", "error"].includes(e.kind) && !e.child) {
    closeOpen(e.kind === "error" || e.kind === "rejected" ? "failed" : "done");
    loading = false; clearInterval(loadIv); dock.loader(false); dock.setStatus("");
    if (e.kind === "published" || e.kind === "reused") dock.setState("success");
  }
});

// ---------------------------------------------------------------- "What are you doing?"
async function graph() {
  if (!st.graph) { try { st.graph = await (await fetch("/api/agents")).json(); paintFoot(); } catch { /* server warming */ } }
  return st.graph;
}
graph();

async function compose(q) {
  const s = q.toLowerCase();
  const g = await graph();
  const names = g ? g.nodes.filter((n) => !n.skill).map((n) => n.id) : [];
  const named = names.find((n) => new RegExp(`\\b${n}\\b`).test(s));
  const r = railNow();
  const recent = st.thoughts.slice(-3);
  if (named) {
    const n = g.nodes.find((x) => x.id === named);
    fly(named);
    const said = [...st.thoughts].reverse().find((t) => t.agent === named);
    return `**${named}** is a uAgent in the Fetch.ai Bureau. ${n.description}${n.llm ? " It asks Llama 4 Maverick." : " It runs plain code, no model call."}${said ? ` Last thing it said: "${said.text}"` : " It has not said anything in this session yet."}`;
  }
  if (/model|llama|backend|who.*(answer|power)/.test(s) && g) {
    return `The agents that call the model (${g.nodes.filter((n) => n.llm).map((n) => n.id).join(", ")}) ask **Llama 4 Maverick**. The backend that actually answered in this process is **${(g.backends || []).join(", ") || "a stub"}**. The Fetch.ai agents run in a local Bureau${g.agentverse && g.agentverse.registered.length ? "; the task agent and the optimizer are registered on Agentverse" : " and are not registered on Agentverse"}.`;
  }
  if (/agent|fetch|uagent|who/.test(s) && g) {
    return `${g.nodes.filter((n) => !n.skill).length} uAgents pass typed messages: ${g.nodes.filter((n) => !n.skill).map((n) => n.id).join(", ")}. Open the **Agents** tab to watch the messages move.`;
  }
  if (/think|why|reason/.test(s)) {
    setThinking(true);
    return recent.length ? `I turned on Thinking. The last thing I did: **${recent.at(-1).agent}** said "${recent.at(-1).text}".` : "I turned on Thinking. Nothing has run yet, so it is empty until you give me a task.";
  }
  if (!loading) {
    return recent.length ? `Nothing is running right now. The last run ended with **${recent.at(-1).agent}**: "${recent.at(-1).text}". Give me another task and I will build it from a sentence.` : "Nothing yet. Type a task in the prompt and I will build the scene, collect demos, train, test on held-out scenes and read the failures.";
  }
  const at = r.live || r.done.at(-1) || "starting";
  const idx = r.all.indexOf(at);
  const next = idx >= 0 && r.all[idx + 1] ? ` Next: ${r.all[idx + 1]}.` : "";
  const lines = recent.map((t) => `- **${t.agent}**: ${t.text}`).join("\n");
  return `I am on the **${at}** step (${r.done.length} of ${r.all.length || STAGES.length} done).${next}${lines ? "\n\nWhat the agents just said:\n" + lines : ""}`;
}
async function answer(q) {
  const id = "q" + ++st.n;
  dock.setState("thinking");
  dock.answerStart(id);
  const text = await compose(q);
  for (const w of text.split(/(\s+)/)) { dock.answerAppend(id, w); if (w.trim()) await new Promise((r) => setTimeout(r, 18)); }
  const names = st.graph ? st.graph.nodes.filter((n) => !n.skill).map((n) => n.id) : [];
  const mentions = names.filter((n) => new RegExp(`\\b${n}\\b`, "i").test(text)).map((n) => n[0].toUpperCase() + n.slice(1)).slice(0, 4);
  const followups = loading ? ["Which model answered?", "What is the critic doing?"] : ["Which model answered?", "List the agents"];
  dock.answerEnd(id, { mentions, followups });
  dock.setState("idle");
  const at = railNow().live;
  const sel = at && TARGET[at.split(" ")[0]];
  if (sel) goTo(sel, { dwell: 3000 });
  else goHomeSoon(1200);
}
addEventListener("forge:ask-bot", () => {
  dock.prefill("What are you doing?");
  const f = $("form", dock.el);
  if (f) f.requestSubmit();
});
try { if (localStorage.getItem("forge-thinking") === "1") setThinking(true); } catch { /* storage may be blocked */ }
window.forgeDock = dock;
