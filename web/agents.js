// The Agents view: the Fetch.ai uAgents Bureau drawn as a graph. Nodes come from /api/agents, message pulses and the
// log rows from the live "agent_msg" events the server records for every uAgents message (agentbus.py). Nodes that call
// the model are labelled "uAgent · Llama 4 Maverick" and the footer says which backend actually answered.
const NS = "http://www.w3.org/2000/svg";
const $ = (s, r = document) => r.querySelector(s);
const mk = (t, a = {}, p) => { const n = document.createElementNS(NS, t); for (const k in a) n.setAttribute(k, a[k]); if (p) p.append(n); return n; };

// fixed layout on a 760 x 430 board (the pipeline left to right; the optimizer hangs under the gateway)
const POS = {
  gateway: [70, 70], task: [200, 70], router: [330, 70], planner: [330, 190], executor: [470, 190], scene: [470, 70],
  data: [600, 70], trainer: [600, 190], eval: [600, 310], critic: [470, 310], registry: [330, 310], optimizer: [70, 310],
};
const W = 112, H = 46;
let graph = null, nodes = {}, edges = [], svg = null, seq = 0, pulses = new Set();
const ROOT = () => $("#view-agents");

function esc(s) { return String(s).replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c])); }
const hhmmss = (ts) => new Date(ts * 1000).toLocaleTimeString([], { hour12: false });

async function load() {
  const root = ROOT();
  if (!root || root.dataset.ready) return;
  root.dataset.ready = "1";
  root.innerHTML = `<div class="doc"><div class="doc-h"><h2>Fetch.ai <em>agents</em></h2>
    <p>Every step of a task is a message between uAgents in one Bureau. The nodes marked Llama call the model; the others are plain code. Pulses are real messages as they are sent.</p></div>
    <div class="ag-view"><div class="ag-graph" id="ag-graph"></div>
    <div class="ag-side"><div class="ag-log" id="ag-log"><div class="ag-empty-log">No messages yet. Run a task and they appear here.</div></div><div class="ag-foot" id="ag-foot"></div></div></div></div>`;
  graph = await (await fetch("/api/agents")).json();
  drawGraph();
  const man = await (await fetch("/api/agents/manifests")).json().catch(() => ({}));
  const past = await (await fetch("/api/agents/messages?limit=40")).json().catch(() => []);
  past.forEach((m) => addRow(m, false));
  foot(man);
}

function drawGraph() {
  const host = $("#ag-graph");
  host.innerHTML = "";
  svg = mk("svg", { viewBox: "0 0 760 400", preserveAspectRatio: "xMidYMid meet" }, host);
  const g = mk("g", {}, svg);
  nodes = {};
  const ids = graph.nodes.filter((n) => POS[n.id]);
  edges = graph.edges.filter(([a, b]) => POS[a] && POS[b]).map(([a, b, label]) => {
    const [x1, y1] = POS[a], [x2, y2] = POS[b];
    const l = mk("line", { class: "ag-edge", x1, y1: y1 - 40 + 0, x2, y2: y2 - 40 + 0 }, g);
    l.dataset.a = a; l.dataset.b = b;
    return { a, b, label, l };
  });
  for (const n of ids) {
    const [x, y] = POS[n.id];
    const gg = mk("g", { class: "ag-node" + (n.llm ? " llm" : ""), transform: `translate(${x - W / 2},${y - 40 - H / 2})` }, g);
    gg.dataset.agent = n.id;
    mk("rect", { width: W, height: H }, gg);
    const t = mk("text", { x: W / 2, y: n.llm ? 20 : 27 }, gg); t.textContent = n.id;
    if (n.llm) { const s = mk("text", { class: "sub", x: W / 2, y: 35 }, gg); s.textContent = "uAgent · Llama 4 Maverick"; }
    const title = mk("title", {}, gg); title.textContent = n.description;
    nodes[n.id] = gg;
  }
  const sk = graph.nodes.filter((n) => n.skill).length;
  const note = mk("text", { x: 380, y: 392, "text-anchor": "middle", style: "font:500 11px var(--ui);fill:var(--ink3)" }, svg);
  note.textContent = `${ids.length} pipeline agents${sk ? ` · ${sk} saved skills are agents too (shown in the log)` : ""}`;
}

function foot(man) {
  const b = (graph.backends || []).join(", ") || "none";
  const nProto = Object.values(man).reduce((n, v) => n + v.length, 0);
  $("#ag-foot").innerHTML = `<b>Model:</b> the calling agents ask <b>Llama 4 Maverick</b>; the backend that answered in this process is <b>${esc(b)}</b>.<br>
    <b>Agentverse:</b> ${esc(graph.agentverse.note)}.<br><span class="ag-man">${nProto} protocol manifests published locally (chat protocol 0.3.0 on every agent, forge-optimize 1.0 on the optimizer).</span>`;
}

function pulse(a, b) {
  if (!svg || !POS[a] || !POS[b]) return;
  const [x1, y1] = POS[a], [x2, y2] = POS[b];
  const c = mk("circle", { class: "ag-pulse", r: 5, cx: x1, cy: y1 - 40 }, svg);
  const t0 = performance.now(), dur = 700;
  const step = (t) => {
    const k = Math.min(1, (t - t0) / dur), e = 1 - Math.pow(1 - k, 3);
    c.setAttribute("cx", x1 + (x2 - x1) * e); c.setAttribute("cy", y1 - 40 + (y2 - y1) * e);
    if (k < 1) requestAnimationFrame(step); else c.remove();
  };
  requestAnimationFrame(step);
  for (const id of [a, b]) { const n = nodes[id]; if (n) { n.classList.add("hot"); setTimeout(() => n.classList.remove("hot"), 900); } }
  const ed = edges.find((e) => (e.a === a && e.b === b) || (e.a === b && e.b === a));
  if (ed) { ed.l.classList.add("hot"); setTimeout(() => ed.l.classList.remove("hot"), 900); }
}

function addRow(m, live = true) {
  const box = $("#ag-log");
  if (!box) return;
  let tb = $("tbody", box);
  if (!tb) {
    box.innerHTML = `<table><thead><tr><th>Time</th><th>From to</th><th>Message</th><th>Summary</th></tr></thead><tbody></tbody></table>`;
    tb = $("tbody", box);
  }
  const tr = document.createElement("tr");
  if (live) tr.className = "new";
  tr.title = m.protocol;
  tr.innerHTML = `<td class="t">${hhmmss(m.ts)}</td><td class="who">${esc(m.sender)} to ${esc(m.receiver)}</td><td class="ty">${esc(m.type)}</td><td>${esc(m.summary)}</td>`;
  tb.prepend(tr);
  while (tb.children.length > 120) tb.lastChild.remove();
  if (m.seq > seq) seq = m.seq;
}

addEventListener("forge:event", (ev) => {
  const e = ev.detail || {};
  if (e.kind !== "agent_msg") return;
  if (ROOT()?.dataset.ready) {
    addRow(e);
    const to = e.receiver;
    pulse(e.sender, to);
  }
});
addEventListener("forge:view", (ev) => { if (ev.detail === "agents") load(); });
