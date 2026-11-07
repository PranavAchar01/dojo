// The Muse Spark + SubQ view: two model lanes side by side. Meta's Muse models write (task, plan, critic, optimizer
// proposals); SubQ Snap reads everything and picks. Each lane shows its calls in flight, its finished calls (backend that
// really answered, tokens, latency), session totals and its guard. Under the lanes, one row per optimizer round: both
// proposals, both SELECT scores, the winner, and the running tally. Every number comes from /api/llm or the live
// "llm_call" / "duel" events; nothing is drawn before it happens.
const $ = (s, r = document) => r.querySelector(s);
const ROOT = () => $("#view-agents");
const esc = (s) => String(s ?? "").replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
const num = (n) => Number(n || 0).toLocaleString();
const secs = (ms) => (ms >= 1000 ? (ms / 1000).toFixed(1) + " s" : Math.round(ms) + " ms");

let S = null; // last /api/llm snapshot, kept current by events
let ticker = 0;

function changeText(c) {
  if (!c) return "none";
  if (typeof c === "string") return c;
  const parts = Object.entries(c).filter(([k]) => !["reason", "diagnosis", "notes", "scene_notes"].includes(k));
  return parts.map(([k, v]) => `${k} ${typeof v === "object" ? JSON.stringify(v) : v}`).join(", ") || "none";
}

function laneHead(lane) {
  const m = S.models?.[lane] || {};
  if (lane === "meta") {
    const ok = m.available;
    const who = ok ? `${esc(m.name || "Muse Spark")} + ${esc(m.optimizer_name || "Muse Wonder")}` : `${esc(m.name || "Muse Spark")} unavailable`;
    const why = ok ? "Meta Model API, api.meta.ai" : esc(m.why || "no reason given") + (m.fallback ? `. Answering instead: ${esc(m.fallback)}` : "");
    const guard = m.cap_usd ? `$${Number(m.spent_usd || 0).toFixed(2)} of $${m.cap_usd} credit` : "";
    return `<div class="ln-h"><span class="ln-dot ${ok ? "on" : "off"}"></span><b>${who}</b><span class="ln-role">writes: task, plan, critic, optimizer proposals</span></div>
      <div class="ln-sub ${ok ? "" : "warn"}">${why}</div><div class="ln-guard">${guard}</div>`;
  }
  const ok = m.available;
  const why = ok ? `Decisions API, ${num(m.context)} token context` : esc(m.why || "no reason given");
  const guard = m.cap_tokens ? `${num(m.tokens_used)} of ${num(m.cap_tokens)} tokens` : "";
  return `<div class="ln-h"><span class="ln-dot ${ok ? "on" : "off"}"></span><b>${esc(m.name || "SubQ Snap")}${ok ? "" : " unavailable"}</b><span class="ln-role">reads everything and picks</span></div>
    <div class="ln-sub ${ok ? "" : "warn"}">${why}</div><div class="ln-guard">${guard}</div>`;
}

function laneTotals(lane) {
  const t = S.totals?.[lane];
  if (!t || !t.calls) return `<div class="ln-tot">No calls yet this session.</div>`;
  const avg = t.ms / t.calls;
  return `<div class="ln-tot"><span><b>${num(t.calls)}</b> calls</span><span><b>${num(t.tokens_in)}</b> in</span><span><b>${num(t.tokens_out)}</b> out</span><span><b>${secs(avg)}</b> avg</span>${t.errors ? `<span class="bad"><b>${t.errors}</b> failed</span>` : ""}</div>`;
}

function callRow(c, open) {
  const label = c.backend && c.backend !== `${c.lane === "subq" ? "subq" : "meta"}:${c.model}` ? c.backend : c.model;
  const right = open
    ? `<span class="ln-spin"></span><span class="ln-ms" data-t0="${c.t0}">${secs(c.elapsed_ms || 0)}</span>`
    : c.error
      ? `<span class="bad">${esc(c.error)}</span><span class="ln-ms">${secs(c.ms)}</span>`
      : `<span class="ln-tok">${num(c.tokens_in)} in${c.tokens_out ? ` / ${num(c.tokens_out)} out` : ""}</span><span class="ln-ms">${secs(c.ms)}</span>`;
  return `<div class="ln-call${open ? " open" : ""}${c.error ? " err" : ""}" title="${esc(c.note || "")}"><span class="ln-p">${esc(c.purpose)}</span><span class="ln-b">${esc(label)}</span>${right}</div>`;
}

function lane(lane) {
  const open = (S.open || []).filter((c) => c.lane === lane);
  const recent = (S.recent || []).filter((c) => c.lane === lane).slice(-14).reverse();
  return `<section class="ln ln-${lane}">${laneHead(lane)}${laneTotals(lane)}
    <div class="ln-list">${open.map((c) => callRow(c, true)).join("")}${recent.map((c) => callRow(c, false)).join("") || (open.length ? "" : `<div class="ln-empty">Run a task or an optimizer round and its calls appear here as they happen.</div>`)}</div></section>`;
}

function duels() {
  const rows = S.duels?.length ? S.duels : S.saved_duels || [];
  const saved = !S.duels?.length && rows.length;
  const t = S.duels?.length ? S.tally : S.saved_tally;
  const tally = t ? `<div class="du-tally"><span class="m">Muse Wonder <b>${t.meta || 0}</b></span><span class="q">SubQ <b>${t.subq || 0}</b></span><span>tie <b>${t.tie || 0}</b></span><span>agreed <b>${t.agreed || 0}</b></span></div>` : "";
  const body = rows.length
    ? rows.slice().reverse().map((d) => {
        const w = d.winner;
        const k = (x) => (x == null ? "" : `${x}/${d.n_select || "?"}`);
        return `<tr class="w-${esc(w)}"><td class="r">${esc(d.round)}</td>
          <td class="${w === "meta" ? "win" : ""}">${esc(changeText(d.meta?.change))}<i>${esc(d.meta?.label || d.meta?.backend || "")}</i></td><td class="k">${k(d.meta?.k)}</td>
          <td class="${w === "subq" ? "win" : ""}">${esc(changeText(d.subq?.change))}<i>${d.subq?.p != null ? "p " + Number(d.subq.p).toFixed(2) : esc(d.subq?.error || "")}</i></td><td class="k">${k(d.subq?.k)}</td>
          <td class="wn">${esc(w)}${d.admitted ? `<i>beat parent ${k(d.parent_k)}</i>` : d.parent_k != null ? `<i>parent ${k(d.parent_k)} kept</i>` : ""}</td></tr>`;
      }).join("")
    : `<tr><td colspan="6" class="du-empty">No optimizer rounds yet. In each round both models propose at once, both changes train from the same parent and are scored on the same SELECT scenes, in simulation.</td></tr>`;
  return `<section class="du"><div class="du-h"><b>Head to head, optimizer rounds</b>${saved ? `<span class="du-note">from the saved optimizer history</span>` : ""}${tally}</div>
    <div class="du-scroll"><table><thead><tr><th>round</th><th>Muse Wonder proposed</th><th>SELECT</th><th>SubQ picked</th><th>SELECT</th><th>winner</th></tr></thead><tbody>${body}</tbody></table></div></section>`;
}

function render() {
  const root = ROOT();
  if (!root || !S) return;
  root.innerHTML = `<div class="doc"><div class="doc-h"><h2>Muse Spark <em>+ SubQ</em></h2>
    <p>Both models work at the same moment, each on what it is built for. Muse writes; SubQ reads every trajectory, every past round and the whole skill library, and picks. Every number below is what the APIs returned.</p></div>
    <div class="ln-view">${lane("meta")}${lane("subq")}</div>${duels()}</div>`;
}

async function load() {
  const root = ROOT();
  if (!root) return;
  try { S = await (await fetch("/api/llm")).json(); } catch { S = { totals: {}, open: [], recent: [], duels: [], models: {} }; }
  root.dataset.ready = "1";
  render();
  clearInterval(ticker);
  ticker = setInterval(() => {
    if (ROOT()?.classList.contains("hidden")) return clearInterval(ticker);
    const now = Date.now() / 1000;
    document.querySelectorAll("#view-agents .ln-ms[data-t0]").forEach((n) => { n.textContent = secs((now - Number(n.dataset.t0)) * 1000); });
  }, 200);
}

function onCall(e) {
  if (e.phase === "start") {
    S.open = [...(S.open || []).filter((x) => x.id !== e.id), { ...e, elapsed_ms: 0 }];
  } else {
    S.open = (S.open || []).filter((x) => x.id !== e.id);
    S.recent = [...(S.recent || []), e].slice(-120);
    const t = (S.totals[e.lane] ||= { calls: 0, errors: 0, tokens_in: 0, tokens_out: 0, ms: 0, backends: {} });
    t.calls += 1; t.errors += e.error ? 1 : 0; t.tokens_in += e.tokens_in || 0; t.tokens_out += e.tokens_out || 0; t.ms += e.ms || 0;
  }
}

addEventListener("forge:event", (ev) => {
  const e = ev.detail || {};
  if (!S || !ROOT()?.dataset.ready) return;
  if (e.kind === "llm_call") onCall(e);
  else if (e.kind === "duel") {
    S.duels = [...(S.duels || []), e];
    const t = (S.tally ||= {});
    t[e.winner] = (t[e.winner] || 0) + 1;
  } else return;
  if (!ROOT().classList.contains("hidden")) render();
});
addEventListener("forge:view", (ev) => { if (ev.detail === "agents") load(); });
