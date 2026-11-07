/* Optimizer step: a language model improves a trained skill, one change per round, scored on the same scenes.
   Mounts into <div id="optimizer-root"></div>. Self-contained: needs only web/optimizer.css (injected if missing)
   and these endpoints: GET /api/optimizer/skills, POST /api/optimizer/start, GET /api/optimizer/runs,
   GET /api/optimizer/runs/{id}, GET /api/optimizer/runs/{id}/img/{path}.
   Everything shown comes from the run record; nothing here is simulated by the page. */
(() => {
  "use strict";
  const root = document.getElementById("optimizer-root");
  if (!root) return;
  if (!document.querySelector('link[href*="optimizer.css"]')) {
    const l = document.createElement("link");
    l.rel = "stylesheet";
    l.href = "/static/optimizer.css";
    document.head.appendChild(l);
  }
  const API = "/api/optimizer";
  const COLS = 10, TILE_W = 96, TILE_H = 72, MOMENTS = 8;
  const el = (tag, cls, text) => {
    const e = document.createElement(tag);
    if (cls) e.className = cls;
    if (text !== undefined) e.textContent = text;
    return e;
  };
  const pct = (v) => `${Math.round(v * 100)}%`;
  const get = (u) => fetch(u).then((r) => (r.ok ? r.json() : Promise.reject(new Error(u + " " + r.status))));

  const S = { run: null, sel: 0, runs: [], skills: [], poll: null, imgs: new Map(), tick: 0, follow: true };

  root.classList.add("opt");
  root.textContent = "";
  const head = el("div", "opt-head");
  const title = el("div", "opt-title");
  title.append(el("h2", "", "Optimize"), el("p", "opt-sub",
    "A language model reads where the skill fails and proposes one change to its training. The change is tried on the same 100 simulated scenes and kept only if success rises."));
  const controls = el("div", "opt-controls");
  const skillSel = el("select", "opt-select");
  skillSel.setAttribute("aria-label", "Skill to optimize");
  const roundsSel = el("select", "opt-select opt-narrow");
  roundsSel.setAttribute("aria-label", "Rounds");
  for (const n of [3, 4, 5, 6]) roundsSel.add(new Option(`${n} rounds`, n, false, n === 5));
  const go = el("button", "opt-btn", "Run optimizer");
  const pastSel = el("select", "opt-select");
  pastSel.setAttribute("aria-label", "Earlier runs");
  controls.append(skillSel, roundsSel, go, pastSel);
  head.append(title, controls);

  const badge = el("div", "opt-badge");
  const main = el("div", "opt-main");
  const gridCard = el("div", "opt-grid-card");
  const gridTop = el("div", "opt-grid-top");
  const gridLabel = el("div", "opt-grid-label");
  const sim = el("span", "opt-pill", "In simulation");
  gridTop.append(gridLabel, sim);
  const wrap = el("div", "opt-canvas-wrap");
  const canvas = el("canvas", "opt-canvas");
  canvas.width = COLS * TILE_W;
  canvas.height = COLS * TILE_H;
  const veil = el("div", "opt-veil");
  wrap.append(canvas, veil);
  const legend = el("div", "opt-legend");
  legend.append(el("span", "opt-key ok", "succeeded"), el("span", "opt-key bad", "failed"),
    el("span", "opt-note", "Each tile is one held-out scene, replayed from the recorded run."));
  gridCard.append(gridTop, wrap, legend);

  const stats = el("div", "opt-stats");
  const ticker = el("div", "opt-ticker");
  const changeBox = el("div", "opt-change");
  const metric = el("div", "opt-metric");
  const spark = el("div", "opt-spark");
  const fails = el("div", "opt-fails");
  const phase = el("div", "opt-phase");
  stats.append(ticker, changeBox, metric, spark, fails, phase);
  main.append(gridCard, stats);

  const replay = el("div", "opt-replay");
  const table = el("div", "opt-table");
  root.append(head, badge, main, replay, table);

  /* ---------- run list and start ---------- */
  async function loadLists() {
    try {
      S.skills = await get(`${API}/skills`);
      skillSel.textContent = "";
      for (const s of S.skills) skillSel.add(new Option(`${s.name} (${s.success_rate * s.n_eval}/${s.n_eval} on its check)`, s.id));
      if (!S.skills.length) skillSel.add(new Option("no skill with headroom", ""));
      S.runs = await get(`${API}/runs`);
      pastSel.textContent = "";
      pastSel.add(new Option(S.runs.length ? "Earlier runs" : "No earlier runs", ""));
      for (const r of S.runs) pastSel.add(new Option(`${r.skill.instruction}, ${r.rounds_done} rounds, ${r.status}`, r.id));
    } catch (e) {
      badge.textContent = "The optimizer endpoints are not available on this server.";
    }
  }
  go.onclick = async () => {
    if (!skillSel.value) return;
    go.disabled = true;
    try {
      const r = await fetch(`${API}/start`, {
        method: "POST", headers: { "content-type": "application/json" },
        body: JSON.stringify({ skill: skillSel.value, rounds: +roundsSel.value }),
      });
      const j = await r.json();
      if (!r.ok) throw new Error(j.detail || r.status);
      S.follow = true;
      open(j.id);
    } catch (e) {
      badge.textContent = `Could not start: ${e.message}`;
    }
    go.disabled = false;
  };
  pastSel.onchange = () => pastSel.value && open(pastSel.value);

  async function open(id) {
    S.imgs.clear();
    S.run = null;
    S.follow = true;
    await refresh(id);
    clearInterval(S.poll);
    S.poll = setInterval(() => refresh(id), 2000);
  }
  async function refresh(id) {
    let r;
    try { r = await get(`${API}/runs/${id}`); } catch { return; }
    const first = !S.run;
    S.run = r;
    const done = r.rounds.filter((x) => x.success !== undefined && x.success !== null);
    if (S.follow || first) S.sel = Math.max(0, done.length - 1);
    if (r.status === "done" || r.status === "error") { clearInterval(S.poll); S.poll = null; loadLists(); }
    render();
  }

  /* ---------- drawing ---------- */
  function sheet(round, k) {
    const key = `${S.run.id}/${round}/${k}`;
    if (!S.imgs.has(key)) {
      const im = new Image();
      im.src = `${API}/runs/${S.run.id}/img/r${round}/m${k}.jpg`;
      S.imgs.set(key, im);
    }
    return S.imgs.get(key);
  }
  function drawGrid() {
    const r = S.run;
    const ctx = canvas.getContext("2d");
    if (!r) { ctx.fillStyle = "#F1F4F7"; ctx.fillRect(0, 0, canvas.width, canvas.height); return; }
    const row = r.rounds[S.sel];
    const ready = row && row.success !== undefined && row.success !== null;
    const round = ready ? S.sel : Math.max(0, S.sel - 1);
    const rr = r.rounds[round];
    if (!rr || rr.success === undefined) return;
    const t = S.tick % (MOMENTS + 3);
    const k = Math.min(t, MOMENTS - 1);
    const im = sheet(round, k);
    ctx.fillStyle = "#F1F4F7";
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    if (im.complete && im.naturalWidth) ctx.drawImage(im, 0, 0);
    const show = t >= MOMENTS - 2;
    const n = rr.outcomes.length;
    for (let i = 0; i < n; i++) {
      const x = (i % COLS) * TILE_W, y = Math.floor(i / COLS) * TILE_H, ok = rr.outcomes[i];
      if (show) {
        ctx.fillStyle = ok ? "rgba(8,102,255,0.16)" : "rgba(217,48,37,0.22)";
        ctx.fillRect(x, y, TILE_W, TILE_H);
      }
      ctx.lineWidth = show ? 3 : 1;
      ctx.strokeStyle = show ? (ok ? "#0866FF" : "#D93025") : "rgba(28,43,51,0.12)";
      ctx.strokeRect(x + 1.5, y + 1.5, TILE_W - 3, TILE_H - 3);
    }
    veil.classList.toggle("on", !ready);
    veil.textContent = ready ? "" : `Round ${S.sel}: ${phaseText(r)}`;
  }
  setInterval(() => { S.tick++; drawGrid(); drawHero(); }, 220);

  function phaseText(r) {
    const n = r.n_scenes;
    return ({
      scoring: `scoring ${n} scenes`, rendering: "rendering the grid", collecting: "collecting demonstrations",
      training: "fine-tuning the policy", proposing: "Llama is reading the failures", replay: "rendering the replay",
    })[r.phase] || r.phase || "working";
  }

  /* ---------- stats card ---------- */
  function render() {
    const r = S.run;
    if (!r) return;
    const answered = r.llm && r.llm.answered;
    badge.className = "opt-badge " + (answered ? (r.llm.real_llama ? "llama" : "fallback") : "");
    badge.textContent = !answered
      ? "Llama proposes the change when the Meta Llama API key works; otherwise the model that answers is named here."
      : r.llm.real_llama
        ? `Llama proposes the change (${answered}).`
        : `Llama (planned), running on fallback: ${answered} proposed these changes.`;
    gridLabel.textContent = `${r.skill.instruction}, ${r.n_scenes} held-out scenes`;

    ticker.textContent = "";
    for (let i = 0; i <= r.n_rounds; i++) {
      const row = r.rounds[i];
      const b = el("button", "opt-chip", `R${i}`);
      b.setAttribute("aria-label", `Round ${i}`);
      const state = !row ? "pending" : row.status === "kept" ? "kept" : row.status === "reverted" ? "reverted" : row.success != null ? "start" : "active";
      b.classList.add(state);
      if (i === S.sel) b.classList.add("sel");
      b.disabled = !row || row.success == null;
      b.onclick = () => { S.sel = i; S.follow = false; render(); };
      ticker.append(b);
    }

    const row = r.rounds[S.sel];
    changeBox.textContent = "";
    if (row) {
      if (S.sel === 0) {
        changeBox.append(el("div", "opt-line", "Start: the skill as trained"));
      } else {
        const c = row.change;
        const l = el("div", "opt-line");
        l.append(el("b", "", "Changed: "), document.createTextNode(c.text));
        changeBox.append(l, el("div", "opt-reason", c.reason));
        const tag = row.status === "kept" ? "Kept" : row.status === "reverted" ? "Reverted" : "Trying";
        const prev = row.vs != null ? ` (best before: ${row.vs}/${r.n_scenes})` : "";
        changeBox.append(el("div", "opt-verdict " + row.status, tag + prev));
      }
    }

    metric.textContent = "";
    const done = row && row.success != null;
    const big = el("div", "opt-big", done ? `${row.success}/${row.n}` : "...");
    big.append(el("span", "opt-bigsub", " succeeded, in simulation"));
    metric.append(big);
    if (done) metric.append(el("div", "opt-ci", `95% CI ${pct(row.ci[0])} to ${pct(row.ci[1])}`));
    drawSpark(r);

    fails.textContent = "";
    if (done) {
      const tags = Object.entries(row.failures || {}).sort((a, b) => b[1] - a[1]);
      fails.append(el("div", "opt-fails-h", "Failures this round"));
      if (!tags.length) fails.append(el("div", "opt-tag", "none"));
      for (const [k, v] of tags) fails.append(el("div", "opt-tag", `${v} ${k}`));
    }
    phase.textContent = r.status === "running" ? `Now: ${phaseText(r)}${r.phase === "training" || r.phase === "scoring" ? `, round ${r.rounds.length - 1}` : ""}` :
      r.status === "done" ? `Finished in ${Math.round(r.seconds)} s. Best: round ${r.best_round}.` :
      r.status === "error" ? `Stopped: ${r.error}` : "";

    renderTable(r);
    renderReplay(r);
    drawGrid();
  }

  function drawSpark(r) {
    spark.textContent = "";
    const pts = r.rounds.filter((x) => x.success != null);
    if (pts.length < 1) return;
    const W = 300, H = 70, P = 8, N = Math.max(r.n_rounds, 1);
    const xs = (i) => P + (i / N) * (W - 2 * P);
    const lo = Math.min(...pts.map((p) => p.ci[0])), hi = Math.max(...pts.map((p) => p.ci[1]));
    const ys = (v) => H - P - ((v - lo) / Math.max(hi - lo, 0.01)) * (H - 2 * P);
    const ns = "http://www.w3.org/2000/svg";
    const svg = document.createElementNS(ns, "svg");
    svg.setAttribute("viewBox", `0 0 ${W} ${H}`);
    svg.setAttribute("role", "img");
    svg.setAttribute("aria-label", "Success rate by round with 95% intervals");
    const mk = (t, a) => { const e = document.createElementNS(ns, t); for (const k in a) e.setAttribute(k, a[k]); svg.append(e); return e; };
    pts.forEach((p) => {
      const i = r.rounds.indexOf(p);
      mk("line", { x1: xs(i), x2: xs(i), y1: ys(p.ci[0]), y2: ys(p.ci[1]), class: "ci" });
    });
    // the line follows the best kept score, the dots are every round's own score
    let best = -1, d = "";
    pts.forEach((p) => {
      const i = r.rounds.indexOf(p);
      if (p.success > best || i === 0) best = p.success;
      d += (d ? "L" : "M") + xs(i) + " " + ys(best / p.n);
    });
    mk("path", { d, class: "line" });
    pts.forEach((p) => {
      const i = r.rounds.indexOf(p);
      mk("circle", { cx: xs(i), cy: ys(p.success / p.n), r: i === S.sel ? 5 : 3.5, class: "dot " + (p.status || "start") });
    });
    spark.append(svg, el("div", "opt-spark-cap", "Line: best kept success. Dots: each round's own score. Bars: 95% CI."));
  }

  function renderTable(r) {
    table.textContent = "";
    const t = el("table");
    const th = el("tr");
    for (const h of ["Round", "Change", "Result", "Success", "95% CI"]) th.append(el("th", "", h));
    t.append(th);
    r.rounds.forEach((row, i) => {
      const tr = el("tr");
      tr.append(el("td", "", `R${i}`),
        el("td", "", i === 0 ? "the skill as trained" : row.change.text),
        el("td", "res " + (row.status || ""), i === 0 ? "start" : row.status),
        el("td", "", row.success != null ? `${row.success}/${row.n}` : ""),
        el("td", "", row.ci ? `${pct(row.ci[0])} to ${pct(row.ci[1])}` : ""));
      t.append(tr);
    });
    table.append(t);
  }

  /* ---------- before / after replay ---------- */
  const heroA = el("canvas", "opt-hero"), heroB = el("canvas", "opt-hero");
  heroA.width = heroB.width = 320; heroA.height = heroB.height = 240;
  const capA = el("div", "opt-hero-cap"), capB = el("div", "opt-hero-cap");
  const hv = { a: null, b: null, id: "" };
  function renderReplay(r) {
    if (!r.hero) { replay.textContent = ""; replay.classList.remove("on"); return; }
    if (hv.id !== r.id) {
      hv.id = r.id;
      for (const [k, f] of [["a", "hero_r0"], ["b", "hero_final"]]) {
        const im = new Image();
        im.src = `${API}/runs/${r.id}/img/${f}.jpg`;
        hv[k] = im;
      }
      replay.textContent = "";
      const h = r.hero;
      replay.append(el("h3", "", `One scene, the first try against the best round (scene ${h.scene + 1} of ${r.n_scenes})`));
      const pair = el("div", "opt-pair");
      const a = el("figure", "opt-fig"), b = el("figure", "opt-fig");
      capA.textContent = `R0, the skill as trained: ${h.a_success ? "succeeded" : "failed, " + h.a_failure}`;
      capB.textContent = `R${h.round_b}, after ${r.rounds.filter((x) => x.status === "kept").length} kept change${r.rounds.filter((x) => x.status === "kept").length === 1 ? "" : "s"}: ${h.b_success ? "succeeded" : "failed, " + h.b_failure}`;
      a.append(heroA, capA); b.append(heroB, capB);
      pair.append(a, b);
      replay.append(pair, el("p", "opt-note", "Same scene, same lighting, in simulation."));
      replay.classList.add("on");
    }
  }
  function drawHero() {
    if (!S.run || !S.run.hero) return;
    const h = S.run.hero, f = Math.min(S.tick % (h.frames + 4), h.frames - 1);
    for (const [c, im] of [[heroA, hv.a], [heroB, hv.b]]) {
      if (im && im.complete && im.naturalWidth) c.getContext("2d").drawImage(im, f * h.w, 0, h.w, h.h, 0, 0, h.w, h.h);
    }
  }

  loadLists().then(async () => {
    const live = S.runs.find((r) => r.status === "running") || S.runs[0];
    if (live) open(live.id);
  });
})();
