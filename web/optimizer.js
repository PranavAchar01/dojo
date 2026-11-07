/* Optimizer step: a language model improves a trained skill, one change per round, scored on the same scenes.
   ES module: export mount(root, ctx) puts a compact panel in #optimizer-root and opens the montage as an overlay.
   ctx.stage(name, state, metric) (optional) drives the rail step. Self-contained: needs only web/optimizer.css (injected if missing)
   and these endpoints: GET /api/optimizer/skills, POST /api/optimizer/start, GET /api/optimizer/runs,
   GET /api/optimizer/runs/{id}, GET /api/optimizer/runs/{id}/img/{path}, POST /api/optimizer/runs/{id}/{stop,test,resume}
   and, for the tile zoom, GET /api/optimizer/runs/{id}/scene and /frame (MuJoCo frames rendered by the server).
   Events: forge:tilezoom {run, scene, round, success} when a tile is opened, forge:tilezoomclose when it is closed.
   Everything shown comes from the run record; nothing here is simulated by the page. */
export function mount(root, ctx = {}) {
  if (!root || root.dataset.optMounted) return;
  root.dataset.optMounted = "1";
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

  let live = false;
  const S = { run: null, sel: 0, runs: [], skills: [], poll: null, imgs: new Map(), tick: 0, follow: true };

  root.classList.add("opt");
  root.textContent = "";
  const head = el("div", "opt-head");
  const title = el("div", "opt-title");
  title.append(el("h2", "", "Optimize"), el("p", "opt-sub",
    "A language model reads where the skill fails and proposes one change to its training. Each change is tried on 100 selection scenes and kept only if success rises there; a separate 100 held-out test scenes, never used to choose, show whether it really helped."));
  const controls = el("div", "opt-controls");
  const skillSel = el("select", "opt-select");
  skillSel.setAttribute("aria-label", "Skill to optimize");
  const roundsSel = el("select", "opt-select opt-narrow");
  roundsSel.setAttribute("aria-label", "Rounds");
  for (const n of [3, 4, 5, 6]) roundsSel.add(new Option(`${n} rounds`, n, false, n === 5));
  roundsSel.add(new Option("until stopped", 0));
  const go = el("button", "opt-btn", "Run optimizer");
  const pastSel = el("select", "opt-select");
  pastSel.setAttribute("aria-label", "Earlier runs");
  const stopBtn = el("button", "opt-btn opt-ghost", "Stop after this round");
  const testBtn = el("button", "opt-btn opt-ghost", "Check TEST now");
  const resumeBtn = el("button", "opt-btn opt-ghost", "Resume");
  for (const b of [stopBtn, testBtn, resumeBtn]) b.hidden = true;
  controls.append(skillSel, roundsSel, go, pastSel, stopBtn, testBtn, resumeBtn);
  head.append(title, controls);

  const badge = el("div", "opt-badge");
  const badge2 = el("div", "opt-badge");
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
    el("span", "opt-note", "Each tile is one selection scene, drawn in Blender from the recorded run. Click a tile to open it."));
  gridCard.append(gridTop, wrap, legend);

  const stats = el("div", "opt-stats");
  const ticker = el("div", "opt-ticker");
  const changeBox = el("div", "opt-change");
  const metric = el("div", "opt-metric");
  const spark = el("div", "opt-spark");
  const fails = el("div", "opt-fails");
  const phase = el("div", "opt-phase");
  const heldCard = el("div", "opt-held");
  const popCard = el("div", "opt-pop");
  stats.append(ticker, changeBox, metric, spark, fails, phase, popCard, heldCard);
  main.append(gridCard, stats);

  const replay = el("div", "opt-replay");
  const table = el("div", "opt-table");
  const summary = el("div", "opt-summary");
  const openBtn = el("button", "opt-btn opt-ghost", "Open the montage");
  const panel = el("div", "opt-panel");
  panel.append(head, badge, summary, openBtn);
  const overlay = el("div", "opt opt-overlay"); // lives on <body>: the app panel is a containing block for fixed children
  overlay.hidden = true;
  const ovHead = el("div", "opt-ov-head");
  const closeBtn = el("button", "opt-btn opt-ghost", "Close");
  ovHead.append(el("h3", "", "Optimizer montage"), closeBtn);
  overlay.append(ovHead, badge2, main, replay, table);
  root.append(panel);
  document.querySelectorAll("body > .opt-overlay").forEach((n) => n.remove());
  document.body.append(overlay);
  const setOpen = (v) => { if (!v && typeof closeZoom === "function") closeZoom(); overlay.hidden = !v; root.dataset.optOpen = v ? "1" : ""; };
  openBtn.onclick = () => setOpen(true);
  closeBtn.onclick = () => setOpen(false);
  document.addEventListener("keydown", (e) => e.key === "Escape" && setOpen(false));
  window.addEventListener("forge:newchat", () => setOpen(false));
  window.addEventListener("forge:optimizer", (e) => e.detail && e.detail.open && loadLists());

  function setBadge(cls, text) {
    for (const b of [badge, badge2]) { b.className = "opt-badge " + cls; b.textContent = text; }
  }
  function report(r) {
    const done = r.rounds.filter((x) => x.success != null);
    const first = done[0], best = r.rounds[r.best_round];
    if (first && best) {
      summary.textContent = `R0 ${first.success}/${first.n}, best R${r.best_round} ${best.success}/${best.n}, in simulation. ` +
        (r.status === "running" ? `Now: ${phaseText(r)}.` : r.status === "error" ? `Stopped: ${r.error}` : r.status === "stopped" ? "Stopped, can be resumed." : "Finished.");
    } else summary.textContent = r.status === "running" ? `Now: ${phaseText(r)}.` : "";
    if (ctx.stage) {
      const m = first && best ? `${first.success} to ${best.success}/${first.n}` : "";
      ctx.stage("optimizer", r.status === "running" ? "live" : r.status === "done" || r.status === "stopped" ? "done" : "fail", m, r.skill && r.skill.id);
    }
  }

  /* ---------- run list and start ---------- */
  async function loadLists() {
    try {
      S.skills = await get(`${API}/skills`);
      const keep = skillSel.value;
      skillSel.textContent = "";
      for (const s of S.skills) skillSel.add(new Option(`${s.name} (${s.success_rate * s.n_eval}/${s.n_eval} on its check)`, s.id));
      if (!S.skills.length) skillSel.add(new Option("no skill with headroom", ""));
      if (keep) skillSel.value = keep;
      S.runs = await get(`${API}/runs`);
      pastSel.textContent = "";
      pastSel.add(new Option(S.runs.length ? "Earlier runs" : "No earlier runs", ""));
      for (const r of S.runs) pastSel.add(new Option(`${r.skill.instruction}, ${r.rounds_done} rounds, ${r.status}`, r.id));
    } catch (e) {
      setBadge("", "The optimizer endpoints are not available on this server.");
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
      setOpen(true);
    } catch (e) {
      setBadge("", `Could not start: ${e.message}`);
    }
    go.disabled = false;
  };
  pastSel.onchange = () => pastSel.value && open(pastSel.value);
  const post = (path) => fetch(`${API}/runs/${S.run.id}/${path}`, { method: "POST" }).then((r) => r.json());
  stopBtn.onclick = async () => { stopBtn.disabled = true; await post("stop").catch(() => {}); stopBtn.disabled = false; };
  testBtn.onclick = async () => { testBtn.disabled = true; await post("test").catch(() => {}); testBtn.disabled = false; };
  resumeBtn.onclick = async () => { await post("resume").catch(() => {}); open(S.run.id); };

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
    if (first && [...skillSel.options].some((o) => o.value === r.skill.id)) skillSel.value = r.skill.id;
    const done = r.rounds.filter((x) => x.success !== undefined && x.success !== null);
    if (S.follow || first) S.sel = Math.max(0, done.length - 1);
    live = r.status === "running" || r.status === "starting";
    stopBtn.hidden = testBtn.hidden = !(live && r.mode === "perpetual");
    resumeBtn.hidden = !(!live && r.mode === "perpetual" && r.status !== "done" && r.engine && r.engine.next_round);
    if (!live) { clearInterval(S.poll); S.poll = null; loadLists(); }
    report(r);
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
      const col = ok ? "#0866FF" : "#D93025";
      ctx.lineWidth = 8;
      ctx.strokeStyle = col;
      ctx.strokeRect(x + 4, y + 4, TILE_W - 8, TILE_H - 8);
      ctx.fillStyle = col;
      ctx.beginPath();
      ctx.arc(x + TILE_W - 14, y + 14, 8, 0, Math.PI * 2);
      ctx.fill();
      ctx.lineWidth = 2;
      ctx.strokeStyle = "#fff";
      ctx.stroke();
    }
    veil.classList.toggle("on", !ready);
    veil.textContent = ready ? "" : `Round ${S.sel}: ${phaseText(r)}`;
  }
  setInterval(() => { S.tick++; drawGrid(); drawHero(); }, 220);

  function phaseText(r) {
    const n = r.n_scenes;
    return ({
      scoring: `scoring ${n} scenes`, rendering: "rendering the grid", collecting: "collecting demonstrations",
      testing: "scoring the held-out TEST scenes", control: "running the control arm",
      training: "fine-tuning the policy", proposing: "Muse Wonder and SubQ are both proposing", replay: "rendering the replay",
    })[r.phase] || r.phase || "working";
  }

  /* ---------- stats card ---------- */
  function render() {
    const r = S.run;
    if (!r) return;
    setBadge("llama", "Muse Wonder reads where the skill fails and writes a change; SubQ reads every round and picks one. Both train, the better one counts.");
    gridLabel.textContent = `${r.skill.instruction}, ${r.n_scenes} selection scenes`;

    ticker.textContent = "";
    for (let i = 0; i <= Math.max(r.n_rounds, r.rounds.length - 1); i++) {
      const row = r.rounds[i];
      const b = el("button", "opt-chip", `R${i}`);
      b.setAttribute("aria-label", `Round ${i}`);
      const state = !row ? "pending" : row.status === "kept" ? "kept" : row.status === "reverted" ? "reverted" : row.success != null ? "start" : "active";
      b.classList.add(state);
      if (i === S.sel) b.classList.add("sel");
      b.disabled = !row || row.success == null;
      b.onclick = () => { S.sel = i; S.follow = false; render(); window.dispatchEvent(new CustomEvent("forge:optround", { detail: { state } })); };
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
    phase.textContent = r.status === "stopped" ? `Stopped after ${r.rounds.length - 1} rounds in ${Math.round(r.seconds)} s. Best: round ${r.best_round}. Can be resumed.` : r.status === "running" ? `Now: ${phaseText(r)}${r.phase === "training" || r.phase === "scoring" ? `, round ${r.rounds.length - 1}` : ""}` :
      r.status === "done" ? `Finished in ${Math.round(r.seconds)} s. Best: round ${r.best_round}.` :
      r.status === "error" ? `Stopped: ${r.error}` : "";

    renderPop(r);
    renderHeld(r);
    renderTable(r);
    renderReplay(r);
    drawGrid();
  }

  function renderHeld(r) {
    heldCard.textContent = "";
    const te = r.test && r.test.compare;
    if (!te) return;
    const ci = (c) => `${pct(c[0])} to ${pct(c[1])}`;
    const sg = (v) => (v > 0 ? "+" : v < 0 ? "-" : "") + Math.abs(Math.round(v * 100));
    heldCard.append(el("div", "opt-held-h", "Held-out test"),
      el("div", "opt-held-sub", `${te.n} scenes the optimizer never saw`),
      el("div", "opt-held-row", `R0 ${te.before.k}/${te.n} (${ci(te.before.ci)}) to final ${te.after.k}/${te.n} (${ci(te.after.ci)})`));
    const c = r.control && r.control.vs_r0;
    if (c) heldCard.append(el("div", "opt-held-row", `Random changes, same rounds: ${c.before.k} to ${c.after.k}`));
    const p = te.mcnemar.p;
    heldCard.append(el("div", "opt-held-line", `${sg(te.delta)} on held-out, ${p < 0.05 ? "significant" : "not yet significant"} (p = ${p.toFixed(2)})`));
    const pk = r.test.peeks || [];
    if (pk.length) {
      heldCard.append(el("div", "opt-held-row", `TEST looked at ${pk.length} time${pk.length === 1 ? "" : "s"}; the number shown is the latest, not the highest. ` +
        pk.map((q) => `R${q.round}: ${q.optimizer}${q.control != null ? ` vs ${q.control}` : ""}`).join(", ")));
    }
  }

  function renderPop(r) {
    popCard.textContent = "";
    const e = r.engine;
    if (!e || !e.population) return;
    popCard.append(el("div", "opt-held-h", "Population"),
      el("div", "opt-held-sub", `${e.population.length} policies kept across ${e.islands} islands. Framework: OpenEvolve population database. Proposals: Muse Wonder and SubQ Snap, head to head.`));
    for (const p of e.population.slice().sort((a, b) => b.success - a.success).slice(0, 5)) {
      const l = el("button", "opt-pop-row" + (p.round === r.best_round ? " best" : ""));
      const bits = [...Object.entries(p.recipe.changed_settings || {}).map(([k, v]) => `${k} ${typeof v === "number" ? +v.toPrecision(2) : v}`), ...(p.recipe.extra_demos || [])];
      l.textContent = `R${p.round}  ${Math.round(p.success)}/${r.n_scenes}  ${bits.length ? bits.join(", ") : "as trained"}`;
      l.onclick = () => { if (r.rounds[p.round]) { S.sel = p.round; S.follow = false; render(); } };
      popCard.append(l);
    }
  }

  function drawSpark(r) {
    spark.textContent = "";
    const pts = r.rounds.filter((x) => x.success != null);
    if (pts.length < 1) return;
    const W = 300, H = 70, P = 8, N = Math.max(r.n_rounds, r.rounds.length - 1, 1);
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
    const te = r.test && r.test.compare;
    if (te) {
      const box = el("div", "opt-test");
      const ci = (c) => `${pct(c[0])} to ${pct(c[1])}`;
      box.append(el("b", "", "Held-out TEST scenes (never seen by the optimizer)"));
      box.append(el("div", "", `R0 ${te.before.k}/${te.n} (${ci(te.before.ci)}), final ${te.after.k}/${te.n} (${ci(te.after.ci)}). Paired change ${te.delta >= 0 ? "+" : ""}${pct(te.delta)} (95% ${ci(te.delta_ci)}), McNemar p ${te.mcnemar.p}: ${te.mcnemar.fixed} fixed, ${te.mcnemar.broken} broken.`));
      const c = r.control && r.control.vs_r0;
      if (c) {
        const kind = r.control.kind === "none" ? "no change, same training" : "a random change each round";
        box.append(el("div", "", `Control (${kind}): ${c.after.k}/${c.n} (${ci(c.after.ci)}), change ${c.delta >= 0 ? "+" : ""}${pct(c.delta)} (95% ${ci(c.delta_ci)}), McNemar p ${c.mcnemar.p}. Optimizer vs control on the same scenes: ${r.control.optimizer_vs_control.delta >= 0 ? "+" : ""}${pct(r.control.optimizer_vs_control.delta)}, p ${r.control.optimizer_vs_control.mcnemar.p}.`));
      }
      if (r.blender && r.blender.engine === "blender-eevee") box.append(el("div", "opt-sub", `Drawn in Blender (${r.blender.renders} renders on up to ${r.blender.workers} workers). Physics is MuJoCo; Blender only renders it.`));
      table.append(box);
    }
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


  /* ---------- tile zoom: the recorded MuJoCo rollout of one scene, replayed live ---------- */
  const Z = { open: false, scene: 0, round: 0, info: null, az: 148, el: -34, dist: 0.46, i: 0, hold: 0, playing: true,
    cache: new Map(), pending: new Set(), drag: null, token: 0 };
  const zoom = el("div", "opt-zoom");
  zoom.hidden = true;
  const zCard = el("div", "opt-zoom-card");
  zCard.setAttribute("role", "dialog");
  zCard.setAttribute("aria-label", "Scene replay");
  const zHead = el("div", "opt-zoom-head");
  const zTitle = el("h3", "", "");
  const zClose = el("button", "opt-btn opt-ghost", "Close");
  zHead.append(zTitle, zClose);
  const zToggle = el("div", "opt-zoom-toggle");
  const zBody = el("div", "opt-zoom-body");
  const zView = el("div", "opt-zoom-view");
  const zCanvas = el("canvas", "opt-zoom-canvas");
  zCanvas.width = 480; zCanvas.height = 360;
  zCanvas.tabIndex = 0;
  zCanvas.setAttribute("aria-label", "MuJoCo replay, drag to orbit, scroll to zoom");
  const zBadge = el("span", "opt-zoom-badge", "");
  const zMsg = el("div", "opt-zoom-msg", "");
  zView.append(zCanvas, zBadge, zMsg);
  const zBar = el("div", "opt-zoom-bar");
  const zPlay = el("button", "opt-btn opt-ghost", "Pause");
  const zScrub = el("input"); zScrub.type = "range"; zScrub.min = 0; zScrub.max = 1; zScrub.value = 0; zScrub.setAttribute("aria-label", "Frame");
  const zOrbitReset = el("button", "opt-btn opt-ghost", "Reset view");
  zBar.append(zPlay, zScrub, zOrbitReset);
  const zLeft = el("div", "opt-zoom-left"); zLeft.append(zView, zBar, el("div", "opt-note", "Live MuJoCo replay of the recorded rollout, drawn by the server at 30 frames a second. Drag to orbit, scroll to zoom. In simulation."));
  const zInfo = el("div", "opt-zoom-info");
  zBody.append(zLeft, zInfo);
  zCard.append(zHead, zToggle, zBody);
  zoom.append(zCard);
  overlay.append(zoom);

  const zurl = (i) => `${API}/runs/${S.run.id}/frame?round=${Z.round}&scene=${Z.scene}&i=${i}&w=480&h=360&az=${Z.az.toFixed(1)}&el=${Z.el.toFixed(1)}&d=${Z.dist.toFixed(3)}`;
  const zkey = (i) => `${Z.round}|${Z.scene}|${Z.az.toFixed(0)}|${Z.el.toFixed(0)}|${Z.dist.toFixed(2)}|${i}`;
  function zWant(i) {
    const k = zkey(i);
    if (Z.cache.has(k) || Z.pending.has(k) || Z.pending.size >= 6) return;
    Z.pending.add(k);
    const tok = Z.token;
    fetch(zurl(i)).then((r) => (r.ok ? r.blob() : Promise.reject(r.status))).then(createImageBitmap)
      .then((bm) => { if (tok === Z.token) { Z.cache.set(k, bm); if (Z.cache.size > 500) Z.cache.delete(Z.cache.keys().next().value); } })
      .catch(() => {}).finally(() => Z.pending.delete(k));
  }
  function zTick() {
    if (!Z.open || !Z.info || !Z.info.available) return;
    const n = Z.info.frames;
    for (let d = 0; d < 6; d++) zWant((Z.i + d) % n);
    const bm = Z.cache.get(zkey(Z.i));
    if (!bm) return;
    zCanvas.getContext("2d").drawImage(bm, 0, 0, zCanvas.width, zCanvas.height);
    zScrub.value = Z.i;
    if (Z.playing) {
      if (Z.i === n - 1 && Z.hold < 20) Z.hold++; // rest on the last frame for a moment, then loop
      else { Z.hold = 0; Z.i = (Z.i + 1) % n; }
    }
  }
  setInterval(zTick, 1000 / 30);

  function zSetPlaying(v) { Z.playing = v; zPlay.textContent = v ? "Pause" : "Play"; }
  zPlay.onclick = () => zSetPlaying(!Z.playing);
  zScrub.oninput = () => { zSetPlaying(false); Z.i = +zScrub.value; };
  zOrbitReset.onclick = () => { Z.az = 148; Z.el = -34; Z.dist = 0.46; Z.cache.clear(); Z.token++; };
  zCanvas.addEventListener("pointerdown", (e) => { Z.drag = { x: e.clientX, y: e.clientY }; zCanvas.setPointerCapture(e.pointerId); });
  zCanvas.addEventListener("pointermove", (e) => {
    if (!Z.drag) return;
    Z.az -= (e.clientX - Z.drag.x) * 0.4;
    Z.el = Math.min(-5, Math.max(-85, Z.el - (e.clientY - Z.drag.y) * 0.3));
    Z.drag = { x: e.clientX, y: e.clientY };
  });
  zCanvas.addEventListener("pointerup", () => { Z.drag = null; });
  zCanvas.addEventListener("wheel", (e) => { e.preventDefault(); Z.dist = Math.min(1.2, Math.max(0.2, Z.dist * (1 + Math.sign(e.deltaY) * 0.08))); }, { passive: false });

  async function openZoom(scene, round) {
    const r = S.run;
    if (!r) return;
    Z.open = true; Z.scene = scene; Z.round = round; Z.info = null; Z.i = 0; Z.hold = 0; Z.token++;
    Z.cache.clear(); Z.pending.clear();
    zSetPlaying(true);
    zoom.hidden = false;
    zTitle.textContent = `Scene ${scene + 1} of ${r.n_scenes}, round ${round}`;
    zMsg.textContent = "Loading the rollout...";
    zMsg.classList.add("on");
    window.dispatchEvent(new CustomEvent("forge:tilezoom", { detail: { run: r.id, scene, round, success: !!(r.rounds[round] && r.rounds[round].outcomes[scene]) } }));
    let info;
    try { info = await get(`${API}/runs/${r.id}/scene?round=${round}&scene=${scene}`); } catch { zMsg.textContent = "That scene could not be loaded."; return; }
    if (!Z.open || Z.scene !== scene || Z.round !== round) return;
    Z.info = info;
    zScrub.max = Math.max(info.frames - 1, 0);
    zMsg.classList.toggle("on", !info.available);
    if (!info.available) zMsg.textContent = info.source;
    renderZoomInfo(info);
  }
  function closeZoom() {
    if (!Z.open) return;
    Z.open = false; Z.token++;
    zoom.hidden = true;
    window.dispatchEvent(new CustomEvent("forge:tilezoomclose", { detail: { run: S.run && S.run.id } }));
  }
  zClose.onclick = closeZoom;
  zoom.addEventListener("click", (e) => { if (e.target === zoom) closeZoom(); });
  document.addEventListener("keydown", (e) => {
    if (e.key === "Escape" && Z.open) { e.stopImmediatePropagation(); closeZoom(); }
  }, true);

  function renderZoomInfo(i) {
    const r = S.run;
    zToggle.textContent = "";
    const opts = [[0, `Before (R0)`]];
    if (S.sel !== 0 && S.sel !== i.best_round) opts.push([S.sel, `This round (R${S.sel})`]);
    opts.push([i.best_round, i.best_round === 0 ? "After (no kept change yet)" : `After (best, R${i.best_round})`]);
    for (const [rd, label] of opts) {
      const b = el("button", "opt-seg" + (rd === Z.round ? " on" : ""), label);
      b.disabled = rd === 0 && label.startsWith("After");
      b.onclick = () => { if (rd !== Z.round) openZoom(Z.scene, rd); };
      zToggle.append(b);
    }
    zBadge.textContent = i.success ? "succeeded" : "failed";
    zBadge.className = "opt-zoom-badge " + (i.success ? "ok" : "bad");
    zInfo.textContent = "";
    const line = (k, v, cls) => { const d = el("div", "opt-zoom-line" + (cls ? " " + cls : "")); d.append(el("b", "", k), document.createTextNode(v)); zInfo.append(d); };
    line("Scene ", `${i.scene + 1} of ${i.n_scenes}, round ${i.round}${i.round === 0 ? " (the skill as trained)" : ""}`);
    line("Outcome ", i.success ? "succeeded" : `failed${i.failure ? ": " + i.failure : ""}`, i.success ? "ok" : "bad");
    if (i.steps) line("Length ", `${i.steps} control steps, ${(i.frames / i.fps).toFixed(1)} s`);
    const c = i.change;
    if (i.round > 0) {
      line("Change this round ", c.text);
      if (c.reason) zInfo.append(el("div", "opt-reason", c.reason));
      zInfo.append(el("div", "opt-verdict " + (c.status || ""), `${c.status === "kept" ? "Kept" : "Reverted"}: ${c.round_success}/${i.n_scenes} on the selection scenes (parent ${c.vs})`));
    }
    const pair = el("div", "opt-zoom-pair");
    const a = i.before, b = i.after;
    pair.textContent = `Same scene: R0 ${a.success ? "succeeded" : "failed"}${b.round !== 0 ? `, best R${b.round} ${b.success ? "succeeded" : "failed"}` : ""}.`;
    zInfo.append(pair);
    const agent = el("div", "opt-zoom-agent");
    agent.append(el("div", "opt-zoom-agent-h", "Muse Wonder + SubQ"));
    if (i.note) {
      agent.append(el("div", "opt-zoom-agent-t", `"${i.note.text}"`),
        el("div", "opt-note", i.note.kind === "scene" ? `Its note on this scene, written in round ${i.note.round} when it chose that change.` : `Its diagnosis of the failure pattern in round ${i.note.round}, not a note on this scene alone.`));
    } else agent.append(el("div", "opt-note", "It has not written about this scene in this run."));
    zInfo.append(agent, el("div", "opt-note", `Source: ${i.source}.`));
  }

  canvas.style.cursor = "pointer";
  canvas.addEventListener("click", (e) => {
    const r = S.run;
    if (!r) return;
    const b = canvas.getBoundingClientRect();
    const x = Math.floor(((e.clientX - b.left) / b.width) * COLS), y = Math.floor(((e.clientY - b.top) / b.height) * COLS);
    const scene = y * COLS + x;
    const row = r.rounds[S.sel];
    const round = row && row.success != null ? S.sel : Math.max(0, S.sel - 1);
    if (scene >= 0 && scene < r.n_scenes) openZoom(scene, round);
  });
  window.addEventListener("forge:newchat", closeZoom);

  loadLists().then(async () => {
    const live = S.runs.find((r) => r.status === "running") || S.runs.find((r) => r.default) || S.runs[0];
    if (live) open(live.id);
  });
}
