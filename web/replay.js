// Replay of a Library card: the skill's recorded development fed through the same handlers a live run uses
// (window.forge.replay.feed), so the step strip, the rail, the loss curve, the eval tally, the mascot and the arm
// all move as they did. It reads /api/skills/<id>/replay-events and writes nothing.
const $ = (s) => document.querySelector(s);
const F = () => window.forge;
const LEAD = 0.4;
const TAIL = 1;
const STAGE_NAME = { task: "Task", router: "Route", scene: "Scene", data: "Demos", trainer: "Train", eval: "Eval", critic: "Critic", optimizer: "Optimize", registry: "Publish" };

let bar = null;
let R = null; // the replay in flight

function buildBar() {
  bar = document.createElement("div");
  bar.className = "replay-bar";
  bar.hidden = true;
  bar.setAttribute("data-keepclear", "");
  bar.setAttribute("role", "group");
  bar.setAttribute("aria-label", "Replay of a recorded run");
  bar.innerHTML = `<span class="rp-tag">Replay</span>
    <button class="rp-play" aria-label="Pause">❚❚</button>
    <span class="rp-speed"><button data-s="1">1x</button><button data-s="2">2x</button><button data-s="4">4x</button></span>
    <input class="rp-scrub" type="range" min="0" max="1000" value="0" step="1" aria-label="Replay position">
    <span class="rp-time">0:00</span>
    <button class="rp-x" aria-label="Close replay" title="Close replay">×</button>
    <div class="rp-note"></div>`;
  $("#view-sim").append(bar);
  bar.querySelector(".rp-play").onclick = () => R && toggle();
  bar.querySelectorAll(".rp-speed button").forEach((b) => (b.onclick = () => R && setSpeed(+b.dataset.s)));
  bar.querySelector(".rp-x").onclick = close;
  const sc = bar.querySelector(".rp-scrub");
  sc.addEventListener("input", () => R && seek((sc.value / 1000) * R.total));
}

// recording seconds -> playback seconds. Waiting is capped (nothing on screen changes while it runs), then the whole
// timeline is time-remapped so a build plays in about GOAL seconds at 1x (a little longer for runs with very many events,
// so every event still gets a beat). Every event keeps its order and is shown; 2x and 4x are relative to this pacing.
const GOAL = 14;
const GAP = 1.5;
function compress(times) {
  const raw = [0];
  times.forEach((t, i) => { if (i) raw.push(raw[i - 1] + Math.min(Math.max(t - times[i - 1], 0), GAP)); });
  const span = raw[raw.length - 1] || 1;
  const goal = GOAL + Math.max(0, times.length - 120) * 0.05;
  const k = Math.min(1, goal / span);
  const at = raw.map((v) => LEAD + v * k);
  return { at, total: at[at.length - 1] + TAIL };
}

const fmt = (s) => `${Math.floor(s / 60)}:${String(Math.floor(s % 60)).padStart(2, "0")}`;

function note(d) {
  const miss = Object.entries(d.stages).filter(([, v]) => v === "not_recorded").map(([k]) => STAGE_NAME[k]);
  const part = Object.entries(d.stages).filter(([, v]) => v === "partial").map(([k]) => STAGE_NAME[k]);
  const rr = d.rerun;
  const pub = rr && `${Math.round(rr.original.success_rate * rr.original.n_eval)}/${rr.original.n_eval}`;
  const src = d.source === "rerun"
    ? `Re-run ${rr.date}: ${rr.succeeded}/${rr.n} on held-out scenes in simulation, captured live. Published result: ${pub}.`
    : d.source === "live"
      ? "Captured live while this skill was trained."
      : "Rebuilt from what was saved on disk, not captured live. The pacing here is for playback; the numbers are the saved ones.";
  const bits = [`<b>${src}</b>`];
  if (part.length) bits.push(`Partly recorded: ${part.join(", ")}.`);
  if (miss.length) bits.push(`Not recorded: ${miss.join(", ")}.`);
  return bits.join(" ");
}

// ---- the RSI example: the optimizer run that improved this tile, played with the replay.
// The grid is the same 100 MuJoCo scenes the Optimize view shows (its own rendered sheets), moving together; the
// replay's position steps it through the rounds, R0 to the final held-out result. Reads /api/optimizer/runs/{id} only.
const pctN = (v) => `${Math.round(v * 100)}%`;
const mkEl = (tag, cls, text) => { const e = document.createElement(tag); if (cls) e.className = cls; if (text !== undefined) e.textContent = text; return e; };
const MOMENTS = 8, COLS = 10, TW = 96, TH = 72;
let rsiUI = null;
// the panel is the next chapter: built at open, hidden, and shown only while the replay sits at its end
function rsiSync() {
  const U = rsiUI; if (!U || U.solo || !R) return;
  const end = R.pos >= R.total;
  if (!end) U.dismissed = false;
  const want = end && !U.dismissed;
  if (want === !U.box.hidden) return;
  U.box.hidden = !want;
  document.body.classList.toggle("rsi-open", want);
  if (want) { U.place(); U.draw(); }
}
function rsiClose() { document.body.classList.remove("rsi-open"); if (rsiUI) { clearInterval(rsiUI.timer); removeEventListener("resize", rsiUI.place); rsiUI.box.remove(); rsiUI = null; } }
async function rsiPanel(rsi, solo = false) {
  const r = await fetch(`/api/optimizer/runs/${rsi.run}`).then((x) => (x.ok ? x.json() : null)).catch(() => null);
  if (!r || (!solo && (!R || !bar)) || r.status === "running") return;
  rsiClose();
  const rows = r.rounds.filter((x) => x.success != null && x.outcomes);
  if (rows.length < 2) return;
  const API = `/api/optimizer/runs/${rsi.run}`;
  const best = r.rounds[r.best_round] || rows[rows.length - 1];
  const c = r.test && r.test.compare, h = r.hero;
  const box = mkEl("section", "rp-rsi");
  box.hidden = !solo;
  if (solo) box.classList.add("rsi-solo");
  box.dataset.keepclear = "8";
  box.setAttribute("data-keepclear", "");
  box.setAttribute("aria-label", "Recursive self-improvement example");
  const top = mkEl("header", "rsi-top");
  top.append(mkEl("span", "rsi-tag", "RSI"), mkEl("h3", "", `The optimizer lifted this skill from ${rows[0].success} to ${best.success} of ${best.n} scenes`));
  const x = mkEl("button", "rsi-x", "×");
  x.setAttribute("aria-label", "Hide the RSI example");
  x.onclick = () => { if (solo) return rsiClose(); rsiUI.dismissed = true; rsiSync(); };
  top.append(x);
  const main = mkEl("div", "rsi-main");
  const left = mkEl("div", "rsi-left");
  const cv = mkEl("canvas", "rsi-grid");
  cv.width = COLS * TW; cv.height = 10 * TH;
  const sims = mkEl("span", "rsi-sims", `${rows[0].n} simulations, in parallel`);
  const wrap = mkEl("div", "rsi-gridwrap");
  const stamp = mkEl("span", "rsi-stamp");
  wrap.append(cv, sims, stamp);
  const cap = mkEl("div", "rsi-cap");
  const capR = mkEl("b", "", ""), capT = mkEl("span", "", "");
  cap.append(capR, capT);
  const meter = mkEl("div", "rsi-meter");
  const big = mkEl("div", "rsi-big");
  const bigN = mkEl("b", "", String(rows[0].success)), bigD = mkEl("span", "", `/ ${rows[0].n} succeeded`);
  const chip = mkEl("span", "rsi-chip", "");
  big.append(bigN, bigD, chip);
  const bar2 = mkEl("div", "rsi-bar"), fill = mkEl("i", "", ""), base = mkEl("u", "", "");
  bar2.append(fill, base);
  meter.append(big, bar2);
  left.append(wrap, cap, meter);
  const right = mkEl("div", "rsi-right");
  const list = mkEl("ol", "rsi-rounds");
  const lis = rows.map((rw, i) => {
    const li = mkEl("li", i ? rw.status || "" : "start");
    li.append(mkEl("span", "rsi-r", `R${i}`), mkEl("span", "rsi-c", i ? rw.change.text : "the skill as trained"), mkEl("span", "rsi-v", i ? (rw.status === "kept" ? "kept" : "reverted") : "start"), mkEl("b", "rsi-n", `${rw.success}`));
    li.onclick = () => rsiUI && rsiUI.pin(i);
    list.append(li);
    return li;
  });
  right.append(mkEl("div", "rsi-h", "Round by round"), list);
  const strips = [];
  if (h) {
    const pair = mkEl("div", "rsi-pair");
    for (const [lab, file, cp] of [["Before", "hero_r0.jpg", h.a_failure || "failed"], ["After", "hero_final.jpg", h.b_success ? "seated" : "failed"]]) {
      const f = mkEl("figure", "");
      const im = mkEl("div", "rsi-strip");
      im.style.backgroundImage = `url(${API}/img/${file})`;
      strips.push(im);
      f.append(im, mkEl("figcaption", "", ""));
      f.lastChild.append(mkEl("b", "", lab), document.createTextNode(` ${cp}`));
      pair.append(f);
    }
    right.append(mkEl("div", "rsi-h", `Scene ${h.scene + 1}, the first try against the best round`), pair);
  }
  const foot = mkEl("div", "rsi-foot");
  if (c) foot.append(mkEl("p", "", `${pctN(c.after.k / c.n)} vs ${pctN(c.before.k / c.n)} on ${c.n} fresh scenes, in simulation. A small gain, not statistically significant (McNemar p ${c.mcnemar.p}).`));
  const open = mkEl("button", "rsi-open", "Open the full Optimize view");
  open.onclick = () => window.forgeOptimizer?.show(rsi.run);
  if (!solo) foot.append(open);
  right.append(foot);
  main.append(left, right);
  box.append(top, main);
  document.body.append(box);
  const place = () => {
    if (solo) { box.style.top = "84px"; box.style.left = "50%"; box.style.width = `${Math.min(900, window.innerWidth - 32)}px`; box.style.maxHeight = `${Math.max(240, window.innerHeight - 112)}px`; return; }
    // under the replay bar, centred in the stage column (never over the side rail)
    const b = bar.getBoundingClientRect(), v = $("#view-sim").getBoundingClientRect();
    const gap = v.width >= 1000 ? 372 : 16; // the stage's floating arm card keeps the left edge
    const w = Math.min(900, v.width - gap - 16);
    box.style.width = `${w}px`;
    box.style.top = `${Math.round(b.bottom + 12)}px`;
    box.style.left = `${Math.round(v.left + gap + w / 2)}px`;
    box.style.maxHeight = `${Math.max(240, window.innerHeight - b.bottom - 28)}px`;
  };
  place();
  addEventListener("resize", place);
  if (R) R.rsi = rsi;

  const imgs = new Map();
  const sheet = (rd, k) => { const key = `${rd}/${k}`; if (!imgs.has(key)) { const im = new Image(); im.src = `${API}/img/r${rd}/m${k}.jpg`; imgs.set(key, im); } return imgs.get(key); };
  for (let k = 0; k < MOMENTS; k++) sheet(0, k);
  const g = cv.getContext("2d");
  const U = { box, tick: 0, round: 0, frac: 0, solo, pinned: null, count: rows[0].success, pin(i) { U.pinned = U.pinned === i ? null : i; draw(); } };
  U.place = place;
  rsiUI = U;
  const roundOf = () => (U.pinned != null ? U.pinned : Math.min(rows.length - 1, Math.floor(U.frac * (rows.length + 1))));
  function draw() {
    const i = roundOf(), rw = rows[i];
    const t = U.tick % (MOMENTS + 4), k = Math.min(t, MOMENTS - 1), settled = t >= MOMENTS - 2;
    g.fillStyle = "#eef1f4"; g.fillRect(0, 0, cv.width, cv.height);
    const im = sheet(rw.round, k);
    if (im.complete && im.naturalWidth) g.drawImage(im, 0, 0);
    for (let n = 0; n < rw.outcomes.length; n++) {
      const tx = (n % COLS) * TW, ty = Math.floor(n / COLS) * TH, ok = rw.outcomes[n];
      if (settled) { g.fillStyle = ok ? "rgba(8,102,255,.14)" : "rgba(217,48,37,.24)"; g.fillRect(tx, ty, TW, TH); }
      g.lineWidth = 6; g.strokeStyle = ok ? "#0866ff" : "#d93025"; g.strokeRect(tx + 3, ty + 3, TW - 6, TH - 6);
    }
    if (i + 1 < rows.length) sheet(rows[i + 1].round, 0);
    capR.textContent = i ? `R${i}` : "R0";
    capT.textContent = i ? rw.change.text : "the skill as trained";
    stamp.textContent = i ? (rw.status === "kept" ? "kept" : "reverted") : "start";
    stamp.className = `rsi-stamp ${i ? rw.status : "start"}`;
    lis.forEach((li, j) => { li.classList.toggle("on", j === i); li.classList.toggle("done", j < i); });
    const tgt = rw.success;
    U.count += Math.sign(tgt - U.count) * Math.max(1, Math.ceil(Math.abs(tgt - U.count) / 3));
    if (Math.abs(tgt - U.count) < 1) U.count = tgt;
    bigN.textContent = String(U.count);
    const d = tgt - rows[0].success;
    chip.textContent = i ? `${d >= 0 ? "+" : ""}${d} vs R0` : "baseline";
    chip.className = `rsi-chip ${d > 0 ? "up" : d < 0 ? "down" : ""}`;
    fill.style.width = `${(U.count / rw.n) * 100}%`;
    base.style.left = `${(rows[0].success / rw.n) * 100}%`;
    const fi = U.tick % 16;
    strips.forEach((s) => { s.style.backgroundPosition = `${(fi / 15) * 100}% 0`; });
    box.classList.toggle("final", i === rows.length - 1);
  }
  U.draw = draw;
  U.timer = setInterval(() => { if (box.hidden) return; U.tick++; if (solo) U.frac = Math.min(1, U.tick / 28); draw(); place(); }, 320);
  draw();
  if (solo) { document.body.append(box); return; }
  rsiSync();
}
window.forgeRSIShow = (rsi) => rsiPanel(rsi, true);
window.forgeRSIHide = () => { if (rsiUI?.solo) rsiClose(); };

async function open(tile) {
  if (R) close();
  if (F().replay.busy()) return F().note?.("A live run is still going. Replay waits until it finishes.");
  const res = await fetch(`/api/skills/${tile.id}/replay-events`);
  if (!res.ok) return;
  const d = await res.json();
  if (!bar) buildBar();
  F().setView("sim");
  const thread = $("#thread");
  const { at, total } = compress(d.events.map((e) => e.ts));
  R = { sid: d.sid, d, at, total, pos: 0, next: 0, gen: 0, speed: 1, playing: true, raf: 0, last: 0, base: thread.children.length, thread };
  bar.querySelector(".rp-note").innerHTML = note(d);
  bar.hidden = false;
  document.body.classList.add("rp-on");
  F().replay.begin(d.prompt);
  setSpeed(1);
  restart();
  tick(performance.now());
  if (window.forgeRSI && tile.id === window.forgeRSI.tile) rsiPanel(window.forgeRSI);
}

const stamp = (e) => ({ ...e, run: `rp:${R.sid}@${R.gen}:${e.run}`, ...(e.child ? { child: `rp:${R.sid}@${R.gen}:${e.child}` } : {}) });

function restart() {
  R.gen += 1;
  R.next = 0;
  R.pos = 0;
  while (R.thread.children.length > R.base) R.thread.lastElementChild.remove();
}

function feedUpTo(pos) {
  while (R.next < R.d.events.length && R.at[R.next] <= pos) {
    F().replay.feed(stamp(R.d.events[R.next]));
    R.next += 1;
  }
}

function seek(target) {
  if (target < R.pos) restart();
  R.pos = target;
  feedUpTo(target);
  paint();
}

function paint() {
  if (rsiUI) { rsiUI.frac = R.pos / R.total; rsiSync(); }
  const sc = bar.querySelector(".rp-scrub");
  if (document.activeElement !== sc) sc.value = String(Math.round((R.pos / R.total) * 1000));
  bar.querySelector(".rp-time").textContent = `${fmt(R.pos)} / ${fmt(R.total)}`;
  const done = R.pos >= R.total;
  bar.querySelector(".rp-play").textContent = done ? "↺" : R.playing ? "❚❚" : "▶";
  bar.querySelector(".rp-play").setAttribute("aria-label", done ? "Play again" : R.playing ? "Pause" : "Play");
}

function tick(now) {
  if (!R) return;
  const dt = R.last ? (now - R.last) / 1000 : 0;
  R.last = now;
  if (R.playing && R.pos < R.total) {
    R.pos = Math.min(R.total, R.pos + dt * R.speed);
    feedUpTo(R.pos);
    if (R.pos >= R.total) R.playing = false;
    paint();
  }
  R.raf = requestAnimationFrame(tick);
}

function toggle() {
  if (R.pos >= R.total) { restart(); R.playing = true; return paint(); }
  R.playing = !R.playing;
  paint();
}

function setSpeed(s) {
  R.speed = s;
  bar.querySelectorAll(".rp-speed button").forEach((b) => b.classList.toggle("on", +b.dataset.s === s));
}

function close() {
  if (!R) return;
  cancelAnimationFrame(R.raf);
  while (R.thread.children.length > R.base) R.thread.lastElementChild.remove();
  R = null;
  rsiClose();
  document.body.classList.remove("rp-on");
  bar.hidden = true;
  F().replay.end();
}

window.forgeReplay = { open, close, active: () => !!R, state: () => R && { pos: R.pos, total: R.total, next: R.next, n: R.d.events.length, playing: R.playing, speed: R.speed } };
