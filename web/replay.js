// Replay of a Library card: the skill's recorded development fed through the same handlers a live run uses
// (window.forge.replay.feed), so the step strip, the rail, the loss curve, the eval tally, the mascot and the arm
// all move as they did. It reads /api/skills/<id>/replay-events and writes nothing.
const $ = (s) => document.querySelector(s);
const F = () => window.forge;
const CAP = 1.2; // longest idle gap kept, in seconds of recording
const TARGET = 85; // seconds a long run is squeezed into at 1x
const LEAD = 0.5;
const TAIL = 1.5;
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

// recording seconds -> playback seconds: idle gaps are capped, and a long run is scaled into TARGET
function compress(times) {
  const gaps = times.map((t, i) => (i ? Math.min(Math.max(t - times[i - 1], 0), CAP) : 0));
  let at = [];
  let acc = LEAD;
  for (const g of gaps) { acc += g; at.push(acc); }
  const k = acc > TARGET ? TARGET / acc : 1;
  at = at.map((a) => a * k);
  return { at, total: acc * k + TAIL };
}

const fmt = (s) => `${Math.floor(s / 60)}:${String(Math.floor(s % 60)).padStart(2, "0")}`;

function note(d) {
  const miss = Object.entries(d.stages).filter(([, v]) => v === "not_recorded").map(([k]) => STAGE_NAME[k]);
  const part = Object.entries(d.stages).filter(([, v]) => v === "partial").map(([k]) => STAGE_NAME[k]);
  const src = d.source === "live"
    ? "Captured live while this skill was trained."
    : "Rebuilt from what was saved on disk, not captured live. The pacing here is for playback; the numbers are the saved ones.";
  const bits = [`<b>${src}</b>`];
  if (part.length) bits.push(`Partly recorded: ${part.join(", ")}.`);
  if (miss.length) bits.push(`Not recorded: ${miss.join(", ")}.`);
  return bits.join(" ");
}

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
  F().replay.begin(d.prompt);
  setSpeed(1);
  restart();
  tick(performance.now());
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
  bar.hidden = true;
  F().replay.end();
}

window.forgeReplay = { open, close, active: () => !!R, state: () => R && { pos: R.pos, total: R.total, next: R.next, n: R.d.events.length, playing: R.playing, speed: R.speed } };
