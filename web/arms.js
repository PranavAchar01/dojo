// Multi-arm tasks. The same chat and the same canvas as a single arm: ask for "another SO-101" or describe a task that
// needs several arms, and the cell grows an arm for each sub-task. Underneath this is the planner and the per-arm
// training and measuring that factory.py already does (an arm is what it used to call a station); every pose, part
// and tote position on screen comes from MuJoCo, nothing is animated by hand.
import { cellBrackets, floorLabel, basePlate } from "./studio.js";

const F = () => window.forge;
const $ = (s) => document.querySelector(s);
const esc = (s) => String(s ?? "").replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" })[c]);
const api = async (url, opt) => {
  const r = await fetch(url, opt);
  if (!r.ok) throw new Error((await r.json().catch(() => ({}))).detail || r.statusText);
  return r.json();
};
const post = (url, body) => api(url, { method: "POST", headers: { "content-type": "application/json" }, body: JSON.stringify(body) });

const MAX_ARMS = 4;
const SPEED = 4; // playback is sped up so a cell of recorded episodes plays in about a minute
const N_PARTS = 12;
const Y0 = 0.32;
const GAP = 0.5; // arms stand close together around one work area // first arm's y; the arms stand along y, one pitch apart
const ADD = /\b(add|bring in|put in|spawn|give me)\s+(another|a second|a third|an? extra|one more)\b[^.]{0,30}\b(so-?101|arms?|robots?)\b/i;
const WANTS = new RegExp(
  [
    ADD.source,
    String.raw`\b(two|three|four|2|3|4|second|third|several|multiple|both)\b[^.:]{0,24}\b(arms?|so-?101s?|robots?)\b`,
    String.raw`\barms?\s*(1|2|3|one|two|three)\b`,
    String.raw`\bone (arm|robot|so-?101)\b[^.]{0,90}\bthe other\b`,
    String.raw`\bassembl\w*`,
    String.raw`\bpallet\b`,
  ].join("|"),
  "i",
);

let active = false;
let line = null; // the planned cell as /api/line returns it
let extra = 0; // arms that have no task yet (planned arms come from the line's stations)
let prevN = 0; // arms already standing before the latest change, so only new ones build in
let taskText = "";
let grp = null;
let cells = [];
let layout = null; // { cy, n }
let token = 0;
let clock = { t: 0, playing: false, last: 0, end: 0, finished: new Set(), badge: "" };
let armRuns = {}; // arm index -> { run, stage, frac }
let previewScene = null;
let saved = null; // what the single-arm view had before the cell took over

// ------------------------------------------------------------------ entry points

function wants(text) { return ADD.test(text) || WANTS.test(text) || (active && extra > 0); }
const total = () => (line ? line.stations.length : 0) + extra;

async function handle(text) {
  if (ADD.test(text)) return addArm();
  F().note("Working out what each arm does.");
  for (const k of ["task", "router", "planner", "scene", "data", "trainer", "eval", "critic", "executor", "registry"]) F().stage(k, "idle", "");
  try {
    const l = await post("/api/line/plan", { prompt: text });
    if (!l.stations.length) {
      F().note(l.plan?.note || "I could not find separate arm tasks in that. Try \"two arms: one seats the housing in the pallet, the other inserts it into the socket\".");
      return;
    }
    taskText = text;
    const had = total();
    extra = Math.max(0, had - l.stations.length);
    await setCell(l);
  } catch (err) {
    F().note(err.message);
  }
}

// adds an arm to whatever is on the canvas, now: the new one builds in beside the arms already standing
async function addArm() {
  const n = total();
  if (n >= MAX_ARMS) { F().note(`The canvas holds up to ${MAX_ARMS} arms.`); return; }
  const first = !active;
  prevN = first ? (F().hasWorld() ? 1 : 0) : n;
  extra = first ? 2 : extra + 1;
  enter();
  await build3D();
  renderPanel();
  F().stage("scene", "done");
  const k = total();
  F().note(`Added an SO-101: arm ${k} has no task yet. Say what it should do, or describe the task the arms do together.`);
}

async function setCell(l) {
  const same = line && line.id === l.id;
  line = l;
  if (!same) armRuns = {};
  if (!same) prevN = 0;
  enter();
  renderPanel();
  for (const k of ["task", "router", "planner"]) F().stage(k, "done");
  await build3D();
  F().stage("scene", "done");
  if (!same) {
    const names = line.stations.map((s, i) => `arm ${i + 1}: ${s.name}`).join("; ");
    F().note(`${line.stations.length} arms working on one task. ${names}.`);
  }
  if (line.result?.playback) startClock();
}

// ------------------------------------------------------------------ switching between one arm and a cell

function enter() {
  if (active) return;
  active = true;
  const f = F();
  f.setMulti(true);
  f.stageIn();
  f.setSingleVisible(false);
  f.controls.maxDistance = 7;
  f.scene3.fog.near = 4.6; f.scene3.fog.far = 12;
  saved = { max: 2.6 };
  new ResizeObserver(fit).observe($("#three").parentElement);
}
function exit() {
  if (!active) return;
  active = false;
  token++;
  const f = F();
  clear();
  line = null; extra = 0; prevN = 0; armRuns = {};
  f.setMulti(false);
  f.setSingleVisible(true);
  f.controls.maxDistance = saved.max;
  f.scene3.fog.near = 2.6; f.scene3.fog.far = 7.0;
  light(0.2, 0, 1, 1.1);
  f.setCamShift(0, 1);
  $("#step-badge").classList.add("hidden");
  renderPanel();
}
function clear() {
  if (grp) { F().scene3.remove(grp); grp = null; }
  cells = [];
  clock.playing = false;
}

// the key light's shadow only covers what it is aimed at: re-aim it at the middle of the cell
function light(x, cy, reach, extent) {
  const { key, rim, fill } = F().lights;
  key.position.set(x + 0.95 * reach, cy - 0.8 * reach, 1.75 * reach);
  key.target.position.set(x, cy, 0);
  Object.assign(key.shadow.camera, { left: -extent, right: extent, top: extent, bottom: -extent });
  key.shadow.camera.updateProjectionMatrix();
  rim.position.set(x - 1.1 * reach, cy + 1.0 * reach, 1.1 * reach);
  fill.position.set(x + 0.6 * reach, cy + 1.2 * reach, 0.8 * reach);
}
// back the camera off so every arm fits the canvas, however wide or tall it is
function fit() {
  if (!active || !layout) return;
  const f = F();
  const aspect = f.camera.aspect || 1.2;
  const width = layout.n * layout.pitch + 0.55;
  const dist = Math.max((0.8 * width) / (2 * Math.tan((15 * Math.PI) / 180) * Math.max(aspect, 0.6)), 1.1);
  f.setCamShift(layout.cy, dist / 1.26);
  light(0.12, layout.cy, 1.1 + 0.25 * layout.n, layout.n * layout.pitch * 0.75 + 0.5);
}

// ------------------------------------------------------------------ the 3D cell

async function build3D() {
  const my = ++token;
  const f = F();
  const { THREE, meshesFor, populate, srgb } = f;
  let data, rows;
  if (line) {
    data = await api(`/api/line/${line.id}/scene`);
    rows = line.stations.map((st, k) => ({ sc: data.cells[k], st }));
  } else data = { block_colors: {} };
  if (extra) {
    previewScene ||= await api("/api/preview");
    const sc = { ...previewScene, blocks: [], bowls: [], pads: [], props: [], bins: [], belt: null };
    rows = [...(rows || []), ...Array.from({ length: extra }, () => ({ sc, st: null }))];
  }
  if (my !== token || !active) return;
  clear();
  grp = new THREE.Group();
  const pitch = GAP;
  const n = rows.length;
  layout = { n, pitch, cy: Y0 + ((n - 1) * pitch) / 2, data };
  cellBrackets(grp, [-0.3, Y0 - pitch / 2, 0.74, Y0 + n * pitch - pitch / 2]);
  const meshes = {};
  for (let k = 0; k < n; k++) {
    const { sc, st } = rows[k];
    const rid = sc.robot_id || st?.robot || "so101";
    meshes[rid] ||= await meshesFor(rid);
    if (my !== token || !active) return;
    const root = new THREE.Group();
    root.position.y = Y0 + k * pitch;
    grp.add(root);
    const { bodies } = populate(sc, root, rid, meshes[rid], { belt: false });
    const belt = null; // no conveyor on screen: the arms share one work area
    root.add(basePlate(0.23 * (f.armScale ? f.armScale(rid) : 1)));
    let cube = null, partMat = null, partBody = -1;
    if (st && sc.blocks[0]) {
      partBody = sc.blocks[0].body;
      bodies[partBody].traverse((o) => { if (!cube && o.isMesh) cube = o; });
      partMat = cube?.material;
    }
    const tag = st ? `ARM ${k + 1}  ${st.controller === "vla" ? "VLA" : "SCRIPTED"}` : `ARM ${k + 1}`;
    floorLabel(grp, tag, 0.5, root.position.y - 0.2, Math.PI / 2, 0.042, st?.controller === "vla" ? "rgba(8,102,255,0.9)" : "rgba(44,56,66,0.9)");
    const cell = { root, bodies, belt, cube, partMat, partBody, home: sc.home || sc.frame, done: [], born: k < prevN ? 0 : performance.now() + (k - prevN) * 160 };
    cells.push(cell);
    applyCell(cell, sc.home || sc.frame, false);
    root.scale.z = k < prevN ? 1 : 0.02;
  }
  f.scene3.add(grp);
  prevN = n;
  fit();
  $("#view-sim").classList.toggle("noplay", !line?.result?.playback);
  $("#vp-caption").textContent = `${n} arms · MuJoCo`;
  $("#crumb-run").textContent = taskText || "multi-arm task";
}

function applyCell(cell, frame, showPart) {
  if (!frame) return;
  for (let i = 0; i < cell.bodies.length; i++) {
    const k = i * 7;
    if (frame[k] === undefined) break;
    cell.bodies[i].position.set(frame[k], frame[k + 1], frame[k + 2]);
    cell.bodies[i].quaternion.set(frame[k + 4], frame[k + 5], frame[k + 6], frame[k + 3]);
  }
  if (cell.partBody >= 0) cell.bodies[cell.partBody].visible = showPart;
  if (cell.belt) cell.belt.tex.offset.y = -(frame[cell.belt.body * 7 + 1] - cell.belt.y0) / 0.06;
}

// ------------------------------------------------------------------ playback on the recorded schedule

function startClock() {
  if (!line?.result?.playback) return;
  clock.t = 0;
  clock.playing = true;
  clock.last = performance.now();
  for (const c of cells) { c.done.forEach((m) => c.root.remove(m)); c.done = []; }
  const eps = line.result.playback.episodes;
  for (const e of eps) e.end = e.start + e.duration;
  clock.end = Math.max(...eps.map((e) => e.end)) + 1.5;
  clock.finished = new Set();
  $("#view-sim").classList.remove("noplay");
  $("#pb-label").textContent = `${cells.length} arms`;
}

function tick(now) {
  if (!active || !cells.length) return;
  for (const c of cells) { // each arm rises from the floor in turn
    const k = c.born === 0 ? 1 : Math.min(1, Math.max(0, (now - c.born) / 700));
    c.root.scale.z = Math.max(0.02, 1 - Math.pow(1 - k, 4) * (1 + 1.6 * (1 - k)));
  }
  if (!clock.playing || !line?.result?.playback) return;
  const dt = Math.max(0, Math.min(0.1, (now - clock.last) / 1000));
  clock.last = now;
  clock.t += dt * SPEED;
  const pb = line.result.playback;
  let badge = "";
  cells.forEach((cell, k) => {
    const e = pb.episodes.find((x) => x.station === k && clock.t >= x.start && clock.t < x.end);
    if (e) {
      const i = Math.min(e.frames.length - 1, Math.floor((clock.t - e.start) * pb.fps));
      if (cell.partMat) cell.partMat.color.copy(F().srgb(colorOf(e.color)));
      applyCell(cell, e.frames[i], true);
      if (!badge) badge = `Try ${e.part + 1} of ${line.result.n_parts} · arm ${k + 1}`;
    } else applyCell(cell, cell.home, false);
  });
  for (const e of pb.episodes) {
    if (e.kind !== "pick" || clock.t < e.end) continue;
    const key = `${e.station}:${e.part}`;
    if (!clock.finished.has(key)) { clock.finished.add(key); restPart(e); }
  }
  if (badge !== clock.badge) {
    clock.badge = badge;
    const sb = $("#step-badge");
    sb.textContent = badge;
    sb.classList.toggle("hidden", !badge);
  }
  $("#pb-fill").style.width = `${(100 * Math.min(clock.t, clock.end)) / clock.end}%`;
  if (clock.t > clock.end + 4) startClock(); // loop the run
}
function colorOf(c) {
  return layout?.data?.block_colors?.[c] || { red: [0.8, 0.12, 0.1], blue: [0.12, 0.32, 0.8], green: [0.14, 0.6, 0.28], yellow: [0.95, 0.78, 0.1] }[c] || [0.5, 0.5, 0.5];
}
function restPart(e) { // a finished part stays where the episode left it (in the tote, or dropped)
  const cell = cells[e.station];
  if (!cell?.cube) return;
  const { THREE, srgb } = F();
  const fr = e.frames[e.frames.length - 1];
  const k = cell.partBody * 7;
  const m = new THREE.Mesh(cell.cube.geometry, new THREE.MeshPhysicalMaterial({ color: srgb(colorOf(e.color)), roughness: 0.5, clearcoat: 0.22, transparent: !e.ok, opacity: e.ok ? 1 : 0.55 }));
  m.position.set(fr[k], fr[k + 1], fr[k + 2]);
  m.quaternion.set(fr[k + 4], fr[k + 5], fr[k + 6], fr[k + 3]);
  m.castShadow = true;
  cell.root.add(m);
  cell.done.push(m);
}
$("#pb-replay").addEventListener("click", () => { if (active) startClock(); });

// ------------------------------------------------------------------ the Arms section of the right panel

const STATUS = { planned: "Planned", train: "Needs training", training: "Training", ready: "Trained", scripted: "Scripted", refused: "Not possible" };
function armRow(st, i) {
  const isVla = st.controller === "vla";
  const tr = (line.training || []).find((t) => t.station === st.index);
  const held = tr ? tr.success_rate : st.skill?.success_rate;
  const nEval = tr ? tr.n_eval : st.skill?.n_eval;
  const ps = line.result?.per_station?.[st.index];
  const run = armRuns[st.index];
  const busy = line.status === "building" || line.status === "running";
  const facts = [];
  if (isVla) facts.push(held != null && nEval ? `${Math.round(held * nEval)}/${nEval} on held-out scenes` : "held-out score after training");
  else facts.push("scripted, no training");
  if (ps) facts.push(`${ps.succeeded}/${ps.parts} in the run`);
  const why = tr?.from ? `fine-tuned from '${tr.from.replace(/-[0-9a-f]{4}$/, "").replaceAll("-", " ")}'` : st.route?.reason || "";
  return `<li class="arm ${isVla ? "vla" : "scripted"} st-${st.status}" data-i="${i}">
    <div class="arm-h"><span class="arm-n">${i + 1}</span><b>${esc(st.name)}</b><span class="arm-st">${esc(STATUS[st.status] || st.status)}</span></div>
    <div class="arm-task">${esc(st.instruction)}</div>
    <div class="arm-meta">${esc(st.robot_label || st.robot)} · ${isVla ? "VLA policy" : "scripted"}</div>
    ${run ? `<div class="arm-train"><div class="arm-bar"><i style="width:${Math.round(run.frac * 100)}%"></i></div><span>${esc(run.stage)}</span></div>` : ""}
    <div class="arm-facts">${facts.map(esc).join(" · ")}</div>
    ${why ? `<div class="arm-why">${esc(why)}</div>` : ""}
    <button type="button" class="arm-toggle" ${busy ? "disabled" : ""}>${isVla ? "Use a script" : "Use a VLA"}</button>
  </li>`;
}
const noTask = (from) => Array.from({ length: extra }, (_, i) => `<li class="arm bare"><div class="arm-h"><span class="arm-n">${from + i + 1}</span><b>SO-101</b><span class="arm-st">No task yet</span></div><div class="arm-meta">Say what it should do, or describe the task the arms do together.</div></li>`).join("");
function renderPanel() {
  const list = $("#arms");
  const go = $("#arms-go");
  if (!active) { list.innerHTML = ""; $("#arms-result").innerHTML = ""; return; }
  if (!line) {
    list.innerHTML = noTask(0);
    $("#arms-note").textContent = `${extra} arms`;
    $("#arms-model").textContent = "";
    $("#arms-result").innerHTML = "";
    go.disabled = true;
    go.textContent = "Train and run";
    return;
  }
  const nTrain = line.stations.filter((s) => s.status === "train").length;
  const refused = line.stations.some((s) => s.status === "refused");
  const busy = line.status === "building" || line.status === "running";
  $("#arms-note").textContent = `${total()} arms`;
  const model = line.plan?.model;
  $("#arms-model").textContent = model ? `Planner: ${F().modelTag(model)}` : "";
  list.innerHTML = line.stations.map(armRow).join("") + noTask(line.stations.length);
  go.disabled = busy || refused || !line.stations.length;
  go.textContent = busy ? (line.status === "building" ? "Training" : "Running") : nTrain ? `Train ${nTrain} arm${nTrain > 1 ? "s" : ""} and run` : "Run";
  renderResult();
}
function renderResult() {
  const r = line.result;
  const box = $("#arms-result");
  if (line.status === "failed") { box.innerHTML = `<p class="arms-bad">${esc(r?.error || "An arm did not publish a skill, so the run did not happen.")}</p>`; return; }
  if (!r || !r.n_parts) { box.innerHTML = ""; return; }
  const fails = (r.fate || []).filter((f) => !f.ok);
  const colors = Object.entries(r.parts.reduce((c, p) => ({ ...c, [p]: (c[p] || 0) + 1 }), {})).map(([k, v]) => `${v} ${k}`).join(", ");
  box.innerHTML = `<p class="arms-line"><b class="num">${r.completed} of ${r.n_parts}</b> tries finished by ${line.stations.length === 2 ? "both arms" : "all arms"}, in simulation.</p>
    <p class="arms-fine">${r.n_parts} tries (${esc(colors)}), seed ${r.seed}.${fails.length ? ` Not finished: ${fails.map((f) => `try ${f.part + 1} (${f.color}) at ${f.station == null ? "the end" : `arm ${f.station + 1}`}${f.why ? `, ${esc(f.why)}` : ""}`).join("; ")}.` : ""}</p>`;
}

$("#arms").addEventListener("click", async (e) => {
  const b = e.target.closest(".arm-toggle");
  if (!b || !line) return;
  const i = +b.closest(".arm").dataset.i;
  b.disabled = true;
  b.textContent = "Routing";
  try {
    const l = await api(`/api/line/${line.id}/stations/${i}`, { method: "PATCH", headers: { "content-type": "application/json" }, body: JSON.stringify({ controller: line.stations[i].controller === "vla" ? "scripted" : "vla" }) });
    await setCell(l);
  } catch (err) {
    F().note(err.message);
    renderPanel();
  }
});
$("#arms-go").addEventListener("click", async () => {
  if (!line) return;
  const nTrain = line.stations.filter((s) => s.status === "train").length;
  try {
    await post(`/api/line/${line.id}/build`, { n_parts: N_PARTS, seed: 7 });
    line.status = "building";
    F().note(nTrain ? `Training ${nTrain} arm${nTrain > 1 ? "s" : ""} that need a skill, then running the task in simulation.` : "Running the task in simulation.");
    F().stage("data", "live"); F().stage("trainer", "live");
    renderPanel();
  } catch (err) {
    F().note(err.message);
  }
});

// ------------------------------------------------------------------ live events from the server

const STAGE = { scene: "Building the arm's scene", demos: "Scripted demos", init: "Starting from the nearest skill", train_start: "Fine-tuning", loss: "Fine-tuning", eval_progress: "Held-out eval", saving: "Saving", saved: "Saved" };
window.addEventListener("forge:event", async (ev) => {
  const e = ev.detail;
  if (!line || !active) return;
  if (e.kind?.startsWith("line_") && e.line === line.id) {
    if (e.kind === "line_station_run") armRuns[e.station] = { run: e.run, stage: "Queued", frac: 0.02 };
    if (e.kind === "line_station") {
      Object.assign(line.stations[e.station.index], e.station);
      if (e.station.status !== "training") delete armRuns[e.station.index];
    }
    if (e.kind === "line_status") {
      line.status = e.status;
      if (e.status === "running") { F().stage("trainer", "done"); F().stage("eval", "live"); }
      if (e.status === "done" || e.status === "failed") {
        const l = await api(`/api/line/${line.id}`);
        await setCell(l);
        const ok = e.status === "done" && l.result?.n_parts;
        for (const k of ["data", "trainer", "eval"]) F().stage(k, ok ? "done" : "fail");
        if (ok) {
          F().stage("registry", "done", `${l.result.completed}/${l.result.n_parts}`);
          F().note(`${l.result.completed} of ${l.result.n_parts} tries finished by ${l.stations.length === 2 ? "both arms" : "all arms"}, in simulation.`);
        } else F().note(l.result?.error || "An arm did not publish a skill, so the run did not happen.");
        return;
      }
    }
    if (e.kind === "line_station_done") line.result = { ...(line.result || {}), progress: e };
    renderPanel();
    return;
  }
  const idx = Object.keys(armRuns).find((k) => armRuns[k].run === e.run);
  if (idx == null || !STAGE[e.kind]) return;
  const s = armRuns[idx];
  s.stage = STAGE[e.kind];
  if (e.kind === "demos") { s.stage = `Scripted demos ${e.done}/${e.n}`; s.frac = 0.05 + 0.25 * (e.done / e.n); }
  if (e.kind === "train_start") { s.frac = Math.max(s.frac, 0.3); s.budget = e.seconds; s.trainT0 = performance.now(); }
  if (e.kind === "loss" && s.budget) s.frac = Math.max(s.frac, 0.3 + 0.5 * Math.min(1, (performance.now() - s.trainT0) / 1000 / s.budget));
  if (e.kind === "eval_progress") { s.stage = `Held-out eval ${e.done}/${e.n}`; s.frac = Math.max(s.frac, 0.8 + 0.18 * (e.done / e.n)); }
  if (e.kind === "saved") s.frac = 1;
  renderPanel();
});
window.addEventListener("forge:newchat", () => exit());
window.addEventListener("forge:view", (e) => { if (e.detail === "sim") setTimeout(fit, 340); });

window.forgeArms = { wants, handle, exit, tick, active: () => active };
