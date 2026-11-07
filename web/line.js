// Line tab: a company describes its line in words, the planner splits it into stations on one conveyor, the router
// checks each station, missing station skills are trained by the agents, and the whole line runs part by part.
// The 3D view lays the stations' cells end to end along one belt and plays back the recorded episodes on the
// line schedule (factory.py): every arm, part and tote position here comes from MuJoCo, nothing is animated by hand.
import { OrbitControls } from "three/addons/controls/OrbitControls.js";

const F = () => window.forge;
const host = document.getElementById("view-line");
const EXAMPLES = [
  "Red parts go to the red tote, blue parts to the blue tote, and a scripted ViperX at the end of the line catches everything else in the gray tote.",
  "Kitting line: pull the yellow parts into the yellow tote, then the green parts into the green tote; anything left goes to gray at the end.",
  "One SO-101 sorts red parts into the red tote; a programmed cell at the end of line takes the rest.",
];
const SPEEDS = [1, 4];

let line = null; // the line as /api/line returns it
let lines = [];
let three = null; // { renderer, scene, camera, controls, cells: [...], belt, parts }
let clock = { t: 0, speed: 4, playing: false, last: 0 };
let stationRuns = {}; // station index -> { run, stage, t0 }

const esc = (s) => String(s ?? "").replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" })[c]);
const pct = (x) => (x == null ? "n/a" : `${Math.round(x * 100)}%`);
const api = async (url, opt) => {
  const r = await fetch(url, opt);
  if (!r.ok) throw new Error((await r.json().catch(() => ({}))).detail || r.statusText);
  return r.json();
};

host.innerHTML = `
<div class="ln">
  <header class="ln-head">
    <span class="mascot md" data-follow></span>
    <div class="ln-title">
      <h1>Factory <em>line</em></h1>
      <p>Describe the line. Each station gets its own trained VLA or a scripted cell, then the whole line runs. All in simulation.</p>
    </div>
    <select class="ln-pick" aria-label="Saved lines"></select>
  </header>
  <form class="ln-ask">
    <textarea rows="2" placeholder="e.g. ${esc(EXAMPLES[0])}"></textarea>
    <button type="submit" class="ln-plan">Plan line</button>
  </form>
  <div class="ln-ex">${EXAMPLES.map((e, i) => `<button type="button" data-i="${i}">${esc(e)}</button>`).join("")}</div>
  <section class="ln-stage">
    <div class="ln-3d"></div>
    <div class="ln-hud">
      <div class="ln-stat"><b class="ln-done">0</b><span>parts done</span></div>
      <div class="ln-stat"><b class="ln-tput">0.0</b><span>parts / min</span></div>
      <div class="ln-stat"><b class="ln-clock">0.0 s</b><span>line time</span></div>
      <button class="ln-speed" type="button" title="Playback speed">4x</button>
    </div>
    <div class="ln-hazard"></div>
    <div class="ln-empty">Plan a line to see it here.</div>
  </section>
  <div class="ln-actions">
    <div class="ln-note"></div>
    <label class="ln-n">Parts <input type="number" min="2" max="30" value="12"></label>
    <button type="button" class="ln-build" disabled>Train stations and run</button>
  </div>
  <div class="ln-cards"></div>
  <div class="ln-result"></div>
</div>`;
const $ = (s) => host.querySelector(s);

// ------------------------------------------------------------------ planning UI

$(".ln-ex").addEventListener("click", (e) => {
  const b = e.target.closest("button");
  if (b) $(".ln-ask textarea").value = EXAMPLES[+b.dataset.i];
});
$(".ln-ask").addEventListener("submit", async (e) => {
  e.preventDefault();
  const text = $(".ln-ask textarea").value.trim();
  if (!text) return;
  const btn = $(".ln-plan");
  btn.disabled = true;
  btn.textContent = "Planning…";
  setNote("The planner is splitting the line into stations; the router is checking each one.", "think");
  try {
    setLine(await api("/api/line/plan", { method: "POST", headers: { "content-type": "application/json" }, body: JSON.stringify({ prompt: text }) }));
    await refreshList();
  } catch (err) {
    setNote(err.message, "fail");
  } finally {
    btn.disabled = false;
    btn.textContent = "Plan line";
  }
});
$(".ln-pick").addEventListener("change", async (e) => {
  if (e.target.value) setLine(await api(`/api/line/${e.target.value}`));
});
$(".ln-speed").addEventListener("click", () => {
  clock.speed = SPEEDS[(SPEEDS.indexOf(clock.speed) + 1) % SPEEDS.length];
  $(".ln-speed").textContent = `${clock.speed}x`;
});
$(".ln-build").addEventListener("click", () => go("build"));

async function go(kind) {
  const n = Math.min(30, Math.max(2, +$(".ln-n input").value || 12));
  try {
    await api(`/api/line/${line.id}/${kind}`, { method: "POST", headers: { "content-type": "application/json" }, body: JSON.stringify({ n_parts: n, seed: 7 }) });
    line.status = kind === "build" ? "building" : "running";
    render();
  } catch (err) {
    setNote(err.message, "fail");
  }
}

function setNote(text, mood = "idle") {
  $(".ln-note").textContent = text;
  $(".ln-head .mascot").dataset.mood = mood;
}

async function refreshList() {
  lines = await api("/api/lines");
  $(".ln-pick").innerHTML = `<option value="">Saved lines (${lines.length})</option>` +
    lines.map((l) => `<option value="${l.id}" ${line && l.id === line.id ? "selected" : ""}>${esc(l.name)}: ${l.stations.length} stations, ${esc(l.status)}</option>`).join("");
}

async function setLine(l) {
  const same = line && line.id === l.id;
  line = l;
  if (!same) stationRuns = {};
  if (line.result?.n_parts) $(".ln-n input").value = line.result.n_parts;
  render();
  await buildLine3D();
  if (line.result?.playback) startClock();
}

// ------------------------------------------------------------------ station cards

const STATUS = { planned: "Planned", train: "Needs training", training: "Training", ready: "Trained", scripted: "Scripted", refused: "Not feasible" };
function stationCard(st) {
  const isVla = st.controller === "vla";
  const route = st.route || {};
  const skill = st.skill;
  const tr = (line.training || []).find((t) => t.station === st.index);
  const held = tr ? tr.success_rate : skill?.success_rate;
  const nEval = tr ? tr.n_eval : skill?.n_eval;
  const ps = line.result?.per_station?.[st.index];
  const run = stationRuns[st.index];
  const busy = line.status === "building" || line.status === "running";
  const rows = [];
  if (isVla) rows.push(`<div><span>Held-out success</span><b>${held != null ? `${pct(held)}${nEval ? ` <i>of ${nEval}</i>` : ""}` : "after training"}</b></div>`);
  if (tr) rows.push(`<div><span>Trained in</span><b>${Math.round(tr.seconds)} s</b></div>`);
  if (ps) rows.push(`<div><span>On the line</span><b>${ps.succeeded}/${ps.parts}</b></div>`, `<div><span>Cycle</span><b>${ps.mean_work_s != null ? `${ps.mean_work_s.toFixed(1)} s` : "n/a"}</b></div>`);
  return `<article class="ln-card ${isVla ? "vla" : "scripted"} st-${st.status}" data-i="${st.index}">
    <header>
      <span class="ln-idx">${String(st.index + 1).padStart(2, "0")}</span>
      <span class="ln-kind">${isVla ? "VLA" : "Scripted"}</span>
      <span class="ln-status">${esc(STATUS[st.status] || st.status)}</span>
    </header>
    <h3>${esc(st.name)}</h3>
    <p class="ln-task">${esc(st.instruction)}</p>
    <p class="ln-robot">${esc(st.robot_label || st.robot)}</p>
    ${run ? `<div class="ln-train"><div class="ln-train-bar"><i style="width:${run.frac * 100}%"></i></div><span>${esc(run.stage)}</span></div>` : ""}
    <div class="ln-nums">${rows.join("")}</div>
    <p class="ln-why">${esc(tr?.from ? `fine-tuned from '${tr.from.replace(/-[0-9a-f]{4}$/, "").replaceAll("-", " ")}'` : route.reason || "")}</p>
    <footer>
      <button type="button" class="ln-toggle" ${busy ? "disabled" : ""} title="Switch between a trained VLA and the scripted cell">${isVla ? "Use scripted cell" : "Use a VLA"}</button>
    </footer>
  </article>`;
}

$(".ln-cards").addEventListener("click", async (e) => {
  const b = e.target.closest(".ln-toggle");
  if (!b) return;
  const i = +b.closest(".ln-card").dataset.i;
  const st = line.stations[i];
  b.disabled = true;
  b.textContent = "Routing…";
  try {
    setLine(await api(`/api/line/${line.id}/stations/${i}`, { method: "PATCH", headers: { "content-type": "application/json" }, body: JSON.stringify({ controller: st.controller === "vla" ? "scripted" : "vla" }) }));
  } catch (err) {
    setNote(err.message, "fail");
    render();
  }
});

function render() {
  const has = !!line;
  $(".ln-empty").classList.toggle("hidden", has && line.stations.length > 0);
  if (!has) return;
  const nTrain = line.stations.filter((s) => s.status === "train").length;
  const refused = line.stations.some((s) => s.status === "refused");
  const busy = line.status === "building" || line.status === "running";
  $(".ln-build").disabled = busy || refused || !line.stations.length;
  $(".ln-build").textContent = nTrain ? `Train ${nTrain} station${nTrain > 1 ? "s" : ""} and run` : "Run the line";
  $(".ln-cards").innerHTML = line.stations.map(stationCard).join("") ||
    `<p class="ln-refused">${esc(line.plan?.note || "No stations in that description.")}</p>`;
  const planned = `Planned by ${esc(line.plan?.model || "the planner")}${line.plan?.note ? `. ${esc(line.plan.note)}` : ""}`;
  if (line.status === "building") setNote("Training the stations that need a skill, one agent run each.", "think");
  else if (line.status === "running") setNote("Running the line in simulation, zone by zone.", "think");
  else if (line.status === "failed") setNote(line.result?.error || "A station did not publish a skill; the line did not run.", "fail");
  else if (refused) setNote("A station is not feasible: see its card.", "fail");
  else setNote(`${line.stations.length} stations on one conveyor. ${planned}`, line.status === "done" ? "ok" : "idle");
  renderResult();
}

function renderResult() {
  const r = line.result;
  if (!r || !r.n_parts) { $(".ln-result").innerHTML = ""; return; }
  const fails = (r.fate || []).filter((f) => !f.ok);
  $(".ln-result").innerHTML = `<div class="ln-res">
    <div><b>${r.completed}/${r.n_parts}</b><span>parts in the right tote</span></div>
    <div><b>${pct(r.success_rate)}</b><span>end-to-end success</span></div>
    <div><b>${r.throughput_per_min.toFixed(1)}</b><span>parts / min (simulated time)</span></div>
    <div><b>${r.makespan_s.toFixed(0)} s</b><span>line time for ${r.n_parts} parts</span></div>
  </div>
  <p class="ln-fine">Measured in MuJoCo: ${r.n_parts} parts (${esc(countColors(r.parts))}) released every ${r.release_every_s} s, seed ${r.seed}. One part per zone at a time.${fails.length ? ` Misses: ${fails.map((f) => `part ${f.part + 1} (${f.color}) at ${f.station == null ? "end of line" : `station ${f.station + 1}`}${f.why ? `, ${esc(f.why)}` : ""}`).join("; ")}.` : ""}</p>`;
}
function countColors(parts) {
  const c = {};
  for (const p of parts) c[p] = (c[p] || 0) + 1;
  return Object.entries(c).map(([k, v]) => `${v} ${k}`).join(", ");
}

// ------------------------------------------------------------------ live events (training and running)

const STAGE = { scene: "Building the station cell", demos: "Scripted demos", init: "Starting from the nearest skill", train_start: "Fine-tuning", loss: "Fine-tuning", eval_progress: "Held-out eval", saving: "Publishing", saved: "Published" };
window.addEventListener("forge:event", async (ev) => {
  const e = ev.detail;
  if (!line) return;
  if (e.kind?.startsWith("line_") && e.line === line.id) {
    if (e.kind === "line_station_run") stationRuns[e.station] = { run: e.run, stage: "Queued", frac: 0.02, t0: performance.now() };
    if (e.kind === "line_station") {
      Object.assign(line.stations[e.station.index], e.station);
      if (e.station.status !== "training") delete stationRuns[e.station.index];
    }
    if (e.kind === "line_status") {
      line.status = e.status;
      if (e.status === "done" || e.status === "failed") return setLine(await api(`/api/line/${line.id}`)).then(refreshList);
    }
    if (e.kind === "line_station_done") line.result = { ...(line.result || {}), progress: e };
    render();
    return;
  }
  const idx = Object.keys(stationRuns).find((k) => stationRuns[k].run === e.run);
  if (idx == null || !STAGE[e.kind]) return;
  const s = stationRuns[idx];
  s.stage = STAGE[e.kind];
  if (e.kind === "demos") { s.stage = `Scripted demos ${e.done}/${e.n}`; s.frac = 0.05 + 0.25 * (e.done / e.n); }
  if (e.kind === "train_start") { s.frac = Math.max(s.frac, 0.3); s.budget = e.seconds; s.trainT0 = performance.now(); }
  if (e.kind === "loss" && s.budget) s.frac = Math.max(s.frac, 0.3 + 0.5 * Math.min(1, (performance.now() - s.trainT0) / 1000 / s.budget));
  if (e.kind === "eval_progress") { s.stage = `Held-out eval ${e.done}/${e.n}`; s.frac = Math.max(s.frac, 0.8 + 0.18 * (e.done / e.n)); }
  if (e.kind === "saved") s.frac = 1;
  render();
});

// ------------------------------------------------------------------ the 3D line

function disposeLine3D() {
  if (!three) return;
  three.stage.dispose();
  three.renderer.dispose();
  three.renderer.domElement.remove();
  three.ro.disconnect();
  three = null;
}

async function buildLine3D() {
  disposeLine3D();
  if (!line?.stations.length) return;
  const { THREE, meshesFor, populate, RoundedBoxGeometry, srgb } = F();
  const { HAZE, setupRenderer, makeEnvironment, makeStage, addStudioLights, buildLab, basePlate, floorLabel, hazardStrip } = await import("./studio.js");
  const data = await api(`/api/line/${line.id}/scene`);
  const box = $(".ln-3d");
  const renderer = new THREE.WebGLRenderer({ antialias: true, preserveDrawingBuffer: true });
  renderer.setPixelRatio(Math.min(2, devicePixelRatio));
  setupRenderer(renderer);
  box.appendChild(renderer.domElement);
  const rect = box.getBoundingClientRect();
  renderer.setSize(Math.max(2, rect.width), Math.max(2, rect.height), false);
  const scene = new THREE.Scene();
  scene.background = HAZE;
  scene.environment = makeEnvironment(renderer); // an environment belongs to the context that built it
  scene.fog = new THREE.Fog(HAZE, 4.5, 11);
  const camera = new THREE.PerspectiveCamera(30, 1, 0.02, 30);
  camera.up.set(0, 0, 1);
  const N = data.cells.length, pitch = data.pitch;
  const y0 = 0.32; // first cell's entry sits at y = 0 in line coordinates
  const span = N * pitch;
  const cy = span / 2;
  const eye = new THREE.Vector3(0.8 + 0.2 * N, -0.36, 0.42 + 0.11 * N); // from the target, in front of the arms
  camera.position.set(0.12 + eye.x, cy + eye.y, 0.05 + eye.z);
  const controls = new OrbitControls(camera, renderer.domElement);
  controls.target.set(0.12, cy, 0.1);
  controls.enableDamping = true;
  controls.maxPolarAngle = Math.PI * 0.48;
  controls.update();

  addStudioLights(scene, { target: [0.12, cy, 0], extent: span * 0.75 + 0.5, reach: 1.1 + 0.25 * N });
  // the lab floor with the whole line taped out as one aisle: a hazard edge down each side and a stencil per cell
  const lab = buildLab(scene, renderer, { cell: [-0.3, y0 - pitch / 2, 0.74, y0 + span - pitch / 2], joints: false });
  hazardStrip(lab, 0.84, y0 - pitch / 2, 0.84, y0 + span - pitch / 2, 0.035);
  hazardStrip(lab, -0.36, y0 - pitch / 2, -0.36, y0 + span - pitch / 2, 0.035);

  // each cell: arm on its floor plate, belt segment, totes, parts; the cells' belts meet end to end
  const cells = [];
  const meshes = {};
  for (let k = 0; k < N; k++) {
    const sc = data.cells[k];
    const rid = sc.robot_id || line.stations[k].robot;
    meshes[rid] ||= await meshesFor(rid);
    const root = new THREE.Group();
    root.position.y = y0 + k * pitch; // cell k's entry meets cell k-1's exit
    scene.add(root);
    const scCut = { ...sc, belt: sc.belt && { ...sc.belt, half: [sc.belt.half[0], pitch / 2, sc.belt.half[2]] } };
    const { bodies, belt } = F().populate(scCut, root, rid, meshes[rid], { belt: true });
    // the cell's belt segment is centred on the cell's own y range
    root.children.filter((c) => c.geometry?.parameters?.height === pitch).forEach((c) => (c.position.y = (-0.32 + 0.34) / 2));
    const plate = basePlate(0.23 * (F().armScale ? F().armScale(rid) : 1));
    root.add(plate);
    const part = sc.blocks[0];
    let cube = null;
    bodies[part.body].traverse((o) => { if (!cube && o.isMesh) cube = o; });
    const partMat = cube.material;
    const st = line.stations[k];
    // the station's stencil on the floor beside its belt, orange for a trained VLA, graphite for a scripted cell
    floorLabel(lab, `STATION ${String(k + 1).padStart(2, "0")}  ${st.controller === "vla" ? "VLA" : "SCRIPTED"}`, 0.5, root.position.y - 0.2, Math.PI / 2, 0.042, st.controller === "vla" ? "rgba(224,86,26,0.95)" : "rgba(44,47,53,0.95)");
    if (k > 0) hazardStrip(lab, -0.3, root.position.y - pitch / 2, 0.74, root.position.y - pitch / 2, 0.012);
    const binBody = sc.bins?.[0]?.body;
    cells.push({ root, bodies, belt, cube, partMat, partBody: part.body, home: sc.home, binBody, done: [] });
    applyCell(cells[k], sc.home, false);
  }
  const stage = makeStage(renderer, scene, camera);
  const ro = new ResizeObserver(() => size());
  ro.observe(box);
  three = { THREE, renderer, scene, camera, controls, cells, ro, data, srgb, eye, stage };
  size();
}

function size() {
  if (!three) return;
  const r = three.renderer.domElement.parentElement.getBoundingClientRect();
  if (!r.width || !r.height) return;
  three.renderer.setSize(r.width, r.height, false);
  three.stage.setSize(r.width, r.height);
  three.camera.aspect = r.width / r.height;
  three.camera.updateProjectionMatrix();
  // portrait screens: look down the line from its entry end, so the stations recede into depth instead of
  // shrinking to fit across; wide screens look at it from the front
  const a = three.camera.aspect, N = three.cells.length;
  const eye = a < 1.3 ? new three.THREE.Vector3(0.75 + 0.1 * N, -(0.75 + 0.42 * N), 0.55 + 0.16 * N) : three.eye;
  const k = Math.max(1, (a < 1.3 ? 0.9 : 1.5) / a);
  const t = three.controls.target;
  three.camera.position.set(t.x + eye.x * k, t.y + eye.y * k, t.z + eye.z * k);
  three.controls.update();
}
window.addEventListener("forge:view", (e) => { if (e.detail === "line") { setTimeout(size, 340); if (!lines.length) refreshList(); } });

function applyCell(cell, frame, showPart) {
  if (!frame) return;
  for (let i = 0; i < cell.bodies.length; i++) {
    const k = i * 7;
    cell.bodies[i].position.set(frame[k], frame[k + 1], frame[k + 2]);
    cell.bodies[i].quaternion.set(frame[k + 4], frame[k + 5], frame[k + 6], frame[k + 3]);
  }
  cell.bodies[cell.partBody].visible = showPart;
  if (cell.belt) cell.belt.tex.offset.y = -(frame[cell.belt.body * 7 + 1] - cell.belt.y0) / 0.06;
}

// ------------------------------------------------------------------ playback on the line clock

function startClock() {
  clock.t = 0;
  clock.playing = true;
  clock.last = performance.now();
  for (const c of three?.cells || []) { c.done.forEach((m) => c.root.remove(m)); c.done = []; }
  const eps = line.result.playback.episodes;
  for (const e of eps) e.end = e.start + e.duration;
  clock.end = Math.max(...eps.map((e) => e.end)) + 1.5;
  clock.finished = new Set();
}

function tick(now) {
  requestAnimationFrame(tick);
  if (!three) return;
  three.controls.update();
  if (clock.playing && line?.result?.playback) {
    const dt = Math.max(0, Math.min(0.1, (now - clock.last) / 1000));
    clock.last = now;
    clock.t += dt * clock.speed;
    const pb = line.result.playback;
    const fps = pb.fps;
    let done = 0;
    three.cells.forEach((cell, k) => {
      const e = pb.episodes.find((x) => x.station === k && clock.t >= x.start && clock.t < x.end);
      if (e) {
        const i = Math.min(e.frames.length - 1, Math.floor((clock.t - e.start) * fps));
        cell.partMat.color.copy(three.srgb(colorOf(e.color)));
        applyCell(cell, e.frames[i], true);
      } else applyCell(cell, cell.home, false);
    });
    for (const e of pb.episodes) {
      if (e.kind !== "pick" || clock.t < e.end) continue;
      if (e.ok) done++;
      const key = `${e.station}:${e.part}`;
      if (!clock.finished.has(key)) { clock.finished.add(key); restPart(e); }
    }
    $(".ln-done").textContent = `${done}/${line.result.n_parts}`;
    $(".ln-tput").textContent = clock.t > 1 ? ((60 * done) / clock.t).toFixed(1) : "0.0";
    $(".ln-clock").textContent = `${Math.min(clock.t, line.result.makespan_s).toFixed(1)} s`;
    if (clock.t > clock.end + 4) startClock(); // loop the run
  }
  three.stage.render();
}
requestAnimationFrame(tick);

const PART_RGB = { red: [0.8, 0.12, 0.1], blue: [0.12, 0.32, 0.8], green: [0.14, 0.6, 0.28], yellow: [0.95, 0.78, 0.1], orange: [1, 0.45, 0.1], purple: [0.5, 0.25, 0.75] };
function colorOf(c) { return three?.data?.block_colors?.[c] || PART_RGB[c] || [0.5, 0.5, 0.5]; }

function restPart(e) { // a finished part stays where the episode left it (in the tote, or dropped)
  const cell = three.cells[e.station];
  const f = e.frames[e.frames.length - 1];
  const k = cell.partBody * 7;
  const m = new three.THREE.Mesh(
    cell.cube.geometry,
    new three.THREE.MeshPhysicalMaterial({ color: three.srgb(colorOf(e.color)), roughness: 0.5, clearcoat: 0.22, transparent: !e.ok, opacity: e.ok ? 1 : 0.55 }),
  );
  m.position.set(f[k], f[k + 1], f[k + 2]);
  m.quaternion.set(f[k + 4], f[k + 5], f[k + 6], f[k + 3]);
  m.castShadow = true;
  cell.root.add(m);
  cell.done.push(m);
}

refreshList().then(() => {
  const last = lines.find((l) => l.status === "done") || lines[0];
  if (last) api(`/api/line/${last.id}`).then(setLine);
});
