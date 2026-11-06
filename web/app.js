import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import { RoomEnvironment } from "three/addons/environments/RoomEnvironment.js";
import { RoundedBoxGeometry } from "three/addons/geometries/RoundedBoxGeometry.js";
import { toCreasedNormals } from "three/addons/utils/BufferGeometryUtils.js";

const $ = (s) => document.querySelector(s);
const el = (tag, cls, text) => {
  const e = document.createElement(tag);
  if (cls) e.className = cls;
  if (text !== undefined) e.textContent = text;
  return e;
};

// ---------------------------------------------------------------- pipeline view
const STAGES = [
  ["task", "Task", "prompt to spec + success check"],
  ["scene", "Scene", "MuJoCo scene, held-out layouts"],
  ["data", "Data", "randomized expert demos"],
  ["trainer", "Trainer", "fine-tune the VLA"],
  ["eval", "Eval", "20 rollouts, measured"],
  ["critic", "Critic", "read failures, plan next round"],
  ["registry", "Registry", "publish as a skill agent"],
];
const stageEls = {};
for (const [key, name, role] of STAGES) {
  const li = el("li", "stage");
  const h = el("div", "stage-h");
  h.append(el("span", "sdot"), el("span", "name", name), el("span", "role", role), el("span", "metric"));
  li.append(h, el("div", "log"));
  $("#pipeline").append(li);
  stageEls[key] = li;
}
function stage(key, state, log, metric) {
  const s = stageEls[key];
  if (!s) return;
  if (state) {
    s.classList.remove("live", "done", "fail");
    if (state !== "idle") s.classList.add(state);
  }
  if (log !== undefined) s.querySelector(".log").textContent = log;
  if (metric !== undefined) s.querySelector(".metric").textContent = metric;
}
function resetStages() {
  for (const [k] of STAGES) stage(k, "idle", "", "");
}

// ---------------------------------------------------------------- 3D
const canvas = $("#three");
const renderer = new THREE.WebGLRenderer({ canvas, antialias: true, preserveDrawingBuffer: true });
renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
renderer.toneMapping = THREE.ACESFilmicToneMapping;
renderer.toneMappingExposure = 1.0;
renderer.shadowMap.enabled = true;
renderer.shadowMap.type = THREE.PCFSoftShadowMap;
const scene3 = new THREE.Scene();
scene3.background = new THREE.Color(0xe9e6e0);
const pmrem = new THREE.PMREMGenerator(renderer);
scene3.environment = pmrem.fromScene(new RoomEnvironment(), 0.04).texture;
scene3.environmentIntensity = 0.75;

const camera = new THREE.PerspectiveCamera(32, 1, 0.01, 20);
camera.up.set(0, 0, 1);
camera.position.set(0.98, -0.66, 0.58);
const controls = new OrbitControls(camera, canvas);
controls.target.set(0.12, 0.0, 0.1);
controls.enableDamping = true;
controls.minDistance = 0.35;
controls.maxDistance = 2.2;
controls.maxPolarAngle = Math.PI * 0.47;
controls.autoRotate = true;
controls.autoRotateSpeed = 0.35;
controls.update();

const key = new THREE.DirectionalLight(0xfff4e6, 2.2);
key.position.set(0.9, -0.6, 1.6);
key.castShadow = true;
key.shadow.mapSize.set(2048, 2048);
key.shadow.camera.left = -0.6; key.shadow.camera.right = 0.6;
key.shadow.camera.top = 0.6; key.shadow.camera.bottom = -0.6;
key.shadow.camera.near = 0.5; key.shadow.camera.far = 3;
key.shadow.radius = 5;
key.shadow.bias = -0.0004;
key.shadow.normalBias = 0.002;
key.target.position.set(0.18, 0, 0);
scene3.add(key, key.target);
const rim = new THREE.DirectionalLight(0xdfe8ff, 0.6);
rim.position.set(-0.8, 0.9, 0.9);
scene3.add(rim);

// table: a real CC0 wood texture (Poly Haven, wood_table_001), softly lit, with a floor that catches shadow
const texLoader = new THREE.TextureLoader();
const woodMap = texLoader.load("/static/assets/wood_diff.jpg");
woodMap.colorSpace = THREE.SRGBColorSpace;
for (const t of [woodMap]) { t.wrapS = t.wrapT = THREE.RepeatWrapping; t.repeat.set(1.2, 1.6); t.anisotropy = 8; }
const tableTop = new THREE.Mesh(
  new RoundedBoxGeometry(0.9, 1.2, 0.04, 4, 0.006),
  new THREE.MeshStandardMaterial({ map: woodMap, color: 0xf3e2cf, roughness: 0.82, envMapIntensity: 0.25 }),
);
tableTop.position.set(0.2, 0, -0.02);
tableTop.receiveShadow = true;
scene3.add(tableTop);
const floor = new THREE.Mesh(new THREE.PlaneGeometry(12, 12), new THREE.ShadowMaterial({ opacity: 0.12 }));
floor.position.z = -0.75;
floor.receiveShadow = true;
scene3.add(floor);
const legMat = new THREE.MeshStandardMaterial({ color: 0x2b2a28, roughness: 0.5, metalness: 0.4 });
for (const [x, y] of [[-0.2, -0.55], [-0.2, 0.55], [0.6, -0.55], [0.6, 0.55]]) {
  const leg = new THREE.Mesh(new THREE.CylinderGeometry(0.018, 0.018, 0.71, 24), legMat);
  leg.rotation.x = Math.PI / 2;
  leg.position.set(x, y, -0.395);
  leg.castShadow = true;
  scene3.add(leg);
}

const MAT = {
  printed: new THREE.MeshPhysicalMaterial({ color: 0xf0bd2c, roughness: 0.42, clearcoat: 0.25, clearcoatRoughness: 0.4, sheen: 0.2 }),
  motor: new THREE.MeshPhysicalMaterial({ color: 0x1b1b1d, roughness: 0.32, metalness: 0.15, clearcoat: 0.4 }),
  other: new THREE.MeshStandardMaterial({ color: 0x9a9a9a, roughness: 0.5 }),
};
const geomCache = new Map();
const meshesP = fetch("/api/meshes").then((r) => r.json());
let MESHES = null;
function meshGeometry(id, data) {
  if (geomCache.has(id)) return geomCache.get(id);
  const v = new Float32Array(Uint8Array.from(atob(data.v), (c) => c.charCodeAt(0)).buffer);
  const f = new Uint32Array(Uint8Array.from(atob(data.f), (c) => c.charCodeAt(0)).buffer);
  let g = new THREE.BufferGeometry();
  g.setAttribute("position", new THREE.BufferAttribute(v, 3));
  g.setIndex(new THREE.BufferAttribute(f, 1));
  g = toCreasedNormals(g, Math.PI / 7);
  geomCache.set(id, g);
  return g;
}
const srgb = (rgb) => new THREE.Color().setRGB(rgb[0], rgb[1], rgb[2], THREE.SRGBColorSpace);

function bowlGeometry(r, h) {
  const t = 0.006;
  const pts = [];
  pts.push(new THREE.Vector2(0, 0.0));
  for (let i = 0; i <= 16; i++) {
    const a = (i / 16) * Math.PI * 0.5;
    pts.push(new THREE.Vector2((r + t) * Math.sin(a) * 0.96 + 0.004, (1 - Math.cos(a)) * h * 1.15));
  }
  pts.push(new THREE.Vector2(r + t * 0.5, h * 1.18 + 0.002));
  for (let i = 16; i >= 0; i--) {
    const a = (i / 16) * Math.PI * 0.5;
    pts.push(new THREE.Vector2(Math.max(r * Math.sin(a) * 0.96, 0.001), (1 - Math.cos(a)) * h * 1.1 + 0.004));
  }
  const g = new THREE.LatheGeometry(pts, 96);
  g.rotateX(Math.PI / 2);
  return g;
}

let world = null; // { root, bodies: Object3D[] }
function buildWorld(sc) {
  if (world) scene3.remove(world.root);
  const root = new THREE.Group();
  const bodies = sc.bodies.map(() => {
    const o = new THREE.Group();
    root.add(o);
    return o;
  });
  for (const g of sc.robot) {
    const geo = meshGeometry(g.mesh, MESHES[String(g.mesh)]);
    const lum = 0.299 * g.rgba[0] + 0.587 * g.rgba[1] + 0.114 * g.rgba[2];
    const mat = lum > 0.5 ? MAT.printed : lum < 0.2 ? MAT.motor : MAT.other;
    const m = new THREE.Mesh(geo, mat);
    m.position.fromArray(g.pos);
    m.quaternion.set(g.quat[1], g.quat[2], g.quat[3], g.quat[0]);
    m.castShadow = true;
    m.receiveShadow = true;
    bodies[g.body].add(m);
  }
  for (const b of sc.blocks) {
    const s = b.half * 2;
    const m = new THREE.Mesh(
      new RoundedBoxGeometry(s, s, s, 4, 0.0025),
      new THREE.MeshPhysicalMaterial({ color: srgb(b.rgb), roughness: 0.38, clearcoat: 0.5, clearcoatRoughness: 0.3 }),
    );
    m.castShadow = m.receiveShadow = true;
    bodies[b.body].add(m);
  }
  for (const b of sc.bowls) {
    const m = new THREE.Mesh(
      bowlGeometry(b.r, b.h),
      new THREE.MeshPhysicalMaterial({ color: srgb(b.rgb), roughness: 0.18, clearcoat: 1.0, clearcoatRoughness: 0.08, side: THREE.DoubleSide }),
    );
    m.castShadow = m.receiveShadow = true;
    bodies[b.body].add(m);
  }
  for (const p of sc.pads) {
    const mat = new THREE.MeshStandardMaterial({ color: srgb(p.rgb), roughness: 0.7 });
    const t = p.half, w = 0.006;
    for (const [x, y, sx, sy] of [[t, 0, w, t + w], [-t, 0, w, t + w], [0, t, t + w, w], [0, -t, t + w, w]]) {
      const m = new THREE.Mesh(new THREE.BoxGeometry(sx * 2, sy * 2, 0.0012), mat);
      m.position.set(x, y, 0.0006);
      m.receiveShadow = true;
      bodies[p.body].add(m);
    }
  }
  scene3.add(root);
  world = { root, bodies, n: sc.bodies.length };
}
function applyFrame(frame) {
  if (!world || !frame) return;
  const n = world.n;
  for (let i = 0; i < n; i++) {
    const k = i * 7; // per body: x y z, then quaternion w x y z (MuJoCo order)
    world.bodies[i].position.set(frame[k], frame[k + 1], frame[k + 2]);
    world.bodies[i].quaternion.set(frame[k + 4], frame[k + 5], frame[k + 6], frame[k + 3]);
  }
}

// playback of the policy's own recorded rollout (body poses at 30 Hz)
let play = null; // { frames, fps, t0 }
function startPlayback(payload) {
  buildWorld(payload.scene);
  play = { frames: payload.frames, fps: payload.fps, t0: performance.now() };
  applyFrame(payload.frames[0]);
}
function resize() {
  const r = canvas.parentElement.getBoundingClientRect();
  renderer.setSize(r.width, r.height, false);
  camera.aspect = r.width / r.height;
  camera.updateProjectionMatrix();
}
new ResizeObserver(resize).observe(canvas.parentElement);
resize();
function tick() {
  if (play) {
    const hold = 1.2;
    const dur = play.frames.length / play.fps;
    const t = ((performance.now() - play.t0) / 1000) % (dur + hold);
    const fi = Math.min(play.frames.length - 1, Math.floor(t * play.fps));
    applyFrame(play.frames[fi]);
    $("#vp-progress div").style.width = `${(100 * Math.min(t, dur)) / dur}%`;
  }
  controls.update();
  renderer.render(scene3, camera);
  requestAnimationFrame(tick);
}
tick();
canvas.addEventListener("pointerdown", () => { controls.autoRotate = false; });

// ---------------------------------------------------------------- run state
let current = null; // run id
let runStart = 0;
let runDone = 0;
let rounds = []; // [{index, successes, eval, plan}]
let sel = { round: -1, index: -1 };
const losses = []; // per round arrays of [t, loss]

function fmt(s) {
  s = Math.max(0, Math.floor(s));
  return `${Math.floor(s / 60)}:${String(s % 60).padStart(2, "0")}`;
}
setInterval(() => {
  if (!current || !runStart) return;
  const end = runDone || Date.now() / 1000;
  $("#elapsed").textContent = fmt(end - runStart);
}, 200);

function drawLoss() {
  const svg = $("#loss");
  const all = losses.flat();
  if (!all.length) { svg.innerHTML = ""; return; }
  const tMax = Math.max(...losses.map((l) => (l.length ? l[l.length - 1][0] : 0))) || 1;
  const vals = all.map((p) => Math.log(p[1]));
  const lo = Math.min(...vals), hi = Math.max(...vals);
  const W = 300, H = 110, pad = 6;
  let off = 0;
  const paths = losses.map((l, ri) => {
    const d = l.map((p, i) => {
      const x = pad + ((off + p[0]) / (tMaxTotal())) * (W - 2 * pad);
      const y = pad + (1 - (Math.log(p[1]) - lo) / Math.max(hi - lo, 1e-6)) * (H - 2 * pad);
      return `${i ? "L" : "M"}${x.toFixed(1)},${y.toFixed(1)}`;
    }).join("");
    off += l.length ? l[l.length - 1][0] : 0;
    return `<path class="l${ri ? 1 : 0}" d="${d}"/>`;
  });
  svg.innerHTML = `<line x1="0" y1="${H - 1}" x2="${W}" y2="${H - 1}"/>` + paths.join("");
  void tMax;
}
function tMaxTotal() {
  return losses.reduce((a, l) => a + (l.length ? l[l.length - 1][0] : 0), 0) || 1;
}

function drawRounds() {
  const box = $("#rounds");
  box.innerHTML = "";
  for (const r of rounds) {
    if (!r.eval) continue;
    const row = el("div", "round");
    row.append(el("span", "", `Round ${r.index + 1}`));
    const bar = el("div", "bar");
    const fill = el("div");
    fill.style.width = `${100 * r.eval.success_rate}%`;
    bar.append(fill);
    row.append(bar, el("span", "pct", `${r.eval.succeeded}/${r.eval.n}`));
    box.append(row);
  }
  const tabs = $("#round-tabs");
  tabs.innerHTML = "";
  const evald = rounds.filter((r) => r.successes);
  if (evald.length > 1) {
    for (const r of evald) {
      const b = el("button", r.index === sel.round ? "on" : "", `Round ${r.index + 1}`);
      b.onclick = () => showRound(r.index);
      tabs.append(b);
    }
  }
}
function showRound(ri, index) {
  const r = rounds[ri];
  if (!r || !r.successes) return;
  sel.round = ri;
  const chips = $("#chips");
  chips.innerHTML = "";
  r.successes.forEach((ok, i) => {
    const c = el("button", `chip ${ok ? "ok" : "bad"}`, String(i + 1));
    c.title = `rollout ${i + 1}: ${ok ? "success" : "failed"}`;
    c.onclick = () => loadRollout(ri, i);
    chips.append(c);
  });
  drawRounds();
  const first = index ?? Math.max(0, r.successes.indexOf(true));
  loadRollout(ri, first);
}
async function loadRollout(ri, i) {
  sel = { round: ri, index: i };
  document.querySelectorAll(".chip").forEach((c, k) => c.classList.toggle("sel", k === i));
  const res = await fetch(`/api/runs/${current}/rollout/${ri}/${i}`);
  if (!res.ok) return;
  const p = await res.json();
  startPlayback(p);
  const badge = $("#vp-badge");
  badge.classList.remove("hidden", "ok", "bad");
  badge.classList.add(p.success ? "ok" : "bad");
  badge.textContent = p.success ? `Rollout ${i + 1} · success` : `Rollout ${i + 1} · ${p.failure}`;
  $("#vp-progress").classList.remove("hidden");
}
function pendingChips(done, n) {
  const chips = $("#chips");
  if (chips.children.length !== n || !chips.dataset.pending) {
    chips.innerHTML = "";
    chips.dataset.pending = "1";
    for (let i = 0; i < n; i++) chips.append(el("span", "chip pending"));
  }
  void done;
}

async function showPreview(rid) {
  const res = await fetch(`/api/runs/${rid}/scene`);
  if (!res.ok) return;
  const sc = await res.json();
  play = null;
  buildWorld(sc);
  applyFrame(sc.frame);
  $("#vp-progress").classList.add("hidden");
  $("#vp-badge").classList.add("hidden");
}
// home pose for the idle scene
async function idleScene() {
  MESHES = await meshesP;
  const st = await (await fetch("/api/state")).json();
  renderSkills(st.skills);
  setReady(st.ready);
  const last = st.runs[st.runs.length - 1];
  if (last && last.status === "published") {
    adoptRun(last);
  } else {
    const sc = await (await fetch("/api/preview")).json();
    buildWorld(sc);
    applyFrame(sc.frame);
  }
}

function adoptRun(summary) {
  current = summary.id;
  $("#run-title").textContent = summary.spec ? summary.spec.instruction : summary.prompt;
  runStart = 1; runDone = 1 + summary.elapsed;
  $("#elapsed").textContent = fmt(summary.elapsed);
  rounds = summary.rounds.map((r) => ({ ...r }));
  losses.length = 0;
  for (const r of summary.rounds) losses.push(r.loss.map(([t, l]) => [t, l]));
  drawLoss();
  const last = summary.rounds[summary.rounds.length - 1];
  if (last && last.train && last.train.steps) $("#loss-steps").textContent = `${last.train.steps} steps`;
  if (last && last.loss.length) $("#loss-now").textContent = last.loss[last.loss.length - 1][1].toFixed(3);
  if (summary.init_from && summary.init_from.reason) $("#loss-init").textContent = `init: ${summary.init_from.reason}`;
  $("#critic").innerHTML = "";
  for (const r of summary.rounds) {
    if (!r.plan || !r.plan.action) continue;
    const p = el("div");
    p.append(el("b", "", `Round ${r.index + 1}: ${r.plan.action}. `), document.createTextNode(r.plan.rationale));
    $("#critic").append(p);
  }
  for (const [k] of STAGES) stage(k, "done");
  drawRounds();
  if (summary.best_round >= 0) showRound(summary.best_round);
}

// ---------------------------------------------------------------- skills
function renderSkills(skills, fresh) {
  const box = $("#skills");
  box.innerHTML = "";
  $("#skill-count").textContent = skills.length ? `${skills.length} skill agent${skills.length > 1 ? "s" : ""}` : "";
  const byId = Object.fromEntries(skills.map((s) => [s.id, s]));
  for (const s of skills) {
    const card = el("div", `skill${s.id === fresh ? " new" : ""}`);
    card.append(el("div", "ins", s.instruction), el("div", "pct", `${Math.round(100 * s.success_rate)}%`));
    const meta = el("div", "meta");
    const parent = s.parent ? (byId[s.parent] ? `from "${byId[s.parent].instruction.slice(0, 22)}…"` : "from a skill") : "from base";
    meta.append(el("span", "", `${Math.round(s.seconds)}s`), el("span", "", parent));
    const b = el("button", "", "Call agent");
    const reply = el("div", "reply");
    b.onclick = async () => {
      b.disabled = true;
      reply.textContent = "running in a fresh scene…";
      const r = await fetch(`/api/skills/${s.id}/run`, { method: "POST" });
      reply.textContent = r.ok ? (await r.json()).reply : "no answer";
      b.disabled = false;
    };
    meta.append(b);
    card.append(meta, reply);
    box.append(card);
  }
}
async function refreshSkills(fresh) {
  const st = await (await fetch("/api/state")).json();
  renderSkills(st.skills, fresh);
}
function setReady(on) {
  $("#ready-dot").classList.toggle("on", on);
  $("#ready-text").textContent = on ? "agents online" : "warming up";
  $("#go").disabled = !on;
}

// ---------------------------------------------------------------- events
const es = new EventSource("/api/events");
es.onmessage = (m) => {
  const e = JSON.parse(m.data);
  if (e.kind === "ready") { setReady(true); return; }
  if (e.kind === "agents") return;
  if (e.run !== current) {
    if (e.kind !== "log" || !e.text.startsWith("chat message")) return;
    beginRun(e.run);
  }
  handle(e);
};
function beginRun(rid) {
  current = rid;
  runStart = Date.now() / 1000;
  runDone = 0;
  rounds = [];
  losses.length = 0;
  drawLoss();
  drawRounds();
  resetStages();
  $("#critic").innerHTML = "";
  $("#chips").innerHTML = "";
  $("#round-tabs").innerHTML = "";
  $("#loss-now").textContent = "";
  $("#loss-steps").textContent = "";
  $("#loss-init").textContent = "";
  $("#vp-badge").classList.add("hidden");
  $("#go").disabled = true;
}
function handle(e) {
  switch (e.kind) {
    case "log":
    {
      const s = stageEls[e.agent];
      if (!s) break;
      const settled = s.classList.contains("done") || s.classList.contains("fail");
      stage(e.agent, settled ? null : "live", e.text);
      break;
    }
    case "task":
      $("#run-title").textContent = e.spec.instruction;
      $("#vp-caption").textContent = `“${e.spec.instruction}”`;
      stage("task", "done");
      break;
    case "rejected":
      stage("task", "fail", e.reason);
      runDone = Date.now() / 1000;
      $("#go").disabled = false;
      break;
    case "error":
      stage(e.agent, "fail", e.error);
      runDone = Date.now() / 1000;
      $("#go").disabled = false;
      break;
    case "scene":
      stage("scene", "done");
      showPreview(e.run);
      break;
    case "demos":
      stage("data", "live", undefined, `${e.ok}/${e.n}`);
      if (e.done === e.n) stage("data", "done");
      break;
    case "init":
      $("#loss-init").textContent = e.skill ? `init: ${e.reason.replace(/^fine-tuning from /, "")}` : "init: base policy";
      break;
    case "loss": {
      while (losses.length <= e.round) losses.push([]);
      losses[e.round].push([e.t, e.loss]);
      $("#loss-now").textContent = e.ema.toFixed(3);
      $("#loss-steps").textContent = `${e.step} steps`;
      stage("trainer", "live", undefined, `${Math.round(e.t)}s`);
      drawLoss();
      break;
    }
    case "eval_progress":
      stage("trainer", "done");
      stage("eval", "live", undefined, `${e.succeeded}/${e.n}`);
      pendingChips(e.done, e.n);
      break;
    case "eval": {
      stage("eval", "done", undefined, `${e.succeeded}/${e.n}`);
      rounds[e.round] = { index: e.round, eval: e, successes: e.successes };
      $("#chips").dataset.pending = "";
      showRound(e.round);
      break;
    }
    case "critic": {
      stage("critic", "done", `[${e.model}] ${e.plan.action}`);
      if (rounds[e.round]) rounds[e.round].plan = e.plan;
      const c = $("#critic");
      const b = el("b", "", e.plan.action === "retrain" ? `Round ${e.round + 1}: retrain. ` : `Round ${e.round + 1}: publish. `);
      const p = el("div");
      p.append(b, document.createTextNode(e.plan.rationale));
      c.append(p);
      if (e.plan.action === "retrain") {
        for (const k of ["data", "trainer", "eval"]) stage(k, "idle");
      }
      break;
    }
    case "published":
      runDone = Date.now() / 1000;
      $("#elapsed").textContent = fmt(e.seconds);
      runDone = runStart + e.seconds;
      stage("registry", "done", undefined, `${Math.round(100 * e.success_rate)}%`);
      refreshSkills(e.skill);
      if (e.best_round !== sel.round) showRound(e.best_round);
      $("#go").disabled = false;
      break;
  }
}

// ---------------------------------------------------------------- prompt
$("#prompt-form").addEventListener("submit", async (ev) => {
  ev.preventDefault();
  const text = $("#prompt").value.trim();
  if (!text) return;
  $("#go").disabled = true;
  await fetch("/api/prompt", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ prompt: text }) });
});
document.querySelectorAll(".ex").forEach((b) => b.addEventListener("click", () => { $("#prompt").value = b.textContent; $("#prompt").focus(); }));

idleScene();

// ---------------------------------------------------------------- live tab capture (?capture=1, demo recording)
if (new URLSearchParams(location.search).has("capture")) {
  let rec = null;
  const chunks = [];
  window.startCapture = async () => {
    const stream = await navigator.mediaDevices.getDisplayMedia({
      video: { frameRate: 30, width: 1920, height: 1200 },
      audio: false,
      preferCurrentTab: true,
    });
    rec = new MediaRecorder(stream, { mimeType: "video/webm;codecs=vp9", videoBitsPerSecond: 12_000_000 });
    rec.ondataavailable = (e) => e.data.size && chunks.push(e.data);
    rec.start(1000);
    return true;
  };
  window.stopCapture = () =>
    new Promise((resolve) => {
      rec.onstop = async () => {
        rec.stream.getTracks().forEach((t) => t.stop());
        const r = await fetch("/api/capture", { method: "POST", body: new Blob(chunks, { type: "video/webm" }) });
        resolve(await r.json());
      };
      rec.stop();
    });
}
