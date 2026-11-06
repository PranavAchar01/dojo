import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import { RoomEnvironment } from "three/addons/environments/RoomEnvironment.js";
import { RoundedBoxGeometry } from "three/addons/geometries/RoundedBoxGeometry.js";
import { toCreasedNormals } from "three/addons/utils/BufferGeometryUtils.js";
import { initBuilder, showBuilder } from "./builder.js";

const $ = (s) => document.querySelector(s);
const el = (tag, cls, text) => {
  const e = document.createElement(tag);
  if (cls) e.className = cls;
  if (text !== undefined) e.textContent = text;
  return e;
};
const svgIcon = (d) => {
  const s = document.createElementNS("http://www.w3.org/2000/svg", "svg");
  s.setAttribute("viewBox", "0 0 20 20");
  s.innerHTML = d;
  return s;
};

// ---------------------------------------------------------------- stage rail (one entry per agent)
const STAGES = [
  ["task", "Task", '<path d="M5 5h10M5 10h10M5 15h6"/>'],
  ["planner", "Plan", '<circle cx="5" cy="5" r="1.8"/><circle cx="5" cy="15" r="1.8"/><circle cx="15" cy="10" r="1.8"/><path d="M6.6 5.8 13.4 9.2M6.6 14.2l6.8-3.4"/>'],
  ["scene", "Scene", '<path d="M3 13.5 10 17l7-3.5M3 9.5 10 13l7-3.5L10 6z"/>'],
  ["data", "Demos", '<path d="M4 16V9M8 16V5M12 16v-5M16 16V7"/>'],
  ["trainer", "Train", '<path d="M3 15c3 0 3-9 7-9s4 6 7 6"/>'],
  ["eval", "Eval", '<path d="m5 10.5 3 3 7-7"/>'],
  ["critic", "Critic", '<circle cx="9" cy="9" r="4.8"/><path d="m12.6 12.6 3.9 3.9"/>'],
  ["executor", "Run", '<path d="M6.5 4.5v11l8.5-5.5z"/>'],
  ["registry", "Publish", '<path d="M10 3.5v9M6 8.5l4 4 4-4M4 16.5h12"/>'],
];
const NAMES = Object.fromEntries(STAGES.map(([k, n]) => [k, n]));
NAMES.trainer = "Trainer"; NAMES.data = "Data"; NAMES.planner = "Planner"; NAMES.executor = "Executor"; NAMES.registry = "Registry";
const rail = {};
for (const [key, name, icon] of STAGES) {
  if (key === "executor") $("#rail").append(el("li", "sep"));
  const li = el("li");
  li.append(svgIcon(icon), el("span", "", name), el("span", "m"));
  if (key === "planner" || key === "executor") li.classList.add("off");
  $("#rail").append(li);
  rail[key] = li;
}
function stage(key, state, metric) {
  const s = rail[key];
  if (!s) return;
  if (state) {
    s.classList.remove("live", "done", "fail");
    if (state !== "idle") s.classList.add(state);
  }
  if (metric !== undefined) s.querySelector(".m").textContent = metric;
}
function resetStages(composite) {
  for (const [k] of STAGES) stage(k, "idle", "");
  rail.planner.classList.toggle("off", !composite);
  rail.executor.classList.toggle("off", !composite);
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
scene3.background = new THREE.Color(0xf1f1f2);
const pmrem = new THREE.PMREMGenerator(renderer);
scene3.environment = pmrem.fromScene(new RoomEnvironment(), 0.04).texture;
scene3.environmentIntensity = 0.75;

const camera = new THREE.PerspectiveCamera(32, 1, 0.01, 20);
camera.up.set(0, 0, 1);
camera.position.set(1.16, -0.78, 0.68);
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
const meshStore = {}; // robot id -> promise of its mesh buffers (fetched once per robot)
const meshesFor = (rid) => (meshStore[rid] ||= fetch(`/api/meshes?robot=${encodeURIComponent(rid)}`).then((r) => r.json()));
const robotMats = new Map();
function robotMaterial(rid, rgba) {
  const lum = 0.299 * rgba[0] + 0.587 * rgba[1] + 0.114 * rgba[2];
  if (rid === "so101") return lum > 0.5 ? MAT.printed : lum < 0.2 ? MAT.motor : MAT.other;
  // other embodiments keep the colours their source models ship with
  const k = rgba.map((x) => x.toFixed(2)).join(",");
  if (!robotMats.has(k)) {
    const dim = lum > 0.8 ? 0.72 / lum : 1; // near-white parts wash out under the key light
    const c = srgb(rgba.slice(0, 3).map((x) => x * dim));
    robotMats.set(k, new THREE.MeshPhysicalMaterial({ color: c, roughness: 0.42, clearcoat: 0.25, clearcoatRoughness: 0.4 }));
  }
  return robotMats.get(k);
}
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
async function buildWorld(sc) {
  const rid = sc.robot_id || "so101";
  const meshes = await meshesFor(rid);
  if (world) scene3.remove(world.root);
  const root = new THREE.Group();
  const bodies = sc.bodies.map(() => {
    const o = new THREE.Group();
    root.add(o);
    return o;
  });
  for (const g of sc.robot) {
    const geo = meshGeometry(`${rid}|${g.mesh}`, meshes[String(g.mesh)]);
    const m = new THREE.Mesh(geo, robotMaterial(rid, g.rgba));
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
async function startPlayback(payload) {
  await buildWorld(payload.scene);
  play = { frames: payload.frames, fps: payload.fps, t0: performance.now(), phases: payload.phase_frames || null, instructions: payload.instructions || [], step: -1 };
  $("#step-badge").classList.add("hidden");
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
const CAMS = {
  orbit: { pos: [1.16, -0.78, 0.68], target: [0.12, 0.0, 0.1], rotate: true },
  front: { pos: [1.2, -0.02, 0.52], target: [0.16, 0.0, 0.06], rotate: false },
  top: { pos: [0.2, -0.06, 1.05], target: [0.2, 0.0, 0.0], rotate: false },
};
function setCam(name) {
  const c = CAMS[name];
  camera.position.fromArray(c.pos);
  controls.target.fromArray(c.target);
  controls.autoRotate = c.rotate;
  controls.update();
  document.querySelectorAll("[data-cam]").forEach((b) => b.classList.toggle("on", b.dataset.cam === name));
}
document.querySelectorAll("[data-cam]").forEach((b) => b.addEventListener("click", () => setCam(b.dataset.cam)));

function tick() {
  if (play) {
    const hold = 1.2;
    const dur = play.frames.length / play.fps;
    const t = ((performance.now() - play.t0) / 1000) % (dur + hold);
    const fi = Math.min(play.frames.length - 1, Math.floor(t * play.fps));
    applyFrame(play.frames[fi]);
    $("#pb-fill").style.width = `${(100 * Math.min(t, dur)) / dur}%`;
    if (play.phases) {
      let k = 0;
      while (k + 1 < play.phases.length && fi >= play.phases[k + 1]) k++;
      if (k !== play.step) {
        play.step = k;
        const sb = $("#step-badge");
        sb.innerHTML = "";
        sb.append(el("b", "", `Step ${k + 1}/${play.phases.length}`), document.createTextNode(` \u00b7 ${play.instructions[k]}`));
        sb.classList.remove("hidden");
      }
    }
  }
  if (!$("#view-sim").classList.contains("hidden")) {
    controls.update();
    renderer.render(scene3, camera);
  }
  requestAnimationFrame(tick);
}
tick();
canvas.addEventListener("pointerdown", () => { controls.autoRotate = false; });

// ---------------------------------------------------------------- run state
let current = null; // the prompt's run id
let composite = false;
const family = new Map(); // child run id -> step index
let runStart = 0;
let runDone = 0;
const roundsBy = {}; // run id -> [{index, eval, successes, plan}]
let shown = { run: null, round: -1, index: -1 }; // rollout on screen
let losses = []; // per round arrays of [t, loss] of the run being trained
let planRows = [];
let lastSkill = null;
let skillsCache = [];

function fmt(s) {
  s = Math.max(0, Math.floor(s));
  return `${Math.floor(s / 60)}:${String(s % 60).padStart(2, "0")}`;
}
setInterval(() => {
  if (!current || !runStart) return;
  const end = runDone || Date.now() / 1000;
  $("#elapsed").textContent = fmt(end - runStart);
}, 200);

// ---------------------------------------------------------------- chat thread
const thread = $("#thread");
const toolEls = {}; // `${run}:${agent}` -> element
const toolLists = {}; // run id -> tools-list element
let forgeMsg = null;
function scrollThread() { thread.scrollTop = thread.scrollHeight; }
function youMsg(text) {
  $("#empty")?.remove();
  const m = el("div", "msg you");
  m.append(el("div", "who", "You"), el("div", "txt", text));
  thread.append(m);
  forgeMsg = el("div", "msg forge");
  forgeMsg.append(el("div", "who", "Forge"));
  const list = el("div", "tools-list");
  forgeMsg.append(list);
  thread.append(forgeMsg);
  scrollThread();
  return list;
}
function forgeNote(text, run) {
  const parent = run && run !== current && toolLists[run] ? toolLists[run] : forgeMsg?.querySelector(".tools-list");
  if (!parent) return;
  parent.append(el("div", "note", text));
  scrollThread();
}
function tool(run, agent, text, state) {
  const list = toolLists[run];
  if (!list || agent === "gateway") return;
  const k = `${run}:${agent}`;
  let t = toolEls[k];
  if (!t) {
    t = el("div", "tool");
    t.append(el("div", "tx"), el("div", "st"));
    list.append(t);
    toolEls[k] = t;
  }
  if (text !== undefined) {
    const tx = t.querySelector(".tx");
    tx.innerHTML = "";
    tx.append(el("b", "", `${NAMES[agent] || agent}: `), document.createTextNode(text));
  }
  if (state) {
    t.classList.remove("live", "done", "fail");
    t.classList.add(state);
    const st = t.querySelector(".st");
    st.innerHTML = state === "live" ? '<span class="spin"></span>working' : state === "done" ? "✓ done" : "failed";
  }
  scrollThread();
}
const settledEarly = {}; // a step's result event can land before its log line; the line then arrives settled
function settle(run, agent, state = "done") {
  const t = toolEls[`${run}:${agent}`];
  if (t) tool(run, agent, undefined, state);
  else settledEarly[`${run}:${agent}`] = state;
}
$("#new-chat").addEventListener("click", () => {
  thread.innerHTML = "";
  for (const k of Object.keys(toolEls)) delete toolEls[k];
});

// ---------------------------------------------------------------- inspector
function drawLoss() {
  const svg = $("#loss");
  const all = losses.flat();
  if (!all.length) {
    svg.innerHTML = "";
    if (composite && planRows.length && planRows.every((p) => p.source === "reuse")) $("#loss-steps").textContent = "no training: every step reused a skill agent";
    return;
  }
  const total = losses.reduce((a, l) => a + (l.length ? l[l.length - 1][0] : 0), 0) || 1;
  const vals = all.map((p) => Math.log(p[1]));
  const lo = Math.min(...vals), hi = Math.max(...vals);
  const W = 300, H = 96, pad = 6;
  let off = 0;
  const paths = losses.map((l, ri) => {
    const d = l.map((p, i) => {
      const x = pad + ((off + p[0]) / total) * (W - 2 * pad);
      const y = pad + (1 - (Math.log(p[1]) - lo) / Math.max(hi - lo, 1e-6)) * (H - 2 * pad);
      return `${i ? "L" : "M"}${x.toFixed(1)},${y.toFixed(1)}`;
    }).join("");
    off += l.length ? l[l.length - 1][0] : 0;
    return `<path class="l${ri ? 1 : 0}" d="${d}"/>`;
  });
  svg.innerHTML = `<line x1="0" y1="${H - 1}" x2="${W}" y2="${H - 1}"/>` + paths.join("");
}
function resetLoss() {
  losses = [];
  drawLoss();
  $("#loss-now").textContent = "";
  $("#loss-steps").textContent = "";
  $("#loss-init").textContent = "";
}
function barRow(label, k, n) {
  const row = el("div", "round");
  row.append(el("span", "", label));
  const bar = el("div", "bar");
  const fill = el("div");
  fill.style.width = `${(100 * k) / Math.max(n, 1)}%`;
  bar.append(fill);
  row.append(bar, el("span", "pct", `${k}/${n}`));
  return row;
}
function drawRounds() {
  const box = $("#rounds");
  box.innerHTML = "";
  if (composite) {
    const ev = (roundsBy[current] || [])[0]?.eval;
    if (ev) {
      ev.per_step.forEach((k, i) => box.append(barRow(`Step ${i + 1}`, k, ev.n)));
      box.append(barRow("All steps", ev.succeeded, ev.n));
    } else {
      box.append(el("span", "dim", "Measured end to end once every step has a skill agent."));
    }
    return;
  }
  for (const r of roundsBy[current] || []) {
    if (r && r.eval) box.append(barRow(`Round ${r.index + 1}`, r.eval.succeeded, r.eval.n));
  }
}
function renderPlan(plan, note) {
  planRows = plan || [];
  const box = $("#plan");
  box.innerHTML = "";
  $("#plan-sec").classList.toggle("hidden", !composite);
  if (note !== undefined) $("#plan-note").textContent = note;
  for (const p of planRows) {
    const li = el("li");
    const trained = p.child && !p.skill_id ? "train" : p.source;
    li.append(el("div", "", `${p.step + 1}. ${p.instruction}`));
    const row = el("div", "row");
    const src = el("span", `src ${trained === "train" ? "train" : ""}`, p.source === "reuse" ? "reuse" : p.skill_id ? `trained ${Math.round(p.seconds)}s` : p.child ? "training" : "to train");
    row.append(src, el("span", "", p.skill_id ? `skill-${p.skill_id.slice(-4)} · ${p.success_rate != null ? Math.round(100 * p.success_rate) + "%" : ""}` : `tier ${p.tier}`));
    li.append(row);
    if (p.child && !p.skill_id) li.classList.add("live");
    box.append(li);
  }
  drawRounds();
}

// ---------------------------------------------------------------- rollouts (bottom tabs + playbar)
function evaluated(run) { return (roundsBy[run] || []).filter((r) => r && r.successes); }
function roundLabel() {
  const rs = evaluated(shown.run);
  const r = (roundsBy[shown.run] || [])[shown.round];
  if (!r) return "Rollouts";
  const step = family.has(shown.run) ? `Step ${family.get(shown.run) + 1} · ` : "";
  if (shown.run === current && composite) return "Chained run";
  return `${step}Round ${rs.indexOf(r) + 1}/${rs.length}`;
}
function showRound(run, ri, index) {
  const r = (roundsBy[run] || [])[ri];
  if (!r || !r.successes) return;
  shown = { run, round: ri, index: -1 };
  const chips = $("#chips");
  chips.innerHTML = "";
  r.successes.forEach((ok, i) => {
    const c = el("button", `rtab ${ok ? "ok" : "bad"}`);
    c.append(el("span", "o"), el("span", "", ok ? "success" : "failed"), el("span", "n", String(i + 1)));
    c.title = `rollout ${i + 1}`;
    c.onclick = () => loadRollout(run, ri, i);
    chips.append(c);
  });
  $("#round-label").textContent = roundLabel();
  drawRounds();
  const first = index ?? Math.max(0, r.successes.indexOf(true));
  loadRollout(run, ri, first);
}
async function loadRollout(run, ri, i) {
  shown = { run, round: ri, index: i };
  document.querySelectorAll(".rtab").forEach((c, k) => c.classList.toggle("sel", k === i));
  $("#chips").children[i]?.scrollIntoView({ block: "nearest", inline: "nearest" });
  const res = await fetch(`/api/runs/${run}/rollout/${ri}/${i}`);
  if (!res.ok) return;
  const p = await res.json();
  if (shown.index !== i || shown.run !== run) return;
  await startPlayback(p);
  const badge = $("#vp-badge");
  badge.classList.remove("hidden", "ok", "bad");
  badge.classList.add(p.success ? "ok" : "bad");
  badge.textContent = p.success ? `Rollout ${i + 1} · success` : `Rollout ${i + 1} · ${p.failure}`;
  $("#pb-label").textContent = `rollout ${i + 1}/${(roundsBy[run][ri].successes || []).length}`;
}
function pendingTabs(n, label) {
  const chips = $("#chips");
  if (chips.dataset.pending === label && chips.children.length === n) return;
  chips.innerHTML = "";
  chips.dataset.pending = label;
  for (let i = 0; i < n; i++) {
    const c = el("span", "rtab pending");
    c.append(el("span", "o"), el("span", "", "running"), el("span", "n", String(i + 1)));
    chips.append(c);
  }
  $("#round-label").textContent = label;
}
$("#pb-prev").onclick = () => {
  const r = (roundsBy[shown.run] || [])[shown.round];
  if (r) loadRollout(shown.run, shown.round, (shown.index - 1 + r.successes.length) % r.successes.length);
};
$("#pb-next").onclick = () => {
  const r = (roundsBy[shown.run] || [])[shown.round];
  if (r) loadRollout(shown.run, shown.round, (shown.index + 1) % r.successes.length);
};
$("#pb-replay").onclick = () => { if (play) { play.t0 = performance.now(); play.step = -1; } };
$("#round-pick").onclick = () => {
  const rs = evaluated(shown.run);
  if (rs.length < 2) return;
  const k = rs.findIndex((r) => r.index === shown.round);
  showRound(shown.run, rs[(k + 1) % rs.length].index);
};

async function showPreview(rid) {
  const res = await fetch(`/api/runs/${rid}/scene`);
  if (!res.ok) return;
  const sc = await res.json();
  play = null;
  await buildWorld(sc);
  applyFrame(sc.frame);
  $("#pb-fill").style.width = "0";
  $("#vp-badge").classList.add("hidden");
  $("#step-badge").classList.add("hidden");
}

// ---------------------------------------------------------------- views
document.querySelectorAll(".seg button").forEach((b) =>
  b.addEventListener("click", () => setView(b.dataset.view)),
);
function setView(v) {
  document.querySelectorAll(".seg button").forEach((b) => b.classList.toggle("on", b.dataset.view === v));
  for (const k of ["sim", "robot", "ladder", "skills"]) $(`#view-${k}`).classList.toggle("hidden", k !== v);
  showBuilder(v === "robot");
  $(".tabs").classList.toggle("hidden", v !== "sim");
  if (v === "ladder") renderLadder();
  if (v === "skills") refreshSkills();
  if (v === "sim") resize();
}
window.setView = setView;
$("#toggle-left").onclick = () => { $("#app").classList.toggle("no-left"); resize(); };
$("#toggle-right").onclick = () => { $("#app").classList.toggle("no-right"); resize(); };

const TIER_NAMES = {
  1: ["Tier 1", "one motion: reach, push, lift"],
  2: ["Tier 2", "carry a block somewhere: place, stack"],
  3: ["Tier 3", "several steps, chained from published skill agents"],
};
async function renderLadder() {
  const st = await (await fetch("/api/state")).json();
  const skills = Object.fromEntries(st.skills.map((s) => [s.id, s]));
  const box = $("#ladder");
  box.innerHTML = "";
  const done = st.runs.filter((r) => r.status === "published");
  if (!done.length) { box.append(el("div", "empty-note", "No prompts finished yet.")); return; }
  const tmax = Math.max(180, ...done.map((r) => r.elapsed));
  for (const tier of [1, 2, 3]) {
    const rows = done.filter((r) => r.tier === tier);
    if (!rows.length) continue;
    const blk = el("div", "tier-block");
    const th = el("div", "th");
    th.append(el("b", "", TIER_NAMES[tier][0]), el("span", "", TIER_NAMES[tier][1]));
    blk.append(th);
    const tb = el("table");
    tb.innerHTML = "<tr><th>Prompt</th><th>Started from</th><th>Time</th><th>Success</th></tr>";
    for (const r of rows) {
      const tr = el("tr");
      const best = r.rounds[r.best_round] || r.rounds[r.rounds.length - 1];
      const ev = best && best.eval;
      let from;
      if (tier === 3) {
        const reuse = r.plan.filter((p) => p.source === "reuse").length;
        const trained = r.plan.filter((p) => p.source === "train");
        from = `calls ${r.plan.length} skill agents; reused ${reuse}` + (trained.length ? `, trained ${trained.length} (${trained.map((p) => Math.round(p.seconds) + "s").join(", ")})` : "");
      } else if (r.parent) {
        const src = r.init_from && r.init_from.skill ? `, fine-tuned from "${skills[r.init_from.skill]?.instruction || r.init_from.skill}"` : ", from the base policy";
        from = `step ${r.step_index + 1} of a tier 3 task${src}`;
      } else if (r.init_from && r.init_from.skill) {
        from = `fine-tuned from "${skills[r.init_from.skill]?.instruction || r.init_from.skill}"`;
      } else {
        from = "the multi-task base policy";
      }
      const rounds = tier === 3 ? "" : ` · ${r.rounds.length} round${r.rounds.length > 1 ? "s" : ""}`;
      const tdP = el("td", "", r.spec ? r.spec.instruction : r.prompt);
      const tdF = el("td", "from", from);
      const tdT = el("td", "num");
      const bar = el("span", "tbar");
      bar.style.width = `${(80 * r.elapsed) / tmax}px`;
      tdT.append(bar, document.createTextNode(fmt(r.elapsed) + rounds));
      const tdS = el("td", "num", ev ? `${ev.succeeded}/${ev.n}` : "");
      tr.append(tdP, tdF, tdT, tdS);
      tb.append(tr);
    }
    blk.append(tb);
    box.append(blk);
  }
}

function renderSkills(skills, fresh) {
  skillsCache = skills;
  const box = $("#skills");
  box.innerHTML = "";
  const byId = Object.fromEntries(skills.map((s) => [s.id, s]));
  for (const s of skills) {
    const card = el("div", `skill${s.id === fresh ? " new" : ""}`);
    card.append(el("div", "lbl", `Tier ${s.tier} · skill-${s.id.slice(-4)}`), el("div", "ins", s.instruction));
    if (s.steps && s.steps.length) {
      const ol = el("ol", "chain");
      for (const st of s.steps) ol.append(el("li", "", `${st.instruction}  (skill-${st.skill_id.slice(-4)})`));
      card.append(ol);
    }
    const meta = el("div", "meta");
    const parent = s.steps?.length ? "chained" : s.parent ? (byId[s.parent] ? `from "${byId[s.parent].instruction.slice(0, 24)}..."` : "from a skill") : "from base";
    meta.append(el("span", "pct", `${Math.round(100 * s.success_rate)}% of ${s.n_eval}`), el("span", "", `${Math.round(s.seconds)}s`), el("span", "", parent));
    if (s.robot && s.robot !== "so101") meta.append(el("span", "robot-tag mono", s.robot));
    const b = el("button", "", "Call agent");
    const reply = el("div", "reply");
    b.onclick = async () => {
      b.disabled = true;
      reply.textContent = "running in a fresh scene...";
      const r = await fetch(`/api/skills/${s.id}/run`, { method: "POST" });
      reply.textContent = r.ok ? (await r.json()).reply : "no answer";
      b.disabled = false;
    };
    meta.append(b);
    card.append(meta, reply);
    box.append(card);
  }
  if (!skills.length) box.append(el("div", "empty-note", "No skill agents yet."));
}
async function refreshSkills(fresh) {
  const st = await (await fetch("/api/state")).json();
  renderSkills(st.skills, fresh);
}
$("#call-last").onclick = async () => {
  if (!lastSkill) return;
  const b = $("#call-last");
  b.disabled = true;
  forgeNote(`Calling skill-${lastSkill.slice(-4)} in a fresh scene...`);
  const r = await fetch(`/api/skills/${lastSkill}/run`, { method: "POST" });
  forgeNote(r.ok ? (await r.json()).reply : "The skill agent did not answer.");
  b.disabled = false;
};
function setReady(on) {
  $("#ready-dot").classList.toggle("on", on);
  $("#ready-text").textContent = on ? "agents online" : "warming up";
  $("#go").disabled = !on;
}

// ---------------------------------------------------------------- page load: show the last finished run
let robotNow = "so101";
window.addEventListener("robot-change", (e) => {
  const changed = e.detail.id !== robotNow;
  robotNow = e.detail.id;
  $("#robot-chip").textContent = `${e.detail.label} · sim`;
  if (changed && !current) {
    $("#vp-caption").textContent = `${e.detail.label} · MuJoCo`;
    fetch("/api/preview").then((r) => r.json()).then(async (sc) => { await buildWorld(sc); applyFrame(sc.frame); });
  }
});
async function idleScene() {
  const st = await (await fetch("/api/state")).json();
  robotNow = st.robot || "so101";
  initBuilder(robotNow);
  renderSkills(st.skills);
  setReady(st.ready);
  const tops = st.runs.filter((r) => !r.parent);
  const last = tops[tops.length - 1];
  if (last && last.status === "published") {
    adoptRun(last, st.runs);
  } else {
    const sc = await (await fetch("/api/preview")).json();
    await buildWorld(sc);
    applyFrame(sc.frame);
  }
}
function adoptRun(summary, all) {
  current = summary.id;
  composite = summary.tier === 3;
  lastSkill = summary.skill_id;
  $("#call-last").disabled = !lastSkill;
  family.clear();
  const list = youMsg(summary.prompt);
  toolLists[summary.id] = list;
  forgeNote(`Published skill-${summary.skill_id.slice(-4)} in ${fmt(summary.elapsed)}.`);
  $("#crumb-run").textContent = summary.prompt;
  $("#vp-caption").textContent = summary.spec ? `“${summary.spec.instruction}”` : `“${summary.prompt}”`;
  setTier(summary.tier);
  runStart = 1; runDone = 1 + summary.elapsed;
  $("#elapsed").textContent = fmt(summary.elapsed);
  roundsBy[summary.id] = summary.rounds.map((r) => ({ ...r }));
  resetStages(composite);
  for (const [k] of STAGES) stage(k, "done");
  resetLoss();
  let lossRun = summary;
  if (composite) {
    for (const p of summary.plan) if (p.child) family.set(p.child, p.step);
    const kids = all.filter((r) => r.parent === summary.id);
    if (kids.length) lossRun = kids[kids.length - 1];
  }
  for (const r of lossRun.rounds) losses.push(r.loss.map(([t, l]) => [t, l]));
  drawLoss();
  if (lossRun.init_from && lossRun.init_from.reason) $("#loss-init").textContent = lossRun.init_from.reason;
  const crit = $("#critic");
  crit.innerHTML = "";
  for (const r of summary.rounds) {
    if (!r.plan || !r.plan.action) continue;
    const p = el("div");
    p.append(el("b", "", `Round ${r.index + 1}: ${r.plan.action}. `), document.createTextNode(r.plan.rationale));
    crit.append(p);
  }
  renderPlan(summary.plan, composite ? `${summary.plan.length} steps` : "");
  drawLoss();
  if (composite && !crit.children.length) crit.append(el("span", "dim", "Tier 3 runs each step's published skill as is; the critic works inside each skill's own training."));
  if (summary.best_round >= 0) showRound(summary.id, summary.best_round);
}
function setTier(t) {
  const e = $("#tier");
  e.textContent = `TIER ${t}`;
  e.classList.toggle("hidden", !t);
}

// ---------------------------------------------------------------- events
const es = new EventSource("/api/events");
es.onmessage = (m) => {
  const e = JSON.parse(m.data);
  if (e.kind === "ready") { setReady(true); return; }
  if (e.kind === "agents") return;
  if (e.run !== current && !family.has(e.run)) {
    if (e.kind !== "log" || !e.text.startsWith("chat message")) return;
    beginRun(e.run);
  }
  handle(e);
};
function beginRun(rid) {
  current = rid;
  composite = false;
  family.clear();
  runStart = Date.now() / 1000;
  runDone = 0;
  roundsBy[rid] = [];
  resetLoss();
  resetStages(false);
  renderPlan([], "");
  $("#critic").innerHTML = "";
  $("#chips").innerHTML = "";
  $("#chips").dataset.pending = "";
  $("#round-label").textContent = "Rollouts";
  $("#vp-badge").classList.add("hidden");
  $("#step-badge").classList.add("hidden");
  $("#go").disabled = true;
  setTier(0);
}
function handle(e) {
  const isChild = family.has(e.run);
  switch (e.kind) {
    case "log": {
      if (e.text.startsWith("chat message")) {
        const prompt = e.text.replace(/^chat message[^:]*:\s*/, "").replace(/^['"“]|['"”]$/g, "");
        if (!toolLists[e.run]) toolLists[e.run] = youMsg(lastPrompt || prompt);
        break;
      }
      const t = toolEls[`${e.run}:${e.agent}`];
      const settled = t && (t.classList.contains("done") || t.classList.contains("fail"));
      const early = settledEarly[`${e.run}:${e.agent}`];
      delete settledEarly[`${e.run}:${e.agent}`];
      tool(e.run, e.agent, e.text, early || (settled ? null : "live"));
      if (!rail[e.agent]?.classList.contains("done")) stage(e.agent, "live");
      break;
    }
    case "task":
      if (e.steps) {
        composite = true;
        resetStages(true);
        setTier(3);
        $("#vp-caption").textContent = `“${$("#crumb-run").textContent}”`;
      } else {
        setTier(e.tier);
        $("#crumb-run").textContent = e.spec.instruction;
        $("#vp-caption").textContent = `“${e.spec.instruction}”`;
      }
      if (e.parse && e.parse.note) forgeNote(e.parse.note);
      settle(e.run, "task");
      stage("task", "done");
      break;
    case "rejected":
      tool(e.run, "task", e.reason, "fail");
      stage("task", "fail");
      runDone = Date.now() / 1000;
      $("#go").disabled = false;
      break;
    case "error":
      tool(e.run, e.agent || "task", e.error, "fail");
      stage(e.agent, "fail");
      if (!isChild) { runDone = Date.now() / 1000; $("#go").disabled = false; }
      break;
    case "plan": {
      const reuse = e.plan.filter((p) => p.source === "reuse").length;
      renderPlan(e.plan, `${e.plan.length} steps · ${reuse} reused`);
      stage("planner", "live", `${reuse}/${e.plan.length}`);
      break;
    }
    case "offer":
      renderPlan(e.plan);
      break;
    case "child": {
      family.set(e.child, e.step);
      renderPlan(e.plan);
      const sub = el("div", "sub-thread");
      sub.append(el("div", "sub-h", `Step ${e.step + 1}: training "${e.plan[e.step].instruction}"`));
      const list = el("div", "tools-list");
      sub.append(list);
      toolLists[current].append(sub);
      toolLists[e.child] = list;
      for (const k of ["scene", "data", "trainer", "eval", "critic", "registry"]) stage(k, "idle", "");
      resetLoss();
      $("#critic").innerHTML = "";
      roundsBy[e.child] = [];
      scrollThread();
      break;
    }
    case "scene":
      settle(e.run, "scene");
      stage("scene", "done");
      showPreview(e.run);
      break;
    case "demos":
      stage("data", "live", `${e.ok}/${e.n}`);
      if (e.done === e.n) { stage("data", "done"); settle(e.run, "data"); }
      break;
    case "init":
      $("#loss-init").textContent = e.reason;
      break;
    case "loss": {
      while (losses.length <= e.round) losses.push([]);
      losses[e.round].push([e.t, e.loss]);
      $("#loss-now").textContent = e.ema.toFixed(3);
      $("#loss-steps").textContent = `${e.step} steps`;
      stage("trainer", "live", `${Math.round(e.t)}s`);
      drawLoss();
      break;
    }
    case "eval_progress":
      stage("trainer", "done");
      settle(e.run, "trainer");
      stage("eval", "live", `${e.succeeded}/${e.n}`);
      pendingTabs(e.n, `${isChild ? `Step ${family.get(e.run) + 1} · ` : ""}Round ${e.round + 1} · evaluating`);
      break;
    case "compose_progress":
      stage("planner", "done");
      settle(current, "planner");
      stage("executor", "live", `t${e.tick}`);
      pendingTabs(e.n, "Chained run · evaluating");
      break;
    case "eval": {
      (roundsBy[e.run] ||= [])[e.round] = { index: e.round, eval: e, successes: e.successes };
      if (e.agent === "executor") {
        stage("executor", "done", `${e.succeeded}/${e.n}`);
        settle(e.run, "executor");
      } else {
        stage("eval", "done", `${e.succeeded}/${e.n}`);
        settle(e.run, "eval");
      }
      $("#chips").dataset.pending = "";
      showRound(e.run, e.round);
      break;
    }
    case "critic": {
      stage("critic", "done", e.plan.action);
      settle(e.run, "critic");
      const rr = roundsBy[e.run] || [];
      if (rr[e.round]) rr[e.round].plan = e.plan;
      const p = el("div");
      p.append(el("b", "", `${isChild ? `Step ${family.get(e.run) + 1}, ` : ""}Round ${e.round + 1}: ${e.plan.action}. `), document.createTextNode(e.plan.rationale));
      $("#critic").append(p);
      if (e.plan.action === "retrain") for (const k of ["data", "trainer", "eval"]) stage(k, "idle");
      break;
    }
    case "published":
      settle(e.run, "registry");
      stage("registry", isChild ? "idle" : "done", isChild ? "" : `${Math.round(100 * e.success_rate)}%`);
      refreshSkills(e.skill);
      if (isChild) {
        forgeNote(`Published skill-${e.skill.slice(-4)}: ${Math.round(100 * e.success_rate)}% in ${fmt(e.seconds)}.`, e.run);
        break;
      }
      lastSkill = e.skill;
      $("#call-last").disabled = false;
      runDone = runStart + e.seconds;
      $("#elapsed").textContent = fmt(e.seconds);
      forgeNote(`Published skill-${e.skill.slice(-4)} in ${fmt(e.seconds)}: ${Math.round(100 * e.success_rate)}% of 20 held-out scenes.`);
      if (!composite && e.best_round !== shown.round) showRound(e.run, e.best_round);
      $("#go").disabled = false;
      break;
  }
}

// ---------------------------------------------------------------- prompt
let lastPrompt = "";
async function submit(text) {
  if (!text) return;
  lastPrompt = text;
  $("#go").disabled = true;
  $("#prompt").value = "";
  $("#crumb-run").textContent = text;
  await fetch("/api/prompt", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ prompt: text }) });
}
$("#prompt-form").addEventListener("submit", (ev) => { ev.preventDefault(); submit($("#prompt").value.trim()); });
$("#prompt").addEventListener("keydown", (ev) => {
  if (ev.key === "Enter" && !ev.shiftKey) { ev.preventDefault(); submit($("#prompt").value.trim()); }
});
document.querySelectorAll(".ex").forEach((b) => b.addEventListener("click", () => {
  $("#prompt").value = b.childNodes[b.childNodes.length - 1].textContent;
  $("#prompt").focus();
}));
window.submitPrompt = submit;

idleScene();

// ---------------------------------------------------------------- live tab capture (?capture=1, demo recording)
window.setSpeed = (on) => { $("#speed").classList.toggle("hidden", !on); };
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
    window.captureT0 = performance.now();
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
