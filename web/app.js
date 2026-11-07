import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import { RoundedBoxGeometry } from "three/addons/geometries/RoundedBoxGeometry.js";
import { toCreasedNormals } from "three/addons/utils/BufferGeometryUtils.js";
import { initBuilder, showBuilder } from "./builder.js";
import { HAZE, setupRenderer, makeEnvironment, makeStage, addStudioLights, buildLab, basePlate, floorLabel, robotMaterial as studioMaterial, partsMesh } from "./studio.js";

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
  ["router", "Route", '<path d="M4 10h5l3-5h4M9 10l3 5h4"/><circle cx="4" cy="10" r="1.6"/>'],
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
NAMES.router = "Router"; NAMES.trainer = "Trainer"; NAMES.data = "Data"; NAMES.planner = "Planner"; NAMES.executor = "Executor"; NAMES.registry = "Registry";
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
  for (const k of ["scene", "data", "trainer", "eval", "critic"]) rail[k].classList.remove("off");
  rail.planner.classList.toggle("off", !composite);
  rail.executor.classList.toggle("off", !composite);
}

// ---------------------------------------------------------------- 3D
// A clean robotics cell: sealed concrete floor, taped cell corners, the arm bolted to a floor plate. No furniture.
const canvas = $("#three");
const renderer = new THREE.WebGLRenderer({ canvas, antialias: true, preserveDrawingBuffer: true, powerPreference: "high-performance" });
renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
setupRenderer(renderer);
const scene3 = new THREE.Scene();
scene3.background = HAZE;
scene3.fog = new THREE.Fog(HAZE, 2.6, 7.0);
scene3.environment = makeEnvironment(renderer);
scene3.environmentIntensity = 0.55;

const camera = new THREE.PerspectiveCamera(30, 1, 0.01, 30);
camera.up.set(0, 0, 1);
camera.position.set(1.37, -0.38, 0.52);
const controls = new OrbitControls(camera, canvas);
controls.target.set(0.14, 0.0, 0.13);
controls.enableDamping = true;
controls.minDistance = 0.35;
controls.maxDistance = 2.6;
controls.maxPolarAngle = Math.PI * 0.47;
controls.update();

addStudioLights(scene3, { target: [0.2, 0, 0], extent: 1.1 });
const lab = buildLab(scene3, renderer, { cell: [-0.34, -0.58, 0.74, 0.58] });
floorLabel(lab, "CELL 01  ·  SO-101", 0.5, -0.545, Math.PI / 2, 0.034);
let plate = basePlate(0.23);
scene3.add(plate);
const postStage = makeStage(renderer, scene3, camera);

const geomCache = new Map();
const meshStore = {}; // robot id -> promise of its mesh buffers (fetched once per robot)
const meshesFor = (rid) => (meshStore[rid] ||= fetch(`/api/meshes?robot=${encodeURIComponent(rid)}`).then((r) => r.json()));
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
// factory-line cell parts (sim/line.py): the totes are robosuite's bin boxes as the MJCF has them; the belt is drawn
// fixed in place and its tread texture scrolls with the slide joint (the MuJoCo belt body itself translates)
function beltTexture() {
  const c = document.createElement("canvas");
  c.width = 64; c.height = 256;
  const g = c.getContext("2d");
  g.fillStyle = "#1c1d21"; g.fillRect(0, 0, 64, 256);
  g.fillStyle = "#2a2c31";
  for (let y = 0; y < 256; y += 32) g.fillRect(0, y, 64, 14);
  const t = new THREE.CanvasTexture(c);
  t.wrapS = t.wrapT = THREE.RepeatWrapping;
  t.colorSpace = THREE.SRGBColorSpace;
  return t;
}
// Conveyor and totes of a factory-line cell (sim/line.py). The belt is drawn fixed in place on a proper frame (side
// rails, end rollers, legs to the floor) and its tread scrolls with the slide joint; totes are robosuite's bin boxes.
const BELT_FRAME = new THREE.MeshStandardMaterial({ color: 0xb9bdc4, roughness: 0.38, metalness: 0.9, envMapIntensity: 0.9 });
const BELT_DARK = new THREE.MeshStandardMaterial({ color: 0x23252a, roughness: 0.6, metalness: 0.5, envMapIntensity: 0.7 });
function addLineParts(sc, root, bodies, { shadows = true, belt: wantBelt = true, halfLen = Infinity } = {}) {
  for (const b of sc.bins || []) {
    const mat = new THREE.MeshPhysicalMaterial({ color: srgb(b.rgb), roughness: 0.5, clearcoat: 0.2, clearcoatRoughness: 0.45, envMapIntensity: 0.6 });
    for (const g of b.boxes) {
      const m = new THREE.Mesh(new RoundedBoxGeometry(g.size[0] * 2, g.size[1] * 2, g.size[2] * 2, 3, 0.0012), mat);
      m.position.set(...g.pos);
      m.castShadow = m.receiveShadow = shadows;
      bodies[b.body].add(m);
    }
  }
  if (!sc.belt || !wantBelt) return null;
  const [w, l0, t] = sc.belt.half;
  const l = Math.min(l0, halfLen);
  const tex = beltTexture();
  tex.repeat.set(1, (l * 2) / 0.06);
  const x = sc.frame ? sc.frame[sc.belt.body * 7] : 0.245;
  const belt = new THREE.Mesh(new THREE.BoxGeometry(w * 2, l * 2, t * 2), new THREE.MeshStandardMaterial({ map: tex, roughness: 0.85 }));
  belt.position.set(x, 0, sc.belt.top - t);
  belt.receiveShadow = shadows;
  root.add(belt);
  const topZ = sc.belt.top;
  for (const s of [-1, 1]) { // side rails and the frame beneath
    const r = new THREE.Mesh(new RoundedBoxGeometry(0.012, l * 2, 0.03, 3, 0.002), BELT_FRAME);
    r.position.set(x + s * (w + 0.006), 0, topZ - 0.013);
    r.castShadow = r.receiveShadow = shadows;
    root.add(r);
    for (let y = -l + 0.1; y <= l - 0.05; y += 0.4) { // legs
      const leg = new THREE.Mesh(new RoundedBoxGeometry(0.02, 0.02, topZ + 0.001, 2, 0.002), BELT_FRAME);
      leg.position.set(x + s * (w + 0.006), y, (topZ - 0.028) / 2 - 0.001);
      leg.scale.z = 0.01 / 0.01;
      leg.geometry = new RoundedBoxGeometry(0.02, 0.02, Math.max(0.005, topZ - 0.028 + 0.012), 2, 0.002);
      leg.position.z = (topZ - 0.028 + 0.012) / 2 - 0.006;
      leg.castShadow = shadows;
      root.add(leg);
    }
  }
  for (const e of [-1, 1]) { // end rollers
    const roll = new THREE.Mesh(new THREE.CylinderGeometry(0.009, 0.009, w * 2 + 0.01, 24), BELT_DARK);
    roll.rotation.z = Math.PI / 2;
    roll.position.set(x, e * (l - 0.004), topZ - 0.009);
    root.add(roll);
  }
  return { tex, body: sc.belt.body, y0: sc.frame ? sc.frame[sc.belt.body * 7 + 1] : 0 };
}
// every drawable of a compiled scene, parented under `root`: the robot's real meshes in the studio materials, blocks,
// bowls, taped squares, totes, fixtures. Returns one Group per MuJoCo body (poses are applied to these).
function populate(sc, root, rid, meshes, { shadows = true, belt: wantBelt = true, beltHalfLen = Infinity } = {}) {
  const bodies = sc.bodies.map(() => {
    const o = new THREE.Group();
    root.add(o);
    return o;
  });
  const shade = (m) => { m.castShadow = shadows; m.receiveShadow = shadows; return m; };
  for (const g of sc.robot) {
    const geo = meshGeometry(`${rid}|${g.mesh}`, meshes[String(g.mesh)]);
    const m = shade(new THREE.Mesh(geo, studioMaterial(g.rgba, g.name)));
    m.position.fromArray(g.pos);
    m.quaternion.set(g.quat[1], g.quat[2], g.quat[3], g.quat[0]);
    bodies[g.body].add(m);
  }
  for (const b of sc.blocks) {
    if (b.parts) {
      const g = partsMesh(b.parts, b.rgb, b.kind || "part");
      g.traverse((o) => { if (o.isMesh) { o.castShadow = o.receiveShadow = shadows; } });
      bodies[b.body].add(g);
      continue;
    }
    const s = b.half * 2;
    const m = shade(new THREE.Mesh(
      new RoundedBoxGeometry(s, s, s, 4, 0.0025),
      new THREE.MeshPhysicalMaterial({ color: srgb(b.rgb), roughness: 0.5, clearcoat: 0.22, clearcoatRoughness: 0.45, envMapIntensity: 0.7 }),
    ));
    bodies[b.body].add(m);
  }
  for (const b of sc.bowls) {
    const m = shade(new THREE.Mesh(
      bowlGeometry(b.r, b.h),
      new THREE.MeshPhysicalMaterial({ color: srgb(b.rgb), roughness: 0.3, clearcoat: 1.0, clearcoatRoughness: 0.06, envMapIntensity: 0.9, side: THREE.DoubleSide }),
    ));
    bodies[b.body].add(m);
  }
  for (const p of sc.pads) {
    const mat = new THREE.MeshStandardMaterial({ color: srgb(p.rgb), roughness: 0.85, envMapIntensity: 0.4 });
    const t = p.half, w = 0.006;
    for (const [x, y, sx, sy] of [[t, 0, w, t + w], [-t, 0, w, t + w], [0, t, t + w, w], [0, -t, t + w, w]]) {
      const m = new THREE.Mesh(new THREE.BoxGeometry(sx * 2, sy * 2, 0.0012), mat);
      m.position.set(x, y, 0.0009);
      m.receiveShadow = shadows;
      bodies[p.body].add(m);
    }
  }
  for (const p of sc.props || []) {
    const g = partsMesh(p.parts, p.rgb, p.kind);
    g.traverse((o) => { if (o.isMesh) { o.castShadow = o.receiveShadow = shadows; } });
    bodies[p.body].add(g);
  }
  const belt = addLineParts(sc, root, bodies, { shadows, belt: wantBelt, halfLen: beltHalfLen });
  return { bodies, belt };
}
async function buildWorld(sc) {
  const rid = sc.robot_id || "so101";
  const meshes = await meshesFor(rid);
  if (world) { scene3.remove(world.root); scene3.remove(world.mirror); }
  if (!world || world.rid !== rid) {
    camFor(rid); // frame a new arm once; keep the viewer's orbit otherwise
    scene3.remove(plate);
    plate = basePlate(0.23 * camScale);
    scene3.add(plate);
  }
  const root = new THREE.Group();
  const lone = populate(sc, root, rid, meshes, { belt: false });
  scene3.add(root);
  // the sealed floor shows a faint mirror of the cell: the same objects below z = 0, no shadows of their own
  const mirror = new THREE.Group();
  mirror.scale.z = -1;
  const twin = populate(sc, mirror, rid, meshes, { shadows: false, belt: false });
  scene3.add(mirror);
  world = { root, mirror, bodies: lone.bodies, twin: twin.bodies, n: sc.bodies.length, rid, belt: null };
}
function applyFrame(frame) {
  if (!world || !frame) return;
  const n = world.n;
  for (let i = 0; i < n; i++) {
    const k = i * 7; // per body: x y z, then quaternion w x y z (MuJoCo order)
    for (const set of [world.bodies, world.twin]) {
      set[i].position.set(frame[k], frame[k + 1], frame[k + 2]);
      set[i].quaternion.set(frame[k + 4], frame[k + 5], frame[k + 6], frame[k + 3]);
    }
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
  if (!r.width || !r.height) return; // the sim view is hidden on the other tabs
  renderer.setSize(r.width, r.height, false);
  postStage.setSize(r.width, r.height);
  camera.aspect = r.width / r.height;
  camera.updateProjectionMatrix();
}
new ResizeObserver(resize).observe(canvas.parentElement);
resize();
const CAMS = {
  orbit: { target: [0.2, 0.0, 0.10], rotate: true },
  front: { pos: [1.3, -0.02, 0.56], target: [0.16, 0.0, 0.06], rotate: false },
  top: { pos: [0.2, -0.06, 1.15], target: [0.2, 0.0, 0.0], rotate: false },
};
// Orbit starts in front of the arm (+x, the side the gripper works toward), a little to the right, and only sways
// about 25 degrees either way, so the camera never swings round behind it.
const SWAY = { az: (-42 * Math.PI) / 180, amp: (12 * Math.PI) / 180, period: 30, r: 1.2, h: 0.4, resume: 4000 };
let userUntil = 0; // sway pauses while the user drags and for a few seconds after
function swayPose(now) {
  const k = camScale;
  const t = CAMS.orbit.target;
  const tz = t[2] + (k - 1) * 0.16;
  const az = SWAY.az + SWAY.amp * Math.sin((2 * Math.PI * now) / (SWAY.period * 1000));
  return [t[0] + Math.cos(az) * SWAY.r * k, t[1] + Math.sin(az) * SWAY.r * k, tz + SWAY.h * k, tz];
}
// bigger arms stand taller and swing wider: back the camera off and lift its aim so the whole arm and the blocks fit
const ARM_SCALE = { so101: 1, so100: 1, koch: 1, wx250s: 1.22, vx300s: 1.42 };
let camName = "orbit";
let camScale = 1;
function camFor(rid) {
  camScale = ARM_SCALE[String(rid || "so101").split(".")[0]] || 1;
  setCam(camName);
}
function setCam(name) {
  const c = CAMS[name];
  camName = name;
  const k = camScale;
  const t = [c.target[0], c.target[1], c.target[2] + (k - 1) * 0.16];
  controls.target.fromArray(t);
  if (name === "orbit") {
    const p = swayPose(performance.now());
    camera.position.set(p[0], p[1], p[2]);
    userUntil = 0;
  } else {
    camera.position.fromArray(c.pos.map((p, i) => t[i] + (p - c.target[i]) * k));
  }
  controls.autoRotate = false;
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
    if (camName === "orbit" && performance.now() > userUntil) {
      const p = swayPose(performance.now());
      camera.position.x += (p[0] - camera.position.x) * 0.06;
      camera.position.y += (p[1] - camera.position.y) * 0.06;
      camera.position.z += (p[2] - camera.position.z) * 0.06;
      controls.target.z += (p[3] - controls.target.z) * 0.06;
    }
    controls.update();
    postStage.render();
  }
  requestAnimationFrame(tick);
}
tick();
controls.addEventListener("start", () => { userUntil = Infinity; });
controls.addEventListener("end", () => { userUntil = performance.now() + SWAY.resume; });

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
  const app = $("#app");
  const prev = app.dataset.view;
  app.dataset.view = v;
  app.classList.toggle("immersive", v !== "sim");
  for (const k of ["sim", "line", "robot", "ladder", "skills"]) $(`#view-${k}`).classList.toggle("hidden", k !== v);
  showBuilder(v === "robot");
  $(".tabs").classList.toggle("hidden", v !== "sim");
  if (v === "ladder") renderLadder();
  if (v === "skills") refreshSkills();
  // the panels slide away over ~300ms; resize the 3D canvases once the layout has settled
  resize();
  if (prev !== v) {
    const again = () => { resize(); window.dispatchEvent(new Event("resize")); };
    setTimeout(again, 340);
  }
  window.dispatchEvent(new CustomEvent("forge:view", { detail: v }));
}
window.setView = setView;
$("#toggle-left").onclick = () => { $("#app").classList.toggle("no-left"); setTimeout(resize, 340); };
$("#toggle-right").onclick = () => { $("#app").classList.toggle("no-right"); setTimeout(resize, 340); };
$("#app").dataset.view = "sim";

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
  window.forgeLib?.render(skills, fresh); // the library (library.js) draws the tiles
}
async function refreshSkills(fresh) {
  if (window.forgeLib) return window.forgeLib.refresh(fresh);
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
  const live = current && !runDone; // a run in flight keeps its own arm on screen
  if (changed && !live) {
    play = null;
    $("#step-badge")?.classList.add("hidden");
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
  window.forgeLib?.onEvent(e, e.run === current || family.has(e.run));
  window.dispatchEvent(new CustomEvent("forge:event", { detail: e })); // the Line tab listens here (one stream per page)
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
    case "route":
      settle(e.run, "router", e.refused ? "fail" : "done");
      stage("router", e.refused ? "fail" : "done", e.refused ? "refused" : e.decision.replace("_", " "));
      break;
    case "rejected":
      tool(e.run, e.agent || "task", e.reason, "fail");
      stage(e.agent || "task", "fail");
      runDone = Date.now() / 1000;
      $("#go").disabled = false;
      break;
    case "reused":
      lastSkill = e.skill;
      $("#call-last").disabled = false;
      runDone = Date.now() / 1000;
      for (const k of ["scene", "data", "trainer", "eval", "critic"]) rail[k].classList.add("off");
      stage("registry", "done", `${Math.round(100 * e.tile.success_rate)}%`);
      $("#go").disabled = false;
      break;
    case "saving":
      stage("registry", "live", "saving");
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
  const how = window.forgeLib?.takeMode() || { mode: "auto", from_skill: "" };
  const res = await fetch("/api/prompt", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ prompt: text, ...how }) });
  if (!res.ok) $("#go").disabled = false;
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
// what the library (library.js) needs from the sim view and the chat thread
window.forge = {
  el, fmt, stage, setView, submit, startPlayback, scrollThread,
  toolList: (run) => toolLists[run],
  current: () => current,
  setBadge(text, ok) {
    const b = $("#vp-badge");
    b.classList.remove("hidden", "ok", "bad");
    b.classList.add(ok ? "ok" : "bad");
    b.textContent = text;
  },
  showScene(sc) { play = null; buildWorld(sc); applyFrame(sc.frame); $("#pb-fill").style.width = "0"; $("#vp-badge").classList.add("hidden"); $("#step-badge").classList.add("hidden"); },
  setCaption(text) { $("#vp-caption").textContent = text; },
  setShown(v) { shown = v; },
  THREE, scene3, camera, controls, renderer,
  meshesFor, meshGeometry, studioMaterial, srgb, addLineParts, RoundedBoxGeometry, postStage,
};

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
