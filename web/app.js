import * as THREE from "three";
import { GLTFLoader } from "three/addons/loaders/GLTFLoader.js";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import { RoundedBoxGeometry } from "three/addons/geometries/RoundedBoxGeometry.js";
import { toCreasedNormals } from "three/addons/utils/BufferGeometryUtils.js";
import { initBuilder, showBuilder } from "./builder.js";
import { loadingNode } from "./loading.js";
import { HAZE, setupRenderer, makeEnvironment, makeStage, addStudioLights, buildLab, basePlate, floorLabel, cellBrackets, robotMaterial as studioMaterial, partsMesh } from "./studio.js";

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
  ["optimizer", "Optimize", '<path d="M4 6h8M15 6h1M4 14h1M8 14h8"/><circle cx="13.5" cy="6" r="1.6"/><circle cx="6.5" cy="14" r="1.6"/>'],
  ["executor", "Run", '<path d="M6.5 4.5v11l8.5-5.5z"/>'],
  ["registry", "Publish", '<path d="M10 3.5v9M6 8.5l4 4 4-4M4 16.5h12"/>'],
];
const NAMES = Object.fromEntries(STAGES.map(([k, n]) => [k, n]));
NAMES.optimizer = "Optimizer"; NAMES.router = "Router"; NAMES.trainer = "Trainer"; NAMES.data = "Data"; NAMES.planner = "Planner"; NAMES.executor = "Executor"; NAMES.registry = "Registry";
const rail = {};
for (const [key, name, icon] of STAGES) {
  if (key === "executor") $("#rail").append(el("li", "sep"));
  const li = el("li");
  li.dataset.stage = key;
  li.append(svgIcon(icon), el("span", "", name), el("span", "m"));
  if (key === "planner" || key === "executor") li.classList.add("off");
  li.classList.add("tap"); li.setAttribute("role", "button"); li.tabIndex = 0; li.title = `Open ${name}`;
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

const lights = addStudioLights(scene3, { target: [0.2, 0, 0], extent: 1.1 });
const lab = buildLab(scene3, renderer, { cell: null, joints: false });
// the taped cell, its label and the arm's floor plate only exist once a task has a scene; before that the canvas is
// the bare floor with a faint grid (like the Etch studio)
const cellMarks = new THREE.Group();
cellBrackets(cellMarks, [-0.34, -0.58, 0.74, 0.58]);
floorLabel(cellMarks, "CELL 01  ·  SO-101", 0.5, -0.545, Math.PI / 2, 0.034);
cellMarks.visible = false;
scene3.add(cellMarks);
let plate = basePlate(0.17);
plate.visible = false;
scene3.add(plate);
const grid = new THREE.GridHelper(6, 60, 0x0866ff, 0x0866ff);
grid.rotation.x = Math.PI / 2;
grid.position.z = 0.0016;
for (const m of [].concat(grid.material)) { m.transparent = true; m.opacity = 0.11; m.depthWrite = false; m.fog = true; }
scene3.add(grid);
const gridMid = new THREE.GridHelper(6, 6, 0x0866ff, 0x0866ff);
gridMid.rotation.x = Math.PI / 2;
gridMid.position.z = 0.0018;
for (const m of [].concat(gridMid.material)) { m.transparent = true; m.opacity = 0.24; m.depthWrite = false; m.fog = true; }
scene3.add(gridMid);
let staged = false; // false: bare floor and grid
let stageT = { t0: 0, from: 0, to: 0 };
function setStaged(on) {
  if (on === staged) return;
  staged = on;
  $("#view-sim").classList.toggle("blank", !on);
  cellMarks.visible = plate.visible = on && !window.forgeArms?.active();
  stageT = { t0: performance.now(), from: on ? 0 : 1, to: on ? 1 : 0 };
}
function stageStep(now) {
  const k = Math.min(1, (now - stageT.t0) / 700);
  const e = 1 - Math.pow(1 - k, 3);
  const v = stageT.from + (stageT.to - stageT.from) * e; // 0 = blank, 1 = built
  for (const g of [grid, gridMid]) {
    g.visible = v < 0.999;
    for (const m of [].concat(g.material)) m.opacity = (g === grid ? 0.11 : 0.24) * (1 - v);
  }
  if (world && k < 1) {
    const up = 0.02 + 0.98 * (1 - Math.pow(1 - k, 4) * (1 + 1.6 * (1 - k)));
    const z = staged ? Math.max(0.02, Math.min(1.02, up)) : 1;
    const xy = staged ? 0.96 + 0.04 * e : 1;
    world.root.scale.set(xy, xy, z);
    world.mirror.scale.set(xy, xy, -z);
    plate.scale.set(xy, xy, z);
  } else if (world && staged) {
    world.root.scale.set(1, 1, 1); world.mirror.scale.set(1, 1, -1); plate.scale.set(1, 1, 1);
  }
}
const postStage = makeStage(renderer, scene3, camera);

const geomCache = new Map();
const meshStore = {}; // robot id -> promise of its mesh buffers (fetched once per robot)
// library models in a scene (the toolbox and its tools): each one's web file, loaded once and cloned per body
const glbStore = { ready: {} };
const glbFor = (id) => (glbStore[id] ||= new GLTFLoader().loadAsync(`/api/assets/${encodeURIComponent(id)}/glb`).then((g) => g.scene));
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
  for (const m of sc.models || []) { // glTF is Y-up, the simulation Z-up: stand the model on the body's own frame
    const src = glbStore.ready[m.model];
    if (!src) continue;
    const g = new THREE.Group();
    g.rotation.x = Math.PI / 2;
    const obj = src.clone(true);
    obj.traverse((o) => { if (o.isMesh) { o.castShadow = o.receiveShadow = shadows; } });
    g.add(obj);
    bodies[m.body].add(g);
  }
  const belt = addLineParts(sc, root, bodies, { shadows, belt: wantBelt, halfLen: beltHalfLen });
  return { bodies, belt };
}
async function buildWorld(sc, quick = false) {
  const rid = sc.robot_id || "so101";
  const meshes = await meshesFor(rid);
  await Promise.all((sc.models || []).map(async (m) => { glbStore.ready[m.model] = await glbFor(m.model); }));
  window.forgeArms?.exit(); // a single-arm scene replaces any multi-arm cell
  if (world) { scene3.remove(world.root); scene3.remove(world.mirror); }
  if (!world || world.rid !== rid) {
    camFor(rid); // frame a new arm once; keep the viewer's orbit otherwise
    scene3.remove(plate);
    plate = basePlate(0.17 * camScale);
    scene3.add(plate);
  }
  plate.visible = cellMarks.visible = true;
  const root = new THREE.Group();
  const lone = populate(sc, root, rid, meshes, { belt: false });
  scene3.add(root);
  // the sealed floor shows a faint mirror of the cell: the same objects below z = 0, no shadows of their own
  const mirror = new THREE.Group();
  mirror.scale.z = -1;
  const twin = populate(sc, mirror, rid, meshes, { shadows: false, belt: false });
  scene3.add(mirror);
  world = { root, mirror, bodies: lone.bodies, twin: twin.bodies, n: sc.bodies.length, rid, belt: null };
  if (!staged && !quick) { // the scene builds in on the bare floor: the arm rises, the grid fades
    world.root.scale.set(1, 1, 0.02);
    world.mirror.scale.set(1, 1, -0.02);
    plate.scale.set(1, 1, 0.02);
    setStaged(true);
  } else if (!staged) { // a recorded rollout: the cell is simply there, no build-in
    setStaged(true);
    stageT = { t0: 0, from: 1, to: 1 };
  }
}
// back to the bare floor (new chat)
function clearWorld() {
  window.forgeArms?.exit();
  if (world) { scene3.remove(world.root); scene3.remove(world.mirror); world = null; }
  play = null;
  plate.visible = cellMarks.visible = false;
  staged = true;
  setStaged(false);
  stageT = { t0: performance.now(), from: 0, to: 0 };
  for (const g of [grid, gridMid]) { g.visible = true; for (const m of [].concat(g.material)) m.opacity = g === grid ? 0.11 : 0.24; }
  $("#step-badge").classList.add("hidden");
  $("#vp-badge").classList.add("hidden");
  $("#pb-fill").style.width = "0";
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
async function startPlayback(payload, auto = false) {
  const tok = ++playTok;
  $("#view-sim").classList.toggle("auto", auto); // saved tiles play on their own: no replay pill
  await buildWorld(payload.scene, true);
  if (tok !== playTok) return;
  play = { frames: payload.frames, fps: payload.fps, t0: performance.now(), phases: payload.phase_frames || null, instructions: payload.instructions || [], step: -1 };
  $("#step-badge").classList.add("hidden");
  applyFrame(payload.frames[0]);
  simLoading(null);
  performance.mark("dojo-frame");
}
let playTok = 0;

// ---------------------------------------------------------------- welcome: with nothing picked, the floor shows the regular loading mascot while the rail follows the published models
let welcomeOn = false, welcomeTimer = 0, welcomeTiles = null, welcomeShown = null;
// a frame from each published model's own rollout, with its measured score, one after another
async function welcomeStep(k) {
  if (!welcomeTiles) {
    try { welcomeTiles = (await (await fetch("/api/library")).json()).tiles.filter((t) => t.thumb && t.replay && t.n_eval > 0).slice(-6); } catch { welcomeTiles = []; }
  }
  if (!welcomeOn || !welcomeTiles.length) return;
  const t = welcomeTiles[k % welcomeTiles.length];
  // the rail, the inspector and any open step card follow the model the loop is showing
  welcomeShown = t;
  const held = `${Math.round(t.success_rate * t.n_eval)}/${t.n_eval}`;
  resetStages(t.tier === 3);
  stage("eval", "done", held);
  stage("registry", "done", held);
  showSavedModel(t);
  if (railCard) { const key = railCardKey; closeRailCard(); openRailCard(key); }
  welcomeTimer = setTimeout(() => welcomeStep(k + 1), 4000);
}
function startWelcome() {
  if (welcomeOn || (current && !runDone) || selSkill || !$("#sim-load").hidden || $("#app").dataset.view !== "sim") return;
  welcomeOn = true;
  $("#welcome").hidden = false;
  $("#view-sim").classList.add("welcoming");
}
function stopWelcome() {
  if (!welcomeOn) return;
  welcomeOn = false;
  $("#welcome").hidden = true;
  $("#view-sim").classList.remove("welcoming");
}

// ---------------------------------------------------------------- new model window
// One window holds the whole prompt: the text, the robot (its builder opens inline), an attached reference (an image
// or a 3D model) and, when there is a reference, the build plan. Panes: main, robot, models.
const nm = $("#new-modal");
const nmCard = $("#nm-card");
let ref = null; // { kind: "image" | "model" | "bridge", name, image?: data URL, thumb?: url }
function setPane(p) {
  nmCard.dataset.pane = p;
  showBuilder(p === "robot");
  if (p === "models") openModels();
  nmCard.scrollTop = 0;
  if (p === "main") $("#nm-prompt").focus();
}
function openNew() {
  $("#nm-prompt").placeholder = $("#prompt").placeholder;
  $("#nm-robot").textContent = $("#robot-chip").textContent;
  setPane("main");
  nm.hidden = false;
  requestAnimationFrame(() => nm.classList.add("on"));
  $("#nm-prompt").focus();
}
function closeNew() {
  if (nm.hidden) return;
  nm.classList.remove("on");
  showBuilder(false);
  $("#nm-menu").hidden = true;
  setTimeout(() => { if (!nm.classList.contains("on")) nm.hidden = true; }, 220);
}
nm.querySelectorAll("[data-close]").forEach((b) => b.addEventListener("click", closeNew));
document.addEventListener("keydown", (ev) => {
  if (ev.key !== "Escape" || nm.hidden) return;
  ev.preventDefault();
  if (!$("#nm-menu").hidden) $("#nm-menu").hidden = true;
  else if (nmCard.dataset.pane !== "main") setPane("main");
  else closeNew();
});
nm.querySelectorAll("[data-pane-go]").forEach((b) => b.addEventListener("click", () => setPane(b.dataset.paneGo)));
$("#nm-prompt").addEventListener("input", () => { $("#nm-go").disabled = !$("#nm-prompt").value.trim(); });
$("#nm-prompt").addEventListener("keydown", (ev) => { if (ev.key === "Enter" && !ev.shiftKey) { ev.preventDefault(); $("#nm-form").requestSubmit(); } });
// Enter inside the plan's fields edits the plan; it never sends the prompt
$("#nm-plan").addEventListener("keydown", (ev) => { if (ev.key === "Enter") ev.preventDefault(); });
$("#nm-form").addEventListener("submit", async (ev) => {
  ev.preventDefault();
  const text = $("#nm-prompt").value.trim();
  if (!text || $("#go").disabled) return;
  const go = $("#nm-go");
  go.disabled = true;
  if (ref && ref.kind !== "model" && window.forgePlan && !(await window.forgePlan.use())) { go.disabled = false; return; } // the plan's build becomes the table task first
  const reference = ref ? { kind: ref.kind, id: ref.id || "" } : null;
  $("#nm-prompt").value = "";
  setRef(null);
  closeNew();
  submit(text, reference);
});
$("#nm-arm").addEventListener("click", () => setPane("robot"));
$("#robot-open").addEventListener("click", () => { openNew(); setPane("robot"); });
$("#dock-plus").addEventListener("click", openNew);
window.addEventListener("robot-chosen", () => { $("#nm-robot").textContent = $("#robot-chip").textContent; if (!nm.hidden) setPane("main"); });

// reference: an image you upload, a 3D model, or the brick bridge suggestion; any of them opens the plan inline
function setRef(r) {
  ref = r;
  $("#nm-ref").hidden = !r;
  const img = $("#nm-ref-img");
  const src = r && (r.image || r.thumb);
  img.hidden = !src;
  if (src) img.src = src;
  if (r) $("#nm-ref-name").textContent = r.name;
  const plan = !!r && r.kind !== "model"; // a 3D model goes into the scene itself; only an image or the bridge opens the plan
  nmCard.classList.toggle("has-plan", plan);
  $("#nm-plan").hidden = !plan;
  if (r && r.kind === "model") $("#nm-ref-name").textContent = `${r.name} (shown in the scene)`;
  if (!plan) return window.forgePlan?.close();
  const note = r.kind === "image" ? `Reference: your image "${r.name}". Read it into steps below.` : "";
  window.forgePlan?.open({ image: r.kind === "image" ? r.image : null, ref: note });
}
$("#nm-attach").addEventListener("click", () => {
  const m = $("#nm-menu");
  m.hidden = !m.hidden;
  $("#nm-attach").setAttribute("aria-expanded", String(!m.hidden));
});
document.addEventListener("click", (ev) => { if (!ev.target.closest(".nm-attach-wrap")) $("#nm-menu").hidden = true; });
$("#nm-pick-img").addEventListener("click", () => { $("#nm-menu").hidden = true; $("#nm-file").click(); });
$("#nm-pick-3d").addEventListener("click", () => { $("#nm-menu").hidden = true; setPane("models"); });
$("#nm-file").addEventListener("change", (ev) => {
  const f = ev.target.files[0];
  ev.target.value = "";
  if (!f || !f.type.startsWith("image/")) return;
  const fr = new FileReader();
  fr.onload = () => setRef({ kind: "image", name: f.name, image: String(fr.result) });
  fr.readAsDataURL(f);
});
$("#nm-ref-x").addEventListener("click", () => setRef(null));
let assetsPanel = null;
async function openModels() {
  const host = $("#nm-assets");
  if (assetsPanel) return assetsPanel.refresh();
  host.replaceChildren(loadingNode("Loading the 3D models...", { layout: "fill col" }));
  const { mountAssets } = await import("/static/assets.js");
  host.replaceChildren();
  assetsPanel = mountAssets(host, {
    onUse(m) { setRef({ kind: "model", id: m.id, name: m.name, thumb: m.has_thumb ? m.thumb : "" }); setPane("main"); },
  });
}
nm.querySelectorAll(".mx").forEach((b) => b.addEventListener("click", () => {
  $("#nm-prompt").value = b.dataset.say;
  $("#nm-go").disabled = false;
  $("#nm-prompt").focus();
  if (b.dataset.ref === "bridge") setRef({ kind: "bridge", name: "Brick bridge" });
}));
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
// Orbit starts in front of the arm (+x, the side the gripper works toward), 20 degrees to the right, so the arm faces
// the camera, and only sways about 8 degrees either way, so the camera never swings round behind it.
let camShift = 0; // y offset of the aim point: a multi-arm cell is centred on its own middle
const SWAY = { az: (-20 * Math.PI) / 180, amp: (8 * Math.PI) / 180, period: 30, r: 1.2, h: 0.4, resume: 4000 };
let userUntil = 0; // sway pauses while the user drags and for a few seconds after
function swayPose(now) {
  const k = camScale;
  const t = CAMS.orbit.target;
  const tz = t[2] + (k - 1) * 0.16;
  const az = SWAY.az + SWAY.amp * Math.sin((2 * Math.PI * now) / (SWAY.period * 1000));
  return [t[0] + Math.cos(az) * SWAY.r * k, t[1] + camShift + Math.sin(az) * SWAY.r * k, tz + SWAY.h * k, tz];
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
  const t = [c.target[0], c.target[1] + camShift, c.target[2] + (k - 1) * 0.16];
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
  stageStep(performance.now());
  window.forgeArms?.tick(performance.now());
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
let savedModel = null; // a Library model open in Sim with no live run: its saved record fills the inspector
let losses = []; // per round arrays of [t, loss] of the run being trained
let planRows = [];
let lastSkill = null;
let selSkill = null;
// recorded rollouts, fetched once per tile recording and kept: a second look, a hover prefetch
const replays = new Map();
function replayData(id, i = 0, src = "") {
  const k = `${id}:${i}:${src}`;
  if (!replays.has(k)) {
    replays.set(k, fetch(`/api/skills/${id}/replay/${i}`, { priority: "high" }).then((r) => (r.ok ? r.json() : null)).catch(() => null));
    if (replays.size > 24) replays.delete(replays.keys().next().value);
  }
  return replays.get(k);
}
// the loading card shows the moment a tile is picked and goes once its first frame is on the floor
function simLoading(text) {
  const n = $("#sim-load");
  n.hidden = !text;
  if (text) $("#sim-load-t").textContent = text;
} // the skill on screen: the run's own once it is saved, or the tile picked in the library
let skillsCache = [];

// "16/20" from the evaluated round when we have it, else the rate on the standard 20 scenes
function heldOut(run, round, rate) {
  const ev = (roundsBy[run] || [])[round]?.eval;
  if (ev) return `${ev.succeeded}/${ev.n}`;
  return `${Math.round(rate * 20)}/20`;
}
// which model answered, as the server reported it
function modelTag(m) {
  if (!m) return "";
  if (/muse|meta:/i.test(m)) return "Muse Spark";
  if (/llama/i.test(m)) return "Llama";
  return m === "rules" ? "keyword rules, no model" : m;
}
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
const dockThread = $("#dock-thread");
const toolEls = {}; // `${run}:${agent}` -> element
const toolLists = {}; // run id -> tools-list element
let forgeMsg = null;
function scrollThread() { thread.scrollTop = thread.scrollHeight; }
function youMsg(text) {
  dockThread.classList.remove("collapsed");
  $("#dock-toggle").setAttribute("aria-expanded", "true");
  const m = el("div", "msg you");
  m.append(el("div", "who", "You"), el("div", "txt", text));
  thread.append(m);
  forgeMsg = el("div", "msg forge");
  forgeMsg.append(el("div", "who", "Dojo"));
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
// the conversation panel only exists while there is a conversation; its header folds it down to one line
function syncDock() {
  const has = thread.children.length > 0;
  dockThread.hidden = !has && !dockThread.querySelector(".wf:not([hidden])");
  const last = [...thread.querySelectorAll(".msg.you .txt")].pop();
  $("#dock-last").textContent = last ? last.textContent : "";
}
new MutationObserver(syncDock).observe(thread, { childList: true });
addEventListener("dock:sync", syncDock);
$("#dock-toggle").addEventListener("click", () => {
  const on = dockThread.classList.toggle("collapsed");
  $("#dock-toggle").setAttribute("aria-expanded", String(!on));
});
$("#new-chat").addEventListener("click", () => {
  thread.innerHTML = "";
  for (const k of Object.keys(toolEls)) delete toolEls[k];
  if (current && !runDone) return; // a run in flight keeps its scene
  window.dispatchEvent(new CustomEvent("forge:newchat"));
  clearWorld();
  current = null;
  $("#crumb-run").textContent = "new task";
  $("#vp-caption").textContent = "SO-101 · MuJoCo";
  selSkill = null;
  setTimeout(startWelcome, 0);
  $("#chips").innerHTML = "";
  $("#round-label").textContent = "Rollouts";
  $("#pb-label").textContent = "rollout ·";
  resetLoss(); resetStages(false); renderPlan([], ""); drawRounds();
  $("#critic").innerHTML = "";
  setTier(0);
});

// ---------------------------------------------------------------- inspector
function drawLoss() {
  const svg = $("#loss");
  const all = losses.flat();
  // empty chart: a quiet placeholder, and a free patch the robot can work in while training warms up
  svg.closest(".sec").dataset.empty = all.length ? "" : "1";
  if (!all.length) {
    svg.innerHTML = `<text class="ph" x="150" y="52" text-anchor="middle">${savedModel ? "loss curve not recorded for this model" : "No training yet"}</text>`;
    if (composite && planRows.length && planRows.every((p) => p.source === "reuse")) $("#loss-steps").textContent = "no training: every step reused a trained model";
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
      box.append(el("span", "dim", "Measured end to end once every step has a trained model."));
    }
    return;
  }
  for (const r of roundsBy[current] || []) {
    if (r && r.eval) box.append(barRow(`Round ${r.index + 1}`, r.eval.succeeded, r.eval.n));
  }
  if (!box.children.length && savedModel) box.append(barRow("Held-out", Math.round(savedModel.sk.success_rate * (savedModel.sk.n_eval || 20)), savedModel.sk.n_eval || 20));
  if (!box.children.length) box.append(el("span", "ph", "Not measured yet. Each round is scored on 20 scenes it never trained on."));
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

// the arm and cell appear once a task has parsed: the run's own scene when it exists, else the default one
async function ensureScene(rid) {
  if (world || window.forgeArms?.active()) return;
  let res = await fetch(`/api/runs/${rid}/scene`);
  let sc = res.ok ? await res.json() : { pending: true };
  if (sc.pending) sc = await (await fetch("/api/preview")).json();
  play = null;
  await buildWorld(sc);
  applyFrame(sc.frame);
}
async function showPreview(rid) {
  const res = await fetch(`/api/runs/${rid}/scene`);
  if (!res.ok) return;
  const sc = await res.json();
  if (sc.pending) return;
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
  // the Library and the Sim keep the prompt docked under them, so a new task can be typed straight from either
  app.classList.toggle("immersive", v !== "sim" && v !== "skills");
  app.classList.toggle("lib", v === "skills");
  for (const k of ["sim", "rsi", "agents", "skills"]) $(`#view-${k}`).classList.toggle("hidden", k !== v);
  $(".tabs").classList.toggle("hidden", v !== "sim");
  if (v === "rsi") renderRsi();
  if (v === "skills") refreshSkills();
  // the panels slide away over ~220ms; resize the 3D canvases once the layout has settled
  resize();
  if (prev !== v) {
    const again = () => { resize(); window.dispatchEvent(new Event("resize")); };
    setTimeout(again, 240);
  }
  window.dispatchEvent(new CustomEvent("forge:view", { detail: v }));
  if (v === "sim") startWelcome();
  else stopWelcome(); // nothing builds worlds behind a hidden floor
}
window.setView = setView;
$("#toggle-right").onclick = () => { $("#app").classList.toggle("no-right"); setTimeout(resize, 240); };
$("#app").dataset.view = "skills";

const TIER_NAMES = {
  1: ["Tier 1", "one motion: reach, push, lift"],
  2: ["Tier 2", "carry a block somewhere: place, stack"],
  3: ["Tier 3", "several steps, chained from published models"],
};
async function renderRsi() {
  const box = $("#ladder");
  if (!box.children.length) box.replaceChildren(loadingNode("Loading the finished prompts...", { layout: "fill col" }));
  const st = await (await fetch("/api/state")).json();
  const skills = Object.fromEntries(st.skills.map((s) => [s.id, s]));
  box.innerHTML = "";
  const done = st.runs.filter((r) => r.status === "published");
  // runs live only as long as the server process; published models are on disk, so they fill in the rows a restart forgot
  const seen = new Set(done.map((r) => r.skill_id || r.skill));
  for (const s of st.skills) {
    if (seen.has(s.id)) continue;
    const n = s.n_eval || 20;
    done.push({
      tier: s.tier, prompt: s.instruction, elapsed: s.seconds || s.train_seconds || 0,
      rounds: [{ eval: { succeeded: Math.round((s.success_rate || 0) * n), n } }], best_round: 0,
      init_from: s.parent ? { skill: s.parent } : null,
      plan: (s.steps || []).map((p) => ({ source: p.source || "reuse", seconds: p.seconds || 0 })),
    });
  }
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
        from = `calls ${r.plan.length} models; reused ${reuse}` + (trained.length ? `, trained ${trained.length} (${trained.map((p) => Math.round(p.seconds) + "s").join(", ")})` : "");
      } else if (r.parent) {
        const src = r.init_from && r.init_from.skill ? `, fine-tuned from "${skills[r.init_from.skill]?.instruction || r.init_from.skill.replace(/-[0-9a-f]{4}$/, "").replace(/-/g, " ")}"` : ", from the base policy";
        from = `step ${r.step_index + 1} of a tier 3 task${src}`;
      } else if (r.init_from && r.init_from.skill) {
        from = `fine-tuned from "${skills[r.init_from.skill]?.instruction || r.init_from.skill.replace(/-[0-9a-f]{4}$/, "").replace(/-/g, " ")}"`;
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
  forgeNote(r.ok ? (await r.json()).reply : "The model did not answer.");
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
  $("#nm-robot").textContent = $("#robot-chip").textContent;
  const live = current && !runDone; // a run in flight keeps its own arm on screen
  if (changed && !live && world) {
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
  // the canvas stays bare until a prompt has a scene; finished runs live in the Library and the RSI page
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
  window.dispatchEvent(new CustomEvent("forge:event", { detail: e })); // the multi-arm cell (arms.js) and the optimizer listen here (one stream per page)
  if (e.kind === "ready") { setReady(true); return; }
  if (e.kind === "agents") return;
  if (e.run !== current && !family.has(e.run)) {
    if (e.kind !== "log" || !e.text.startsWith("chat message")) return;
    beginRun(e.run);
  }
  handle(e);
};
function beginRun(rid) {
  stopWelcome();
  current = rid;
  selSkill = null;
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
      savedModel = null;
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
      if (e.parse && e.parse.model) $("#plan-model").textContent = `Planner: ${modelTag(e.parse.model)}`;
      settle(e.run, "task");
      stage("task", "done");
      if (!e.steps) ensureScene(e.run);
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
      lastSkill = selSkill = e.skill;
      $("#call-last").disabled = false;
      runDone = Date.now() / 1000;
      for (const k of ["scene", "data", "trainer", "eval", "critic"]) rail[k].classList.add("off");
      stage("registry", "done", `${Math.round(e.tile.success_rate * (e.tile.n_eval || 20))}/${e.tile.n_eval || 20}`);
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
      if (e.model || e.plan.model) $("#critic-model").textContent = modelTag(e.model || e.plan.model);
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
      const held = heldOut(e.run, e.best_round, e.success_rate);
      stage("registry", isChild ? "idle" : "done", isChild ? "" : held);
      refreshSkills(e.skill);
      if (isChild) {
        forgeNote(`Saved skill-${e.skill.slice(-4)}: ${held} on held-out scenes, in simulation.`, e.run);
        break;
      }
      lastSkill = selSkill = e.skill;
      $("#call-last").disabled = false;
      runDone = runStart + e.seconds;
      $("#elapsed").textContent = fmt(e.seconds);
      forgeNote(`Saved skill-${e.skill.slice(-4)}: ${held} on held-out scenes, in simulation.`);
      if (!composite && e.best_round !== shown.round) showRound(e.run, e.best_round);
      $("#go").disabled = false;
      break;
  }
}

// ---------------------------------------------------------------- prompt
let lastPrompt = "";
async function submit(text, reference = null) {
  if (!text) return;
  lastPrompt = text;
  if ($("#app").dataset.view !== "sim") setView("sim");
  $("#go").disabled = true;
  $("#prompt").value = "";
  $("#crumb-run").textContent = text;
  const how = window.forgeLib?.takeMode() || { mode: "auto", from_skill: "" };
  if (how.mode === "auto" && window.forgeArms?.wants(text)) {
    youMsg(text);
    $("#go").disabled = false;
    await window.forgeArms.handle(text);
    return;
  }
  const res = await fetch("/api/prompt", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ prompt: text, ...how, ...(reference ? { reference } : {}) }) });
  if (!res.ok) $("#go").disabled = false;
}
$("#prompt-form").addEventListener("submit", (ev) => { ev.preventDefault(); submit($("#prompt").value.trim()); });
$("#prompt").addEventListener("keydown", (ev) => {
  if (ev.key === "Enter" && !ev.shiftKey) { ev.preventDefault(); submit($("#prompt").value.trim()); }
});
window.submitPrompt = submit;
// what the library (library.js) needs from the sim view and the chat thread
window.forge = {
  el, fmt, stage, setView, submit, openNew, startPlayback, scrollThread,
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
  // a finished tile picked in the library: the rail, the bubble and the bot show that skill as saved
  replayData,
  simLoading,
  selectSkill(t) {
    stopWelcome();
    selSkill = t.id;
    if (current && !runDone) return;
    const held = `${Math.round(t.success_rate * (t.n_eval || 20))}/${t.n_eval || 20}`;
    resetStages(t.tier === 3);
    stage("eval", "done", held);
    stage("registry", "done", held);
    showSavedModel(t);
    $("#crumb-run").textContent = t.name;
    window.dispatchEvent(new CustomEvent("dojo:skill", { detail: { id: t.id, name: t.name, held, seconds: t.train_seconds ?? t.seconds } }));
  },
  note: forgeNote, modelTag, lights, cellMarks, plate: () => plate, grid, gridMid,
  setCamShift(y, scale) { camShift = y; camScale = scale; setCam(camName); },
  setSingleVisible(v) { if (world) world.root.visible = world.mirror.visible = v; plate.visible = cellMarks.visible = v && staged; },
  stageIn() { setStaged(true); },
  hasWorld: () => !!world,
  lastTask: () => { const t = skillsCache.find((x) => x.id === lastSkill); return t ? { instruction: t.instruction, success_rate: t.success_rate, n_eval: t.n_eval || 20 } : null; },
  setMulti(on) { $("#app").classList.toggle("multi", on); $("#view-sim").classList.toggle("multi", on); },
  THREE, scene3, camera, controls, renderer,
  meshesFor, meshGeometry, studioMaterial, srgb, addLineParts, RoundedBoxGeometry, postStage, applyFrame, buildWorld, populate, armScale: (rid) => ARM_SCALE[String(rid || "so101").split(".")[0]] || 1,
};

// ---------------------------------------------------------------- optimizer mount (web/optimizer.js + optimizer.css, optional)
// DOM contract: #optimizer-sec (hidden until the Optimize step is clicked) holds #optimizer-root, an empty div the
// optimizer owns. If the files are absent the section opens with a one-line note and the app is otherwise unchanged.
const optSec = $("#optimizer-sec"), optRoot = $("#optimizer-root");
function optimizerOpen(on) {
  optSec.hidden = !on;
  rail.optimizer.classList.toggle("sel", on);
  if (on) { $("#app").classList.remove("no-right"); optSec.scrollIntoView({ block: "nearest", behavior: "smooth" }); }
  optRoot.dataset.open = on ? "1" : "";
  window.dispatchEvent(new CustomEvent("forge:optimizer", { detail: { open: on } }));
}
// the Plan and Run steps scroll to their inspector card; the others open the step card below
const DETAIL = { planner: "#plan-sec", executor: "#arms-sec" };
function pulse(n) { n.classList.remove("pulse"); void n.offsetWidth; n.classList.add("pulse"); setTimeout(() => n.classList.remove("pulse"), 1300); }
// ---- the step card: one card beside the rail with that step's own data, from /api/state (skill and run records)
const NOT_REC = "not recorded for this model";
let railCard = null, railCardKey = "";
function closeRailCard() {
  railCard?.remove(); railCard = null; railCardKey = "";
  for (const li of Object.values(rail)) if (li.dataset.stage !== "optimizer") li.classList.remove("sel");
}
const ruleText = (r) => `${String(r.op).replace(/_/g, " ")}: ${Object.entries(r).filter(([k]) => k !== "op").map(([k, v]) => `${k} ${v}`).join(", ")}`;
const when = (t) => new Date(t * 1000).toLocaleString([], { month: "short", day: "numeric", hour: "numeric", minute: "2-digit" });
// a published model's saved record (engine records.py) shaped like a run, so the cards read one thing
function recordRun(sk) {
  const rec = sk?.record;
  if (!rec || !Object.keys(rec).length) return null;
  const n = Math.max(rec.eval?.length || 0, rec.demos?.per_round?.length || 0, rec.critic?.length || 0, rec.train?.rounds || 0, 1);
  const rounds = Array.from({ length: n }, (_, i) => ({
    index: i,
    demos: rec.demos?.per_round?.[i]?.ok || 0,
    train: i === 0 && rec.train ? { steps: rec.train.steps, seconds: rec.train.seconds } : {},
    eval: rec.eval?.find((e) => e.round === i) || {},
    successes: rec.eval?.find((e) => e.round === i)?.successes || [],
    plan: rec.critic?.find((c) => c.round === i) || {},
    loss: (rec.train?.loss_curve || []).filter((p) => p[0] === i).map((p) => [p[1], p[2]]),
  }));
  return { route: rec.route, rounds, best_round: rec.best_round ?? n - 1, layout_seed: rec.layout_seed, n_scenes: rec.n_scenes, spec: null, saved: true, demos: rec.demos };
}
function stepRows(k, sk, run) {
  const rows = []; // [label, text] or [label, null] for "not recorded"
  const add = (l, v) => rows.push([l, v === undefined || v === null || v === "" ? null : String(v)]);
  const spec = run?.spec || sk?.spec;
  const rounds = run?.rounds || [];
  const best = rounds[run?.best_round] || rounds[rounds.length - 1];
  const btn = [];
  if (k === "task") {
    add("Instruction", spec?.instruction || run?.prompt || sk?.instruction);
    if (spec) {
      add("Skill", spec.skill); add("Object", spec.obj); add("Target", spec.target); add("Distractors", spec.distractors);
      (spec.success || []).forEach((r, i) => add(i ? "" : "Success rule", ruleText(r)));
    }
    if (run?.parse?.model) add("Parsed by", modelTag(run.parse.model));
    const flat = (v) => (typeof v === "object" && v ? (Array.isArray(v) ? v.map(flat).join("; ") : Object.entries(v).map(([a, b]) => `${a.replace(/_/g, " ")} ${flat(b)}`).join(", ")) : String(v));
    for (const key of ["feasibility", "routing"]) {
      const fe = spec?.[key] ?? sk?.[key];
      if (fe == null) continue;
      const cap = key[0].toUpperCase() + key.slice(1);
      if (typeof fe === "object" && !Array.isArray(fe)) Object.entries(fe).forEach(([a, b], i) => add(i ? a.replace(/_/g, " ") : `${cap}: ${a.replace(/_/g, " ")}`, flat(b)));
      else add(cap, flat(fe));
    }
  } else if (k === "router") {
    const r = run?.route;
    if (r && r.decision) {
      add("Route", r.decision.replace(/_/g, " ")); add("Why", r.reason);
      if (r.expected != null) add("Expected success", `${Math.round(r.expected * 100)}%`);
      if (r.estimate_s) add("Estimated time", `${Math.round(r.estimate_s)}s`);
      if (r.skill_id) add("Starts from", `skill-${r.skill_id.slice(-4)}`);
      (r.checks || []).forEach((c) => add(c.name, `${c.ok ? "ok" : "failed"}: ${c.detail}`));
    } else {
      add("Route", null);
      if (sk) { add("Asked from", sk.origin); if (sk.parent) add("Started from", `skill-${sk.parent.slice(-4)}`); }
    }
  } else if (k === "scene") {
    add("Robot", run?.robot || sk?.robot);
    if (spec) { add("Block", spec.obj); add("Target", spec.target); add("Distractors", spec.distractors); }
    add("Held-out scenes", sk?.n_eval || (best?.eval?.n));
    add("Replays from", sk?.replay_source);
    add("Layout seed", run?.layout_seed);
    btn.push(["Set orbit camera", () => setCam("orbit")]);
  } else if (k === "data") {
    const n = rounds.reduce((a, r) => a + (r.demos || 0), 0);
    add("Demonstrations", n || null);
    if (rounds.length) add("Collected", "scripted expert in randomized scenes");
    if (rounds.length > 1) add("Rounds", `${rounds.length} (${rounds.map((r) => r.demos).join(" + ")})`);
    const d = run?.demos;
    add("Expert success", d?.asked ? `${d.ok}/${d.asked} demos succeeded` : null);
  } else if (k === "trainer") {
    const secs = rounds.reduce((a, r) => a + (r.train?.seconds || 0), 0) || sk?.train_seconds;
    add("Train time", secs ? `${(+secs).toFixed(1)}s` : null);
    if (rounds.length) add("Rounds", rounds.length);
    const steps = rounds.reduce((a, r) => a + (r.train?.steps || 0), 0);
    add("Steps", steps || null);
    const lastLoss = [...rounds].reverse().find((r) => r.loss?.length)?.loss.at(-1);
    add("Final loss", lastLoss ? lastLoss[1].toFixed(3) : null);
    if (sk?.seconds) add("Prompt to model", `${fmt(sk.seconds)}`);
  } else if (k === "eval") {
    const n = sk?.n_eval || best?.eval?.n || 0;
    const ok = sk ? Math.round(sk.success_rate * n) : best?.eval?.succeeded;
    add("Succeeded", n && ok != null ? `${ok}/${n} on held-out scenes` : null);
    if (best?.successes?.length) add("Per scene", best.successes.map((x) => (x ? "\u25CF" : "\u25CB")).join(" "));
    else add("Per scene", null);
    if (sk?.tests?.length) add("Test runs", `${sk.tests.length} on fresh scenes`);
  } else if (k === "critic") {
    const ps = rounds.filter((r) => r.plan && r.plan.action);
    if (ps.length) ps.forEach((r) => add(`Round ${r.index + 1}: ${r.plan.action}`, r.plan.rationale));
    else add("Verdict", null);
  } else if (k === "registry") {
    if (sk) {
      add("Model", sk.name || `skill-${sk.id.slice(-4)}`); add("Id", sk.id);
      add("Success", `${Math.round(sk.success_rate * (sk.n_eval || 20))}/${sk.n_eval || 20} held-out`);
      if (sk.created) add("Published", when(sk.created));
      if (sk.agent_address) add("Agent", `${sk.agent_address.slice(0, 14)}...`);
      btn.push(["Open the Library card", () => { closeRailCard(); window.forgeLib?.openTile(sk.id); }]);
    } else add("Model", null);
  }
  return { rows, btn };
}
async function openRailCard(k) {
  if (railCard && railCardKey === k) return closeRailCard();
  closeRailCard();
  railCardKey = k;
  rail[k].classList.add("sel");
  let st = { skills: [], runs: [] };
  try { st = await (await fetch("/api/state")).json(); } catch { /* the card still opens with what the page holds */ }
  if (railCardKey !== k) return; // closed or switched while loading
  const sid = selSkill || lastSkill || (welcomeOn && welcomeShown?.id) || "";
  const live = st.runs.find((r) => r.id === current && !runDone) || st.runs.find((r) => sid && r.skill_id === sid) || st.runs.find((r) => r.id === current);
  const sk = st.skills.find((x) => x.id === (sid || live?.skill_id));
  const run = live || recordRun(sk);
  const { rows, btn } = sk || run ? stepRows(k, sk, run) : { rows: [], btn: [] };
  const c = el("div", "rail-card");
  c.setAttribute("role", "dialog"); c.setAttribute("aria-label", NAMES[k]);
  const x = el("button", "wf-x", "\u00D7"); x.type = "button"; x.setAttribute("aria-label", "Close"); x.onclick = closeRailCard;
  c.append(x, el("div", "rc-t", NAMES[k]));
  if (!sk && !run) c.append(el("p", "rc-empty", "Pick a model from the Library or describe a task."));
  const body = el("dl", "rc-rows");
  for (const [l, v] of rows) { body.append(el("dt", "", l)); body.append(el("dd", v === null ? "none" : "", v === null ? NOT_REC : v)); }
  c.append(body);
  for (const [t, fn] of btn) { const b = el("button", "rc-btn", t); b.type = "button"; b.onclick = fn; c.append(b); }
  const host = $("#view-sim"), li = rail[k];
  host.append(c);
  const hr = host.getBoundingClientRect(), lr = li.getBoundingClientRect(), rr = $("#rail").getBoundingClientRect();
  c.style.right = `${hr.right - rr.left + 10}px`;
  c.style.top = `${Math.max(8, Math.min(lr.top - hr.top - 6, hr.height - c.offsetHeight - 80))}px`;
  railCard = c;
}
document.addEventListener("keydown", (ev) => { if (ev.key === "Escape" && railCard) closeRailCard(); });
async function showSavedModel(sk) {
  savedModel = { sk };
  let run = null;
  try {
    const st = await (await fetch("/api/state")).json();
    run = st.runs.find((r) => r.skill_id === sk.id) || recordRun(st.skills.find((x) => x.id === sk.id)) || null;
  } catch { /* the saved skill record still fills the cards */ }
  if (!savedModel || savedModel.sk !== sk) return;
  savedModel.run = run;
  losses = (run?.rounds || []).map((r) => r.loss || []).filter((l) => l.length);
  drawLoss(); drawRounds();
  const last = losses.at(-1)?.at(-1);
  $("#loss-now").textContent = last ? last[1].toFixed(3) : "";
  $("#loss-steps").textContent = sk.train_seconds != null ? `trained ${(+sk.train_seconds).toFixed(1)}s` : "";
  $("#loss-init").textContent = "";
  const box = $("#critic");
  box.innerHTML = "";
  const ps = (run?.rounds || []).filter((r) => r.plan && r.plan.action);
  if (!ps.length) box.append(el("span", "dim", "critic verdict not recorded for this model"));
  for (const r of ps) { const d = el("div"); d.append(el("b", "", `Round ${r.index + 1}: ${r.plan.action}. `), document.createTextNode(r.plan.rationale || "")); box.append(d); }
}
function openStage(k) {
  if (k === "optimizer") { closeRailCard(); return optimizerOpen(optSec.hidden); }
  setView("sim");
  if (DETAIL[k]) {
    closeRailCard();
    $("#app").classList.remove("no-right");
    const sec = $(DETAIL[k]);
    sec.scrollIntoView({ block: "nearest", behavior: "smooth" });
    return pulse(sec);
  }
  openRailCard(k);
}
for (const [k, li] of Object.entries(rail)) {
  li.addEventListener("click", () => openStage(k));
  li.addEventListener("keydown", (ev) => { if (ev.key === "Enter" || ev.key === " ") { ev.preventDefault(); openStage(k); } });
}
// the optimizer runs in the background on one skill at a time; the rail only shows it on that skill
function optStage(state, metric, skill) {
  if (skill && skill !== selSkill && optSec.hidden) return stage("optimizer", "idle", "");
  stage("optimizer", state, metric);
}
$("#optimizer-close").addEventListener("click", () => optimizerOpen(false));
(async function loadOptimizer() {
  const have = async (url, type) => {
    try {
      const r = await fetch(url, { method: "HEAD" });
      return r.ok && (r.headers.get("content-type") || "").includes(type);
    } catch { return false; }
  };
  if (await have("/static/optimizer.css", "css")) {
    const l = document.createElement("link");
    l.rel = "stylesheet"; l.href = "/static/optimizer.css";
    document.head.append(l);
  }
  if (!(await have("/static/optimizer.js", "javascript"))) { optRoot.dataset.loaded = "absent"; return; }
  try {
    const mod = await import("/static/optimizer.js");
    optRoot.dataset.loaded = "1";
    // optional named export: mount(root, ctx). A module that wires itself to #optimizer-root needs no export.
    if (typeof mod.mount === "function") mod.mount(optRoot, { forge: window.forge, stage: (k, st, m, skill) => (k === "optimizer" ? optStage(st, m, skill) : stage(k, st, m)), open: optimizerOpen, current: () => current });
  } catch (err) {
    optRoot.dataset.loaded = "error";
    optRoot.textContent = `The optimizer did not load: ${err.message}`;
  }
})();

idleScene();
setView("skills"); // the library is the first thing after the landing page
// warm the arm's meshes and build their geometry in idle slices, so the first tile draws at once
meshesFor("so101").then(async (m) => {
  for (const k of Object.keys(m)) {
    meshGeometry(`so101|${k}`, m[k]);
    await new Promise((r) => setTimeout(r, 0));
  }
});

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

// empty inspector cards start with their quiet placeholders
drawLoss();
drawRounds();

// the bottom-right voice pill and the docked prompt bar float over scrolling pages: reserve their height so the last row stays readable
function reserveBottom() {
  const root = document.documentElement, vh = window.innerHeight;
  const top = (sel) => { const n = document.querySelector(sel); if (!n || !n.offsetParent && getComputedStyle(n).position !== "fixed") return vh; return n.getBoundingClientRect().top; };
  const pill = vh - top(".ag-pill");
  const dock = $("#app").classList.contains("immersive") ? null : $("#dock").getBoundingClientRect();
  const pr = document.querySelector(".ag-pill")?.getBoundingClientRect();
  // the prompt bar owns a strip under the views; it only steps up when the voice pill would sit under it
  const lift = dock && pr && pr.width && Math.min(pr.right, dock.right) - Math.max(pr.left, dock.left) > 12 ? pill + 8 : 0;
  // with the inspector folded away the rounds strip runs under the voice pill: it ends where the pill begins
  const tabs = document.querySelector(".tabs"), cv = document.querySelector(".canvas");
  const tr = tabs?.getBoundingClientRect(), cr = cv?.getBoundingClientRect();
  const tabsClear = tr && cr && pr && pr.width && pr.top < tr.bottom && pr.bottom > tr.top && pr.left < cr.right ? Math.max(0, cr.right - pr.left + 10) : 0;
  root.style.setProperty("--tabs-clear", `${Math.round(tabsClear)}px`);
  root.style.setProperty("--dock-lift", `${Math.max(0, lift)}px`);
  root.style.setProperty("--pill-pad", `${Math.max(0, pill) + 16}px`);
  root.style.setProperty("--clear", `${(dock ? 0 : Math.max(pill, 0)) + 16}px`);
}
addEventListener("resize", reserveBottom);
new MutationObserver(reserveBottom).observe($("#app"), { attributes: true, attributeFilter: ["data-view", "class"] });
const pillWatch = setInterval(() => { const p = document.querySelector(".ag-pill"); if (!p) return; clearInterval(pillWatch); new ResizeObserver(reserveBottom).observe(p); reserveBottom(); }, 400);
