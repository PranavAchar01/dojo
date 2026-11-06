// Robot tab: pick an arm, a wrist and an end effector from sourced models, see each part and the assembled robot,
// run the checks (MJCF compiles, joint limits, finger opening, reach over the table, holding torque), then make it
// the embodiment the next skill trains on.
import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import { RoomEnvironment } from "three/addons/environments/RoomEnvironment.js";
import { toCreasedNormals } from "three/addons/utils/BufferGeometryUtils.js";

const $ = (s) => document.querySelector(s);
const el = (tag, cls, text) => {
  const e = document.createElement(tag);
  if (cls) e.className = cls;
  if (text !== undefined) e.textContent = text;
  return e;
};

const SLOTS = [
  ["arm", "Arm", "base, shoulder, elbow and the arm's own wrist and jaw", "arms"],
  ["wrist", "Wrist", "keep every wrist joint, or lock the roll for a smaller action space", "wrists"],
  ["effector", "End effector", "keep the arm's jaw or mount a sourced gripper on the last link", "effectors"],
];
const SKILL_NAMES = { reach: "Reach", push: "Push", lift: "Lift", pick_place: "Pick and place", stack: "Stack" };

let CAT = null;
const sel = { arm: "so101", wrist: "native", effector: "native" };
let active = "so101";

export function robotId(s = sel) {
  return s.arm === "so101" && s.wrist === "native" && s.effector === "native" ? "so101" : `${s.arm}.${s.wrist}.${s.effector}`;
}
function parse(rid) {
  if (rid === "so101") return { arm: "so101", wrist: "native", effector: "native" };
  const [arm, wrist, effector] = rid.split(".");
  return { arm, wrist, effector };
}
export function robotLabel(rid) {
  if (!CAT) return rid === "so101" ? "SO-101" : rid;
  const p = parse(rid);
  const arm = CAT.arms.find((a) => a.id === p.arm);
  const ee = CAT.effectors.find((e) => e.id === p.effector);
  let s = arm ? arm.short || arm.name : p.arm;
  if (p.wrist === "locked") s += ", roll locked";
  if (p.effector !== "native") s += ` + ${ee ? ee.short || ee.name : p.effector}`;
  return s;
}

// ---------------------------------------------------------------- mesh buffers -> three.js
const geoms = new Map();
function geometry(key, data) {
  if (geoms.has(key)) return geoms.get(key);
  const v = new Float32Array(Uint8Array.from(atob(data.v), (c) => c.charCodeAt(0)).buffer);
  const f = new Uint32Array(Uint8Array.from(atob(data.f), (c) => c.charCodeAt(0)).buffer);
  let g = new THREE.BufferGeometry();
  g.setAttribute("position", new THREE.BufferAttribute(v, 3));
  g.setIndex(new THREE.BufferAttribute(f, 1));
  g = toCreasedNormals(g, Math.PI / 7);
  geoms.set(key, g);
  return g;
}
const mats = new Map();
function material(rgba) {
  const key = rgba.map((x) => x.toFixed(2)).join(",");
  if (!mats.has(key)) {
    // near-white source colours wash out on the pale stage: pull them down a little so the shape reads
    const lum = 0.299 * rgba[0] + 0.587 * rgba[1] + 0.114 * rgba[2];
    const k = lum > 0.8 ? 0.72 / lum : 1;
    const c = new THREE.Color().setRGB(rgba[0] * k, rgba[1] * k, rgba[2] * k, THREE.SRGBColorSpace);
    mats.set(key, new THREE.MeshPhysicalMaterial({ color: c, roughness: 0.45, clearcoat: 0.25, clearcoatRoughness: 0.4 }));
  }
  return mats.get(key);
}
function partGroup(view) {
  const g = new THREE.Group();
  for (const p of view.geoms) {
    const data = view.meshes[String(p.mesh)];
    // the same source mesh shows up in several previews (every SO-101 variant): build its geometry once
    const id = `${data.v.length}:${data.f.length}:${data.v.slice(0, 48)}`;
    const m = new THREE.Mesh(geometry(id, data), material(p.rgba));
    m.position.fromArray(p.pos);
    m.quaternion.set(p.quat[1], p.quat[2], p.quat[3], p.quat[0]);
    m.castShadow = m.receiveShadow = true;
    g.add(m);
  }
  return g;
}
const views = new Map();
const fetchJSON = (url) => {
  if (!views.has(url)) {
    views.set(url, fetch(url).then((r) => {
      if (!r.ok) throw new Error(`${r.status}`);
      return r.json();
    }));
  }
  return views.get(url);
};
// the wrist options and the native jaw are not parts of their own: they are shown on the selected arm,
// framed close on the wrist and jaw (the stock SO-101 if that arm cannot be built with them)
const onArm = (slot, id) => slot === "wrist" || (slot === "effector" && id === "native");
function partURL(slot, id, arm = sel.arm) {
  if (slot === "arm") return `/api/parts/arm/${id}/view`;
  if (!onArm(slot, id)) return `/api/parts/effector/${id}/view`;
  const rid = robotId({ arm, wrist: slot === "wrist" ? id : "native", effector: "native" });
  return `/api/robots/${rid}/view`;
}
function closeUp(v) { // a box around the tool point big enough for the wrist and the jaw
  const lo = new THREE.Vector3().fromArray(v.bbox[0]);
  const hi = new THREE.Vector3().fromArray(v.bbox[1]);
  const h = 0.17 * hi.distanceTo(lo);
  const c = new THREE.Vector3().fromArray(v.tcp);
  c.z += 0.4 * h;
  return [[c.x - h, c.y - h, c.z - h], [c.x + h, c.y + h, c.z + h]];
}

function stage(canvas) {
  const r = new THREE.WebGLRenderer({ canvas, antialias: true, preserveDrawingBuffer: true, alpha: true });
  r.setPixelRatio(Math.min(window.devicePixelRatio, 2));
  r.toneMapping = THREE.ACESFilmicToneMapping;
  r.shadowMap.enabled = true;
  const s = new THREE.Scene();
  const pm = new THREE.PMREMGenerator(r);
  s.environment = pm.fromScene(new RoomEnvironment(), 0.04).texture;
  s.environmentIntensity = 0.8;
  const key = new THREE.DirectionalLight(0xfff4e6, 2.0);
  key.position.set(0.9, -0.6, 1.6);
  s.add(key);
  const rim = new THREE.DirectionalLight(0xdfe8ff, 0.6);
  rim.position.set(-0.8, 0.9, 0.9);
  s.add(rim);
  const cam = new THREE.PerspectiveCamera(30, 1, 0.005, 20);
  cam.up.set(0, 0, 1);
  return { r, s, cam };
}
function frame(cam, bbox, dir = [1.1, -0.9, 0.7], pad = 1.0) {
  const lo = new THREE.Vector3().fromArray(bbox[0]);
  const hi = new THREE.Vector3().fromArray(bbox[1]);
  const c = lo.clone().add(hi).multiplyScalar(0.5);
  const size = Math.max(hi.clone().sub(lo).length(), 0.05);
  const d = new THREE.Vector3().fromArray(dir).normalize();
  const dist = (pad * size) / (2 * Math.tan((cam.fov * Math.PI) / 360));
  cam.position.copy(c).addScaledVector(d, dist);
  cam.lookAt(c);
  return c;
}

// one shared offscreen renderer draws every part tile to an image
let thumb = null;
const thumbQueue = [];
let thumbBusy = false;
function queueThumb(img, slot, id) {
  thumbQueue.push([img, slot, id]);
  if (!thumbBusy) drainThumbs();
}
async function drainThumbs() {
  thumbBusy = true;
  if (!thumb) {
    const c = document.createElement("canvas");
    c.width = 360; c.height = 240;
    thumb = stage(c);
    thumb.r.setPixelRatio(1);
    thumb.r.setSize(360, 240, false);
    thumb.cam.aspect = 1.5;
    thumb.cam.updateProjectionMatrix();
  }
  while (thumbQueue.length) {
    const [img, slot, id] = thumbQueue.shift();
    try {
      let v;
      try {
        v = await fetchJSON(partURL(slot, id));
      } catch (err) {
        if (!onArm(slot, id) || sel.arm === "so101") throw err;
        v = await fetchJSON(partURL(slot, id, "so101"));
      }
      const g = partGroup(v);
      thumb.s.add(g);
      if (onArm(slot, id) && v.tcp) frame(thumb.cam, closeUp(v), [1.1, -0.9, 0.55]);
      else frame(thumb.cam, v.bbox);
      thumb.r.render(thumb.s, thumb.cam);
      img.src = thumb.r.domElement.toDataURL("image/png");
      img.classList.add("in");
      thumb.s.remove(g);
    } catch {
      img.alt = "preview failed";
      img.classList.add("in", "bad");
    }
  }
  thumbBusy = false;
}

// ---------------------------------------------------------------- the assembled robot, live
let asm = null;
let asmRobot = null;
let visible = false;
function initAssembly() {
  const canvas = $("#asm");
  asm = stage(canvas);
  asm.s.background = new THREE.Color(0xf7f6fb);
  asm.controls = new OrbitControls(asm.cam, canvas);
  asm.controls.enableDamping = true;
  asm.controls.autoRotate = true;
  asm.controls.autoRotateSpeed = 0.6;
  canvas.addEventListener("pointerdown", () => { asm.controls.autoRotate = false; });
  // the table workspace the checks are measured over, drawn as a thin plate
  const plate = new THREE.Mesh(
    new THREE.RingGeometry(0.125, 0.285, 48, 1, -0.663, 1.326),
    new THREE.MeshBasicMaterial({ color: 0xe4e4e7, side: THREE.DoubleSide }),
  );
  plate.position.z = -0.0005;
  asm.s.add(plate);
  const grid = new THREE.GridHelper(1.0, 20, 0xd4d4d8, 0xe4e4e7);
  grid.rotation.x = Math.PI / 2;
  grid.position.z = -0.001;
  asm.s.add(grid);
  const resize = () => {
    const b = canvas.parentElement.getBoundingClientRect();
    if (!b.width) return;
    asm.r.setSize(b.width, b.height, false);
    asm.cam.aspect = b.width / b.height;
    asm.cam.updateProjectionMatrix();
  };
  new ResizeObserver(resize).observe(canvas.parentElement);
  resize();
  asm.cam.position.set(0.72, -0.62, 0.48);
  asm.controls.target.set(0.13, 0, 0.1);
  asm.controls.update();
  const loop = () => {
    if (visible) {
      asm.controls.update();
      asm.r.render(asm.s, asm.cam);
    }
    requestAnimationFrame(loop);
  };
  loop();
}
const css = (v) => getComputedStyle(document.documentElement).getPropertyValue(v).trim();
const OK = new THREE.MeshBasicMaterial({ color: new THREE.Color(css("--ok") || "#30a46c") });
const MISS = new THREE.MeshBasicMaterial({ color: new THREE.Color(css("--bad") || "#e5484d") });
const dotGeo = new THREE.SphereGeometry(0.0045, 12, 8);
function showAssembly(rid, view) {
  if (asmRobot) asm.s.remove(asmRobot);
  const g = partGroup(view);
  for (const p of view.reach.points) {
    const d = new THREE.Mesh(dotGeo, p.grasp && p.hover ? OK : MISS);
    d.position.set(p.x, p.y, 0.004);
    g.add(d);
  }
  const tcp = new THREE.Mesh(new THREE.SphereGeometry(0.006, 16, 12), new THREE.MeshBasicMaterial({ color: new THREE.Color(css("--text") || "#18181b") }));
  tcp.position.fromArray(view.tcp);
  g.add(tcp);
  asm.s.add(g);
  asmRobot = g;
}

// ---------------------------------------------------------------- page
function card(slot, p) {
  const b = el("button", "part");
  b.type = "button";
  b.dataset.slot = slot;
  b.dataset.id = p.id;
  const fig = el("div", "part-fig");
  const img = el("img");
  img.alt = p.name;
  fig.append(img, el("span", "spin"));
  const name = el("div", "part-name", p.name);
  const by = el("div", "part-by", onArm(slot, p.id) ? `on the ${armName()}` : p.maker);
  const meta = el("div", "part-meta mono");
  if (slot === "arm") meta.textContent = `${p.dof} arm joints + jaw · ${Math.round(p.reach_m * 100)} cm reach`;
  else if (p.kind) meta.textContent = p.kind;
  const prov = el("div", "prov");
  const lic = el("span", "lic mono", p.licence);
  prov.append(lic);
  if (p.url) {
    const a = el("a", "src", "source");
    a.href = p.url;
    a.target = "_blank";
    a.rel = "noopener noreferrer";
    a.title = p.path || p.url;
    a.addEventListener("click", (e) => e.stopPropagation());
    prov.append(a);
  }
  b.append(fig, name, by);
  if (meta.textContent) b.append(meta);
  if (slot !== "arm" && p.notes) b.append(el("div", "part-note", p.notes));
  b.append(prov);
  b.onclick = () => pick(slot, p.id);
  queueThumb(img, slot, p.id);
  return b;
}

function armName(id = sel.arm) {
  const a = CAT.arms.find((x) => x.id === id);
  return a ? a.short || a.name : id;
}
function redrawOnArm() { // wrist and native-jaw tiles follow the selected arm
  document.querySelectorAll(".part").forEach((c) => {
    if (!onArm(c.dataset.slot, c.dataset.id)) return;
    c.querySelector(".part-by").textContent = `on the ${armName()}`;
    const img = c.querySelector(".part-fig img");
    img.classList.remove("in", "bad");
    queueThumb(img, c.dataset.slot, c.dataset.id);
  });
}
function pick(slot, id) {
  const armChanged = slot === "arm" && sel.arm !== id;
  sel[slot] = id;
  document.querySelectorAll(`.part[data-slot="${slot}"]`).forEach((c) => c.classList.toggle("on", c.dataset.id === id));
  if (armChanged) redrawOnArm();
  assemble();
}

let assembling = 0;
async function assemble() {
  const rid = robotId();
  const n = ++assembling;
  $("#asm-id").textContent = rid;
  $("#asm-name").textContent = robotLabel(rid);
  $("#asm-load").classList.remove("hidden");
  $("#use-robot").disabled = true;
  $("#checks").innerHTML = "";
  $("#feas").innerHTML = "";
  $("#asm-fail").classList.add("hidden");
  $("#asm-sum").textContent = "assembling and checking...";
  const [rep, view] = await Promise.allSettled([fetchJSON(`/api/robots/${rid}/report`), fetchJSON(`/api/robots/${rid}/view`)]);
  if (n !== assembling) return;
  $("#asm-load").classList.add("hidden");
  if (view.status === "fulfilled") showAssembly(rid, view.value);
  if (rep.status !== "fulfilled") {
    failCard("The server could not assemble these parts.", String(rep.reason));
    $("#asm-sum").textContent = "";
    return;
  }
  renderReport(rid, rep.value);
}

function checkRow(ok, name, detail, level) {
  const li = el("li", ok ? "ok" : level === "block" ? "bad" : "warn");
  const ic = el("span", "ic", ok ? "✓" : level === "block" ? "✕" : "!");
  const tx = el("div", "tx");
  tx.append(el("b", "", name), el("div", "d", detail));
  li.append(ic, tx);
  return li;
}
function failCard(title, detail) {
  const f = $("#asm-fail");
  f.innerHTML = "";
  f.append(el("div", "lbl", "Cannot train on this robot"), el("b", "", title), el("div", "d", detail));
  f.classList.remove("hidden");
}
function renderReport(rid, rep) {
  const box = $("#checks");
  for (const c of rep.checks) box.append(checkRow(c.ok, c.name, c.detail, c.level));
  const passed = rep.checks.filter((c) => c.ok).length;
  $("#asm-sum").textContent = `${passed}/${rep.checks.length} passed`;
  const fb = $("#feas");
  let any = false;
  for (const [k, v] of Object.entries(rep.skills || {})) {
    fb.append(checkRow(v.ok, SKILL_NAMES[k] || k, v.reason, "block"));
    any ||= v.ok;
  }
  const facts = [];
  if (rep.act_dim) facts.push(`${rep.act_dim}-D actions`);
  if (rep.stroke_mm) facts.push(`fingers ${rep.stroke_mm[0]}-${rep.stroke_mm[1]} mm`);
  if (rep.mass_kg) facts.push(`${rep.mass_kg} kg`);
  $("#asm-facts").textContent = facts.join(" · ");
  if (!rep.ok) failCard(rep.failure ? rep.failure.split(":")[0] : "A check failed", rep.failure ? rep.failure.split(": ").slice(1).join(": ") : "");
  else if (!any) failCard("No skill fits this robot", "Every skill fails a reach or grip check; see the list below.");
  const ub = $("#use-robot");
  ub.disabled = !rep.ok || !any || rid === active;
  ub.textContent = rid === active ? "Skills train on this robot" : "Train the next skill on this robot";
  const note = $("#use-note");
  note.textContent = rep.ok && rep.act_dim && rep.act_dim !== 6
    ? `${rep.act_dim} action dimensions instead of 6: the policy keeps its vision and language trunk and retrains its action layers on this robot's demos.`
    : rep.ok && rid !== "so101"
      ? "Same 6 action dimensions: the policy keeps every layer and fine-tunes on this robot's own demos."
      : "";
}

async function useRobot() {
  const rid = robotId();
  const b = $("#use-robot");
  b.disabled = true;
  const r = await fetch("/api/robot/use", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ robot: rid }),
  });
  if (!r.ok) {
    failCard("The server refused this robot.", (await r.json()).detail || "");
    return;
  }
  setActive(rid);
  b.textContent = "Skills train on this robot";
}

function setActive(rid) {
  active = rid;
  window.dispatchEvent(new CustomEvent("robot-change", { detail: { id: rid, label: robotLabel(rid) } }));
}

let built = false;
export async function initBuilder(activeId) {
  active = activeId || "so101";
  Object.assign(sel, parse(active));
  CAT = await (await fetch("/api/robots")).json();
  setActive(active);
}

export function showBuilder(on) {
  visible = on;
  if (!on || built) return;
  built = true;
  initAssembly();
  const box = $("#slots");
  for (const [slot, title, sub, key] of SLOTS) {
    const s = el("section", "slot");
    const h = el("div", "slot-h");
    h.append(el("span", "lbl", title), el("span", "dim", sub));
    const grid = el("div", "parts");
    for (const p of CAT[key]) grid.append(card(slot, p));
    s.append(h, grid);
    box.append(s);
    grid.querySelectorAll(".part").forEach((c) => c.classList.toggle("on", c.dataset.id === sel[slot]));
  }
  const no = $("#not-offered");
  for (const p of CAT.not_offered) {
    const li = el("li");
    li.append(el("b", "", p.name), document.createTextNode(`: ${p.reason}`));
    no.append(li);
  }
  $("#use-robot").onclick = useRobot;
  assemble();
}
