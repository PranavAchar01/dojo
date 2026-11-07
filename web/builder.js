// Robot tab: pick an arm, a wrist and an end effector from sourced models, see each part and the assembled robot,
// run the checks (MJCF compiles, joint limits, finger opening, reach over the table, holding torque), then make it
// the embodiment the next skill trains on.
import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import { setupRenderer, makeEnvironment, addStudioLights, robotMaterial as studioMaterial } from "./studio.js";

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
const SKILL_NAMES = { reach: "Reach", push: "Push", lift: "Lift", pick_place: "Pick and place", stack: "Stack", kit: "Kit a tray slot", insert: "Insert", rotate: "Rotate and insert" };

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
const bytes = (b64) => {
  if (Uint8Array.fromBase64) return Uint8Array.fromBase64(b64).buffer;
  const bin = atob(b64);
  const u = new Uint8Array(bin.length);
  for (let i = 0; i < u.length; i++) u[i] = bin.charCodeAt(i);
  return u.buffer;
};
// Creased normals on typed arrays: each face corner takes the area-weighted normals of the faces around its vertex
// that lie within `angle` of its own face. three's toCreasedNormals does the same through string-keyed maps and
// took seconds per mesh; this is a few milliseconds.
function creased(pos, idx, angle) {
  const nF = idx.length / 3;
  const nV = pos.length / 3;
  const fn = new Float32Array(nF * 3); // area-weighted face normals
  const unit = new Float32Array(nF * 3);
  for (let f = 0; f < nF; f++) {
    const a = idx[3 * f] * 3, b = idx[3 * f + 1] * 3, c = idx[3 * f + 2] * 3;
    const ux = pos[b] - pos[a], uy = pos[b + 1] - pos[a + 1], uz = pos[b + 2] - pos[a + 2];
    const vx = pos[c] - pos[a], vy = pos[c + 1] - pos[a + 1], vz = pos[c + 2] - pos[a + 2];
    const x = uy * vz - uz * vy, y = uz * vx - ux * vz, z = ux * vy - uy * vx;
    const l = Math.hypot(x, y, z) || 1;
    fn[3 * f] = x; fn[3 * f + 1] = y; fn[3 * f + 2] = z;
    unit[3 * f] = x / l; unit[3 * f + 1] = y / l; unit[3 * f + 2] = z / l;
  }
  const start = new Uint32Array(nV + 1); // faces around each vertex, packed
  for (let i = 0; i < idx.length; i++) start[idx[i] + 1]++;
  for (let v = 0; v < nV; v++) start[v + 1] += start[v];
  const fill = start.slice(0, nV);
  const around = new Uint32Array(idx.length);
  for (let f = 0; f < nF; f++) for (let k = 0; k < 3; k++) around[fill[idx[3 * f + k]]++] = f;
  const cos = Math.cos(angle);
  const P = new Float32Array(nF * 9), N = new Float32Array(nF * 9);
  for (let f = 0; f < nF; f++) {
    for (let k = 0; k < 3; k++) {
      const v = idx[3 * f + k];
      let x = 0, y = 0, z = 0;
      for (let j = start[v]; j < start[v + 1]; j++) {
        const g = around[j];
        if (unit[3 * g] * unit[3 * f] + unit[3 * g + 1] * unit[3 * f + 1] + unit[3 * g + 2] * unit[3 * f + 2] >= cos) {
          x += fn[3 * g]; y += fn[3 * g + 1]; z += fn[3 * g + 2];
        }
      }
      const l = Math.hypot(x, y, z) || 1;
      const o = 9 * f + 3 * k;
      P[o] = pos[3 * v]; P[o + 1] = pos[3 * v + 1]; P[o + 2] = pos[3 * v + 2];
      N[o] = x / l; N[o + 1] = y / l; N[o + 2] = z / l;
    }
  }
  const g = new THREE.BufferGeometry();
  g.setAttribute("position", new THREE.BufferAttribute(P, 3));
  g.setAttribute("normal", new THREE.BufferAttribute(N, 3));
  return g;
}
function geometry(key, data) {
  if (geoms.has(key)) return geoms.get(key);
  const g = creased(new Float32Array(bytes(data.v)), new Uint32Array(bytes(data.f)), Math.PI / 7);
  geoms.set(key, g);
  return g;
}
function partGroup(view) {
  const g = new THREE.Group();
  for (const p of view.geoms) {
    const data = view.meshes[String(p.mesh)];
    // the same source mesh shows up in several previews (every SO-101 variant): build its geometry once
    const id = `${data.v.length}:${data.f.length}:${data.v.slice(0, 48)}`;
    const m = new THREE.Mesh(geometry(id, data), studioMaterial(p.rgba, p.name));
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
    // a failed fetch (server restarting, network blip) is forgotten so the next look retries it
    // network errors (server restarting) retry a few times; anything that still fails is forgotten so the next look retries
    const get = (tries) => fetch(url).catch((e) => {
      if (tries <= 0) throw e;
      return new Promise((ok) => setTimeout(ok, 1500)).then(() => get(tries - 1));
    });
    views.set(url, get(4).then((r) => {
      if (!r.ok) throw new Error(`${r.status}`);
      return r.json();
    }).catch((e) => { views.delete(url); throw e; }));
  }
  return views.get(url);
};
// the wrist options and the native jaw are not parts of their own: they are shown on the selected arm,
// framed close on the wrist and jaw (the stock SO-101 if that arm cannot be built with them)
const onArm = (slot, id) => slot === "wrist" || (slot === "effector" && id === "native");
const prefetch = (rid) => { fetchJSON(`/api/robots/${rid}/view`).catch(() => {}); fetchJSON(`/api/robots/${rid}/report`).catch(() => {}); };

function stage(canvas) {
  const r = new THREE.WebGLRenderer({ canvas, antialias: true, preserveDrawingBuffer: true, alpha: true });
  r.setPixelRatio(Math.min(window.devicePixelRatio, 2));
  setupRenderer(r);
  const s = new THREE.Scene();
  s.environment = makeEnvironment(r);
  s.environmentIntensity = 1.0;
  addStudioLights(s, { target: [0.1, 0, 0.1], extent: 0.7, reach: 0.8 });
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

// card images are rendered once on the server (assets/thumbs, long-cached): a card is an <img>, nothing assembles
// in the browser until an arm is picked
const thumbURL = (slot, id) => (onArm(slot, id) ? `/api/thumbs/${slot}/${id}?arm=${sel.arm}` : `/api/thumbs/${slot}/${id}`);
function loadThumb(fig, img, slot, id, tries = 0) {
  fig.classList.add("sk");
  img.onload = () => { fig.classList.remove("sk"); img.classList.add("in"); };
  img.onerror = () => { // not drawn yet (first start renders them in the background): ask again shortly
    if (tries < 40) setTimeout(() => loadThumb(fig, img, slot, id, tries + 1), 3000);
    else { fig.classList.remove("sk"); img.alt = "preview unavailable"; }
  };
  img.src = thumbURL(slot, id);
}

// ---------------------------------------------------------------- the assembled robot, live
let asm = null;
let asmRobot = null;
let visible = false;
function initAssembly() {
  const canvas = $("#asm");
  asm = stage(canvas);
  asm.s.background = new THREE.Color(0xfafafa);
  asm.controls = new OrbitControls(asm.cam, canvas);
  asm.controls.enableDamping = true;
  asm.controls.autoRotate = true;
  asm.controls.autoRotateSpeed = 0.6;
  canvas.addEventListener("pointerdown", () => { asm.controls.autoRotate = false; });
  // the table workspace the checks are measured over, drawn as a thin plate
  const plate = new THREE.Mesh(
    new THREE.RingGeometry(0.125, 0.285, 48, 1, -0.663, 1.326),
    new THREE.MeshStandardMaterial({ color: 0xb9bcc1, roughness: 0.9, side: THREE.DoubleSide }),
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
  if (asm.framed !== rid.split(".")[0]) { // a new arm: frame it and its reach map, then let the viewer orbit
    asm.framed = rid.split(".")[0];
    const lo = view.bbox[0].map((v, i) => Math.min(v, [0.1, -0.2, 0][i]));
    const hi = view.bbox[1].map((v, i) => Math.max(v, [0.3, 0.2, 0][i]));
    asm.controls.target.copy(frame(asm.cam, [lo, hi], [1.1, -0.95, 0.62], 1.05));
    asm.controls.update();
  }
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
  fig.append(img);
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
  loadThumb(fig, img, slot, p.id);
  b.addEventListener("pointerenter", () => { if (slot === "arm") prefetch(robotId({ ...sel, arm: p.id, wrist: "native", effector: "native" })); }, { once: true });
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
    img.classList.remove("in");
    loadThumb(c.querySelector(".part-fig"), img, c.dataset.slot, c.dataset.id);
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
  const slow = setTimeout(() => { if (n === assembling) $("#asm-load").classList.remove("hidden"); }, 150);
  $("#use-robot").disabled = true;
  $("#checks").innerHTML = "";
  $("#feas").innerHTML = "";
  $("#asm-fail").classList.add("hidden");
  $("#asm-sum").textContent = "assembling and checking...";
  performance.mark(`rv-start-${rid}`);
  const [rep, view] = await Promise.allSettled([fetchJSON(`/api/robots/${rid}/report`), fetchJSON(`/api/robots/${rid}/view`)]);
  clearTimeout(slow);
  if (n !== assembling) return;
  performance.mark(`rv-fetched-${rid}`);
  $("#asm-load").classList.add("hidden");
  if (view.status === "fulfilled") showAssembly(rid, view.value);
  performance.mark(`rv-shown-${rid}`);
  if (rep.status !== "fulfilled") {
    failCard(rep.reason instanceof TypeError ? "Could not reach the server." : "The server could not assemble these parts.", rep.reason instanceof TypeError ? "It may be restarting. Pick the part again in a few seconds." : String(rep.reason));
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
function failCard(title, detail, partial = false) {
  const f = $("#asm-fail");
  f.innerHTML = "";
  f.classList.toggle("partial", partial);
  f.append(el("div", "lbl", partial ? "Some models are off on this robot" : "Cannot train on this robot"), el("b", "", title), el("div", "d", detail));
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
  const usable = rep.usable ?? (rep.ok && any);
  if (!rep.ok) failCard(rep.failure ? rep.failure.split(":")[0] : "A check failed", rep.failure ? rep.failure.split(": ").slice(1).join(": ") : "", usable);
  else if (!any) failCard("No model fits this robot", "Every model fails a reach or grip check; see the list below.");
  const ub = $("#use-robot");
  ub.disabled = !usable || rid === active;
  ub.textContent = rid === active ? "Models train on this robot" : "Train a model on this robot";
  const note = $("#use-note");
  const limits = usable && rep.usable_reason && rep.usable_reason !== "every skill fits" ? `It ${rep.usable_reason}. ` : "";
  note.textContent = limits + (usable && rep.act_dim && rep.act_dim !== 6
    ? `${rep.act_dim} action dimensions instead of 6: the policy keeps its vision and language trunk and retrains its action layers on this robot's demos.`
    : usable && rid !== "so101"
      ? "Same 6 action dimensions: the policy keeps every layer and fine-tunes on this robot's own demos."
      : "");
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
  b.textContent = "Models train on this robot";
}

function setActive(rid) {
  active = rid;
  window.dispatchEvent(new CustomEvent("robot-change", { detail: { id: rid, label: robotLabel(rid) } }));
}

let built = false;
const catReady = fetch("/api/robots").then((r) => r.json());
export async function initBuilder(activeId) {
  active = activeId || "so101";
  Object.assign(sel, parse(active));
  CAT = await catReady;
  setActive(active);
}

export async function showBuilder(on) {
  visible = on;
  if (!on || built) return;
  built = true;
  performance.mark("rv-tab");
  prefetch(robotId()); // the selected robot's preview and checks download while the cards draw
  const box = $("#slots");
  // never a blank left side: grey placeholder cards hold the grid until the catalog (a few ms) is in
  const sk = el("div", "parts");
  sk.append(...Array.from({ length: 8 }, () => { const c = el("div", "part part-sk"); c.append(el("div", "part-fig sk")); return c; }));
  box.replaceChildren(sk);
  CAT ||= await catReady;
  box.replaceChildren();
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
  await new Promise((r) => requestAnimationFrame(() => setTimeout(r)));
  initAssembly(); // the WebGL setup is the slow part on a weak GPU: after the cards are on screen
  assemble();
}
