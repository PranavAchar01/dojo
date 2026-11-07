// Plays the factory line the engine produced (GET /api/factory = assets_hero.run): the same scene export, the same
// recorded frames and the same renderer stage as the Sim tab. Nothing here is placed by hand: every body, mesh and
// pose comes from the JSON, and the counts shown are the engine's own.
import * as THREE from "three";
import { toCreasedNormals } from "three/addons/utils/BufferGeometryUtils.js";
import { createStage, loadModel } from "/static/stage.js";
import { robotMaterial, partsMesh } from "/static/studio.js";

const $ = (s) => document.querySelector(s);
const data = await (await fetch("/api/factory")).json();
const meshes = await (await fetch("/api/meshes?robot=so101")).json();
const stage = createStage($("#c"), { target: [0.25, data.pitch / 2, 0.05], extent: 1.3, reach: 1.1 });
const { scene, camera } = stage;

const geoms = new Map();
function geometry(id) {
  if (geoms.has(id)) return geoms.get(id);
  const d = meshes[String(id)];
  const v = new Float32Array(Uint8Array.from(atob(d.v), (c) => c.charCodeAt(0)).buffer);
  const f = new Uint32Array(Uint8Array.from(atob(d.f), (c) => c.charCodeAt(0)).buffer);
  let g = new THREE.BufferGeometry();
  g.setAttribute("position", new THREE.BufferAttribute(v, 3));
  g.setIndex(new THREE.BufferAttribute(f, 1));
  g = toCreasedNormals(g, Math.PI / 7);
  geoms.set(id, g);
  return g;
}
const shade = (m) => { m.castShadow = m.receiveShadow = true; return m; };
const place = (o, pos, q) => { o.position.fromArray(pos); o.quaternion.set(q[1], q[2], q[3], q[0]); return o; };

// one cell: a group per scene body, the arm's meshes and the fixtures attached to them
function buildCell(sc, shiftY) {
  const root = new THREE.Group();
  root.position.y = shiftY;
  scene.add(root);
  const bodies = sc.bodies.map(() => { const o = new THREE.Group(); root.add(o); return o; });
  for (const g of sc.robot) bodies[g.body].add(place(shade(new THREE.Mesh(geometry(g.mesh), robotMaterial(g.rgba, g.name))), g.pos, g.quat));
  for (const p of sc.props) bodies[p.body].add(partsMesh(p.parts, p.rgb, p.kind));
  const product = sc.bodies.map((n, i) => (/^product/.test(n) ? i : -1)).filter((i) => i >= 0);
  return { sc, root, bodies, product };
}
function pose(cell, frame) {
  for (let i = 0; i < cell.bodies.length; i++) {
    const o = cell.bodies[i], k = i * 7;
    o.position.set(frame[k], frame[k + 1], frame[k + 2]);
    o.quaternion.set(frame[k + 4], frame[k + 5], frame[k + 6], frame[k + 3]);
  }
}
const cells = data.cells.map((c, i) => buildCell(c.scene, i * data.pitch));

// the dressing (table, guard, shelf, belt, library models) from the engine's compile of the cell
const dress = new THREE.Group();
scene.add(dress);
const mats = new Map();
function dressMat(d) {
  if (!mats.has(d.mat)) {
    const [r, g, b, a] = d.rgba;
    const glass = a < 0.99;
    mats.set(d.mat, new THREE.MeshPhysicalMaterial({
      color: new THREE.Color(r, g, b), roughness: glass ? 0.1 : 0.45, metalness: /alu/.test(d.mat) ? 0.7 : 0,
      transparent: glass, opacity: a, depthWrite: !glass, envMapIntensity: 0.8,
    }));
  }
  return mats.get(d.mat);
}
const modelOnce = new Set();
for (const d of data.dressing) {
  if (d.front) continue; // the guard run on the open side, which the engine's own hero view also looks through
  const parent = d.body < 0 ? dress : cells[0].bodies[d.body];
  if (d.kind === "mesh") {
    const key = `${d.body}|${d.id}`;
    if (modelOnce.has(key)) continue;
    modelOnce.add(key);
    const m = await loadModel(`/api/assets/${d.id}/glb`, d.scale);
    const holder = new THREE.Group();
    holder.add(m);
    parent.add(holder);
    continue;
  }
  const geo = d.kind === "box"
    ? new THREE.BoxGeometry(d.size[0] * 2, d.size[1] * 2, d.size[2] * 2)
    : new THREE.CylinderGeometry(d.size[0], d.size[0], d.size[1] * 2, 24).rotateX(Math.PI / 2);
  parent.add(place(shade(new THREE.Mesh(geo, dressMat(d))), d.pos, d.quat));
}

// timeline: for each part, each station's arrive frames then its recorded episode
const fps = 30;
const timeline = [];
for (const part of data.parts) for (const st of part.stations) {
  const cell = cells[st.station - 1];
  for (const f of [...st.arrive, ...st.frames]) timeline.push({ cell, frame: f, part: part.part, task: st.task });
}
for (const c of cells) pose(c, data.parts[0].stations[c === cells[0] ? 0 : 1].arrive[0]);

const c0 = new THREE.Vector3(0.25, data.pitch / 2, 0.05);
// the engine's wide view: from the open side of the cell, looking down about 30 degrees
camera.position.set(c0.x + 1.25, c0.y - 0.15, c0.z + 0.8);
camera.lookAt(c0);
const resize = () => stage.resize(innerWidth, innerHeight);
addEventListener("resize", resize);
resize();

let t0 = performance.now();
$("#title").textContent = `${data.n_parts} parts, ${data.cell.replace(/_/g, " ")}`;
$("#replay").onclick = () => { t0 = performance.now(); };
function tick(now) {
  const i = Math.min(timeline.length - 1, Math.floor(((now - t0) / 1000) * fps));
  const s = timeline[i];
  pose(s.cell, s.frame);
  for (const c of cells) for (const b of c.product) c.bodies[b].visible = c === s.cell;
  $("#fill").style.width = `${(100 * i) / (timeline.length - 1)}%`;
  $("#status").textContent = `part ${s.part + 1} of ${data.n_parts}, ${s.task}`;
  stage.render();
  requestAnimationFrame(tick);
}
requestAnimationFrame(tick);
window.factoryReady = true;
