// Build plan: the bridge as stud-grid steps, checked by rules, with a live three.js render of the real LDraw bricks.
// Left: the instruction image (rendered from the rules, or one you upload) that Muse Spark can read into steps.
// Right: the live render. Below: the rules you can change, each rule's live pass/fail, the editable step list and
// "fix in words". Every row is labelled by who made it: the rules, Muse Spark's read, or your edit.
import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import { toCreasedNormals } from "three/addons/utils/BufferGeometryUtils.js";

const $ = (s, r = document) => r.querySelector(s);
const esc = (s) => String(s ?? "").replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
const ROOT = () => $("#view-plan");
const STUD = 0.008, PLATE = 0.0032;
const SRC = { rules: "From the rules", muse: "Muse Spark's read", user: "Your edit" };

let S = null; // server state: parts, colours, muse status
let rules = { gap: 4, pillar: 2, deck: 6, tier: true };
let bricks = [];
let upto = 0; // how many steps the render shows
let pending = null; // a proposal (Muse read or fix) waiting for Apply
let imageData = null; // an uploaded image as a data URL
let three = null;
const meshCache = new Map();
let checkTimer = 0, checkSeq = 0;

const api = async (path, body) => {
  const r = await fetch("/api/build" + path, body ? { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) } : undefined);
  const j = await r.json().catch(() => ({}));
  if (!r.ok) throw new Error(typeof j.detail === "string" ? j.detail : "request failed");
  return j;
};

function dom() {
  ROOT().innerHTML = `
  <div class="bp doc">
    <div class="doc-h bp-head"><h2>Build <em>plan</em></h2>
      <p>A bridge made from rules: two pillars, a deck across them and an optional brick on top. Each rule below is checked by code, live. Change a number and the steps are rebuilt from the rules; edit a step by hand and the rules tell you what broke.</p></div>
    <div class="bp-grid">
      <section class="bp-card bp-img">
        <div class="bp-lbl">Instruction image <span id="bp-img-src" class="bp-dim"></span></div>
        <div class="bp-imgwrap"><img id="bp-img" alt="Instruction image of the bridge"></div>
        <div class="bp-row">
          <label class="bp-btn">Use my photo<input id="bp-file" type="file" accept="image/*" hidden></label>
          <button class="bp-btn" id="bp-reset-img" type="button">Rendered image</button>
          <button class="bp-btn bp-primary" id="bp-read" type="button">Read it with Muse Spark</button>
        </div>
        <div id="bp-muse-note" class="bp-dim"></div>
      </section>
      <section class="bp-card bp-view">
        <div class="bp-lbl">Live render <span class="bp-dim">real LDraw bricks, drag to turn</span></div>
        <div id="bp-canvas"></div>
        <div class="bp-row bp-steprow"><span class="bp-dim">Step</span><input id="bp-upto" type="range" min="0" max="1" step="1"><b id="bp-upto-n"></b></div>
      </section>
    </div>
    <div class="bp-grid bp-grid2">
      <section class="bp-card">
        <div class="bp-lbl">Rules <span class="bp-dim">change these and the steps are rebuilt</span></div>
        <div class="bp-rules">
          <label>Gap <small>studs</small><input id="r-gap" type="number" min="1" max="6"></label>
          <label>Pillar height <small>bricks</small><input id="r-pillar" type="number" min="1" max="4"></label>
          <label>Deck length <small>studs</small><select id="r-deck"><option>4</option><option>6</option><option>8</option></select></label>
          <label class="bp-chk"><input id="r-tier" type="checkbox"> Brick on the deck</label>
        </div>
        <div id="bp-rule-err" class="bp-err"></div>
        <div class="bp-lbl bp-mt">Checks <span id="bp-score" class="bp-dim"></span></div>
        <ul id="bp-checks" class="bp-checks"></ul>
      </section>
      <section class="bp-card">
        <div class="bp-lbl">Fix in words <span class="bp-dim">Muse Spark revises the steps, the checker validates them</span></div>
        <div class="bp-row"><input id="bp-fix" class="bp-text" type="text" maxlength="400" placeholder="e.g. make the pillars one brick taller and move the top brick to the left end of the deck"><button class="bp-btn bp-primary" id="bp-fix-go" type="button">Revise</button></div>
        <div id="bp-proposal"></div>
      </section>
    </div>
    <section class="bp-card">
      <div class="bp-lbl">Steps <span class="bp-dim">bottom up; nudge by one stud or one plate level</span>
        <span class="bp-grow"></span><button class="bp-btn" id="bp-add" type="button">Add brick</button></div>
      <div class="bp-tablewrap"><table class="bp-steps"><thead><tr><th>#</th><th>Part</th><th>Colour</th><th>Stud col</th><th>Stud row</th><th>Plate level</th><th>Turn</th><th>Who</th><th></th></tr></thead><tbody id="bp-body"></tbody></table></div>
      <div class="bp-row bp-use"><button class="bp-btn bp-primary" id="bp-use" type="button">Use this build</button><span id="bp-use-note" class="bp-dim"></span></div>
      <img id="bp-table" class="bp-table" alt="" hidden>
    </section>
    <p class="bp-foot bp-dim" id="bp-foot"></p>
  </div>`;
}

// ------------------------------------------------------------------ three.js
async function geometry(part) {
  if (!meshCache.has(part)) {
    meshCache.set(part, api(`/mesh/${part}`).then((m) => {
      const g = new THREE.BufferGeometry();
      g.setAttribute("position", new THREE.BufferAttribute(new Float32Array(m.v), 3));
      g.setIndex(m.f);
      return toCreasedNormals(g, 0.6);
    }));
  }
  return meshCache.get(part);
}

function initThree() {
  const host = $("#bp-canvas");
  const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
  renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
  host.appendChild(renderer.domElement);
  const scene = new THREE.Scene();
  const cam = new THREE.PerspectiveCamera(32, 1, 0.5, 200);
  cam.position.set(-10, 9, 13);
  scene.add(new THREE.HemisphereLight(0xffffff, 0xb8c4d0, 1.1));
  const sun = new THREE.DirectionalLight(0xffffff, 1.6);
  sun.position.set(-6, 14, 8);
  scene.add(sun);
  const floor = new THREE.GridHelper(24, 30, 0xc9d3de, 0xe1e7ee);
  scene.add(floor);
  const world = new THREE.Group(); // metres, Z up -> scene units of 100 per metre, Y up
  world.rotation.x = -Math.PI / 2;
  world.scale.setScalar(100);
  scene.add(world);
  const controls = new OrbitControls(cam, renderer.domElement);
  controls.enableDamping = true;
  controls.target.set(2.5, 1.2, -0.6);
  const resize = () => {
    const w = host.clientWidth || 600, h = Math.max(320, Math.round(w * 0.62));
    renderer.setSize(w, h);
    cam.aspect = w / h;
    cam.updateProjectionMatrix();
  };
  new ResizeObserver(resize).observe(host);
  resize();
  let raf = 0;
  const loop = () => {
    raf = requestAnimationFrame(loop);
    if (ROOT().classList.contains("hidden")) return;
    controls.update();
    renderer.render(scene, cam);
  };
  loop();
  three = { world, cam, controls, scene, renderer, raf };
}

async function draw() {
  if (!three) return;
  const { world } = three;
  world.clear();
  const list = bricks.slice(0, upto);
  const geoms = await Promise.all(list.map((b) => geometry(b.part)));
  list.forEach((b, i) => {
    const [nx, ny] = S.parts[b.part]?.footprint ?? [1, 1];
    const [fx, fy] = b.yaw % 180 ? [ny, nx] : [nx, ny];
    const c = S.colours[b.colour] ?? [0.7, 0.7, 0.7, 1];
    const mat = new THREE.MeshStandardMaterial({ color: new THREE.Color(c[0], c[1], c[2]), roughness: 0.42, metalness: 0 });
    if (i === upto - 1) { mat.emissive = new THREE.Color(0x0866ff); mat.emissiveIntensity = 0.28; }
    const m = new THREE.Mesh(geoms[i], mat);
    m.position.set((b.col + fx / 2) * STUD, (b.row + fy / 2) * STUD, b.lvl * PLATE);
    m.rotation.z = (b.yaw * Math.PI) / 180;
    world.add(m);
  });
}

// ------------------------------------------------------------------ steps table
const sel = (opts, v, k, i) => `<select data-k="${k}" data-i="${i}">${opts.map((o) => `<option value="${o}"${o == v ? " selected" : ""}>${esc(S.parts[o]?.name ?? o)}</option>`).join("")}</select>`;
const num = (v, k, i, lo, hi) => `<span class="bp-nudge"><button data-nudge="${k}" data-d="-1" data-i="${i}" type="button" aria-label="${k} down">-</button><input type="number" data-k="${k}" data-i="${i}" value="${v}" min="${lo}" max="${hi}"><button data-nudge="${k}" data-d="1" data-i="${i}" type="button" aria-label="${k} up">+</button></span>`;

function renderSteps() {
  $("#bp-body").innerHTML = bricks.map((b, i) => `<tr class="${i === upto - 1 ? "cur" : ""}" data-row="${i}">
    <td>${i + 1}</td><td>${sel(Object.keys(S.parts), b.part, "part", i)}</td>
    <td><select data-k="colour" data-i="${i}">${Object.keys(S.colours).map((c) => `<option${c === b.colour ? " selected" : ""}>${c}</option>`).join("")}</select></td>
    <td>${num(b.col, "col", i, -4, 24)}</td><td>${num(b.row, "row", i, -4, 24)}</td><td>${num(b.lvl, "lvl", i, 0, 40)}</td>
    <td><select data-k="yaw" data-i="${i}"><option value="0"${b.yaw ? "" : " selected"}>along</option><option value="90"${b.yaw ? " selected" : ""}>across</option></select></td>
    <td><span class="bp-who bp-src-${b.source}">${SRC[b.source] ?? SRC.user}</span></td>
    <td><button class="bp-x" data-del="${i}" type="button" aria-label="Remove step ${i + 1}">x</button></td></tr>`).join("");
  const sl = $("#bp-upto");
  sl.max = bricks.length;
  sl.value = upto;
  $("#bp-upto-n").textContent = `${upto} of ${bricks.length}`;
}

function edited() {
  upto = Math.min(Math.max(upto, 1), bricks.length) || bricks.length;
  renderSteps();
  draw();
  clearTimeout(checkTimer);
  checkTimer = setTimeout(runChecks, 250);
}

async function runChecks() {
  const seq = ++checkSeq;
  const ul = $("#bp-checks");
  try {
    const r = await api("/check", { rules, bricks });
    if (seq !== checkSeq) return;
    if (!r.results.length) {
      ul.innerHTML = `<li class="bad"><span class="bp-ic">x</span><div><b>Cannot check yet</b><div class="bp-dim">${esc(r.problems.join("; "))}</div></div></li>`;
      $("#bp-score").textContent = "";
      return;
    }
    ul.innerHTML = r.results.map((x) => `<li class="${x.ok ? "ok" : "bad"}"><span class="bp-ic">${x.ok ? "&#10003;" : "&#10007;"}</span><div><b>${esc(x.label)}</b><div class="bp-dim">${esc(x.detail)}</div></div></li>`).join("");
    const n = r.results.filter((x) => x.ok).length;
    $("#bp-score").textContent = `${n} of ${r.results.length} pass`;
  } catch (e) {
    ul.innerHTML = `<li class="bad"><span class="bp-ic">x</span><div class="bp-dim">${esc(e.message)}</div></li>`;
  }
}

function syncRules() {
  $("#r-gap").value = rules.gap;
  $("#r-pillar").value = rules.pillar;
  $("#r-deck").value = String(rules.deck);
  $("#r-tier").checked = rules.tier;
  $("#bp-img").src = imageData || `/api/build/photo.png?gap=${rules.gap}&pillar=${rules.pillar}&deck=${rules.deck}&tier=${rules.tier}`;
  $("#bp-img-src").textContent = imageData ? "your upload" : "rendered from the rules";
}

async function regenerate() {
  const err = $("#bp-rule-err");
  err.textContent = "";
  try {
    const r = await api("/generate", rules);
    bricks = r.bricks;
    upto = bricks.length;
    pending = null;
    renderProposal();
    syncRules();
    edited();
  } catch (e) {
    err.textContent = e.message; // the rules cannot be met together, say which
  }
}

// ------------------------------------------------------------------ proposals (Muse read / fix in words)
function renderProposal() {
  const host = $("#bp-proposal");
  if (!pending) { host.innerHTML = ""; return; }
  const p = pending;
  host.innerHTML = `<div class="bp-prop">
    <div class="bp-lbl">${esc(p.title)} <span class="bp-dim">${esc(p.meta)}</span></div>
    ${p.summary ? `<p class="bp-sum">${esc(p.summary)}</p>` : ""}
    ${p.score ? `<p class="bp-scoreline">${esc(p.score)}</p>` : ""}
    ${p.diff.length ? `<ul class="bp-diff">${p.diff.map((d) => `<li>${esc(d)}</li>`).join("")}</ul>` : `<p class="bp-dim">No difference from the current steps.</p>`}
    ${p.problems?.length ? `<p class="bp-err">Not applied: ${esc(p.problems.join("; "))}</p>` : `<div class="bp-row"><button class="bp-btn bp-primary" id="bp-apply" type="button">Apply</button><button class="bp-btn" id="bp-discard" type="button">Discard</button></div>`}
  </div>`;
  $("#bp-apply")?.addEventListener("click", () => { bricks = p.bricks; upto = bricks.length; pending = null; renderProposal(); edited(); });
  $("#bp-discard")?.addEventListener("click", () => { pending = null; renderProposal(); });
}

const keys = ["part", "colour", "col", "row", "lvl", "yaw"];
function diffRows(a, b) {
  const out = [];
  for (let i = 0; i < Math.max(a.length, b.length); i++) {
    if (!a[i]) out.push(`step ${i + 1} added: ${b[i].part} ${b[i].colour} at col ${b[i].col}, row ${b[i].row}, level ${b[i].lvl}`);
    else if (!b[i]) out.push(`step ${i + 1} removed: ${a[i].part} ${a[i].colour}`);
    else { const ch = keys.filter((k) => a[i][k] !== b[i][k]).map((k) => `${k} ${a[i][k]} to ${b[i][k]}`); if (ch.length) out.push(`step ${i + 1}: ${ch.join(", ")}`); }
  }
  return out;
}

async function doRead() {
  const btn = $("#bp-read");
  btn.disabled = true; btn.textContent = "Muse Spark is reading...";
  try {
    const r = await api("/read", { rules, image: imageData });
    const sc = r.score ? `Muse Spark's read matches ${r.score.exact_matches} of ${r.score.bricks_expected} bricks exactly and passes ${r.score.rules_passed} of ${r.score.rules_total} checks.` : "";
    pending = { title: "Muse Spark's read of the image", meta: `${r.model}, ${r.usage.prompt_tokens ?? 0} in / ${r.usage.completion_tokens ?? 0} out tokens`, summary: r.summary, score: sc, bricks: r.bricks, problems: r.problems, diff: diffRows(bricks, r.bricks) };
    renderProposal();
  } catch (e) {
    pending = { title: "Muse Spark could not read the image", meta: "", diff: [], problems: [e.message], bricks: [] };
    renderProposal();
  } finally {
    btn.disabled = false; btn.textContent = "Read it with Muse Spark";
  }
}

async function doFix() {
  const text = $("#bp-fix").value.trim();
  if (text.length < 3) return;
  const btn = $("#bp-fix-go");
  btn.disabled = true; btn.textContent = "Revising...";
  try {
    const r = await api("/fix", { rules, bricks, instruction: text });
    pending = r.ok
      ? { title: `Muse Spark's revision: "${text}"`, meta: `${r.usage.prompt_tokens} in / ${r.usage.completion_tokens} out tokens, validated`, summary: r.summary, diff: r.diff, bricks: r.bricks }
      : { title: `Muse Spark's revision: "${text}"`, meta: "", diff: [], problems: r.problems, bricks: [] };
  } catch (e) {
    pending = { title: "Muse Spark could not revise the steps", meta: "", diff: [], problems: [e.message], bricks: [] };
  } finally {
    btn.disabled = false; btn.textContent = "Revise";
    renderProposal();
  }
}

async function useBuild() {
  const note = $("#bp-use-note");
  note.textContent = "Sending to the table...";
  try {
    const r = await api("/use", { rules, bricks });
    note.textContent = `Sent: ${r.bricks} bricks are now the MuJoCo task. Rendering the table...`;
    const img = $("#bp-table");
    img.onload = () => { img.hidden = false; note.textContent = `Sent: ${r.bricks} bricks are the MuJoCo task (two SO-101 arms, loose bricks on the table).`; };
    img.onerror = () => { note.textContent = `Sent: ${r.bricks} bricks are the MuJoCo task. The table preview failed to render.`; };
    img.src = r.table + "?t=" + Date.now();
  } catch (e) {
    note.textContent = "Not sent: " + e.message;
  }
}

// ------------------------------------------------------------------ wiring
function wire() {
  const rd = () => { rules = { gap: +$("#r-gap").value, pillar: +$("#r-pillar").value, deck: +$("#r-deck").value, tier: $("#r-tier").checked }; regenerate(); };
  ["#r-gap", "#r-pillar", "#r-deck", "#r-tier"].forEach((s) => $(s).addEventListener("change", rd));
  $("#bp-upto").addEventListener("input", (e) => { upto = +e.target.value; $("#bp-upto-n").textContent = `${upto} of ${bricks.length}`; renderSteps(); draw(); });
  $("#bp-body").addEventListener("change", (e) => {
    const t = e.target, i = +t.dataset.i, k = t.dataset.k;
    if (k == null) return;
    bricks[i] = { ...bricks[i], [k]: ["col", "row", "lvl", "yaw"].includes(k) ? +t.value : t.value, source: "user" };
    edited();
  });
  $("#bp-body").addEventListener("click", (e) => {
    const t = e.target.closest("button");
    if (!t) return;
    if (t.dataset.nudge) { const i = +t.dataset.i, k = t.dataset.nudge; bricks[i] = { ...bricks[i], [k]: bricks[i][k] + +t.dataset.d, source: "user" }; edited(); }
    if (t.dataset.del != null) { bricks.splice(+t.dataset.del, 1); upto = bricks.length; edited(); }
  });
  $("#bp-add").addEventListener("click", () => {
    bricks.push({ name: `step${bricks.length + 1}`, part: "3003", colour: "blue", col: 0, row: 4, lvl: 0, yaw: 0, role: "added by you", source: "user" });
    upto = bricks.length; edited();
  });
  $("#bp-file").addEventListener("change", (e) => {
    const f = e.target.files[0];
    if (!f) return;
    const fr = new FileReader();
    fr.onload = () => { imageData = String(fr.result); syncRules(); };
    fr.readAsDataURL(f);
  });
  $("#bp-reset-img").addEventListener("click", () => { imageData = null; syncRules(); });
  $("#bp-read").addEventListener("click", doRead);
  $("#bp-fix-go").addEventListener("click", doFix);
  $("#bp-fix").addEventListener("keydown", (e) => { if (e.key === "Enter") doFix(); });
  $("#bp-use").addEventListener("click", useBuild);
}

let booted = false;
async function boot() {
  if (booted) return;
  booted = true;
  dom();
  S = await api("/state");
  rules = S.rules;
  bricks = S.bricks;
  upto = bricks.length;
  const m = S.muse;
  $("#bp-muse-note").textContent = m.live ? "Muse Spark is on: reads and revisions are real model calls." : `Muse Spark is off here: ${m.why}. Everything else works.`;
  $("#bp-read").disabled = $("#bp-fix-go").disabled = !m.live;
  $("#bp-foot").textContent = `Parts: LDraw Parts Library (${S.licence}). ${S.attribution} Settle test: ${S.settle.seconds} s of gravity, no brick may move more than ${S.settle.max_mm} mm.`;
  wire();
  initThree();
  syncRules();
  edited();
}

window.addEventListener("forge:view", (e) => { if (e.detail === "plan") boot(); });
if ($("#app")?.dataset.view === "plan") boot();
