// Models panel: the real-model library plus "My models" imports.
//
//   import { mountAssets } from "/static/assets.js";
//   const panel = mountAssets(element, { onUse(model) { ... } });   // panel.refresh(), panel.destroy()
//
// Needs /static/assets.css, the importmap from index.html (three + three/addons), and the /api/assets routes.
// Every model comes from a named open-source library (name, author, licence, link on each card) or is the
// user's own import. Nothing here executes uploaded content: files go to the server as raw bytes.
import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import { GLTFLoader } from "three/addons/loaders/GLTFLoader.js";
import { RoomEnvironment } from "three/addons/environments/RoomEnvironment.js";

const API = "/api/assets";
const MAX_MB = 50;
const ACCEPT = ".glb,.gltf,.obj,.stl,.zip";
const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[c]);
const cap = (s) => String(s ?? "").charAt(0).toUpperCase() + String(s ?? "").slice(1);
const cm = (m) => (m >= 1 ? `${m.toFixed(2)} m` : `${(m * 100).toFixed(m < 0.1 ? 1 : 0)} cm`);
const NICE = { conveyor: "Conveyors", pallet: "Pallets", container: "Totes and bins", shelving: "Shelving", workbench: "Benches and carts", handling: "Handling", machine: "Machines", part: "Parts", tool: "Tools", lighting: "Lighting", safety: "Safety", utility: "Utility", structure: "Structure", mine: "Imported", product: "Products" };
const SOURCE_SHORT = { "Poly Haven": "Poly Haven", "Gazebo Fuel": "Gazebo Fuel", user: "Your file" };

function licenceChip(m) {
  const s = m.source || {};
  const lic = s.license || "user import";
  const cls = m.origin === "user" ? "as-chip user" : "as-chip lic";
  return `<span class="${cls}" title="${esc(s.name)} / ${esc(s.author)}">${esc(lic)}</span>`;
}

export function mountAssets(root, opts = {}) {
  root.classList.add("as-root");
  root.innerHTML = `
    <div class="as-bar">
      <input class="as-q" type="search" placeholder="Search ${"models, parts, authors"}" aria-label="Search models">
      <div class="as-origin" role="group" aria-label="Origin">
        <button data-o="" class="on">All</button><button data-o="library">Library</button><button data-o="user">My models</button>
      </div>
      <button class="as-import" type="button"><svg viewBox="0 0 24 24" width="16" height="16"><path d="M12 16V4M7 9l5-5 5 5M5 20h14"/></svg>Import a 3D model</button>
    </div>
    <div class="as-cats" role="group" aria-label="Category"></div>
    <div class="as-count"></div>
    <div class="as-grid" aria-live="polite"></div>
    <div class="as-drop" hidden><div><b>Drop to import</b><span>GLB, glTF, OBJ, STL, or a zip (up to ${MAX_MB} MB)</span></div></div>
    <input class="as-file" type="file" accept="${ACCEPT}" hidden>
    <div class="as-modal" hidden></div>`;
  const $ = (s) => root.querySelector(s);
  const state = { q: "", cat: "", origin: "", items: [], cats: [] };
  let preview = null;
  let timer = 0;

  async function load() {
    const p = new URLSearchParams();
    if (state.q) p.set("q", state.q);
    if (state.cat && state.cat !== "mine") p.set("category", state.cat);
    if (state.origin) p.set("origin", state.origin);
    const [items, cats] = await Promise.all([fetch(`${API}?${p}`).then((r) => r.json()), fetch(`${API}/categories`).then((r) => r.json())]);
    state.items = items; state.cats = cats;
    renderCats(); renderGrid();
  }

  function renderCats() {
    const total = state.cats.reduce((a, c) => a + c.count, 0);
    $(".as-cats").innerHTML =
      `<button data-c="" class="${state.cat ? "" : "on"}">All <i>${total}</i></button>` +
      state.cats.map((c) => `<button data-c="${esc(c.category)}" class="${state.cat === c.category ? "on" : ""}">${esc(NICE[c.category] || c.category)} <i>${c.count}</i></button>`).join("");
  }

  function renderGrid() {
    const n = state.items.length;
    $(".as-count").textContent = `${n} model${n === 1 ? "" : "s"}`;
    if (!n) {
      $(".as-grid").innerHTML = `<div class="as-empty">${state.origin === "user" ? "Nothing imported yet. Drop a GLB, OBJ, STL or a zip anywhere on this page." : "No model matches."}</div>`;
      return;
    }
    $(".as-grid").innerHTML = state.items.map((m) => {
      const mine = m.origin === "user";
      return `<button class="as-card" data-id="${esc(m.id)}" type="button">
        <span class="as-thumb">${m.has_thumb ? `<img loading="lazy" src="${esc(m.thumb)}" alt="">` : `<span class="as-nothumb">no preview yet</span>`}
          <span class="as-kind">${esc(cap(m.kind))}</span>${mine ? `<span class="as-mine">My model</span>` : ""}</span>
        <span class="as-meta"><b>${esc(m.name)}</b>
          <span class="as-sub">${esc(NICE[m.category] || m.category)} &middot; ${m.size_m ? esc(m.size_m.map((v) => cm(v)).join(" x ")) : ""}</span>
          <span class="as-row">${licenceChip(m)}<span class="as-src">${esc(SOURCE_SHORT[m.source.name] || m.source.name)}</span></span></span>
      </button>`;
    }).join("");
  }

  // ---------------------------------------------------------------- detail with 3D preview
  function closeModal() {
    if (preview) { preview.dispose(); preview = null; }
    const m = $(".as-modal"); m.hidden = true; m.innerHTML = "";
  }

  function openModel(id) {
    const m = state.items.find((x) => x.id === id);
    if (!m) return;
    const s = m.source;
    const mine = m.origin === "user";
    const link = (u, t) => (u ? `<a href="${esc(u)}" target="_blank" rel="noopener noreferrer">${esc(t)}</a>` : esc(t));
    const modal = $(".as-modal");
    modal.hidden = false;
    modal.innerHTML = `<div class="as-sheet" role="dialog" aria-label="${esc(m.name)}">
      <button class="as-x" aria-label="Close">&times;</button>
      <div class="as-view"><canvas></canvas><span class="as-hint">drag to orbit, scroll to zoom</span></div>
      <div class="as-info">
        <h3>${esc(m.name)}</h3>
        <p class="as-desc">${esc(m.description)}</p>
        <dl>
          <dt>Source</dt><dd>${link(s.url, s.name)}</dd>
          <dt>Author</dt><dd>${esc(s.author || "you")}</dd>
          <dt>Licence</dt><dd>${link(s.license_url, s.license || "user import")}</dd>
          <dt>Size</dt><dd>${m.size_m ? esc(m.size_m.map(cm).join(" x ")) : "-"}</dd>
          <dt>Physics</dt><dd>${esc(cap(m.kind))}${m.mass_kg ? `, ${m.mass_kg.toFixed(2)} kg` : ""}, ${m.hulls ?? "?"} convex hulls</dd>
          <dt>Files</dt><dd><a href="${esc(m.glb)}" download="${esc(m.id)}.glb">GLB</a> &middot; <a href="${API}/${esc(m.id)}/mjcf" target="_blank">MJCF</a></dd>
        </dl>
        <div class="as-tags">${(m.tags || []).map((t) => `<span>${esc(t)}</span>`).join("")}</div>
        <div class="as-actions">
          ${opts.onUse ? `<button class="as-use" type="button">Use in scene</button>` : ""}
          ${mine ? `<button class="as-del" type="button">Remove</button>` : ""}
        </div>
      </div></div>`;
    preview = makePreview(modal.querySelector("canvas"), m.glb);
    modal.querySelector(".as-x").onclick = closeModal;
    modal.onclick = (e) => { if (e.target === modal) closeModal(); };
    const use = modal.querySelector(".as-use");
    if (use) use.onclick = () => { opts.onUse(m); closeModal(); };
    const del = modal.querySelector(".as-del");
    if (del) del.onclick = async () => { await fetch(`${API}/${encodeURIComponent(m.id)}`, { method: "DELETE" }); closeModal(); load(); };
  }

  // ---------------------------------------------------------------- import
  function chooseSource() {
    const modal = $(".as-modal");
    modal.hidden = false;
    modal.innerHTML = `<div class="as-sheet as-imp" role="dialog" aria-label="Import a 3D model">
      <button class="as-x" aria-label="Close">&times;</button>
      <h3>Import a 3D model</h3>
      <p class="as-desc">Bring in a GLB, glTF, OBJ, STL or a zip (up to ${MAX_MB} MB). It is scaled to metres, split into convex pieces and given physics defaults, then it joins the grid.</p>
      <div class="as-actions"><button class="as-pick" type="button">Choose a file</button></div>
      <label>Or paste a link<input class="as-url" type="url" placeholder="https://example.com/model.glb" maxlength="500"></label>
      <div class="as-status" aria-live="polite"></div>
      <div class="as-actions"><button class="as-fetch" type="button">Use this link</button></div></div>`;
    modal.querySelector(".as-x").onclick = closeModal;
    modal.querySelector(".as-pick").onclick = () => $(".as-file").click();
    modal.querySelector(".as-fetch").onclick = () => {
      const u = modal.querySelector(".as-url").value.trim();
      if (!/^https?:\/\/\S+$/i.test(u)) { const st = modal.querySelector(".as-status"); st.className = "as-status bad"; st.textContent = "Paste a web link that starts with http or https."; return; }
      importFile({ url: u, name: decodeURIComponent(u.split("?")[0].split("/").pop() || "model"), size: 0 });
    };
  }

  async function importFile(file) {
    const modal = $(".as-modal");
    modal.hidden = false;
    const ext = file.name.split(".").pop().toLowerCase();
    modal.innerHTML = `<div class="as-sheet as-imp" role="dialog" aria-label="Import">
      <button class="as-x" aria-label="Close">&times;</button>
      <h3>Import ${esc(file.name)}</h3>
      <p class="as-desc">${file.url ? "From a link." : `${(file.size / 1048576).toFixed(1)} MB.`} The file is checked, scaled to metres, split into convex pieces and given physics defaults.</p>
      <label>Name<input class="as-name" value="${esc(file.name.replace(/\.[^.]+$/, ""))}" maxlength="60"></label>
      <label>Units<select class="as-units"><option value="auto">Detect</option><option value="m">Metres</option><option value="cm">Centimetres</option><option value="mm">Millimetres</option><option value="in">Inches</option></select></label>
      <label>Physics<select class="as-kindsel"><option value="dynamic">Dynamic (can be picked up)</option><option value="static">Static (fixed in place)</option></select></label>
      <div class="as-status" aria-live="polite"></div>
      <div class="as-actions"><button class="as-go" type="button">Import</button></div></div>`;
    modal.querySelector(".as-x").onclick = closeModal;
    const status = modal.querySelector(".as-status");
    if (file.size > MAX_MB * 1048576) { status.className = "as-status bad"; status.textContent = `That file is over ${MAX_MB} MB.`; modal.querySelector(".as-go").disabled = true; return; }
    if (!ACCEPT.split(",").includes("." + ext)) { status.className = "as-status bad"; status.textContent = `.${ext} files are not supported. Use GLB, glTF, OBJ, STL or zip.`; modal.querySelector(".as-go").disabled = true; return; }
    modal.querySelector(".as-go").onclick = async () => {
      const go = modal.querySelector(".as-go"); go.disabled = true;
      const q = new URLSearchParams({ name: modal.querySelector(".as-name").value, units: modal.querySelector(".as-units").value, kind: modal.querySelector(".as-kindsel").value });
      status.className = "as-status"; status.textContent = file.url ? "Downloading..." : "Uploading...";
      try {
        const r = file.url
          ? await fetch(`${API}/import-url`, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ url: file.url, name: modal.querySelector(".as-name").value, units: modal.querySelector(".as-units").value, kind: modal.querySelector(".as-kindsel").value }) })
          : await fetch(`${API}/import?${q}`, { method: "POST", headers: { "X-Filename": encodeURIComponent(file.name) }, body: file });
        const j = await r.json();
        if (!r.ok) throw new Error(j.detail || r.statusText);
        status.textContent = "Converting: scaling, convex decomposition, physics...";
        for (;;) {
          await new Promise((res) => setTimeout(res, 900));
          const s = await (await fetch(`${API}/import/${j.id}`)).json();
          if (s.state === "error") throw new Error(s.message);
          if (s.state === "done") { status.className = "as-status ok"; status.textContent = "Done."; state.origin = "user"; syncOrigin(); await load(); closeModal(); openModel(s.model.id); return; }
        }
      } catch (e) {
        status.className = "as-status bad"; status.textContent = String(e.message || e); go.disabled = false;
      }
    };
  }

  function syncOrigin() { root.querySelectorAll(".as-origin button").forEach((b) => b.classList.toggle("on", b.dataset.o === state.origin)); }

  // ---------------------------------------------------------------- wiring
  $(".as-q").addEventListener("input", (e) => { clearTimeout(timer); timer = setTimeout(() => { state.q = e.target.value.trim(); load(); }, 160); });
  $(".as-cats").addEventListener("click", (e) => { const b = e.target.closest("button"); if (b) { state.cat = b.dataset.c; load(); } });
  $(".as-origin").addEventListener("click", (e) => { const b = e.target.closest("button"); if (b) { state.origin = b.dataset.o; syncOrigin(); load(); } });
  $(".as-grid").addEventListener("click", (e) => { const c = e.target.closest(".as-card"); if (c) openModel(c.dataset.id); });
  $(".as-import").onclick = chooseSource;
  $(".as-file").onchange = (e) => { const f = e.target.files[0]; e.target.value = ""; if (f) importFile(f); };
  const onKey = (e) => { if (e.key === "Escape") closeModal(); };
  document.addEventListener("keydown", onKey);
  let depth = 0;
  const dz = $(".as-drop");
  const hasFiles = (e) => [...(e.dataTransfer?.types || [])].includes("Files");
  const onEnter = (e) => { if (hasFiles(e)) { e.preventDefault(); depth++; dz.hidden = false; } };
  const onOver = (e) => { if (hasFiles(e)) e.preventDefault(); };
  const onLeave = (e) => { if (hasFiles(e) && --depth <= 0) { depth = 0; dz.hidden = true; } };
  const onDrop = (e) => { if (!hasFiles(e)) return; e.preventDefault(); depth = 0; dz.hidden = true; const f = e.dataTransfer.files[0]; if (f) importFile(f); };
  root.addEventListener("dragenter", onEnter); root.addEventListener("dragover", onOver);
  root.addEventListener("dragleave", onLeave); root.addEventListener("drop", onDrop);

  load();
  return { refresh: load, open: openModel, destroy() { closeModal(); document.removeEventListener("keydown", onKey); root.innerHTML = ""; } };
}

// A small orbit viewer: neutral studio light, soft shadow, the model sized to fill the view.
function makePreview(canvas, url) {
  const r = new THREE.WebGLRenderer({ canvas, antialias: true });
  r.setPixelRatio(Math.min(devicePixelRatio, 2));
  r.toneMapping = THREE.ACESFilmicToneMapping;
  r.shadowMap.enabled = true; r.shadowMap.type = THREE.PCFSoftShadowMap;
  const scene = new THREE.Scene();
  scene.background = new THREE.Color(0xf1f4f7);
  const pm = new THREE.PMREMGenerator(r);
  scene.environment = pm.fromScene(new RoomEnvironment(), 0.04).texture;
  const sun = new THREE.DirectionalLight(0xffffff, 1.5); sun.castShadow = true; sun.shadow.mapSize.set(1024, 1024); scene.add(sun);
  const floor = new THREE.Mesh(new THREE.CircleGeometry(30, 48), new THREE.ShadowMaterial({ opacity: 0.2 }));
  floor.rotation.x = -Math.PI / 2; floor.receiveShadow = true; scene.add(floor);
  const cam = new THREE.PerspectiveCamera(32, 1, 0.01, 200);
  const ctl = new OrbitControls(cam, canvas); ctl.enableDamping = true; ctl.autoRotate = true; ctl.autoRotateSpeed = 1.2;
  ctl.addEventListener("start", () => { ctl.autoRotate = false; });
  let alive = true;
  new GLTFLoader().load(url, (g) => {
    if (!alive) return;
    g.scene.traverse((o) => { if (o.isMesh) { o.castShadow = true; o.receiveShadow = true; } });
    scene.add(g.scene);
    const box = new THREE.Box3().setFromObject(g.scene);
    const c = box.getCenter(new THREE.Vector3()); const rad = Math.max(0.05, box.getSize(new THREE.Vector3()).length() / 2);
    const d = (rad / Math.sin((cam.fov * Math.PI) / 360)) * 1.05;
    cam.position.set(c.x + d * 0.62, c.y + d * 0.5, c.z + d * 0.62); ctl.target.copy(c);
    cam.near = d / 100; cam.far = d * 20; cam.updateProjectionMatrix();
    sun.position.set(c.x + rad * 2, c.y + rad * 3, c.z + rad * 1.5); sun.target.position.copy(c);
    const sh = rad * 1.6; Object.assign(sun.shadow.camera, { left: -sh, right: sh, top: sh, bottom: -sh, near: 0.01, far: rad * 8 }); sun.shadow.camera.updateProjectionMatrix();
    floor.position.y = box.min.y; ctl.maxDistance = d * 4; ctl.minDistance = d * 0.2;
  });
  const fit = () => { const w = canvas.clientWidth, h = canvas.clientHeight; if (w && h) { r.setSize(w, h, false); cam.aspect = w / h; cam.updateProjectionMatrix(); } };
  const ro = new ResizeObserver(fit); ro.observe(canvas); fit();
  r.setAnimationLoop(() => { ctl.update(); r.render(scene, cam); });
  return { dispose() { alive = false; ro.disconnect(); r.setAnimationLoop(null); ctl.dispose(); r.dispose(); scene.traverse((o) => { o.geometry?.dispose(); }); } };
}

// Standalone page (web/assets.html) mounts itself.
const auto = document.getElementById("assets-root");  // standalone page only
if (auto) mountAssets(auto);
