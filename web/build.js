// The loading screen as a build: while a run is under way the Dojo bot puts together THIS run's arm and task in the
// empty space, and every change follows a real event from the stream (re-broadcast as "forge:event"):
//   task         the task pieces in the run's real colours go onto the mat; the arm shows as a dashed outline
//   scene (log)  the base and first link go in; the rest wait for the scene event, so the arm is whole when the
//                scene is really built
//   demos        the counter shows the real done/ok/n, and the arm traces the task once per update, sped up
//   loss         the arm's fill follows the loss (lower = more solid); the bot spots it like a sparring partner
//   eval_progress the arm attempts the task; new passes flash green, new misses red, with the real counts
//   published    the arm is complete and the bot bows beside it
// A missing field holds the last frame. Drawn in SVG; nothing animates while no event is being played.
import { tick, clamp } from "./spring.js";

const NS = "http://www.w3.org/2000/svg";
const VW = 320, VH = 180, MAT = 150;
const BLOCK = { red: "#db241c", green: "#299e45", blue: "#2152db", yellow: "#f2c71a", orange: "#f57514", purple: "#7a3db8" };
const BOWL = { white: "#edebe0", blue: "#3366cc", green: "#409461", pink: "#e68c9e", black: "#1f1f21" };
const PAD = { blue: "#198cf2", green: "#26b359", orange: "#fa8c1a", white: "#f2f2f2" };
// link lengths in drawing units, by arm family (the outline only needs the proportions)
function dims(robot = "") {
  if (/wx250s|vx300s/.test(robot)) return { l1: 58, l2: 54, base: 14, grip: 13 };
  if (/koch/.test(robot)) return { l1: 44, l2: 40, base: 12, grip: 11 };
  return { l1: 50, l2: 46, base: 12, grip: 12 };
}
const LABEL = { so101: "SO-101", so100: "SO-100", koch: "Koch", wx250s: "WidowX 250 s", vx300s: "ViperX 300 s" };
const label = (r) => LABEL[Object.keys(LABEL).find((k) => r.startsWith(k)) || ""] || r || "";
const PARTS = ["base", "l1", "l2", "grip"];
const ease = (t) => (t < 0.5 ? 2 * t * t : 1 - (-2 * t + 2) ** 2 / 2);

let robot = "so101";
addEventListener("robot-change", (e) => { if (e.detail?.id) robot = e.detail.id; });
fetch("/api/robots").then((r) => r.json()).then((j) => { if (j.active) robot = j.active; }).catch(() => {});

// ---------------------------------------------------------------- the run being built
const S = { run: null, spec: null, robot, built: {}, placed: {}, demos: null, L0: 0, ema: null, ev: null, saved: null, endAt: 0, sceneLive: false };
const queue = [];
let wake = null;
const push = (f) => { queue.push(f); if (queue.length > 6) queue.splice(0, queue.length - 6); if (wake) { wake(); wake = null; } };

function reset(run, spec) {
  Object.assign(S, { run, spec: spec || null, robot, built: {}, placed: {}, demos: null, L0: 0, ema: null, ev: null, saved: null, endAt: 0, sceneLive: false });
  queue.length = 0;
  site.redraw();
  push(placePieces);
}
addEventListener("forge:event", (ev) => {
  const e = ev.detail || {};
  if (e.kind === "task" && !e.steps && e.spec) { reset(e.run, e.spec); return; }
  if (e.kind === "child" && e.child) { const st = e.plan?.[e.step] || {}; reset(e.child, st.spec || (st.obj ? st : null)); return; }
  if (!S.run || e.run !== S.run) return;
  switch (e.kind) {
    case "log":
      if (e.agent === "scene" && !S.sceneLive && !S.built.l1) { S.sceneLive = true; push((a) => placeParts(a, ["base", "l1"])); }
      break;
    case "scene": push((a) => placeParts(a, PARTS)); break;
    case "demos": {
      if (e.n == null) break;
      const was = S.demos ? S.demos.done : 0;
      S.demos = { done: e.done, ok: e.ok, n: e.n };
      site.redraw();
      push((a) => trace(a, Math.min(2, Math.max(1, e.done - was)), null));
      break;
    }
    case "loss":
      if (typeof e.ema !== "number") break;
      if (!S.L0 || e.round > (S.round ?? 0)) { S.L0 = S.L0 || e.ema; S.round = e.round; }
      S.ema = e.ema;
      site.redraw();
      if (!queue.length && performance.now() - (S.spotAt || 0) > 1500) { S.spotAt = performance.now(); push(spot); }
      break;
    case "eval_progress": {
      if (e.n == null) break;
      const was = S.ev || { done: 0, succeeded: 0 };
      S.ev = { done: e.done, succeeded: e.succeeded, n: e.n };
      const ok = e.succeeded - was.succeeded, bad = (e.done - e.succeeded) - (was.done - was.succeeded);
      if (ok > 0 || bad > 0) push((a) => trace(a, 1, ok > 0 ? "ok" : "bad", e));
      else site.redraw();
      break;
    }
    case "eval": if (e.n != null) { S.ev = { done: e.n, succeeded: e.succeeded, n: e.n }; site.redraw(); } break;
    case "published":
      S.saved = { rate: e.success_rate, n: S.ev?.n || 20 };
      push(finish);
      break;
    case "error": case "rejected": S.endAt = performance.now(); break;
  }
});
// the build is shown while its run is going and for a few seconds after it is saved
export const buildState = S;
export const buildCaption = () => (S.saved ? "Skill saved" : "");
export const buildActive = () => !!S.run && !!S.spec && (!S.endAt || performance.now() - S.endAt < 5000);

// ---------------------------------------------------------------- geometry (drawing units)
function rig() {
  const d = dims(S.robot), bx = 70, by = MAT;
  return { ...d, bx, by, sx: bx, sy: by - d.base - 3, obj: { x: 172, y: MAT - 7 }, tgt: { x: 248, y: MAT - 3 } };
}
const REST = (g) => ({ x: g.sx + 34, y: g.sy - 48 });
// two-link IK, elbow up, gripper hanging straight down to the tip
function pose(g, tipP) {
  const W = { x: tipP.x, y: tipP.y - g.grip };
  let dx = W.x - g.sx, dy = W.y - g.sy, d = Math.hypot(dx, dy);
  const dm = clamp(d, Math.abs(g.l1 - g.l2) + 2, g.l1 + g.l2 - 0.5);
  if (dm !== d) { dx *= dm / d; dy *= dm / d; d = dm; }
  const a = Math.atan2(dy, dx), b = Math.acos(clamp((g.l1 * g.l1 + d * d - g.l2 * g.l2) / (2 * g.l1 * d), -1, 1));
  const E = { x: g.sx + g.l1 * Math.cos(a - b), y: g.sy + g.l1 * Math.sin(a - b) };
  return { E, W: { x: g.sx + dx, y: g.sy + dy } };
}

// ---------------------------------------------------------------- the drawing
const site = (() => {
  let root = null, svg = null, cap = null, el = {}, map = { s: 1, ox: 0, oy: 0 }, compact = false;
  let tipP = null, objP = null, flashAt = 0;
  const mk = (tag, cls, attrs = {}) => { const n = document.createElementNS(NS, tag); if (cls) n.setAttribute("class", cls); for (const k in attrs) n.setAttribute(k, attrs[k]); return n; };
  function mount(c) {
    compact = !!c;
    root = document.createElement("div");
    root.className = "bd" + (compact ? " compact" : "");
    root.setAttribute("aria-live", "polite");
    svg = mk("svg", "", { viewBox: `0 0 ${VW} ${VH}`, "aria-hidden": "true" });
    svg.append(mk("rect", "bd-mat", { x: 14, y: MAT, width: VW - 28, height: 9, rx: 4.5 }));
    el.tgt = mk("rect", "bd-piece hidden"); el.obj = mk("rect", "bd-piece hidden", { width: 14, height: 14, rx: 2.5 });
    svg.append(el.tgt);
    el.parts = {};
    for (const p of PARTS) {
      const g = mk("g", `bd-part ${p} ghost`);
      for (const c2 of ["o", "i", "s"]) g.append(mk("path", c2));
      el.parts[p] = g; svg.append(g);
    }
    el.joints = [mk("circle", "bd-joint", { r: 3 }), mk("circle", "bd-joint", { r: 2.6 })];
    svg.append(...el.joints, el.obj);
    el.flash = mk("circle", "bd-flash", { r: 16 });
    el.read = mk("text", "bd-read", { x: 14, y: 22 });
    el.count = mk("text", "bd-count", { x: VW - 14, y: MAT + 24, "text-anchor": "end" });
    svg.append(el.flash, el.read, el.count);
    cap = document.createElement("span");
    cap.className = "bd-cap";
    cap.innerHTML = '<span class="ag-spin"></span><span class="t"></span>';
    root.append(svg, cap);
    document.body.append(root);
    tipP = null; objP = null;
    redraw();
  }
  function unmount() { const r = root; if (!r) return; root = null; r.classList.add("out"); setTimeout(() => r.remove(), 320); }
  function layout(r) {
    if (!root) return;
    const pad = compact ? 8 : 0, top = compact ? 26 : 40;
    const s = Math.min((r.width - 2 * pad) / VW, (r.height - top - pad) / VH, 1.7);
    const w = VW * s, h = VH * s;
    const ox = r.left + (r.width - w) / 2, oy = r.bottom - pad - h;
    if (Math.abs(ox - map.ox) + Math.abs(oy - map.oy) + Math.abs(s - map.s) < 0.2 && root.style.width) return;
    map = { s, ox, oy };
    if (compact) Object.assign(root.style, { left: `${r.left}px`, top: `${r.top}px`, width: `${r.width}px`, height: `${r.height}px` });
    else Object.assign(root.style, { left: "0px", top: "0px", width: "0px", height: "0px" });
    const bx = compact ? r.left : 0, byy = compact ? r.top : 0;
    Object.assign(svg.style, { left: `${ox - bx}px`, top: `${oy - byy}px`, width: `${w}px`, height: `${h}px` });
    cap.style.left = `${r.left + r.width / 2 - bx}px`;
    cap.style.top = `${(compact ? r.top + 4 : oy - 34) - byy}px`;
    cap.style.maxWidth = `${Math.max(120, r.width - 24)}px`;
  }
  const page = (p) => ({ x: map.ox + p.x * map.s, y: map.oy + p.y * map.s });
  const local = (p) => ({ x: (p.x - map.ox) / map.s, y: (p.y - map.oy) / map.s });
  function redraw() {
    if (!root) return;
    const g = rig(), sp = S.spec || {};
    const t = tipP || REST(g), { E, W } = pose(g, t);
    const path = {
      base: `M${g.bx - 14} ${g.by - 1}H${g.bx + 14}M${g.bx} ${g.by - 2}V${g.sy}`,
      l1: `M${g.sx} ${g.sy}L${E.x.toFixed(1)} ${E.y.toFixed(1)}`,
      l2: `M${E.x.toFixed(1)} ${E.y.toFixed(1)}L${W.x.toFixed(1)} ${W.y.toFixed(1)}`,
      grip: `M${(W.x - 5).toFixed(1)} ${W.y.toFixed(1)}H${(W.x + 5).toFixed(1)}M${(W.x - 4).toFixed(1)} ${W.y.toFixed(1)}v${g.grip}M${(W.x + 4).toFixed(1)} ${W.y.toFixed(1)}v${g.grip}`,
    };
    // lower loss = more solid; with no loss yet the arm is an outline
    const solid = S.saved ? Math.max(0.35, solidity()) : solidity();
    for (const p of PARTS) {
      const n = el.parts[p];
      for (const c of n.children) c.setAttribute("d", path[p]);
      n.classList.toggle("ghost", !S.built[p]);
      n.lastChild.setAttribute("stroke-opacity", solid.toFixed(3));
    }
    el.joints[0].setAttribute("cx", g.sx); el.joints[0].setAttribute("cy", g.sy);
    el.joints[1].setAttribute("cx", E.x.toFixed(1)); el.joints[1].setAttribute("cy", E.y.toFixed(1));
    el.joints[0].classList.toggle("hidden", !S.built.l1); el.joints[1].classList.toggle("hidden", !S.built.l2);
    // the task pieces in the run's colours
    const oc = BLOCK[sp.obj];
    const o = objP || g.obj;
    el.obj.classList.toggle("hidden", !oc || !S.placed.obj);
    if (oc) { el.obj.setAttribute("fill", oc); el.obj.setAttribute("x", (o.x - 7).toFixed(1)); el.obj.setAttribute("y", (o.y - 7).toFixed(1)); }
    const kind = { push: "pad", kit: "pad", pick_place: "bowl", stack: "block", insert: "socket", rotate: "socket" }[sp.skill];
    const tc = kind === "pad" ? PAD[sp.target] : kind === "bowl" ? BOWL[sp.target] : kind === "block" ? BLOCK[sp.target] : kind === "socket" ? "none" : null;
    el.tgt.classList.toggle("hidden", !tc || !S.placed.tgt);
    if (tc) {
      const r = kind === "pad" ? [g.tgt.x - 14, MAT - 3, 28, 3, 1] : kind === "bowl" ? [g.tgt.x - 15, MAT - 10, 30, 10, 5] : kind === "block" ? [g.tgt.x - 7, MAT - 14, 14, 14, 2.5] : [g.tgt.x - 9, MAT - 9, 18, 9, 1];
      ["x", "y", "width", "height", "rx"].forEach((k, i) => el.tgt.setAttribute(k, r[i]));
      el.tgt.setAttribute("fill", tc);
    }
    el.flash.setAttribute("cx", g.tgt.x); el.flash.setAttribute("cy", MAT - 8);
    el.flash.style.transformOrigin = `${g.tgt.x}px ${MAT - 8}px`;
    // the readout: only numbers the run reported
    const words = { push: "push", pick_place: "put in bowl", stack: "stack", reach: "reach", lift: "lift", kit: "kit", insert: "insert", rotate: "rotate" }[sp.skill] || "";
    let read = [label(S.robot), words, sp.obj && `${sp.obj} block`, sp.target && `${sp.target} ${kind === "bowl" ? "bowl" : kind === "block" ? "block" : kind === "pad" ? "square" : ""}`.trim()].filter(Boolean).join(" · ");
    if (S.ema != null && !S.ev) read = `Loss ${S.ema.toFixed(3)}`;
    if (S.ev) read = `${S.ev.succeeded}/${S.ev.done} passed · ${S.ev.n} held-out scenes`;
    if (S.saved) read = `Saved · ${Math.round(S.saved.rate * S.saved.n)}/${S.saved.n} held-out, in simulation`;
    el.read.textContent = read;
    el.count.textContent = S.demos ? `Demos ${S.demos.done}/${S.demos.n} · ${S.demos.ok} ok` : "";
  }
  function solidity() { return S.ema == null || !S.L0 ? 0 : clamp((S.L0 - S.ema) / (S.L0 * 0.85), 0, 1); }
  function flash(kind) { el.flash.setAttribute("class", "bd-flash"); void el.flash.getBBox; requestAnimationFrame(() => el.flash.setAttribute("class", `bd-flash ${kind}`)); flashAt = performance.now(); }
  function pop(p) { const n = el.parts[p]; n.classList.remove("pop"); requestAnimationFrame(() => n.classList.add("pop")); }
  return {
    mount, unmount, layout, redraw, page, local, flash, pop,
    caption(t) { if (cap) cap.querySelector(".t").textContent = t; },
    set tip(p) { tipP = p; }, get tip() { return tipP; },
    set obj(p) { objP = p; }, get obj() { return objP; },
    get mounted() { return !!root; },
    start: (api) => loop(api),
    get flashAt() { return flashAt; },
  };
})();
export const buildSite = site;

// ---------------------------------------------------------------- what the bot does, one queued event at a time
let running = null;
async function loop(api) {
  if (running && running.on) return;
  running = api;
  site.redraw();
  await api.sleep(80);
  while (api.on) {
    const f = queue.shift();
    if (f) { try { await f(api); } catch (err) { if (api.on) setTimeout(() => { throw err; }); } continue; }
    await idle(api);
    if (!queue.length) await new Promise((r) => { wake = r; setTimeout(r, 4000); });
  }
  running = null;
}
const k = () => clamp(site.page({ x: 1, y: 0 }).x - site.page({ x: 0, y: 0 }).x, 0.3, 1.6) * 0.62;
async function idle(api) {
  // perch beside the arm, watching it
  const g = rig();
  api.scale(clamp(k(), 0.36, 0.9));
  api.hand(false);
  await api.moveTo(() => { const p = site.page({ x: g.bx - 36, y: g.sy - 62 }); return p; }, 90);
  api.look(site.page({ x: g.sx + 20, y: g.sy - 20 }));
}
async function carryPiece(api, which, to) {
  const from = { x: which === "obj" ? 20 : 300, y: MAT - 50 };
  api.scale(clamp(k(), 0.36, 0.9));
  api.hand(true);
  if (which === "obj") await api.moveTo(() => api.reach(site.page(from), 6), 260);
  if (!api.on) return;
  // the block rides on the hand; the target goes down where the hand stops
  let carrying = which === "obj";
  if (carrying) {
    S.placed.obj = true;
    tick(() => { if (!site.mounted || !carrying) return false; const t = site.local(api.tip()); site.obj = { x: t.x, y: t.y + 7 }; site.redraw(); return true; });
  }
  await api.moveTo(() => api.reach(site.page({ x: to.x, y: to.y - 7 }), 0), 240);
  carrying = false;
  site.obj = null;
  S.placed[which] = true;
  site.redraw();
  api.kick(0.5);
  await api.sleep(150);
  
}
async function placePieces(api) {
  if (!S.spec) return;
  const g = rig();
  api.look(null);
  if (BLOCK[S.spec.obj]) await carryPiece(api, "obj", g.obj);
  const hasT = S.spec.target || /insert|rotate/.test(S.spec.skill);
  if (hasT && api.on) await carryPiece(api, "tgt", { x: g.tgt.x, y: MAT });
  site.redraw();
}
// carry each still-missing part from the edge of the mat to its place on the outline
async function placeParts(api, which) {
  const g = rig(), { E, W } = pose(g, site.tip || REST(g));
  const at = { base: { x: g.bx, y: g.by - 4 }, l1: { x: (g.sx + E.x) / 2, y: (g.sy + E.y) / 2 }, l2: { x: (E.x + W.x) / 2, y: (E.y + W.y) / 2 }, grip: { x: W.x, y: W.y + g.grip / 2 } };
  api.scale(clamp(k(), 0.36, 0.9));
  for (const p of which) {
    if (S.built[p] || !api.on) continue;
    api.hand(true);
    await api.moveTo(() => api.reach(site.page(at[p]), 4), 260);
    S.built[p] = true;
    site.redraw(); site.pop(p);
    api.kick(0.4); api.react("blink");
    await api.sleep(120);
  }
}
// sweep the tip along the task path, sped up; the block rides along between grab and release
function path(g) {
  const o = g.obj, t = g.tgt, sk = S.spec?.skill;
  if (sk === "reach") return [[REST(g), 0], [{ x: o.x, y: o.y - 22 }, 0], [{ x: o.x, y: o.y - 8 }, 0], [REST(g), 0]];
  if (sk === "lift") return [[REST(g), 0], [{ x: o.x, y: o.y - 8 }, 0], [{ x: o.x, y: o.y - 40 }, 1], [{ x: o.x, y: o.y - 8 }, 1], [REST(g), 0]];
  if (sk === "push") return [[REST(g), 0], [{ x: o.x - 10, y: o.y - 20 }, 0], [{ x: o.x - 10, y: o.y }, 0], [{ x: t.x - 10, y: o.y }, 2], [{ x: t.x - 10, y: o.y - 24 }, 0], [REST(g), 0]];
  return [[REST(g), 0], [{ x: o.x, y: o.y - 24 }, 0], [{ x: o.x, y: o.y - 8 }, 0], [{ x: o.x, y: o.y - 34 }, 1], [{ x: t.x, y: o.y - 34 }, 1], [{ x: t.x, y: o.y - 12 }, 1], [REST(g), 0]];
}
function sweep(ms, fail) {
  const g = rig(), pts = path(g);
  return new Promise((res) => {
    const t0 = performance.now();
    tick(() => {
      if (!site.mounted) { res(); return false; }
      const u = Math.min(1, (performance.now() - t0) / ms), f = u * (pts.length - 1), i = Math.min(pts.length - 2, Math.floor(f)), e = ease(f - i);
      const a = pts[i][0], b = pts[i + 1][0], hold = pts[i + 1][1];
      // a miss lets go halfway through the carry
      const tip = { x: a.x + (b.x - a.x) * e, y: a.y + (b.y - a.y) * e };
      site.tip = tip;
      const slip = fail && u > 0.55;
      if (hold === 1 && !slip) site.obj = { x: tip.x, y: tip.y + 8 };
      else if (hold === 2 && !slip) site.obj = { x: tip.x + 10, y: g.obj.y };
      site.redraw();
      if (u >= 1) { site.tip = null; site.obj = null; site.redraw(); res(); return false; }
      return true;
    });
  });
}
async function trace(api, times, verdict) {
  if (!S.built.grip) await placeParts(api, PARTS);
  const g = rig();
  api.hand(false);
  api.scale(clamp(k(), 0.36, 0.9));
  await api.moveTo(() => site.page({ x: g.bx - 36, y: g.sy - 62 }), 110);
  api.look(site.page(g.obj));
  for (let n = 0; n < times && api.on; n++) await sweep(api.reduced() ? 1 : 900, verdict === "bad");
  if (verdict) { site.flash(verdict); api.emote(verdict === "ok" ? "pleased" : "wince"); }
  site.redraw();
}
async function spot(api) {
  if (!S.built.grip) return;
  const g = rig(), { E } = pose(g, REST(g));
  api.hand(false);
  await api.moveTo(() => site.page({ x: E.x + 30, y: E.y - 34 }), 90);
  api.point(site.page(E)); api.look(site.page(E));
  api.kick(0.3); api.react("nod");
}
async function finish(api) {
  if (!S.built.grip) await placeParts(api, PARTS);
  site.redraw();
  const g = rig();
  api.hand(false);
  await api.moveTo(() => site.page({ x: g.bx + 110, y: g.sy - 30 }), 100);
  api.look(site.page({ x: g.sx, y: g.sy - 30 }));
  api.bow();
  S.endAt = performance.now();
}
