// The roaming Dojo bot: the Etch eye rig (eyes.js, mounted by mascot.js) on a small robot body that flies around the
// page on Etch's springs. Etch's dock keeps its character in a pill and only lets the eyes look at the part being
// worked on (dock.js lookAt); roaming is new here and uses the same physics (spring.js presets, wobble, squash) and
// the same emotes. The robot touches (Meta): side sensor pods, chin vents, a two-finger gripper arm on a shoulder
// servo, and hover jets under a short gi torso whose belt colour ranks the skills the library has passed (Dojo).
//
//   const r = createRoamer({ mountBot, slot })     slot: the dock pill's eye slot, where the bot rests
//   r.visit(target, { dwell })                      fly beside an element (never on it) and point at it
//   r.work(target, { caption, inset })              train in an empty region: carry blocks across a mat, as reps
//   r.stopWork()                                    leave the region (content arrived), back to the dock
//   r.home() / r.pointAt(target) / r.bow() / r.setBelt(rank)
//
// Keep clear: anything marked [data-keepclear] (plus the chat, cards and bars below) is scored as an obstacle; the
// bot picks the free spot beside its target with the least overlap and moves again if something appears under it.
import { spring, tick, reducedMotion, clamp } from "./spring.js";

const W = 84, H = 66;
// shoulder servo on the right pod, relative to the bot's centre (unscaled); the left one mirrors it
const SH = { x: 40, y: 8.5 }, ARM = 55;
const KEEP = "[data-keepclear], .msg, .skill, .lib-bar, .top, .tabs, .composer, .sec, .ag-panel, .ag-pill, .pill:not(.hidden), #three";
const BELTS = ["#ffffff", "#f5c518", "#ff8a1f", "#2fb36b", "#1877f2", "#8a5a3c", "#1c2b33"];

const ARM_SVG = `<svg viewBox="0 0 64 28" aria-hidden="true">
  <rect class="rb-up" x="2" y="9" width="30" height="10" rx="5"/><path class="rb-hl" d="M8 11.4h20"/>
  <rect class="rb-fore" x="26" y="10" width="22" height="8" rx="4"/>
  <rect class="rb-servo" x="30" y="7.5" width="10" height="13" rx="3"/><circle class="rb-hub" cx="35" cy="14" r="1.6"/>
  <path class="rb-claw" d="M46 9.2h9.5q4.5 0 6.5 3.4l-5.3.9q-1 .15-1.7.9l-.6 1.2H46z"/>
  <path class="rb-claw b" d="M46 18.8h9q3.2 0 5-2.4l-4.2.2q-.9 0-1.5-.7l-.8-1H46z"/>
  <circle class="rb-servo" cx="7" cy="14" r="6.4"/><circle class="rb-hub" cx="7" cy="14" r="2.3"/>
</svg>`;
const SHELL_SVG = `<svg class="rb-shell" viewBox="0 0 92 72" aria-hidden="true">
  <g class="rb-pod"><rect x="-2.5" y="38" width="9" height="15" rx="4.5"/><circle cx="2" cy="45.5" r="1.7" class="rb-led"/></g>
  <g class="rb-pod"><rect x="85.5" y="38" width="9" height="15" rx="4.5"/><circle cx="90" cy="45.5" r="1.7" class="rb-led"/></g>
  <g class="rb-vent"><path d="M38 64.5h16M40 67h12"/></g>
</svg>`;
// a short gi: white jacket with crossed lapels, the belt and its knot, two jet nozzles underneath
const TORSO_SVG = `<svg class="rb-torso" viewBox="0 0 48 26" aria-hidden="true">
  <path class="rb-gi" d="M7 1.5h34q5 0 4.2 5L42.6 20q-.7 3.5-4.3 3.5H9.7Q6.1 23.5 5.4 20L2.8 6.5Q2 1.5 7 1.5z"/>
  <path class="rb-lapel" d="M15 1.8 24 11.5M33 1.8 21.6 9"/>
  <path class="rb-belt" d="M3.9 12.2h40.2l-.8 4.2H4.7z"/>
  <path class="rb-belt" d="M21 12.6h6v4h-6z"/><path class="rb-belt tail" d="M22.4 16.2l-1.6 5 2.2.4 1.2-5zM25.6 16.2l1.6 5-2.2.4-1.2-5z"/>
  <ellipse class="rb-noz" cx="15" cy="24.2" rx="3.6" ry="1.6"/><ellipse class="rb-noz" cx="33" cy="24.2" rx="3.6" ry="1.6"/>
</svg>`;

const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
function rectOf(t) {
  const el = typeof t === "string" ? document.querySelector(t) : t;
  if (!el || !el.isConnected) return null;
  const r = el.getBoundingClientRect();
  if (r.width < 4 || r.height < 4) return null;
  return { el, r };
}
const onScreen = (r) => r.bottom > 40 && r.top < innerHeight - 20 && r.right > 0 && r.left < innerWidth;
const ov = (a, b) => Math.max(0, Math.min(a.r, b.right) - Math.max(a.l, b.left)) * Math.max(0, Math.min(a.b, b.bottom) - Math.max(a.t, b.top));
// the bot's footprint around its centre: head, pods, torso and jets
const box = (x, y, k = 1) => ({ l: x - 48 * k, r: x + 48 * k, t: y - 38 * k, b: y + 50 * k });

export function createRoamer({ mountBot, slot = null } = {}) {
  const root = document.createElement("div");
  root.className = "rb";
  root.dataset.mode = "home";
  const shadow = document.createElement("span"); shadow.className = "rb-shadow";
  const bob = document.createElement("span"); bob.className = "rb-bob";
  const jets = document.createElement("span"); jets.className = "rb-jets"; jets.innerHTML = "<i></i><i></i>";
  const arm = document.createElement("span"); arm.className = "rb-arm"; arm.innerHTML = ARM_SVG;
  const body = document.createElement("span"); body.className = "rb-body";
  body.insertAdjacentHTML("beforeend", TORSO_SVG);
  const host = document.createElement("span");
  host.className = "mascot lg rb-head";
  host.setAttribute("data-follow", "");
  host.setAttribute("role", "button");
  host.setAttribute("tabindex", "0");
  host.setAttribute("aria-label", "What are you doing?");
  host.title = "What are you doing?";
  body.append(host);
  bob.append(jets, body, arm);
  root.append(shadow, bob);
  document.body.append(root);
  const bot = mountBot(host);
  host.insertAdjacentHTML("beforeend", SHELL_SVG);

  const small = () => innerWidth < 760;
  const pos = spring({ x: innerWidth - 60, y: innerHeight - 40 }, { preset: { stiffness: 70, damping: 15 }, precision: 0.05, onUpdate: paint });
  const ang = spring({ a: 62, e: 0.35 }, { preset: { stiffness: 260, damping: 18 }, precision: 0.01, onUpdate: paint });
  const sq = spring({ x: 1, y: 1 }, { preset: "wobbly", precision: 0.0008, onUpdate: paint });
  const ks = spring(1, { preset: { stiffness: 140, damping: 20 }, precision: 0.002, onUpdate: paint });
  const bw = spring(0, { preset: { stiffness: 120, damping: 14 }, precision: 0.004, onUpdate: paint });
  let tilt = 0, vis = 0, lastX = pos.value.x, side = 1;

  function paint() {
    const p = pos.value, a = ang.value, q = sq.value, k = ks.value, b = bw.value;
    const vx = p.x - lastX; lastX = p.x;
    tilt += (clamp(vx * 1.3, -16, 16) - tilt) * 0.25;
    root.style.transform = `translate(${(p.x - W / 2).toFixed(1)}px, ${(p.y - H / 2).toFixed(1)}px) scale(${k.toFixed(3)})`;
    // a bow dips the head and leans the body in; the squash spring keeps Etch's wobble on top
    body.style.transform = `translateY(${(b * 7).toFixed(2)}px) rotate(${tilt.toFixed(1)}deg) scale(${q.x.toFixed(3)}, ${(q.y * (1 - b * 0.07)).toFixed(3)})`;
    const s = 0.55 + 0.45 * a.e;
    arm.style.transform = `translate(${(W / 2 + SH.x * side).toFixed(1)}px, ${(H / 2 + SH.y + b * 6).toFixed(1)}px) rotate(${a.a.toFixed(1)}deg) scale(${s.toFixed(3)}, ${(s * side).toFixed(3)})`;
    const speed = Math.hypot(pos.velocity.x, pos.velocity.y);
    const on = clamp((speed - 60) / 260);
    vis += (on - vis) * 0.3;
    root.style.setProperty("--jet", Math.max(vis, root.dataset.mode === "work" ? 0.35 : 0).toFixed(3));
  }
  const kick = (s = 1) => sq.impulse({ x: 0.9 * s, y: -1.4 * s });
  // the claw's tip in page coordinates (what a carried block hangs from)
  function tip() {
    const p = pos.value, k = ks.value, a = ang.value, s = 0.55 + 0.45 * a.e, r = (a.a * Math.PI) / 180;
    return { x: p.x + k * (SH.x * side + ARM * s * Math.cos(r)), y: p.y + k * (SH.y + ARM * s * Math.sin(r)) };
  }

  // ---- keep clear: every candidate spot beside the target is scored against what must stay visible ----
  function obstacles(target) {
    const out = [];
    for (const el of document.querySelectorAll(KEEP)) {
      if (root.contains(el) || el.closest(".mb-bubbles, .rb-site")) continue;
      if (target && (el.contains(target) || target.contains(el))) continue;
      const r = el.getBoundingClientRect();
      if (r.width < 2 || r.height < 2 || r.bottom < 0 || r.top > innerHeight || r.right < 0 || r.left > innerWidth) continue;
      // the sim canvas only counts once something is drawn on it, and then softly (the floor has room to spare)
      const w = el.id === "three" ? (el.closest(".blank") ? 0 : 0.25) : +(el.dataset.keepclear || 1) || 1;
      if (w) out.push({ r, w });
    }
    return out;
  }
  function scoreAt(x, y, t, obs, big) {
    const b = box(x, y), A = (b.r - b.l) * (b.b - b.t);
    let s = 0;
    for (const o of obs) s += (o.w * ov(b, o.r)) / A;
    s += ((big ? 0.06 : 4) * ov(b, t)) / A;
    const off = Math.max(0, 6 - b.l) + Math.max(0, b.r - innerWidth + 6) + Math.max(0, 6 - b.t) + Math.max(0, b.b - innerHeight + 6);
    s += off / 16;
    s += Math.hypot(x - (t.left + t.width / 2), y - (t.top + t.height / 2)) / 4000;
    const c = pos.target;
    return s + Math.hypot(x - c.x, y - c.y) / 8000;
  }
  function spotFor(el, r) {
    const obs = obstacles(el);
    const big = r.width * r.height > innerWidth * innerHeight * 0.16;
    const cx = r.left + r.width / 2, cy = r.top + r.height / 2;
    const cand = [];
    for (const m of [10, 34, 80, 150]) {
      for (const y of [r.top + 30, cy, r.bottom - 40]) cand.push([r.right + m + 48, y], [r.left - m - 48, y]);
      for (const x of [r.left + 48, cx, r.right - 48]) cand.push([x, r.top - m - 50], [x, r.bottom + m + 38]);
    }
    if (big) for (const fx of [0.14, 0.5, 0.86]) for (const fy of [0.16, 0.5, 0.84]) cand.push([r.left + r.width * fx, r.top + r.height * fy]);
    let best = null;
    for (const [x, y] of cand) { const s = scoreAt(x, y, r, obs, big); if (!best || s < best.s) best = { x, y, s }; }
    const inside = best.x > r.left && best.x < r.right && best.y > r.top && best.y < r.bottom;
    const ix = Math.min(40, r.width / 3), iy = Math.min(40, r.height / 3);
    best.at = inside ? [cx, cy] : [clamp(best.x, r.left + ix, r.right - ix), clamp(best.y, r.top + iy, r.bottom - iy)];
    best.cur = scoreAt(pos.target.x, pos.target.y, r, obs, big);
    return best;
  }
  function aimArm(from, at, e = 1) {
    side = at[0] < from.x - 6 ? -1 : 1;
    const sx = from.x + SH.x * side, sy = from.y + SH.y;
    let a = (Math.atan2(at[1] - sy, at[0] - sx) * 180) / Math.PI;
    if (side > 0) a = clamp(a, -80, 100);
    else { if (a < 0) a += 360; a = clamp(a, 80, 260); }
    ang.set({ a, e });
  }
  function setLook(at) { if (bot) bot.eyes.lookAt(at ? at[0] : null, at ? at[1] : null); }

  // ---- where it rests, where it goes ----
  let cur = null;          // { el, mode }
  let homeT = 0, queued = null, dwellEnd = 0, flying = false, spot = null, checkAt = 0;
  const slotCenter = () => {
    const s = slot && slot.isConnected ? slot.getBoundingClientRect() : null;
    return s && s.width ? { x: s.left + s.width / 2, y: s.top + s.height / 2 } : { x: innerWidth - 70, y: innerHeight - 46 };
  };

  tick((dt, now = performance.now()) => {
    if (job) { layoutSite(); return true; }
    // while resting it tracks the dock slot (the pill wobbles and moves); while out it keeps beside its element
    if (!cur) {
      const c = slotCenter();
      const g = pos.target;
      if (Math.abs(g.x - c.x) > 0.3 || Math.abs(g.y - c.y) > 0.3) { if (!pos.moving) pos.jump(c); else pos.set(c); }
      return true;
    }
    const t = rectOf(cur.el);
    if (!t || !onScreen(t.r)) return true;
    if (now >= checkAt) {
      // every 400 ms: is something important under it now, or has its element moved? then step to the best free spot
      checkAt = now + 400;
      const b = spotFor(t.el, t.r);
      if (!spot || (Math.hypot(b.x - spot.x, b.y - spot.y) > 6 && b.s < b.cur - 0.03)) { spot = b; pos.set({ x: b.x, y: b.y }); }
      else spot.at = b.at;
    }
    if (cur.mode === "point" && spot && !flying) { aimArm(pos.value, spot.at); setLook(spot.at); }
    return true;
  });

  function setMode(m) { root.dataset.mode = m; }
  function arrive(mode) {
    flying = false;
    setMode(mode);
    kick(0.7);
    if (bot) bot.eyes.emote("hop");
    dispatchEvent(new CustomEvent("forge:roam", { detail: { phase: "arrived", mode } }));
  }

  function visit(target, o = {}) {
    if (small()) return false;
    const t = rectOf(target);
    if (!t || !onScreen(t.r)) return false;
    const now = performance.now();
    if (cur && now < dwellEnd - 400 && !o.force) { queued = { target, o }; return true; } // let the last stop be seen
    if (job) stopWork(true);
    queued = null;
    clearTimeout(homeT);
    cur = { el: t.el, mode: "point" };
    const dwell = o.dwell ?? 5200;
    dwellEnd = now + Math.min(1100, dwell);
    spot = spotFor(t.el, t.r);
    checkAt = now + 900;
    flying = true;
    setMode("fly");
    ks.set(1);
    pos.config({ stiffness: 70, damping: 15 }).set({ x: spot.x, y: spot.y });
    kick(-0.5);
    ang.set({ a: -20, e: 0.3 });
    const chk = setInterval(() => {
      const v = pos.value;
      if (!cur || cur.el !== t.el) { clearInterval(chk); return; }
      if (Math.hypot(v.x - spot.x, v.y - spot.y) < 14) { clearInterval(chk); arrive("perch"); aimArm(pos.value, spot.at); setLook(spot.at); }
    }, 80);
    if (dwell > 0) homeT = setTimeout(home, dwell + 1100);
    return true;
  }
  function pointAt(target) {
    const t = rectOf(target);
    if (!t) return;
    const at = [t.r.left + t.r.width / 2, t.r.top + t.r.height / 2];
    aimArm(pos.value, at); setLook(at);
    clearTimeout(pointAt.t);
    pointAt.t = setTimeout(() => { setLook(null); if (!cur && !job) { side = 1; ang.set({ a: 62, e: 0.35 }); } }, 2200);
  }
  function home() {
    clearTimeout(homeT);
    if (job) { stopWork(); return; }
    if (queued && performance.now() < dwellEnd + 4000) { const q = queued; queued = null; dwellEnd = 0; visit(q.target, { ...q.o, force: true }); return; }
    if (!cur) return;
    cur = null; spot = null;
    flyHome();
  }
  function flyHome() {
    flying = true;
    setMode("fly");
    setLook(null);
    side = 1;
    ang.set({ a: 62, e: 0.35 });
    ks.set(1);
    pos.config({ stiffness: 90, damping: 16 }).set(slotCenter());
    const chk = setInterval(() => {
      if (cur || job) { clearInterval(chk); return; }
      const c = slotCenter(), v = pos.value;
      if (Math.hypot(v.x - c.x, v.y - c.y) < 10) { clearInterval(chk); flying = false; setMode("home"); kick(0.8); dispatchEvent(new CustomEvent("forge:roam", { detail: { phase: "home" } })); }
    }, 80);
  }

  // ---- work: reps on a mat in an empty region, the loading screen for whatever is still on its way ----
  let job = null; // { el, site, mat, blocks, cap, on, inset, g }
  function makeSite(el, inset) {
    const site = document.createElement("div");
    site.className = "rb-site";
    site.setAttribute("aria-live", "polite");
    const mat = document.createElement("span"); mat.className = "rb-mat";
    const pad = document.createElement("span"); pad.className = "rb-padmark";
    const cap = document.createElement("span"); cap.className = "rb-cap";
    cap.innerHTML = '<span class="ag-spin"></span><span class="t"></span>';
    const blocks = [0, 1, 2].map((i) => { const b = document.createElement("span"); b.className = `rb-blk c${i}`; site.append(b); return { el: b, at: { home: i }, held: false }; });
    site.prepend(mat, pad);
    site.append(cap);
    document.body.append(site);
    el.classList.add("rb-busy");
    return { el, site, mat, pad, blocks, cap, on: true, inset: { l: 0, r: 0, t: 0, b: 0, ...inset }, g: null };
  }
  function geo(j) {
    const t = rectOf(j.el);
    if (!t) return null;
    const i = j.inset;
    const r = { left: t.r.left + i.l, right: t.r.right - i.r, top: t.r.top + i.t, bottom: t.r.bottom - i.b };
    r.width = r.right - r.left; r.height = r.bottom - r.top;
    const k = clamp(Math.min(r.width / 420, (r.height - 40) / 200), 0.36, 1);
    const sz = Math.round(18 * k + 4);
    const floor = r.bottom - 8 - 8 * k;
    const capB = r.top + 12 + 22;
    // how many blocks fit in a stack under the bot's reach
    const levels = clamp(Math.floor((floor - capB - 120 * k) / sz), 1, 3);
    const x0 = r.left + Math.max(18, r.width * 0.14) + 40 * k;
    const px = r.left + r.width * 0.68 + 20 * k;
    return {
      r, k, sz, floor, levels,
      home: (n) => ({ x: x0 + n * (sz + 7), y: floor - sz / 2 }),
      stack: (n) => ({ x: px, y: floor - sz / 2 - n * sz }),
    };
  }
  function blockPos(g, b) { return b.at.home != null ? g.home(b.at.home) : g.stack(b.at.stack); }
  function layoutSite() {
    const j = job;
    const g = geo(j);
    if (!g) return;
    j.g = g;
    const { r, sz, floor, k } = g;
    Object.assign(j.mat.style, { left: `${r.left + 10}px`, width: `${r.width - 20}px`, top: `${floor}px`, height: `${Math.max(6, 8 * k)}px` });
    const p0 = g.stack(0);
    Object.assign(j.pad.style, { left: `${p0.x - sz * 0.8}px`, width: `${sz * 1.6}px`, top: `${floor - 2}px` });
    j.cap.style.left = `${r.left + r.width / 2}px`;
    j.cap.style.top = `${r.top + 12}px`;
    j.cap.style.maxWidth = `${Math.max(120, r.width - 24)}px`;
    j.cap.classList.toggle("sm", k < 0.6);
    const tp = tip();
    for (const b of j.blocks) {
      const c = b.held ? { x: tp.x, y: tp.y + sz / 2 - 1 } : blockPos(g, b);
      b.el.style.width = b.el.style.height = `${sz}px`;
      b.el.style.transform = `translate(${(c.x - sz / 2).toFixed(1)}px, ${(c.y - sz / 2).toFixed(1)}px)`;
    }
  }
  // the bot's centre for its claw tip to touch the top of a block centred at c, `lift` above it
  const reach = (g, c, lift = 0) => ({ x: c.x - SH.x * g.k, y: c.y - g.sz / 2 - (SH.y + ARM) * g.k - lift });
  async function flyTo(j, f, stiff = 120) {
    pos.config({ stiffness: stiff, damping: 17 });
    for (let n = 0; n < 80 && j.on; n++) {
      const g = j.g || geo(j);
      if (!g) return;
      const to = f(g);
      if (Math.hypot(pos.target.x - to.x, pos.target.y - to.y) > 0.8) pos.set(to);
      const v = pos.value;
      if (Math.hypot(v.x - to.x, v.y - to.y) < 2.5) return;
      await sleep(50);
    }
  }
  async function carry(j, b, dest) {
    const lift = (g) => 26 * g.k + g.sz * 0.6;
    ang.set({ a: 90, e: 1 });
    await flyTo(j, (g) => reach(g, blockPos(g, b), lift(g)));
    if (!j.on) return;
    setLook([b.el.getBoundingClientRect().left, b.el.getBoundingClientRect().top]);
    await flyTo(j, (g) => reach(g, blockPos(g, b)), 160);
    if (!j.on) return;
    b.held = true; kick(0.35); if (bot) bot.eyes.react("blink");
    await sleep(160);
    await flyTo(j, (g) => reach(g, blockPos(g, b), lift(g)), 140);
    const was = b.at; b.at = dest;
    await flyTo(j, (g) => reach(g, blockPos(g, b), lift(g)), 110);
    await flyTo(j, (g) => reach(g, blockPos(g, b)), 160);
    b.held = false;
    if (!j.on) { b.at = was.home != null ? was : dest; return; }
    kick(0.6); b.el.classList.remove("set"); void b.el.offsetWidth; b.el.classList.add("set");
    await sleep(180);
    await flyTo(j, (g) => reach(g, blockPos(g, b), lift(g)), 140);
  }
  async function reps(j) {
    // a few frames for the site to lay out, then fly in from wherever the bot was
    await sleep(60);
    if (!j.on) return;
    if (reducedMotion()) { const g = j.g || geo(j); if (g) { ks.jump(g.k); pos.jump(reach(g, g.home(0), 30 * g.k)); ang.jump({ a: 90, e: 1 }); } return; }
    while (j.on) {
      const g = j.g || geo(j);
      if (!g) return;
      const L = g.levels;
      for (let n = 0; n < L && j.on; n++) await carry(j, j.blocks[n], { stack: n });
      if (!j.on) return;
      // a beat on top of the finished stack, the eyes pleased with it, then put everything back
      if (bot) bot.eyes.emote("pleased");
      await flyTo(j, (gg) => ({ x: gg.stack(0).x - 70 * gg.k, y: gg.stack(L - 1).y - 80 * gg.k }), 90);
      setLook(null);
      await sleep(700);
      for (let n = L - 1; n >= 0 && j.on; n--) await carry(j, j.blocks[n], { home: n });
      await sleep(400);
    }
  }
  function work(target, o = {}) {
    const t = rectOf(target);
    if (!t) return false;
    if (job && job.el === t.el) { if (o.caption != null) setCaption(o.caption); return true; }
    if (job) stopWork(true);
    clearTimeout(homeT); queued = null; cur = null; spot = null;
    job = makeSite(t.el, o.inset);
    setCaption(o.caption || "");
    layoutSite();
    root.classList.add("working");
    setMode("work");
    flying = false;
    side = 1;
    ks.set(job.g ? job.g.k : 0.7);
    reps(job);
    return true;
  }
  function setCaption(text) { if (job) job.cap.querySelector(".t").textContent = text; }
  function stopWork(quiet = false) {
    const j = job;
    if (!j) return;
    j.on = false;
    job = null;
    j.el.classList.remove("rb-busy");
    j.site.classList.add("out");
    setTimeout(() => j.site.remove(), 320);
    root.classList.remove("working");
    if (!quiet) flyHome();
  }
  function bow() {
    if (reducedMotion()) return;
    bw.set(1);
    if (bot) { bot.eyes.emote("pleased"); bot.eyes.react("nod"); }
    setTimeout(() => bw.set(0), 650);
  }
  function setBelt(rank) { root.style.setProperty("--belt", BELTS[clamp(rank | 0, 0, BELTS.length - 1)]); root.dataset.belt = String(rank | 0); }

  host.addEventListener("click", () => dispatchEvent(new CustomEvent("forge:ask-bot")));
  host.addEventListener("keydown", (e) => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); dispatchEvent(new CustomEvent("forge:ask-bot")); } });
  addEventListener("resize", () => { if (!cur && !job) pos.jump(slotCenter()); });

  setBelt(0);
  pos.jump(slotCenter());
  paint();
  return {
    host, bot, root, visit, home, pointAt, kick, work, stopWork, setCaption, bow, setBelt,
    get away() { return !!cur || !!job; },
    get working() { return job ? job.el : null; },
    setSlot(s) { slot = s; },
    get position() { const v = pos.value; return { x: v.x, y: v.y }; },
    get reduced() { return reducedMotion(); },
  };
}
