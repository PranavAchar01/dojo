// The roaming Dojo bot: the Etch eye rig (eyes.js, mounted by mascot.js) on a small robot body that flies around the
// page on Etch's springs. Etch's dock keeps its character in a pill and only lets the eyes look at the part being
// worked on (dock.js lookAt); roaming is new here and uses the same physics (spring.js presets, wobble, squash) and
// the same emotes. The body is the flat face alone; a single stroke is its hand.
//
//   const r = createRoamer({ mountBot, slot })     slot: the dock pill's eye slot, where the bot rests
//   r.visit(target, { dwell })                      fly beside an element (never on it) and point at it
//   r.work(target, { site, caption, inset })        build in an empty region; the site (build.js) drives the bot
//   r.stopWork()                                    leave the region (content arrived), back to the dock
//   r.home() / r.pointAt(target) / r.bow()
//
// Keep clear: anything marked [data-keepclear] (plus the chat, cards and bars below) is scored as an obstacle; the
// bot picks the free spot beside its target with the least overlap and moves again if something appears under it.
import { spring, tick, reducedMotion, clamp } from "./spring.js";

const W = 84, H = 66;
// the pointing hand: one stroke from the right side of the face, mirrored on the left
const SH = { x: 38, y: 6 }, ARM = 51;
const KEEP = "[data-keepclear], .msg, .skill, .lib-bar, .top, .tabs, .composer, .dock-thread, .sec, .ag-panel, .ag-pill, .pill:not(.hidden), #three";
const ARM_SVG = `<svg viewBox="0 0 64 28" aria-hidden="true"><path class="rb-hand" d="M7 14H58"/></svg>`;

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
// the bot's footprint around its centre
const box = (x, y, k = 1) => ({ l: x - 44 * k, r: x + 44 * k, t: y - 36 * k, b: y + 40 * k });

export function createRoamer({ mountBot, slot = null } = {}) {
  const root = document.createElement("div");
  root.className = "rb";
  root.dataset.mode = "home";
  const shadow = document.createElement("span"); shadow.className = "rb-shadow";
  const bob = document.createElement("span"); bob.className = "rb-bob";
  const arm = document.createElement("span"); arm.className = "rb-arm"; arm.innerHTML = ARM_SVG;
  const body = document.createElement("span"); body.className = "rb-body";
  const host = document.createElement("span");
  host.className = "mascot lg rb-head";
  host.setAttribute("data-follow", "");
  host.setAttribute("role", "button");
  host.setAttribute("tabindex", "0");
  host.setAttribute("aria-label", "What are you doing?");
  host.title = "What are you doing?";
  body.append(host);
  bob.append(body, arm);
  root.append(shadow, bob);
  document.body.append(root);
  const bot = mountBot(host);

  const small = () => innerWidth < 760;
  const pos = spring({ x: innerWidth - 60, y: innerHeight - 40 }, { preset: { stiffness: 70, damping: 15 }, precision: 0.05, onUpdate: paint });
  const ang = spring({ a: 62, e: 0.35 }, { preset: { stiffness: 260, damping: 18 }, precision: 0.01, onUpdate: paint });
  const sq = spring({ x: 1, y: 1 }, { preset: "wobbly", precision: 0.0008, onUpdate: paint });
  const ks = spring(1, { preset: { stiffness: 140, damping: 20 }, precision: 0.002, onUpdate: paint });
  const bw = spring(0, { preset: { stiffness: 120, damping: 14 }, precision: 0.004, onUpdate: paint });
  let tilt = 0, lastX = pos.value.x, side = 1;

  function paint() {
    const p = pos.value, a = ang.value, q = sq.value, k = ks.value, b = bw.value;
    const vx = p.x - lastX; lastX = p.x;
    tilt += (clamp(vx * 1.3, -16, 16) - tilt) * 0.25;
    root.style.transform = `translate(${(p.x - W / 2).toFixed(1)}px, ${(p.y - H / 2).toFixed(1)}px) scale(${k.toFixed(3)})`;
    // a bow dips the head and leans the body in; the squash spring keeps Etch's wobble on top
    body.style.transform = `translateY(${(b * 7).toFixed(2)}px) rotate(${tilt.toFixed(1)}deg) scale(${q.x.toFixed(3)}, ${(q.y * (1 - b * 0.07)).toFixed(3)})`;
    const s = 0.55 + 0.45 * a.e;
    arm.style.transform = `translate(${(W / 2 + SH.x * side).toFixed(1)}px, ${(H / 2 + SH.y + b * 6).toFixed(1)}px) rotate(${a.a.toFixed(1)}deg) scale(${s.toFixed(3)}, ${(s * side).toFixed(3)})`;
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
    if (job) { const r = geo(job); if (r) job.site.layout(r, api); return true; }
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

  // ---- work: a site (build.js) draws in an empty region and drives the bot through this small api ----
  let job = null; // { el, site, on, inset }
  function geo(j) {
    const t = rectOf(j.el);
    if (!t) return null;
    const i = j.inset;
    const r = { left: t.r.left + i.l, right: t.r.right - i.r, top: t.r.top + i.t, bottom: t.r.bottom - i.b };
    r.width = r.right - r.left; r.height = r.bottom - r.top;
    return r.width > 60 && r.height > 60 ? r : null;
  }
  const api = {
    get on() { return !!job && job.on; },
    scale(k) { ks.set(k); },
    get k() { return ks.target; },
    tip,
    // the bot's centre so that its hand, pointing down, touches page point p from `lift` above
    reach(p, lift = 0) { const k = ks.target; return { x: p.x - SH.x * k, y: p.y - (SH.y + ARM) * k - lift }; },
    async moveTo(f, stiff = 120) {
      const j = job;
      pos.config({ stiffness: stiff, damping: 17 });
      for (let n = 0; n < 22 && j && j.on; n++) {
        const to = typeof f === "function" ? f() : f;
        if (!to) return;
        if (Math.hypot(pos.target.x - to.x, pos.target.y - to.y) > 0.8) pos.set(to);
        const v = pos.value;
        if (Math.hypot(v.x - to.x, v.y - to.y) < 4) return;
        await sleep(50);
      }
    },
    hand(down = true) { side = 1; ang.set(down ? { a: 90, e: 1 } : { a: 62, e: 0.35 }); },
    point(at) { aimArm(pos.value, [at.x, at.y]); },
    look(at) { setLook(at ? [at.x, at.y] : null); },
    kick, bow,
    emote(n) { if (bot) bot.eyes.emote(n); },
    react(n) { if (bot) bot.eyes.react(n); },
    sleep,
    reduced: reducedMotion,
  };
  function work(target, o = {}) {
    const t = rectOf(target);
    if (!t || !o.site) return false;
    const inset = { l: 0, r: 0, t: 0, b: 0, ...o.inset };
    if (job && job.el === t.el && job.site === o.site) { job.inset = inset; if (o.caption != null) o.site.caption(o.caption); return true; }
    if (job) stopWork(true);
    clearTimeout(homeT); queued = null; cur = null; spot = null;
    job = { el: t.el, site: o.site, on: true, inset };
    o.site.mount(o.compact);
    o.site.caption(o.caption || "");
    t.el.classList.add("rb-busy");
    root.classList.add("working");
    setMode("work");
    flying = false;
    side = 1;
    const r = geo(job);
    if (r) o.site.layout(r, api);
    o.site.start(api);
    return true;
  }
  function setCaption(text) { if (job) job.site.caption(text); }
  function stopWork(quiet = false) {
    const j = job;
    if (!j) return;
    j.on = false;
    job = null;
    j.el.classList.remove("rb-busy");
    j.site.unmount();
    root.classList.remove("working");
    if (!quiet) flyHome();
  }
  function bow() {
    if (reducedMotion()) return;
    bw.set(1);
    if (bot) { bot.eyes.emote("pleased"); bot.eyes.react("nod"); }
    setTimeout(() => bw.set(0), 650);
  }

  host.addEventListener("click", () => dispatchEvent(new CustomEvent("forge:ask-bot")));
  host.addEventListener("keydown", (e) => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); dispatchEvent(new CustomEvent("forge:ask-bot")); } });
  addEventListener("resize", () => { if (!cur && !job) pos.jump(slotCenter()); });

  pos.jump(slotCenter());
  paint();
  return {
    host, bot, root, visit, home, pointAt, kick, work, stopWork, setCaption, bow,
    get away() { return !!cur || !!job; },
    get working() { return job ? job.el : null; },
    get site() { return job ? job.site : null; },
    setSlot(s) { slot = s; },
    get position() { const v = pos.value; return { x: v.x, y: v.y }; },
    get reduced() { return reducedMotion(); },
  };
}
