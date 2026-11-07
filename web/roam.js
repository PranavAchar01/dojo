// The roaming forge bot: the Etch eye rig (eyes.js, mounted by mascot.js) in a small robot body that flies around the
// page on Etch's springs. Etch's dock keeps its character in a pill and only lets the eyes look at the part being
// worked on (dock.js lookAt); roaming is new here and uses the same physics (spring.js presets, wobble, squash) and
// the same emotes. The robot touches (Meta): side sensor pods, a visor seam, chin vents, a two-finger gripper arm it
// points with, and hover jets that light while it flies.
//
//   const r = createRoamer({ mountBot, slot })     slot: the dock pill's eye slot, where the bot rests
//   r.visit(target, { dwell, mode, say })           fly to an element (selector or node), perch on it and point at it
//   r.home()                                        fly back to the dock
//   r.pointAt(target)                               swing the arm without moving
import { spring, tick, reducedMotion, clamp } from "./spring.js";

const W = 84, H = 66;
const ARM_SVG = `<svg viewBox="0 0 64 28" aria-hidden="true">
  <rect class="rb-up" x="2" y="9" width="30" height="10" rx="5"/>
  <circle class="rb-servo" cx="7" cy="14" r="6.2"/><circle class="rb-hub" cx="7" cy="14" r="2.2"/>
  <rect class="rb-fore" x="26" y="10" width="22" height="8" rx="4"/>
  <rect class="rb-servo" x="30" y="7.5" width="10" height="13" rx="3"/>
  <path class="rb-claw" d="M46 9.2h9.5q4.5 0 6.5 3.4l-5.3.9q-1 .15-1.7.9l-.6 1.2H46z"/>
  <path class="rb-claw" d="M46 18.8h9.5q4.5 0 6.5-3.4l-5.3-.9q-1-.15-1.7-.9l-.6-1.2H46z" opacity=".001"/>
  <path class="rb-claw b" d="M46 18.8h9q3.2 0 5-2.4l-4.2.2q-.9 0-1.5-.7l-.8-1H46z"/>
</svg>`;
const SHELL_SVG = `<svg class="rb-shell" viewBox="0 0 92 72" aria-hidden="true">
  <g class="rb-pod"><rect x="-2.5" y="38" width="9" height="15" rx="4.5"/><circle cx="2" cy="45.5" r="1.7" class="rb-led"/></g>
  <g class="rb-pod"><rect x="85.5" y="38" width="9" height="15" rx="4.5"/><circle cx="90" cy="45.5" r="1.7" class="rb-led"/></g>
  <path class="rb-seam" d="M17 60.5q29 5.5 58 0"/>
  <g class="rb-vent"><path d="M38 65.5h16M40 68h12"/></g>
  <circle class="rb-rivet" cx="12.5" cy="29" r="1.2"/><circle class="rb-rivet" cx="79.5" cy="29" r="1.2"/>
</svg>`;

function rectOf(t) {
  const el = typeof t === "string" ? document.querySelector(t) : t;
  if (!el || !el.isConnected) return null;
  const r = el.getBoundingClientRect();
  if (r.width < 4 || r.height < 4) return null;
  return { el, r };
}
const onScreen = (r) => r.bottom > 40 && r.top < innerHeight - 20 && r.right > 0 && r.left < innerWidth;

export function createRoamer({ mountBot, slot = null } = {}) {
  const root = document.createElement("div");
  root.className = "rb";
  root.dataset.mode = "home";
  const jets = document.createElement("span");
  jets.className = "rb-jets";
  jets.innerHTML = "<i></i><i></i>";
  const arm = document.createElement("span");
  arm.className = "rb-arm";
  arm.innerHTML = ARM_SVG;
  const body = document.createElement("span");
  body.className = "rb-body";
  const host = document.createElement("span");
  host.className = "mascot lg rb-head";
  host.setAttribute("data-follow", "");
  host.setAttribute("role", "button");
  host.setAttribute("tabindex", "0");
  host.setAttribute("aria-label", "What are you doing?");
  host.title = "What are you doing?";
  body.append(host);
  root.append(jets, body, arm);
  document.body.append(root);
  const bot = mountBot(host);
  host.insertAdjacentHTML("beforeend", SHELL_SVG);

  const small = () => innerWidth < 760;
  const pos = spring({ x: innerWidth - 60, y: innerHeight - 40 }, { preset: { stiffness: 70, damping: 15 }, precision: 0.05, onUpdate: paint });
  const ang = spring({ a: 62, e: 0.35 }, { preset: { stiffness: 260, damping: 18 }, precision: 0.01, onUpdate: paint });
  const sq = spring({ x: 1, y: 1 }, { preset: "wobbly", precision: 0.0008, onUpdate: paint });
  let tilt = 0, vis = 0, scale = 1, lastX = pos.value.x;

  function paint() {
    const p = pos.value, a = ang.value, q = sq.value;
    const vx = p.x - lastX; lastX = p.x;
    tilt += (clamp(vx * 1.3, -16, 16) - tilt) * 0.25;
    root.style.transform = `translate(${(p.x - W / 2).toFixed(1)}px, ${(p.y - H / 2).toFixed(1)}px)`;
    body.style.transform = `rotate(${tilt.toFixed(1)}deg) scale(${(q.x * scale).toFixed(3)}, ${(q.y * scale).toFixed(3)})`;
    arm.style.transform = `translate(${W - 14}px, ${H * 0.52}px) rotate(${a.a.toFixed(1)}deg) scale(${(0.55 + 0.45 * a.e).toFixed(3)})`;
    const speed = Math.hypot(pos.velocity.x, pos.velocity.y);
    const on = clamp((speed - 60) / 260);
    vis += (on - vis) * 0.3;
    root.style.setProperty("--jet", vis.toFixed(3));
  }
  const kick = (s = 1) => sq.impulse({ x: 0.9 * s, y: -1.4 * s });

  // ---- where it rests, where it goes ----
  let cur = null;          // { el, mode, until }
  let homeT = 0, queued = null, dwellEnd = 0, flying = false;
  const slotCenter = () => {
    const s = slot && slot.isConnected ? slot.getBoundingClientRect() : null;
    return s && s.width ? { x: s.left + s.width / 2, y: s.top + s.height / 2 } : { x: innerWidth - 70, y: innerHeight - 46 };
  };
  function perchSpot(r) {
    // above the element's top edge when there is room, else inside its top-right corner
    if (r.top > H + 60) return { x: clamp(r.left + Math.min(70, r.width * 0.25), W / 2 + 6, innerWidth - W / 2 - 6), y: r.top - H * 0.42, at: [r.left + r.width / 2, r.top + Math.min(r.height / 2, 120)] };
    return { x: clamp(r.right - W * 0.75 - 6, W / 2 + 6, innerWidth - W / 2 - 6), y: Math.max(r.top + H * 0.62 + 6, 96), at: [r.left + r.width / 2, r.top + r.height / 2] };
  }
  function aimArm(from, at, e = 1) {
    const sx = from.x + W / 2 - 14, sy = from.y - H / 2 + H * 0.52;
    const a = (Math.atan2(at[1] - sy, at[0] - sx) * 180) / Math.PI;
    ang.set({ a: clamp(a, -80, 100), e });
  }
  function setLook(at) { if (bot) bot.eyes.lookAt(at ? at[0] : null, at ? at[1] : null); }

  tick(() => {
    // while resting it tracks the dock slot (the pill wobbles and moves); while perched it tracks its element
    if (!cur) {
      const c = slotCenter();
      const g = pos.target;
      if (Math.abs(g.x - c.x) > 0.3 || Math.abs(g.y - c.y) > 0.3) { if (!pos.moving) pos.jump(c); else pos.set(c); }
      return true;
    }
    const t = rectOf(cur.el);
    if (!t || !onScreen(t.r)) return true;
    const sp = perchSpot(t.r);
    const tg = pos.target;
    if (Math.abs(tg.x - sp.x) > 1.5 || Math.abs(tg.y - sp.y) > 1.5) pos.set({ x: sp.x, y: sp.y });
    if (cur.mode === "point") { aimArm(pos.value, sp.at); setLook(sp.at); }
    return true;
  });

  function setMode(m) { root.dataset.mode = m; }
  function arrive(mode, r) {
    flying = false;
    setMode(mode);
    kick(0.7);
    if (bot) { bot.eyes.emote("hop"); }
    dispatchEvent(new CustomEvent("forge:roam", { detail: { phase: "arrived", mode } }));
  }

  function visit(target, o = {}) {
    if (small()) return false;
    const t = rectOf(target);
    if (!t || !onScreen(t.r)) return false;
    const now = performance.now();
    if (cur && now < dwellEnd - 400 && !o.force) { queued = { target, o }; return true; } // let the last stop be seen
    queued = null;
    clearTimeout(homeT);
    cur = { el: t.el, mode: "point" };
    const dwell = o.dwell ?? 5200;
    dwellEnd = now + Math.min(1100, dwell);
    const sp = perchSpot(t.r);
    flying = true;
    setMode("fly");
    pos.config({ stiffness: 70, damping: 15 }).set({ x: sp.x, y: sp.y });
    kick(-0.5);
    ang.set({ a: -20, e: 0.3 });
    // arrive when the spring settles near the spot
    const chk = setInterval(() => {
      const v = pos.value;
      if (Math.hypot(v.x - sp.x, v.y - sp.y) < 14 || !flying) {
        clearInterval(chk);
        if (cur && cur.el === t.el) { arrive("perch"); aimArm(pos.value, sp.at); setLook(sp.at); }
      }
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
    pointAt.t = setTimeout(() => { setLook(null); if (!cur) ang.set({ a: 62, e: 0.35 }); }, 2200);
  }
  function home() {
    clearTimeout(homeT);
    if (queued && performance.now() < dwellEnd + 4000) { const q = queued; queued = null; dwellEnd = 0; visit(q.target, { ...q.o, force: true }); return; }
    if (!cur) return;
    cur = null; flying = true;
    setMode("fly");
    setLook(null);
    ang.set({ a: 62, e: 0.35 });
    pos.config({ stiffness: 90, damping: 16 }).set(slotCenter());
    const chk = setInterval(() => {
      const c = slotCenter(), v = pos.value;
      if (Math.hypot(v.x - c.x, v.y - c.y) < 10) { clearInterval(chk); flying = false; setMode("home"); kick(0.8); dispatchEvent(new CustomEvent("forge:roam", { detail: { phase: "home" } })); }
    }, 80);
  }
  host.addEventListener("click", () => dispatchEvent(new CustomEvent("forge:ask-bot")));
  host.addEventListener("keydown", (e) => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); dispatchEvent(new CustomEvent("forge:ask-bot")); } });
  addEventListener("resize", () => { if (!cur) pos.jump(slotCenter()); });

  pos.jump(slotCenter());
  paint();
  return {
    host, bot, root, visit, home, pointAt, kick,
    get away() { return !!cur; },
    setSlot(s) { slot = s; },
    get position() { const v = pos.value; return { x: v.x, y: v.y }; },
    get reduced() { return reducedMotion(); },
  };
}
