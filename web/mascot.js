// The forge bot: an original robot drawn for VLA Forge, in the Meta palette. A chamfered head plate in Meta blue, smart
// glasses (dark wayfarer-style lenses with a small camera dot on the corner), two tall eyes in the lenses, and an
// antenna that ends in a small open loop (the thing it learns to use is still a gripper: the loop opens and closes).
// Head ink is currentColor, the eyes are var(--paper) and the glasses are var(--frame), so the same mark works on
// light pages and in a tinted tile. It reads the page's own state (agent rail, rollout badge, ready dot) and shows one
// mood: idle, think (agents working or warming up), ok (a finished rollout) or fail.

const EYES = `<g class="mb-eyes open"><rect x="14.5" y="24" width="6" height="12" rx="3"/><rect x="27.5" y="24" width="6" height="12" rx="3"/></g>
    <g class="mb-eyes happy"><path d="M14 31.5 L17.5 27 L21 31.5"/><path d="M27 31.5 L30.5 27 L34 31.5"/></g>
    <g class="mb-eyes sad"><rect x="14" y="30" width="7" height="4" rx="2"/><rect x="27" y="30" width="7" height="4" rx="2"/></g>`;

export function mascotSVG() {
  return `<svg viewBox="0 0 48 48" aria-hidden="true">
  <g class="mb-body">
    <g class="mb-antenna"><path class="mb-arm" d="M24 16 V10"/><g class="mb-bulb"><path class="mb-loop" d="M24 6.2 C22.2 2.2 16.4 2.2 16.4 6.2 C16.4 10.2 22.2 10.2 24 6.2 C25.8 2.2 31.6 2.2 31.6 6.2 C31.6 10.2 25.8 10.2 24 6.2 Z"/></g></g>
    <path class="mb-head" d="M12 16 H36 L44 24 V37 L37 44 H11 L4 37 V24 Z"/>
    <g class="mb-glasses"><rect class="mb-lens" x="10.5" y="21" width="13.5" height="18" rx="5.5"/><rect class="mb-lens" x="24" y="21" width="13.5" height="18" rx="5.5"/><path class="mb-temple" d="M10.8 25 L4.6 24.4 M37.2 25 L43.4 24.4"/><circle class="mb-cam" cx="36" cy="23.2" r="1.1"/></g>
    <g class="mb-look">${EYES}</g>
  </g>
</svg>`;
}

function mount(host) {
  if (host.querySelector(":scope > svg")) return host;
  host.insertAdjacentHTML("afterbegin", mascotSVG());
  if (!host.dataset.mood) host.dataset.mood = "idle";
  return host;
}
function make(cls) {
  const s = document.createElement("span");
  s.className = `mascot ${cls}`;
  return mount(s);
}

// ---------------------------------------------------------------- global mood
const $ = (s) => document.querySelector(s);
let mood = "idle";
function readMood() {
  if (!$("#ready-dot")?.classList.contains("on")) return "think";
  if (document.querySelector("#rail li.live, .tool.live")) return "think";
  const badge = $("#vp-badge");
  if (badge && !badge.classList.contains("hidden")) {
    if (badge.classList.contains("ok")) return "ok";
    if (badge.classList.contains("bad")) return "fail";
  }
  return "idle";
}
function followers() {
  const forge = document.querySelectorAll(".msg.forge .who .mascot");
  return [...document.querySelectorAll(".mascot[data-follow]"), ...(forge.length ? [forge[forge.length - 1]] : [])];
}
function apply() {
  const next = readMood();
  const changed = next !== mood;
  mood = next;
  document.documentElement.dataset.mood = mood;
  for (const m of followers()) {
    if (m.dataset.mood === mood && !changed) continue;
    m.dataset.mood = mood;
    if (changed && mood === "ok") {
      m.classList.remove("cheer");
      void m.offsetWidth; // restart the cheer when two successes play back to back
      m.classList.add("cheer");
      setTimeout(() => m.classList.remove("cheer"), 1600);
    }
  }
  const boot = $("#boot");
  if (boot && !boot.classList.contains("gone") && $("#ready-dot")?.classList.contains("on")) {
    boot.classList.add("gone");
    setTimeout(() => boot.remove(), 600);
  }
}

// ---------------------------------------------------------------- placements
function decorate(root) {
  root.querySelectorAll?.(".mascot").forEach(mount);
  root.querySelectorAll?.(".msg.forge .who").forEach((w) => {
    if (!w.querySelector(".mascot")) w.prepend(make("xs"));
  });
  // every published skill is its own agent, so each card gets its own little bot
  root.querySelectorAll?.(".skill > .lbl").forEach((l) => {
    if (!l.querySelector(".mascot")) l.prepend(make("xs"));
  });
  root.querySelectorAll?.(".empty-note").forEach((n) => {
    if (!n.querySelector(".mascot")) {
      const m = make("md");
      m.dataset.mood = "idle";
      n.prepend(m);
    }
  });
}

document.querySelectorAll(".mascot").forEach(mount);
decorate(document);
new MutationObserver((recs) => {
  for (const r of recs) for (const n of r.addedNodes) if (n.nodeType === 1) decorate(n.parentElement || n);
  apply();
}).observe(document.body, { subtree: true, childList: true, attributes: true, attributeFilter: ["class"] });
apply();

// ---------------------------------------------------------------- eyes follow the pointer (Etch's --look-x)
let px = 0, py = 0, raf = 0;
addEventListener("pointermove", (e) => {
  px = e.clientX; py = e.clientY;
  if (raf) return;
  raf = requestAnimationFrame(() => {
    raf = 0;
    for (const m of document.querySelectorAll(".mascot")) {
      const r = m.getBoundingClientRect();
      if (!r.width) continue;
      const dx = px - (r.left + r.width / 2), dy = py - (r.top + r.height / 2);
      const d = Math.hypot(dx, dy) || 1, k = Math.min(1, d / 260);
      m.style.setProperty("--look-x", `${((dx / d) * 2.6 * k).toFixed(2)}px`);
      m.style.setProperty("--look-y", `${((dy / d) * 2 * k).toFixed(2)}px`);
    }
  });
}, { passive: true });
