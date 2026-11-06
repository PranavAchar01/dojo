// The forge bot: a small violet robot with a white visor and two tall ink eyes.
// It reads the page's own state (agent rail, rollout badge, ready dot) and shows one mood:
// idle, think (agents working or warming up), ok (a successful rollout) or fail.
// Motion is ported from Etch's bot (blink, float, cheer, ripple, thought cloud).

let uid = 0;
export function mascotSVG() {
  const g = `mb-g${++uid}`;
  return `<svg viewBox="0 0 64 64" aria-hidden="true">
  <defs>
    <linearGradient id="${g}" x1="0" y1="0" x2="0.35" y2="1">
      <stop offset="0" stop-color="var(--accent-400)"/><stop offset="1" stop-color="var(--accent-600)"/>
    </linearGradient>
  </defs>
  <g class="mb-body">
    <g class="mb-antenna"><path d="M32 15V7.5"/><circle class="mb-bulb" cx="32" cy="6" r="3.6"/></g>
    <rect class="mb-ear" x="3.5" y="29" width="6" height="13" rx="3"/>
    <rect class="mb-ear" x="54.5" y="29" width="6" height="13" rx="3"/>
    <rect class="mb-head" x="8" y="14" width="48" height="42" rx="17" fill="url(#${g})"/>
    <ellipse class="mb-shine" cx="19" cy="20.5" rx="6" ry="2.6" transform="rotate(-18 19 20.5)"/>
    <rect class="mb-visor" x="14" y="22" width="36" height="27" rx="12"/>
    <g class="mb-look">
      <g class="mb-eyes open"><ellipse cx="25" cy="35.5" rx="3.7" ry="6.2"/><ellipse cx="39" cy="35.5" rx="3.7" ry="6.2"/></g>
      <g class="mb-eyes happy"><path d="M21 37.5q4-6.5 8 0M35 37.5q4-6.5 8 0"/></g>
      <g class="mb-eyes sad"><path d="M21.5 35.5h7M35.5 35.5h7"/></g>
      <ellipse class="mb-cheek" cx="19.5" cy="43" rx="3.2" ry="1.9"/>
      <ellipse class="mb-cheek" cx="44.5" cy="43" rx="3.2" ry="1.9"/>
    </g>
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
