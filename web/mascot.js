// The forge bot: a little robot built from square tiles (safety-orange shell, hazard stripe, dark visor, two tall white eyes).
// It reads the page's own state (agent rail, rollout badge, ready dot) and shows one mood:
// idle, think (agents working or warming up), ok (a successful rollout) or fail.
// Motion is ported from Etch's bot (blink, float, cheer, ripple, thought cloud).

// Tile grid: 11 x 11 cells of 6 units. Every part of the bot is one or more square tiles.
const CELL = 6, OY = 3;
const tile = (c, r, cls) => `<rect class="t ${cls}" x="${c * CELL}" y="${OY + r * CELL}" width="${CELL}" height="${CELL}"/>`;
const row = (r, c0, c1, cls) => { let o = ""; for (let c = c0; c <= c1; c++) o += tile(c, r, typeof cls === "function" ? cls(c) : cls); return o; };

function buildTiles() {
  let head = "";
  head += row(2, 1, 9, (c) => (c % 2 ? "hz-a" : "hz-k"));            // hard-hat hazard stripe
  for (let r = 3; r <= 6; r++) {                                       // side frame + dark visor
    head += tile(1, r, "o") + tile(9, r, "o");
    head += row(r, 2, 8, (c) => (r === 3 && c < 4 ? "v2" : "v1"));
  }
  head += row(7, 1, 9, "o") + row(8, 1, 9, "od");                      // jaw plate
  const ears = tile(0, 4, "od") + tile(0, 5, "od") + tile(10, 4, "od") + tile(10, 5, "od");
  const body = row(9, 3, 7, "k") + tile(5, 9, "hz-a");                 // shoulders with a lamp tile
  const antenna = `<g class="mb-antenna">${tile(5, 1, "k")}${tile(5, 0, "bulb mb-bulb")}</g>`;
  const eyes = `<g class="mb-eyes open">${tile(3, 4, "e")}${tile(3, 5, "e")}${tile(7, 4, "e")}${tile(7, 5, "e")}</g>` +
    `<g class="mb-eyes happy">${tile(3, 4, "ea")}${tile(3, 5, "ea")}${tile(7, 4, "ea")}${tile(7, 5, "ea")}${tile(2, 6, "o")}${tile(8, 6, "o")}</g>` +
    `<g class="mb-eyes sad">${tile(3, 5, "e")}${tile(7, 5, "e")}</g>`;
  return { head, ears, body, antenna, eyes };
}
const TILES = buildTiles();

export function mascotSVG() {
  return `<svg viewBox="0 0 66 66" shape-rendering="crispEdges" aria-hidden="true">
  <g class="mb-body">
    ${TILES.antenna}${TILES.ears}${TILES.body}
    <g class="mb-head">${TILES.head}</g>
    <g class="mb-look">${TILES.eyes}</g>
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
