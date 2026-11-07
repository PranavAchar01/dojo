// One loading look for the whole page: the mascot looking side to side and blinking, with a short line.
// mascot.js mounts the eyes on any `.mascot[data-loading]` that appears and keeps them in the "waiting" look; under
// prefers-reduced-motion the eyes hold still and only blink.
export function loadingNode(text, { size = "md", layout = "" } = {}) {
  const n = document.createElement("div");
  n.className = `ld ${layout}`.trim();
  n.setAttribute("role", "status");
  const m = document.createElement("span");
  m.className = `mascot ${size}`;
  m.setAttribute("data-loading", "");
  const t = document.createElement("span");
  t.className = "ld-t";
  t.textContent = text;
  n.append(m, t);
  return n;
}
