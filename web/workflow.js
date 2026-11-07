// The run's workflow at the bottom of the stage, the way Etch shows a turn: Etch's own timeline (trace.js timeline(),
// the rows its dock uses) with one live row carrying the pixel loader, a checkmark as each step finishes, finished
// steps folding into one summary line, and the whole run folding when it is published. Driven only by the stage
// rail app.js keeps from the SSE stream; the optimizer has its own view and is left out.
import { el, timeline } from "./trace.js";

// row -> [rail stages it covers, running words, done words]
const ROWS = [
  ["task", ["task"], "Reading the task", "Task parsed"],
  ["plan", ["router", "planner"], "Planning the steps", "Plan ready"],
  ["scene", ["scene"], "Building the scene", "Scene ready"],
  ["data", ["data"], "Collecting demos", "Demos collected"],
  ["train", ["trainer"], "Training the policy", "Policy trained"],
  ["eval", ["eval"], "Testing on held-out scenes", "Measured on held-out scenes"],
  ["critic", ["critic"], "Reading the failures", "Critic done"],
  ["run", ["executor"], "Running the plan", "Plan ran"],
  ["publish", ["registry"], "Saving the skill", "Skill saved"],
];

const rail = document.querySelector("#rail");
const canvas = document.querySelector(".canvas");
if (rail && canvas) {
  const card = el("div", "wf");
  card.dataset.keepclear = "";
  card.setAttribute("aria-label", "What the run is doing");
  card.hidden = true;
  canvas.append(card);
  let tl = null, last = "", hideT = 0, t0 = 0;
  const li = (k) => rail.querySelector(`li[data-stage="${k}"]`);
  const st = (k) => { const n = li(k); return !n || n.classList.contains("off") ? "" : n.classList.contains("live") ? "live" : n.classList.contains("fail") ? "fail" : n.classList.contains("done") ? "done" : ""; };
  const metric = (k) => (li(k)?.querySelector(".m")?.textContent || "").trim();

  function read() {
    const out = [];
    for (const [id, keys, run, done] of ROWS) {
      const s = keys.map(st);
      const status = s.includes("fail") ? "failed" : s.includes("live") ? "running" : s.some((x) => x === "done") ? "done" : null;
      if (!status) continue;
      const m = keys.map(metric).filter(Boolean).join(" · ");
      out.push({ id, kind: "step", status, title: status === "running" ? run : status === "failed" ? `${run.split(" ")[0]} failed` : done, detail: m });
    }
    return out;
  }
  function sync() {
    const rows = read();
    const sig = JSON.stringify(rows);
    if (sig === last) return;
    last = sig;
    const anyLive = rows.some((r) => r.status === "running");
    // a new run starts when the task goes live again, or rows appear after the rail was cleared
    if (!rows.length) { if (tl && !tl.finished) tl.done(); scheduleHide(1500); return; }
    if (!tl || (tl.finished && anyLive && rows.find((r) => r.id === "task")?.status === "running") || (tl.finished && rows.length < tl.rows.size)) {
      tl = timeline({ verb: "Ran for" });
      card.replaceChildren(tl.el);
      t0 = performance.now();
    }
    clearTimeout(hideT);
    card.hidden = false;
    card.classList.remove("gone");
    for (const r of rows) tl.row(r);
    // a row that dropped back to idle is no longer part of this run's story; leave it folded
    const published = rows.find((r) => r.id === "publish")?.status === "done";
    const failedEnd = rows.some((r) => r.status === "failed") && !anyLive;
    if ((published || failedEnd) && !anyLive && !tl.finished) { tl.done((performance.now() - t0) / 1000); scheduleHide(12000); }
  }
  function scheduleHide(ms) {
    clearTimeout(hideT);
    hideT = setTimeout(() => { if (card.matches(":hover") || (tl && !tl.finished)) return; card.classList.add("gone"); setTimeout(() => { if (card.classList.contains("gone")) card.hidden = true; }, 260); }, ms);
  }
  card.addEventListener("pointerleave", () => { if (tl && tl.finished) scheduleHide(4000); });
  // sit above the playbar when it shows
  const bar = document.querySelector("#playbar"), sim = document.querySelector("#view-sim");
  const phone = matchMedia("(max-width: 760px)");
  const place = () => {
    card.classList.toggle("lift", !!(bar && !bar.classList.contains("hidden") && sim && !sim.classList.contains("hidden") && !sim.classList.contains("blank")));
    // phones: the stage rail runs along the bottom of the canvas, so the card sits just above it
    const q = rail.getBoundingClientRect();
    card.style.bottom = phone.matches && q.height ? `${Math.round(canvas.getBoundingClientRect().bottom - q.top + 10)}px` : "";
  };
  addEventListener("resize", place);
  const mo = new MutationObserver(() => { sync(); place(); });
  mo.observe(rail, { subtree: true, attributes: true, attributeFilter: ["class"], childList: true, characterData: true });
  for (const n of [bar, sim]) if (n) mo.observe(n, { attributes: true, attributeFilter: ["class"] });
  sync(); place();
}
