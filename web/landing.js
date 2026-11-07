// Landing page behaviour: reveal on scroll, nav border, and the hero rollout loop, which pauses off screen or when motion is reduced.
document.documentElement.classList.add("js");

const reduced = matchMedia("(prefers-reduced-motion: reduce)").matches;
const io = new IntersectionObserver((entries) => {
  for (const e of entries) {
    if (!e.isIntersecting) continue;
    e.target.classList.add("in");
    io.unobserve(e.target);
  }
}, { threshold: 0.12, rootMargin: "0px 0px -6% 0px" });
document.querySelectorAll(".rv, .pipe, .chain").forEach((el) => io.observe(el));
document.querySelectorAll(".pipe li").forEach((li, i) => li.style.setProperty("--i", i));

const nav = document.getElementById("nav");
const onScroll = () => nav.classList.toggle("scrolled", scrollY > 8);
addEventListener("scroll", onScroll, { passive: true });
onScroll();

// The roaming bot guides the page: it greets, then flies to each section's heading as it scrolls into view and says
// one line about it. It is the same robot as in the app (roam.js through mascot.js). Skipped on phones and with
// reduced motion, where the page reads the same without it.
if (!reduced && innerWidth >= 760) {
  import("/static/mascot.js").then(({ roamer, goTo, goHomeSoon, say }) => {
    const LINES = {
      how: "One sentence, eight steps.",
      agents: "Muse Spark and SubQ, in parallel.",
      arms: "Another arm gets its own sub-task.",
      skills: "Each saved skill is an agent too.",
      robots: "Every part has a real source.",
      models: "What runs today. Muse Spark plans.",
    };
    setTimeout(() => { say("Hi, I will point things out.", 3600); }, 1400);
    let last = "", t = 0;
    const seen = new IntersectionObserver((es) => {
      for (const e of es) {
        if (!e.isIntersecting || e.target.id === last) continue;
        last = e.target.id;
        clearTimeout(t);
        t = setTimeout(() => {
          const w = e.target.querySelector(".wrap") || e.target;
          let h = w.querySelector(".tour-pin");
          if (!h) { h = document.createElement("span"); h.className = "tour-pin"; h.style.cssText = "position:absolute;right:8%;top:130px;width:24px;height:24px;pointer-events:none"; w.style.position = "relative"; w.append(h); }
          if (h && goTo(h, { dwell: 5200, force: true })) { say(LINES[last], 4200); goHomeSoon(5600); }
        }, 700);
      }
    }, { threshold: 0.45 });
    document.querySelectorAll("section.band").forEach((s) => seen.observe(s));
    roamer.host.addEventListener("click", () => say("Open the app and give it a task.", 4200));
  });
}

// Every number and picture below comes from the published models (/api/library): success_rate and n_eval from the
// held-out evaluation, train_seconds from the run, the frame from the model's own recorded rollout.
const el = (tag, cls, html) => { const n = document.createElement(tag); if (cls) n.className = cls; if (html != null) n.innerHTML = html; return n; };
const esc = (t) => String(t).replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
const hits = (t) => Math.round(t.success_rate * t.n_eval);
const thumb = (t) => `/api/skills/${encodeURIComponent(t.id)}/thumb.png?v=${encodeURIComponent((t.replay_source || "") + "." + t.thumb_v)}`;

async function fill() {
  let tiles;
  try { tiles = (await (await fetch("/api/library")).json()).tiles; } catch { return; }
  const pub = tiles.filter((t) => t.n_eval > 0 && t.thumb);
  if (!pub.length) return;

  // the proof row: counts and means over every published model
  const evals = pub.reduce((a, t) => a + t.n_eval, 0), ok = pub.reduce((a, t) => a + hits(t), 0);
  const best = pub.reduce((a, t) => (t.success_rate > a.success_rate ? t : a));
  const tiers = new Set(pub.map((t) => t.tier)).size;
  const dl = $("proof-nums");
  dl.innerHTML = [
    ["Published models", `${pub.length}`, ""],
    ["Held-out scenes passed", `${ok}`, ` / ${evals}`],
    ["Best model, held-out scenes", `${hits(best)}`, ` / ${best.n_eval}`],
    ["Task tiers covered", `${tiers}`, ""],
  ].map(([k, v, of]) => `<div><dt>${k}</dt><dd class="num">${v}<span class="of">${of}</span></dd></div>`).join("");
  $("proof-fine").textContent = "Live from this Dojo's library, in simulation. Each model is scored on layouts it never trained on; the best one is \"" + best.instruction + "\".";

  // the hero: a frame from each model's own rollout, one after another
  const hero = pub.filter((t) => t.replay).slice(0, 6);
  const img = $("hero-img");
  if (hero.length && img) {
    let k = 0;
    const show = () => {
      const t = hero[k % hero.length]; k++;
      img.src = thumb(t); img.hidden = false;
      $("hero-task").textContent = t.instruction;
      $("hero-sub").textContent = `${hits(t)} / ${t.n_eval} on held-out scenes, in simulation`;
    };
    show();
    if (!reduced && hero.length > 1) setInterval(show, 4000);
  }

  // the model library: the newest model at each tier
  const list = $("tile-list");
  for (const tier of [1, 2, 3]) {
    for (const t of pub.filter((x) => x.tier === tier).slice(-2).reverse()) {
      const li = el("li", "tile rv in", `<img src="${thumb(t)}" alt="Simulator view of the model: ${esc(t.instruction)}" width="480" height="360" loading="lazy"><div class="tb"><span class="t${tier}">Tier ${tier}</span><h3>${esc(t.instruction)}</h3><p class="res"><b class="num">${hits(t)} / ${t.n_eval}</b> in simulation</p></div>`);
      list.append(li);
    }
  }

  // the line: stations that were trained for a factory cell; the section is hidden when there are none
  const st = pub.filter((t) => t.origin === "station").slice(-3);
  if (st.length) {
    const chain = $("chain");
    st.forEach((t, i) => chain.append(el("article", "st", `<header><span class="idx">0${i + 1}</span><span class="kind vla">VLA</span></header><h3>${esc(t.instruction)}</h3><p>Trained in ${t.train_seconds.toFixed(1)} s.</p><div class="meter"><b style="--w:${Math.round(t.success_rate * 100)}%"></b></div><p class="res"><b class="num">${hits(t)} / ${t.n_eval}</b> on held-out scenes, in simulation</p>`)));
    chain.style.gridTemplateColumns = `repeat(${st.length}, 1fr)`;
    $("arms").hidden = false;
    io.observe(chain);
  }
}
function $(id) { return document.getElementById(id); }
fill();
