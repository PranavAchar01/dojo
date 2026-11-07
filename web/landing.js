// Landing page behaviour: reveal on scroll, nav border, and a hero video that pauses off screen or when motion is reduced.
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

const video = document.getElementById("hero-video");
if (video && video.play) {
  if (reduced) {
    video.removeAttribute("autoplay");
    video.pause();
  } else {
    new IntersectionObserver(([e]) => {
      if (e.isIntersecting) video.play().catch(() => {});
      else video.pause();
    }, { threshold: 0.2 }).observe(video);
  }
}

// The roaming bot guides the page: it greets, then flies to each section's heading as it scrolls into view and says
// one line about it. It is the same robot as in the app (roam.js through mascot.js). Skipped on phones and with
// reduced motion, where the page reads the same without it.
if (!reduced && innerWidth >= 760) {
  import("/static/mascot.js").then(({ roamer, goTo, goHomeSoon, say }) => {
    const LINES = {
      how: "One sentence goes in. Eight steps hand it down the pipe.",
      agents: "These are the Fetch.ai agents. Llama thinks behind three of them.",
      arms: "Ask for another SO-101 and each arm gets its own sub-task.",
      skills: "Every finished skill is saved and becomes an agent too.",
      robots: "Every part here comes from a real source, with its licence.",
      models: "This is what runs today. Llama plans and critiques.",
    };
    setTimeout(() => { say("Hi, I am the forge bot. I will point things out as you scroll.", 3600); }, 1400);
    let last = "", t = 0;
    const seen = new IntersectionObserver((es) => {
      for (const e of es) {
        if (!e.isIntersecting || e.target.id === last) continue;
        last = e.target.id;
        clearTimeout(t);
        t = setTimeout(() => {
          const h = e.target.querySelector("h2");
          if (h && goTo(h, { dwell: 5200, force: true })) { say(LINES[last], 4200); goHomeSoon(5600); }
        }, 700);
      }
    }, { threshold: 0.45 });
    document.querySelectorAll("section.band").forEach((s) => seen.observe(s));
    roamer.host.addEventListener("click", () => say("Open the app and give the arm a task. Ask me what I am doing there.", 4200));
  });
}
