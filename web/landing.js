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
