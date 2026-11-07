// Voice for the dock. Etch's adapter (js/ui/voice-adapter.js) sits on Etch's own speech engine and a backend route;
// the forge has neither, so this keeps the same surface (toggle, release, start, stop, say, listening, created) over
// the browser's SpeechRecognition. Hold Space or the mic to talk, click to toggle. It never speaks out loud: answers
// show as captions and bubbles (say() returns false and the dock does that part). Where the browser has no speech
// recognition the mic says so and typing works as before.
export function speakable(md, max = 2) {
  const t = String(md || "").replace(/`([^`]+)`/g, "$1").replace(/\*\*([^*]+)\*\*/g, "$1").replace(/\s+/g, " ").trim();
  const s = t.split(/(?<=[.!?])\s+/).slice(0, max).join(" ");
  return s.length > 380 ? s.slice(0, 377).replace(/\s+\S*$/, "") + "…" : s;
}

export function createVoiceAdapter(h) {
  const SR = window.SpeechRecognition || window.webkitSpeechRecognition;
  let rec = null, listening = false, held = false, created = false, lvT = 0, text = "", pressAt = 0;
  const finish = () => {
    clearInterval(lvT); h.onLevel?.("you", 0);
    const was = listening; listening = false;
    if (!was) return;
    if (text.trim()) { h.onState?.("transcribing"); h.onHeard?.(text.trim()); h.onState?.("idle"); }
    else h.onState?.("idle");
  };
  function start(o = {}) {
    created = true;
    if (!SR) { h.onState?.("off", "This browser has no speech recognition. Type your question instead."); return false; }
    if (listening) return true;
    held = !!o.held; text = "";
    rec = new SR(); rec.lang = "en-US"; rec.interimResults = true; rec.continuous = true;
    rec.onresult = (e) => { text = Array.from(e.results).map((r) => r[0].transcript).join(" "); h.onInterim?.(text); };
    rec.onerror = (e) => { if (e.error === "not-allowed" || e.error === "service-not-allowed") { listening = false; clearInterval(lvT); h.onState?.("off", "The microphone is blocked for this page."); } };
    rec.onend = () => { if (listening) finish(); };
    try { rec.start(); } catch { return false; }
    listening = true;
    h.onState?.("listening", { held });
    lvT = setInterval(() => h.onLevel?.("you", 0.2 + Math.random() * 0.5), 90);
    return true;
  }
  function stop() { if (rec && listening) { try { rec.stop(); } catch { /* already stopped */ } finish(); } }
  return {
    get created() { return created; },
    get listening() { return listening; },
    get held() { return held; },
    get talked() { return false; },
    get available() { return !!SR; },
    toggle() { if (listening) stop(); else { pressAt = performance.now(); start({}); } },
    release() { if (listening && performance.now() - pressAt > 600) stop(); },
    start, stop,
    say() { return false; },
    answer() {},
    destroy() { stop(); },
  };
}
