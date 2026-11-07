"""Cut the live recording into the demo: 4x on the logged waits only (training, part previews), narration placed on top.

    python3 -I scripts/edit_demo.py demo/vla-forge-YYYYMMDD-HHMMSS.webm demo/edit.mp4

Reads <webm>.marks.json (written by record_demo.mjs) and demo/narration/<name>.wav. The landing-page capture the
marks name under "landing" (its own webm, 1x) opens the video and everything else is shifted after it. Every other second of the
recording plays at 1x. Narration cues are given in seconds of the recording, so they follow the speed-up.
BADGE=1 burns demo/speed-badge.png (the page's own 4x badge) onto the sped-up cuts, where the page put it, for
takes recorded while the badge lived inside the Sim view and so never showed on the Line tab.
"""

import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FAST = 4.0

# narration line -> where it starts, as (anchor in marks, offset in output seconds after that anchor)
CUES = [
    ("hook", "zero", 0.5),
    ("describe", "describe", 0.3),
    ("train", "speed0", 0.4),
    ("round2", "mid", 0.0),
    ("trained", "trained", 0.3),
    ("addarm", "addarm", 0.3),
    ("twoarm", "twoarm", 0.3),
    ("armsresult", "armsresult", 0.3),
    ("optimize", "optimize", 0.3),
    ("montage", "montage", 2.3),
    ("ticker", "ticker", 5.0),
    ("replay", "replay", 0.3),
    ("close", "close", 0.5),
]


# idle holds trimmed at the end of a segment, in recording seconds: (anchor in marks, offset after it, length).
# "speed_last_end" is where the last logged wait ends. Narration cues all sit before these.
DROPS = [
    ("trained", 5.5, 1.5),
    ("close", 3.0, 1.0),
]


def has_retry() -> bool:
    return json.loads((ROOT / "demo/v7-results.json").read_text())["retry"]


def main() -> None:
    src, out = Path(sys.argv[1]), Path(sys.argv[2])
    marks = json.loads(src.with_suffix(".marks.json").read_text())
    spans = marks["speed"]
    at = marks["at"]
    last_end = max(b for _a, b in spans)
    drops = [
        ((last_end if k == "speed_last_end" else at[k]) + o, 0.0, ln)
        for k, o, ln in DROPS
    ]
    segs = sorted(
        [(a, b, FAST) for a, b in spans] + [(a, a + ln, 0.0) for a, _z, ln in drops],
        key=lambda x: x[0],
    )

    def to_out(t: float) -> float:
        saved = 0.0
        for a, b, f in segs:
            if f == 1.0:
                continue
            keep = 1 / f if f else 0.0
            saved += (min(t, b) - a) * (1 - keep) if t > a else 0.0
        return t - saved

    anchors = {
        "zero": 0.0,
        "mid": (spans[0][0] + max(b for _a, b in spans if b <= at["trained"] + 0.5))
        / 2,
        **({"ladder": marks["ladder"]} if "ladder" in marks else {}),
        **marks.get("at", {}),
        **{f"speed{i}": a for i, (a, _b) in enumerate(spans)},
        **{f"speed{i}_end": b for i, (_a, b) in enumerate(spans)},
        **{f"start{i}": p["start"] for i, p in enumerate(marks["prompts"])},
        **{f"published{i}": p["published"] for i, p in enumerate(marks["prompts"])},
    }
    # tab capture can open on a frame of a different size before the page settles: size every cut like the
    # frame one second in
    probe = ["ffprobe", "-v", "error", "-select_streams", "v", "-read_intervals", "1%+#1",
             "-show_entries", "frame=width,height", "-of", "csv=p=0", str(src)]  # fmt: skip
    w, h = (
        subprocess.run(probe, check=True, capture_output=True, text=True)
        .stdout.split()[0]
        .split(",")[:2]
    )
    fit = f"scale={w}:{h}:force_original_aspect_ratio=decrease,pad={w}:{h}:(ow-iw)/2:(oh-ih)/2,setsar=1"
    cuts, t = [], 0.0
    for a, b, f in segs:
        cuts.append((t, a, 1.0)) if a > t else None
        if f:
            cuts.append((a, b, f))
        t = b
    cuts.append((t, marks["end"], 1.0))
    landing = ROOT / "demo" / marks["landing"]
    lead = json.loads(landing.with_suffix(".marks.json").read_text())["landing_len"]
    # BADGE: style.css .speed sits fixed at top 64px, right 28px. It goes on the uncut source, lit only inside the
    # logged spans. The PNG loops at 30 fps: a single still frame stops drawing after its own 0.04 s
    badge = os.environ.get("BADGE") == "1"
    li = 2 if badge else 1  # input index of the landing capture
    parts, labels, feed = [], [], ["[0:v]"] * len(cuts)
    if badge:
        on = "+".join(f"between(t,{a:.3f},{b:.3f})" for a, b in spans)
        feed = [f"[i{i}]" for i in range(len(cuts))]
        parts.append(
            f"[0:v][1:v]overlay=W-w-28:64:shortest=1:enable='{on}',split={len(cuts)}{''.join(feed)}"
        )
    for i, (a, b, f) in enumerate(cuts):
        parts.append(
            f"{feed[i]}trim={a:.3f}:{b:.3f},setpts=(PTS-STARTPTS)/{f},fps=30,{fit}[v{i}]"
        )
        labels.append(f"[v{i}]")
    parts.append(f"[{li}:v]trim=0:{lead:.3f},setpts=PTS-STARTPTS,fps=30,{fit}[vl]")
    graph = (
        ";".join(parts) + f";[vl]{''.join(labels)}concat=n={len(cuts) + 1}:v=1:a=0[v]"
    )
    # narration is mixed in its own pass, then muxed: one graph for both lost all but the first half second of
    # audio on a 6-minute capture
    inputs, agraph, amix = [], [], []
    cues = [
        c
        for c in CUES
        if (ROOT / "demo/narration" / f"{c[0]}.wav").exists()
        and (c[0] != "round2" or has_retry())
    ]
    for k, (name, anchor, off) in enumerate(cues):
        at = to_out(anchors[anchor]) + off + (0.0 if name == "hook" else lead)
        inputs += ["-i", str(ROOT / "demo/narration" / f"{name}.wav")]
        ms = int(at * 1000)
        agraph.append(f"[{k}:a]aresample=48000,adelay={ms}|{ms}[a{k}]")
        amix.append(f"[a{k}]")
        print(f"{name:9s} {at:6.1f}s")
    agraph.append(f"{''.join(amix)}amix=inputs={len(amix)}:normalize=0,apad[a]")
    total = lead + to_out(marks["end"])
    mix = out.with_suffix(".narration.wav")
    subprocess.run(
        ["ffmpeg", "-y", "-v", "error", *inputs, "-filter_complex", ";".join(agraph), "-map", "[a]",
         "-t", f"{total:.3f}", "-ac", "2", str(mix)],
        check=True,
    )  # fmt: skip
    cmd = [
        "ffmpeg", "-y", "-v", "error", "-i", str(src),
        *(["-loop", "1", "-framerate", "30", "-i", str(ROOT / "demo/speed-badge.png")] if badge else []),
        "-i", str(landing), "-i", str(mix),
        "-filter_complex", graph, "-map", "[v]", "-map", f"{li + 1}:a", "-t", f"{total:.3f}",
        "-c:v", "libx264", "-preset", "medium", "-crf", "18", "-pix_fmt", "yuv420p",
        "-c:a", "aac", "-b:a", "192k", "-movflags", "+faststart", str(out),
    ]  # fmt: skip
    subprocess.run(cmd, check=True)
    mix.unlink()
    print(f"total {total:.1f}s -> {out}")


if __name__ == "__main__":
    main()
