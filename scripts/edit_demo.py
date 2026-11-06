"""Cut the live recording into the demo: 4x on the logged training waits only, narration placed on top.

    python3 -I scripts/edit_demo.py demo/vla-forge-YYYYMMDD-HHMMSS.webm demo/edit.mp4

Reads <webm>.marks.json (written by record_demo.mjs) and demo/narration/<name>.wav. Every other second of the
recording plays at 1x. Narration cues are given in seconds of the recording, so they follow the speed-up.
"""

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FAST = 4.0

# narration line -> where it starts, as (anchor in marks, offset in output seconds after that anchor)
CUES = [
    ("hook", "zero", 0.5),
    ("t1_train", "speed0", 0.5),
    ("t1_done", "simple_done", -0.2),
    ("chain", "chain", 2.5),
    ("far", "far", 2.5),
    ("library", "library", 0.5),
    ("robot", "robot", 3.0),
    ("widowx", "widowx", 2.0),
    ("custom", "custom", 0.5),
    ("close", "ladder", 0.4),
]


def main() -> None:
    src, out = Path(sys.argv[1]), Path(sys.argv[2])
    marks = json.loads(src.with_suffix(".marks.json").read_text())
    spans = marks["speed"]

    def to_out(t: float) -> float:
        saved = 0.0
        for a, b in spans:
            if t >= b:
                saved += (b - a) * (1 - 1 / FAST)
            elif t > a:
                saved += (t - a) * (1 - 1 / FAST)
        return t - saved

    anchors = {
        "zero": 0.0,
        **({"ladder": marks["ladder"]} if "ladder" in marks else {}),
        **marks.get("at", {}),
        **{f"speed{i}": a for i, (a, _b) in enumerate(spans)},
        **{f"speed{i}_end": b for i, (_a, b) in enumerate(spans)},
        **{f"start{i}": p["start"] for i, p in enumerate(marks["prompts"])},
        **{f"published{i}": p["published"] for i, p in enumerate(marks["prompts"])},
    }
    cuts, t = [], 0.0
    for a, b in spans:
        cuts += [(t, a, 1.0), (a, b, FAST)]
        t = b
    cuts.append((t, marks["end"], 1.0))
    parts, labels = [], []
    for i, (a, b, f) in enumerate(cuts):
        parts.append(
            f"[0:v]trim={a:.3f}:{b:.3f},setpts=(PTS-STARTPTS)/{f},fps=30[v{i}]"
        )
        labels.append(f"[v{i}]")
    graph = ";".join(parts) + f";{''.join(labels)}concat=n={len(cuts)}:v=1:a=0[v]"
    inputs, amix = [], []
    for k, (name, anchor, off) in enumerate(CUES, start=1):
        at = to_out(anchors[anchor]) + off
        inputs += ["-i", str(ROOT / "demo/narration" / f"{name}.wav")]
        ms = int(at * 1000)
        graph += f";[{k}:a]aresample=48000,adelay={ms}|{ms}[a{k}]"
        amix.append(f"[a{k}]")
        print(f"{name:9s} {at:6.1f}s")
    graph += f";{''.join(amix)}amix=inputs={len(amix)}:normalize=0,apad[a]"
    total = to_out(marks["end"])
    cmd = [
        "ffmpeg", "-y", "-v", "error", "-i", str(src), *inputs,
        "-filter_complex", graph, "-map", "[v]", "-map", "[a]", "-t", f"{total:.3f}",
        "-c:v", "libx264", "-preset", "medium", "-crf", "18", "-pix_fmt", "yuv420p",
        "-c:a", "aac", "-b:a", "192k", "-ac", "2", "-movflags", "+faststart", str(out),
    ]  # fmt: skip
    subprocess.run(cmd, check=True)
    print(f"total {total:.1f}s -> {out}")


if __name__ == "__main__":
    main()
