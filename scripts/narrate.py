"""Demo narration: one gpt-realtime-2.1 take per script line (voice cedar), acted, checked, trimmed, sped up.

Reads demo/narration.tsv (name<TAB>mood<TAB>line). Numbers in a line are {placeholders} filled from the run
measured for this cut (demo/v9-results.json), spelled out as words; the
"round2" line is only voiced when a station really needed a second round. Each take's own transcript must match the line word for
word (case and punctuation aside) or it is voiced again. Writes demo/narration/1x/<name>.wav (as spoken) and
demo/narration/<name>.wav (sped to SPEED), 48 kHz mono. Adapted from meta-sight's narration script.
The key comes from $OPENAI_API_KEY, else the macOS Keychain (OPENAI_API_KEY) and is never written anywhere.

    python3 -I scripts/narrate.py [name ...]
"""

import asyncio
import base64
import hashlib
import json
import os
import re
import subprocess
import sys
import wave
from pathlib import Path

import websockets

ROOT = Path(__file__).resolve().parents[1]
LINES = ROOT / os.environ.get(
    "NARRATION", "demo/narration.tsv"
)  # NARRATION=demo/narration-ui.tsv for the UI cut
NARR = ROOT / "demo/narration"
OUT_1X, OUT = NARR / "1x", NARR
CACHE = NARR / "cache"
MODEL, VOICE, SPEED, TRIES, PARALLEL = "gpt-realtime-2.1", "cedar", 1.75, 4, 6
TRIM = (  # leading and trailing silence off; the reverse pass trims the tail
    "silenceremove=start_periods=1:start_threshold=-45dB:start_silence=0.05,"
    "areverse,silenceremove=start_periods=1:start_threshold=-45dB:start_silence=0.08,areverse"
)
BASE = (
    "You are a voice actor reading one line of narration for a short product demo. Read the user's text "
    "aloud exactly, word for word, and say nothing else: no greeting, no reply, no extra words. "
    "The narrator is Pranav, an 18-year-old builder showing a robot-learning tool he made. "
    "Keep it natural and human, like talking to a friend over his shoulder. Don't overact. "
    "Direction for this line: "
)
MOODS = {
    "hook": "Curious and upbeat, inviting the listener in.",
    "explain": "Clear and relaxed, explaining what is on screen.",
    "wait": "Light and a little amused while the machine works.",
    "proud": "Quietly proud, reporting a result you measured.",
    "honest": "Plain and candid about a limit, no drama.",
    "close": "Warm and confident, wrapping up.",
}


ONES = [
    "zero",
    "one",
    "two",
    "three",
    "four",
    "five",
    "six",
    "seven",
    "eight",
    "nine",
    "ten",
    "eleven",
    "twelve",
    "thirteen",
    "fourteen",
    "fifteen",
    "sixteen",
    "seventeen",
    "eighteen",
    "nineteen",
]
TENS = [
    "_",
    "_",
    "twenty",
    "thirty",
    "forty",
    "fifty",
    "sixty",
    "seventy",
    "eighty",
    "ninety",
]


def words(n: int) -> str:
    if n < 20:
        return ONES[n]
    if n < 100:
        return TENS[n // 10] + (f"-{ONES[n % 10]}" if n % 10 else "")
    h, r = divmod(n, 100)
    return f"{ONES[h]} hundred" + (f" and {words(r)}" if r else "")


def decimal(x: float) -> str:  # 7.2 -> "seven point two"
    whole, frac = f"{x:.1f}".split(".")
    return words(int(whole)) + (f" point {ONES[int(frac)]}" if frac != "0" else "")


def fills() -> dict[str, str] | None:
    f = ROOT / "demo/v9-results.json"
    if not f.exists():
        return None
    r = json.loads(f.read_text())
    sim, arms, opt = r["sim_skill"], r["arms"], r["optimizer"]
    best = opt["rounds"][opt["best_round"]]
    out = {
        "t": words(round(sim["train_seconds"])),
        "s": words(sim["rounds"][-1]["succeeded"]),
        "n": words(sim["n_eval"]),
        "ac": words(arms["completed"]),
        "an": words(arms["tries"]),
        "o0": words(opt["rounds"][0]["success"]),
        "ob": words(best["success"]),
        "h0": words(opt["test"]["r0"]["k"]),
        "hf": words(opt["test"]["final"]["k"]),
        "c0": words(opt["test"]["random_control"]["before"]["k"]),
        "cf": words(opt["test"]["random_control"]["after"]["k"]),
    }
    if r["retry"]:
        first = sim["rounds"][0]
        out |= {"r1": words(first["succeeded"]), "rn": words(first["n"])}
    return out


def norm(s: str) -> str:
    s = s.lower().replace("%", " percent").replace("’", "'")
    s = re.sub(r"(?<=\w)-(?=\w)", "", s)  # ESP-32 and ESP32, keep-or-drop
    return " ".join(re.sub(r"[^a-z0-9' ]+", " ", s).split())


async def take(key: str, text: str, direction: str) -> tuple[bytes, str]:
    url = f"wss://api.openai.com/v1/realtime?model={MODEL}"
    hdr = {"Authorization": "Bearer " + key}
    async with websockets.connect(url, additional_headers=hdr, max_size=None) as ws:
        session = {
            "type": "realtime",
            "output_modalities": ["audio"],
            "audio": {
                "output": {
                    "voice": VOICE,
                    "format": {"type": "audio/pcm", "rate": 24000},
                }
            },
            "instructions": BASE + direction,
        }
        item = {
            "type": "message",
            "role": "user",
            "content": [{"type": "input_text", "text": text}],
        }
        await ws.send(json.dumps({"type": "session.update", "session": session}))
        await ws.send(json.dumps({"type": "conversation.item.create", "item": item}))
        await ws.send(json.dumps({"type": "response.create"}))
        pcm, tr = b"", ""
        async for m in ws:
            e = json.loads(m)
            t = e.get("type", "")
            if t in ("response.output_audio.delta", "response.audio.delta"):
                pcm += base64.b64decode(e["delta"])
            elif t.endswith("audio_transcript.delta"):
                tr += e.get("delta", "")
            elif t == "error":
                raise RuntimeError(e.get("error", {}).get("message", "realtime error"))
            elif t == "response.done":
                break
    return pcm, tr.strip()


def ff(src: Path, dst: Path, af: str) -> None:
    cmd = ["ffmpeg", "-nostdin", "-v", "error", "-y", "-i", str(src), "-af", af]
    subprocess.run(
        cmd + ["-ar", "48000", "-ac", "1", "-c:a", "pcm_s16le", str(dst)], check=True
    )


async def voice(
    key: str, name: str, mood: str, text: str, gate: asyncio.Semaphore
) -> dict:
    direction = MOODS[mood]
    tag = hashlib.sha1(f"{MODEL}|{VOICE}|{direction}|{text}".encode()).hexdigest()[:12]
    raw, meta = CACHE / f"{name}-{tag}.wav", CACHE / f"{name}-{tag}.json"
    if meta.exists():
        info = json.loads(meta.read_text())
    else:
        async with gate:
            for n in range(1, TRIES + 1):
                pcm, tr = await take(key, text, direction)
                info = {"tries": n, "match": norm(tr) == norm(text), "transcript": tr}
                if info["match"]:
                    break
        with wave.open(str(raw), "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(24000)
            w.writeframes(pcm)
        meta.write_text(json.dumps(info))
    ff(raw, OUT_1X / f"{name}.wav", TRIM)
    ff(OUT_1X / f"{name}.wav", OUT / f"{name}.wav", f"atempo={SPEED}")
    print(name, "ok" if info["match"] else "MISMATCH", info["tries"], flush=True)
    return {"name": name, "direction": direction, **info}


def keychain_key() -> str:
    if os.environ.get("OPENAI_API_KEY"):
        return os.environ["OPENAI_API_KEY"]
    return subprocess.run(
        ["security", "find-generic-password", "-s", "OPENAI_API_KEY", "-w"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()


async def main(key: str) -> None:
    only = set(sys.argv[1:])
    for d in (OUT_1X, OUT, CACHE):
        d.mkdir(parents=True, exist_ok=True)
    rows = [r.split("\t", 2) for r in LINES.read_text().splitlines() if r.strip()]
    fill = fills() if "{" in LINES.read_text() else None
    if fill is not None:
        rows = [
            [n, m, t.format(**fill)]
            for n, m, t in rows
            if n != "round2" or "r1" in fill
        ]
    gate = asyncio.Semaphore(PARALLEL)
    jobs = [voice(key, n, m, t, gate) for n, m, t in rows if not only or n in only]
    done = await asyncio.gather(*jobs)
    log = OUT_1X / "takes.json"
    old = json.loads(log.read_text()) if log.exists() else {}
    old.update({d["name"]: d for d in done})
    log.write_text(json.dumps(old, indent=1))


if __name__ == "__main__":
    asyncio.run(main(keychain_key()))
