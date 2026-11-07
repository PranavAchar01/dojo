"""Multi-arm demos in the Library: a finished factory line (several arms working one task) kept as an ordinary card.

The card is a record under skills/runs/line-<line id>.json (records.py) with the arms, what each was and how it
scored, and the line's own end-to-end count. The saved replay stays where the engine put it, in runs/lines/<id>.json
(the per-arm pose frames), and opening the card plays that in the Sim with every arm moving. The thumbnail is
skills/tiles/line-<line id>/thumb.png, shot through the shared stage (scripts/multi_thumbs.mjs).

Nothing here is estimated: a number is either read from the line's saved result and training rows, from the skill
record of an arm that reused a library skill, or absent ("not measured").
"""

from __future__ import annotations

import json
from pathlib import Path

from . import records
from .registry import SKILLS_DIR, TILES_DIR

PREFIX = "line-"
LINES_DIR = SKILLS_DIR.parent / "runs" / "lines"


def card_id(line_id: str) -> str:
    return PREFIX + line_id


def _held_out(st: dict, line: dict, skills: dict) -> dict | None:
    """Held-out score of one trained arm: this line's own training row, else the library skill it reused."""
    for t in line.get("training") or []:
        if t.get("station") == st["index"] and t.get("n_eval"):
            return {
                "rate": t["success_rate"],
                "n": t["n_eval"],
                "k": round(t["success_rate"] * t["n_eval"]),
            }
    sk = skills.get(st.get("skill_id") or "")
    if sk is not None and sk.n_eval:
        return {
            "rate": sk.success_rate,
            "n": sk.n_eval,
            "k": round(sk.success_rate * sk.n_eval),
        }
    return None


def key(line: dict) -> str:
    """One card per prompt and arm line-up: a rerun of the same demo replaces the older card."""
    return (
        line["prompt"]
        + "|"
        + ",".join(f"{s['robot']}/{s['controller']}" for s in line["stations"])
    )


def record_from_line(line: dict, skills: dict) -> dict | None:
    """The card record for a finished line, or None when it never produced a result."""
    res = line.get("result") or {}
    if (
        line.get("status") != "done"
        or not res.get("n_parts")
        or not line.get("stations")
    ):
        return None
    per = {p["station"]: p for p in res.get("per_station") or []}
    arms = []
    for st in line["stations"]:
        vla = st["controller"] == "vla"
        p = per.get(st["index"])
        arms.append(
            {
                "index": st["index"],
                "name": st["name"],
                "robot": st["robot"],
                "controller": "trained VLA" if vla else "scripted",
                "instruction": st.get("instruction") or "",
                "held_out": _held_out(st, line, skills) if vla else None,
                "in_run": {"succeeded": p["succeeded"], "parts": p["parts"]}
                if p
                else None,
            }
        )
    n, done = res["n_parts"], res["completed"]
    return {
        "kind": "multi",
        "line": line["id"],
        "key": key(line),
        "name": line["name"],
        "prompt": line["prompt"],
        "created": line.get("created", 0.0),
        "assembly": bool(line.get("assembly")),
        "arms": arms,
        "measured": {
            "tries": n,
            "completed": done,
            "seed": res.get("seed"),
            "makespan_s": res.get("makespan_s"),
            "source": "line run in simulation",
        },
    }


def register(line: dict, skills: dict) -> str | None:
    """Save the card for a finished line; an older card of the same demo gives way to it. Returns the card id."""
    rec = record_from_line(line, skills)
    if rec is None:
        return None
    for old in cards():
        if (
            old["key"] == rec["key"]
            and old["line"] != rec["line"]
            and old["created"] <= rec["created"]
        ):
            (records.RECORDS / f"{card_id(old['line'])}.json").unlink(missing_ok=True)
    if any(c["key"] == rec["key"] and c["created"] > rec["created"] for c in cards()):
        return None
    records.save(card_id(line["id"]), rec)
    return card_id(line["id"])


def sync(skills: dict, lines_dir: Path = LINES_DIR) -> list[str]:
    """Cards for every finished line on disk (oldest first, so the newest of each demo wins)."""
    made = []
    for f in sorted(
        lines_dir.glob("*.json"),
        key=lambda p: json.loads(p.read_text()).get("created", 0),
    ):
        line = json.loads(f.read_text())
        # a run where no arm finished a part is the build failing, not a demo
        if (line.get("result") or {}).get("completed") and (
            cid := register(line, skills)
        ):
            made.append(cid)
    return made


def cards() -> list[dict]:
    out = []
    for f in (
        sorted(records.RECORDS.glob(f"{PREFIX}*.json"))
        if records.RECORDS.exists()
        else []
    ):
        try:
            rec = json.loads(f.read_text())
        except (OSError, ValueError):
            continue
        if rec.get("kind") == "multi":
            out.append(rec)
    return out


def has_thumb(cid: str) -> bool:
    return (TILES_DIR / cid / "thumb.png").exists()


def tile(rec: dict) -> dict:
    """The card as the library page sees it: the same fields as any tile, plus the arms and the line to replay."""
    m = rec["measured"]
    cid = card_id(rec["line"])
    return {
        "id": cid,
        "name": rec["name"],
        "instruction": rec["prompt"],
        "prompt": rec["prompt"],
        "tier": 3,
        "robot": "multi",
        "origin": "arms",
        "parent": None,
        "success_rate": m["completed"] / m["tries"],
        "n_eval": m["tries"],
        "seconds": 0,
        "train_seconds": 0,
        "created": rec["created"],
        "steps": [],
        "thumb": has_thumb(cid),
        "replay": False,
        "replay_source": "line run",
        "tests": [],
        "agent_address": "",
        "line": rec["line"],
        "arms": rec["arms"],
        "measured": m,
        "assembly": rec["assembly"],
    }


def tiles() -> list[dict]:
    return [tile(r) for r in cards()]
