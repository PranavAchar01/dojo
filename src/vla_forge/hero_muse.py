"""Muse Spark on the build plan: read an instruction image into steps, or revise the step list from a sentence.

Both calls go to Meta's model directly (never the stub), so a reply labelled Muse Spark really is one. The reply is
validated before anything is applied: parts must exist, bricks must not overlap and each must rest on studs. Costs go
through the repo's credit guard (VLA_FORGE_META_CREDITS=1)."""

from __future__ import annotations

from . import hero_bricks as hb
from . import hero_rules as hr
from . import llama

MODEL = "muse-spark-1.3"

STEP_PROPS = {
    "part": {"type": "string", "enum": sorted(hb.PART_FOOTPRINT)},
    "colour": {"type": "string", "enum": sorted(hb.COLOURS)},
    "col": {
        "type": "integer",
        "description": "stud column of the brick's low-x end, 0 = left end of the bridge",
    },
    "row": {
        "type": "integer",
        "description": "stud row of the brick's low-y side, 0 = front",
    },
    "lvl": {
        "type": "integer",
        "description": "plate level of the brick's bottom face; table = 0, a brick is 3 plates tall",
    },
    "yaw": {"type": "integer", "enum": [0, 90], "description": "0 = long side along x"},
}
SCHEMA = {
    "type": "object",
    "properties": {
        "summary": {"type": "string"},
        "steps": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": STEP_PROPS,
                "required": list(STEP_PROPS),
            },
        },
    },
    "required": ["summary", "steps"],
}
GRID = (
    "Grid: x runs left to right along the bridge, y is the depth, one stud is 8 mm, one plate level is 3.2 mm and a "
    "brick is 3 levels tall. A brick's (col, row) is its low-x, low-y stud and `lvl` the level of its bottom face. "
    "Parts: 3003 is a 2x2 brick, 3001 is 2x4, 2456 is 2x6, 3007 is 2x8; yaw 0 puts the long side along x. "
    "List the bricks bottom up, one per step."
)
SYSTEM = (
    "You write brick assembly steps for a robot that builds LEGO models. Use only what you are given. "
    + GRID
)


def available() -> tuple[bool, str]:
    if llama.api_key() is None:
        return False, "no Meta key"
    if not llama.meta_allowed():
        return (
            False,
            "Meta calls are off (VLA_FORGE_META_CREDITS=1 turns them on) or the credit cap is reached",
        )
    return True, ""


def _call(user: str, images: list[bytes] | None = None) -> tuple[dict, dict]:
    ok, why = available()
    if not ok:
        raise RuntimeError(why)
    client = llama.MetaClient(llama.api_key() or "", MODEL)
    req = client.request
    client.request = lambda *a, **k: {
        **req(*a, **k),
        "max_tokens": 16000,
    }  # reasoning tokens count too
    reply = client.chat_json(SYSTEM, user, SCHEMA, images)
    return reply, dict(client.usage)


def validate(rows: list[dict], source: str) -> tuple[tuple[hb.Brick, ...], list[str]]:
    """Rows to bricks (names step1.. in the order given) plus every problem found. Problems mean: do not apply."""
    problems: list[str] = []
    out: list[hb.Brick] = []
    for i, r in enumerate(rows):
        try:
            part, colour = str(r["part"]), str(r["colour"])
            col, row, lvl, yaw = (
                int(r["col"]),
                int(r["row"]),
                int(r["lvl"]),
                int(r.get("yaw", 0)) % 180,
            )
        except (KeyError, TypeError, ValueError):
            problems.append(f"step {i + 1} is missing a field")
            continue
        if part not in hb.PART_FOOTPRINT:
            problems.append(
                f"step {i + 1}: part {part} is not one of {sorted(hb.PART_FOOTPRINT)}"
            )
            continue
        if colour not in hb.COLOURS:
            problems.append(
                f"step {i + 1}: colour {colour} is not one of {sorted(hb.COLOURS)}"
            )
            continue
        if yaw not in (0, 90):
            problems.append(f"step {i + 1}: yaw must be 0 or 90")
            continue
        if not (0 <= lvl <= 40 and -4 <= col <= 24 and -4 <= row <= 24):
            problems.append(f"step {i + 1}: position is off the plate")
            continue
        name = str(r.get("name") or f"step{i + 1}")
        out.append(
            hb.Brick(
                name,
                part,
                colour,
                col,
                row,
                lvl,
                yaw,
                str(r.get("role") or ""),
                str(r.get("source") or source),
            )
        )
    names = [b.name for b in out]
    if len(set(names)) != len(names):
        out = [hb.replace_name(b, f"step{i + 1}") for i, b in enumerate(out)]
    if not out and not problems:
        problems.append("the step list is empty")
    problems += hb.check_assembly(out)
    return tuple(out), problems


def read_image(png: bytes) -> dict:
    reply, usage = _call(
        "This image shows a finished LEGO bridge built from plain bricks: two pillars, a deck across them and possibly "
        "one brick on the deck. Give the steps that rebuild exactly what the image shows. Return JSON.",
        [png],
    )
    return {"reply": reply, "usage": usage, "model": MODEL}


def revise(rows: list[dict], rules: hb.Rules, instruction: str) -> dict:
    """One revision, retried once with the validator's complaints. Returns bricks only if they validated."""
    import json

    base = json.dumps(
        [
            {k: r[k] for k in ("part", "colour", "col", "row", "lvl", "yaw")}
            for r in rows
        ]
    )
    ask = (
        f"Current steps: {base}\nRules in force: gap {rules.gap} studs between pillars, {rules.pillar} bricks per pillar, "
        f"deck {rules.deck} studs long, tier brick {'on' if rules.tier else 'off'}.\n"
        f"The user asks, in their words: {instruction!r}\nReturn the complete revised step list as JSON."
    )
    usage_total = {"prompt_tokens": 0, "completion_tokens": 0}
    problems: list[str] = []
    for _ in range(2):
        reply, usage = _call(ask)
        for k in usage_total:
            usage_total[k] += usage.get(k, 0)
        bricks, problems = validate(reply["steps"], "muse")
        if not problems:
            return {
                "ok": True,
                "summary": reply["summary"],
                "bricks": hb.to_json(bricks),
                "usage": usage_total,
                "model": MODEL,
            }
        ask += (
            "\nYour last answer was refused: "
            + "; ".join(problems)
            + ". Fix those and return the full list again."
        )
    return {"ok": False, "problems": problems, "usage": usage_total, "model": MODEL}


def diff(old: list[dict], new: list[dict]) -> list[str]:
    """Human-readable differences between two step lists, matched by position in build order."""
    key = ("part", "colour", "col", "row", "lvl", "yaw")
    out = []
    for i in range(max(len(old), len(new))):
        a = old[i] if i < len(old) else None
        b = new[i] if i < len(new) else None
        if a is None:
            out.append(
                f"step {i + 1} added: {b['part']} {b['colour']} at col {b['col']}, row {b['row']}, level {b['lvl']}"
            )
        elif b is None:
            out.append(f"step {i + 1} removed: {a['part']} {a['colour']}")
        else:
            ch = [f"{k} {a[k]} to {b[k]}" for k in key if a[k] != b[k]]
            if ch:
                out.append(f"step {i + 1}: " + ", ".join(ch))
    return out


def score(
    read: tuple[hb.Brick, ...], reference: tuple[hb.Brick, ...], rules: hb.Rules
) -> dict:
    """How the model's read compares to the rule-generated build: exact brick matches and the rule checks on the read."""
    want = {(b.part, b.colour, b.col, b.row, b.lvl, b.yaw_deg) for b in reference}
    got = {(b.part, b.colour, b.col, b.row, b.lvl, b.yaw_deg) for b in read}
    results = hr.check(read, rules) if read else []
    return {
        "bricks_expected": len(want),
        "bricks_read": len(read),
        "exact_matches": len(want & got),
        "rules_passed": sum(r.ok for r in results),
        "rules_total": len(results),
        "rules": hr.to_json(results),
    }
