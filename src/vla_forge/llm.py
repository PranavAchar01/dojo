"""Language-model calls: turning a sentence into a TaskSpec, and the critic's plan for the next round.

Every call goes to Llama through llama.py (the Meta Llama API client, with a stub backend until a key works).
Model output is data: every field is validated and clamped before it touches the pipeline.
"""

from __future__ import annotations

import json
import re
import time

from . import llama
from .compose import Step, validate
from .sim.props import KIT_SLOTS
from .sim.scene import BLOCK_COLORS, BOWL_COLORS, PAD_COLORS
from .sim.tasks import PREDICATES, SKILLS, TaskSpec, default_success

MODEL = llama.MODEL


def _chat_json(
    system: str, user: str, schema: dict, kind: str = "task"
) -> tuple[dict, str]:
    """Returns (parsed JSON, which model answered). kind: task, plan, critic or optimizer."""
    return llama.chat_json(kind, system, user, schema)


# ------------------------------------------------------------------ task parsing

TASK_SYSTEM = f"""You turn a request for a robot into a task for a simulated SO-101 arm on a tabletop.
The arm can learn exactly eight skills:
- reach: move the closed gripper down to hover just above a block ("touch", "point at", "go to"). target = null.
- push: push a block across the table onto a taped square. target = square colour, one of {list(PAD_COLORS)}.
- lift: pick a block up and hold it in the air. target = null.
- pick_place: pick a block and put it in a bowl. target = bowl colour, one of {list(BOWL_COLORS)}.
- stack: put one block on top of another block. target = the lower block's colour.
- kit: put a block into one slot of a 2 x 2 kitting tray ("in the front-left slot of the tray"). target = the
  slot, one of {list(KIT_SLOTS)} (front is toward the camera, left is the viewer's left; default "front-left").
- insert: push a block down into a square socket that fits it with a few millimetres to spare ("insert the
  block into the socket", "peg in hole"). target = null.
- rotate: pick a block up, turn it a quarter-turn-ish and seat it in the socket that is turned 45 degrees
  ("rotate it and set it in the diamond socket"). target = null.
Blocks are 3 cm cubes coloured one of {list(BLOCK_COLORS)}.

kind = "single" when the request is one skill: fill `steps` with exactly that one step.
kind = "composite" when the request needs several skills in a row ("sort the blocks into the bowls of the
same colour", "clear the table", "put X in the bowl and push Y onto the square", "build a tower"). Then
`steps` lists 2-3 steps in the order the arm should do them, each one of push / pick_place / stack only (kit, insert and rotate are single tasks).
Composite rules: each block is moved once; each bowl or square receives one block; towers are two blocks.
If the request is vague ("clear the table"), choose concrete colours yourself and say which in `note`.

If the request names an object or colour the scene does not have, choose the closest available one and say so
in `note`. Set feasible=false only when no skill is even close (for example "fold the laundry"), and explain in
`note`.
distractors: how many other blocks share the table in a single task (0-2); use 2 unless the user asks for an
empty table.
checks (single tasks only): extra predicates for the success check from {json.dumps(PREDICATES)};
usually [] or ["undisturbed"] when the user asks not to touch the other blocks.
mentions: every physical object or place the request names, exactly as the user wrote it (phrase), with kind
"block", "bowl", "square", "tray", "socket" or "slot" when it is one of those, else "other" (a cup, the floor, a shelf, the laundry, a
drawer...), and the colour word the user used for it (null if none). Report what the user said here, not the
substitute you picked: a "pink block" stays color "pink" even though the steps use another colour.
"""

TASK_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": [
        "feasible",
        "note",
        "kind",
        "steps",
        "distractors",
        "checks",
        "mentions",
    ],
    "properties": {
        "feasible": {"type": "boolean"},
        "note": {"type": "string"},
        "kind": {"type": "string", "enum": ["single", "composite"]},
        "steps": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["skill", "object_color", "target_color"],
                "properties": {
                    "skill": {"type": "string", "enum": list(SKILLS)},
                    "object_color": {"type": "string", "enum": list(BLOCK_COLORS)},
                    "target_color": {"type": ["string", "null"]},
                },
            },
        },
        "distractors": {"type": "integer"},
        "checks": {
            "type": "array",
            "items": {"type": "string", "enum": list(PREDICATES)},
        },
        "mentions": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["phrase", "kind", "color"],
                "properties": {
                    "phrase": {"type": "string"},
                    "kind": {
                        "type": "string",
                        "enum": [
                            "block",
                            "bowl",
                            "square",
                            "tray",
                            "socket",
                            "slot",
                            "other",
                        ],
                    },
                    "color": {"type": ["string", "null"]},
                },
            },
        },
    },
}


def _valid_target(skill: str, color: str | None) -> str | None:
    table = {
        "pick_place": BOWL_COLORS,
        "push": PAD_COLORS,
        "stack": BLOCK_COLORS,
        "kit": dict.fromkeys(KIT_SLOTS),
    }.get(skill)
    if table is None:
        return None
    if skill == "kit" and isinstance(color, str):
        color = color.lower().replace(" ", "-")
    if color in table:
        return color
    return next(iter(table))


def _clean_success(skill: str, extra: list[str]) -> list[dict]:
    """The skill's own predicates plus the extra checks that make sense for it."""
    out = default_success(skill)
    allowed = {"still", "undisturbed"} if skill != "lift" else {"undisturbed"}
    for op in extra or []:
        q = {"op": op, "block": "target"}
        if op in allowed and q not in out:
            out.append(q)
    return out


def parse_task(prompt: str) -> tuple[TaskSpec | list[Step] | None, dict]:
    """Sentence -> (TaskSpec for one skill, a list of Steps for a composed task, or None if infeasible; info)."""
    t0 = time.time()
    try:
        raw, model = _chat_json(TASK_SYSTEM, prompt, TASK_SCHEMA)
    except RuntimeError as e:
        spec = rule_parse(prompt)
        return spec, {
            "model": "rules",
            "note": f"language model unavailable ({e}); used the keyword parser",
            "kind": "single",
            "motion_word": rule_motion(prompt)[1],
            "raw_steps": 1,
        }
    info = {
        "model": model,
        "seconds": round(time.time() - t0, 2),
        "note": raw.get("note", ""),
        "mentions": [
            {
                "phrase": str(m.get("phrase", ""))[:60],
                "kind": m.get("kind"),
                "color": m.get("color"),
            }
            for m in (raw.get("mentions") or [])[:12]
        ],
    }
    steps_raw = raw.get("steps") or []
    info["raw_steps"] = len(steps_raw)
    if not raw.get("feasible", True) or not steps_raw:
        return None, info
    if raw.get("kind") == "composite" and len(steps_raw) > 1:
        steps, dropped = validate(steps_raw)
        info["dropped"] = dropped
        if dropped:
            info["note"] = (info["note"] + ". " if info["note"] else "") + dropped
        if len(steps) > 1:
            info["kind"] = "composite"
            return steps, info
        steps_raw = [
            s.to_json() | {"object_color": s.obj, "target_color": s.target}
            for s in steps
        ] or steps_raw
    r = steps_raw[0]
    skill = r["skill"] if r.get("skill") in SKILLS else "pick_place"
    obj = r["object_color"] if r.get("object_color") in BLOCK_COLORS else "red"
    target = _valid_target(skill, r.get("target_color"))
    if skill == "stack" and target == obj:
        target = next(c for c in BLOCK_COLORS if c != obj)
    spec = TaskSpec(
        skill=skill,
        obj=obj,
        target=target,
        distractors=int(min(max(raw.get("distractors", 2), 0), 2)),
        success=_clean_success(skill, raw.get("checks", [])),
    )
    info["kind"] = "single"
    return spec, info


MOTION_WORDS = (
    ("stack", ("stack", " on top")),
    ("rotate", ("rotate", "diamond", "twist", "turn it")),
    ("insert", ("insert", "socket", "peg", "plug in")),
    ("kit", ("slot", "tray", "kitting", "kit ")),
    ("push", ("push", "slide", "shove", "nudge")),
    ("pick_place", ("bowl", "put", "place", "drop")),
    ("reach", ("touch", "reach", "point")),
    ("lift", ("lift", "pick up", "raise", "hold")),
)


def rule_motion(prompt: str) -> tuple[str, str | None]:
    """(skill, the word that chose it, or None when nothing in the sentence names a motion: lift is a guess)."""
    p = " " + prompt.lower()
    for skill, words in MOTION_WORDS:
        for w in words:
            if w in p:
                return skill, w.strip()
    return "lift", None


def rule_parse(prompt: str) -> TaskSpec:
    """Keyword fallback when no language model is reachable."""
    p = prompt.lower()
    colors = [
        c
        for c in re.findall(r"[a-z]+", p)
        if c in BLOCK_COLORS or c in BOWL_COLORS or c in PAD_COLORS
    ]
    skill, _word = rule_motion(prompt)
    obj = next((c for c in colors if c in BLOCK_COLORS), "red")
    rest = colors[colors.index(obj) + 1 :] if obj in colors else colors
    target = _valid_target(skill, rest[0] if rest else None)
    if skill == "kit":
        fb = "back" if re.search(r"\b(back|rear|far)\b", p) else "front"
        lr = "right" if re.search(r"\bright\b", p) else "left"
        target = f"{fb}-{lr}"
    if skill == "stack" and target == obj:
        target = next(c for c in BLOCK_COLORS if c != obj)
    return TaskSpec(skill=skill, obj=obj, target=target)


# ------------------------------------------------------------------ critic

CRITIC_SYSTEM = """You are the critic in a loop that trains a vision-language-action policy for a simulated
robot arm. You get the evaluation of the policy that was just trained (success rate, failure modes with the
start positions that failed), the history of earlier rounds, the current training settings and the seconds
left in the run. Decide the next round.
- action "publish" when the policy is good enough (success_rate >= target) or no round fits in the time left.
- action "retrain" otherwise. Choose extra_demos (expert demonstrations to add; seeded at the failed start
  positions when focus_failures is true), train_seconds, learning rate, and domain-randomization strength.
A round costs about 0.35 s per extra demo, train_seconds, and about 20 s of evaluation. Never plan a round
that exceeds the time left. Keep `rationale` to one or two plain sentences about what failed and why the change
should help."""

CRITIC_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": [
        "action",
        "extra_demos",
        "focus_failures",
        "train_seconds",
        "lr",
        "look_strength",
        "rationale",
    ],
    "properties": {
        "action": {"type": "string", "enum": ["publish", "retrain"]},
        "extra_demos": {"type": "integer"},
        "focus_failures": {"type": "boolean"},
        "train_seconds": {"type": "number"},
        "lr": {"type": "number"},
        "look_strength": {"type": "number"},
        "rationale": {"type": "string"},
    },
}


def critic_plan(
    report: dict,
    history: list[dict],
    settings: dict,
    seconds_left: float,
    target: float,
) -> tuple[dict, str]:
    user = json.dumps(
        {
            "evaluation": report,
            "history": history,
            "settings": settings,
            "seconds_left": round(seconds_left),
            "target_success_rate": target,
        }
    )
    try:
        raw, model = _chat_json(CRITIC_SYSTEM, user, CRITIC_SCHEMA, "critic")
    except RuntimeError:
        raw, model = _rule_critic(report, settings, seconds_left, target), "rules"
    plan = {
        "action": raw.get("action")
        if raw.get("action") in ("publish", "retrain")
        else "publish",
        "extra_demos": int(min(max(raw.get("extra_demos", 30), 0), 80)),
        "focus_failures": bool(raw.get("focus_failures", True)),
        "train_seconds": float(min(max(raw.get("train_seconds", 30), 8), 90)),
        "lr": float(min(max(raw.get("lr", 2e-4), 1e-5), 1e-3)),
        "look_strength": float(min(max(raw.get("look_strength", 1.0), 0.3), 1.5)),
        "rationale": str(raw.get("rationale", ""))[:400],
    }
    if report.get("success_rate", 0) >= target:
        plan["action"] = "publish"
    return plan, model


def _rule_critic(
    report: dict, settings: dict, seconds_left: float, target: float
) -> dict:
    retrain = report.get("success_rate", 0) < target and seconds_left > 45
    return {
        "action": "retrain" if retrain else "publish",
        "extra_demos": 30,
        "focus_failures": True,
        "train_seconds": max(10.0, min(30.0, seconds_left - 35)),
        "lr": settings.get("lr", 2e-4),
        "look_strength": settings.get("look_strength", 1.0),
        "rationale": "rule fallback: add demonstrations at the failed starts and keep training",
    }
