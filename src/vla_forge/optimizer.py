"""The recursive optimizer: a language model improves a trained skill, one change per round, on a fair three-way split.

Scenes come from three disjoint seed sets. TRAIN seeds generate every demonstration. SELECT (100 scenes) is scored each
round and decides keep or revert. TEST (100 scenes) is never shown to the optimizer and is scored only at R0 and on the
final kept policy, with 95% Wilson intervals and a paired comparison (McNemar exact, paired bootstrap) on the same scenes.
A control arm repeats the same number of rounds with a random valid change each round (or no change at all), under the
same keep/revert rule, so a gain can be attributed to the optimizer and not to extra training.

Every round two models propose at the same moment (lanes.both). Muse Wonder (llama.py: Meta's Model API if its key
works, else OpenAI behind the same interface, labelled as such) reads the per-failure evidence and rendered frames and
writes one change from a fixed menu; SubQ Snap (subq.py) reads every past round, every scored scene and the recorded
trajectories, and picks one change from the same menu, enumerated. Both are validated here and may not repeat a change
that was just reverted. Both children are trained from the same parent and scored on the same SELECT scenes; the round
records both, the winner and the backend that really answered. Each SELECT scoring
pass is rendered in Blender (blender_montage.py) from the recorded MuJoCo trajectories. Every round is written to
runs/opt-<id>/run.json.
"""

from __future__ import annotations

import contextlib
import copy
import io
import json
import math
import os
import time
from collections.abc import Callable
from concurrent.futures import as_completed
from dataclasses import dataclass, field
from pathlib import Path
from uuid import uuid4

import numpy as np

from . import blender_montage, lanes, llama, montage, subq
from .datagen import demo_job, pool
from .encoders import device
from .learn import SCOPES, Demos
from .policy import VLA
from .registry import ROOT, Skill
from .rollout import EVAL_SEED0, _goal_xy, evaluate
from .sim.tasks import sample_layout

N_SCENES = int(
    os.environ.get("VLA_FORGE_OPT_SCENES", "100")
)  # fewer only for smoke tests
N_TEST = int(os.environ.get("VLA_FORGE_OPT_TEST_SCENES", "100"))
SEED0 = (
    EVAL_SEED0 + 200_000
)  # SELECT: held-out, never a training seed, never the 20 scenes a skill was published on
TEST_SEED0 = (
    EVAL_SEED0 + 300_000
)  # TEST: a second held-out block the optimizer never sees
START_SEED0 = 50_000  # TRAIN: the demonstrations every fine-tune starts from
ROUND_SEED0 = (
    60_000  # TRAIN: round r draws its extra demonstrations from ROUND_SEED0 + 1000 * r
)
MAX_ROUNDS = 8
CHUNK = 25
KEEP_FRAMES = 32
START_DEMOS = 60  # fresh demonstrations every fine-tune starts from, before any change
RUNS = ROOT / "runs"
DEFAULT_RUN = "064106-9ad6"  # the run the Optimize panel opens on (a running one wins)

DEFAULTS = {
    "lr": 1e-4,
    "train_seconds": 40.0,
    "state_noise": 0.03,
    "replay_frac": 0.3,
    "batch": 256,
    "weight_decay": 1e-4,
    "scope": "all",
}
# knob -> (low, high, unit text); the demo knobs add demonstrations for this round and every later one
KNOBS = {
    "extra_demos_near_failures": (10, 60, "demonstrations at the failed starts"),
    "extra_demos_random": (10, 60, "demonstrations in fresh random scenes"),
    "lr": (5e-5, 6e-4, "learning rate"),
    "train_seconds": (20, 75, "seconds of training per round"),
    "state_noise": (0.0, 0.08, "noise added to the joint readings"),
    "replay_frac": (0.0, 0.7, "share of each batch replayed from the base data"),
    "batch": (64, 512, "batch size (64, 128, 256 or 512)"),
    "weight_decay": (1e-5, 1e-2, "weight decay"),
    "scope": (0, 0, "which layers learn: all, action_layers or perception_layers"),
}
SCHEMA = {
    "type": "object",
    "properties": {
        "diagnosis": {"type": "string"},
        "knob": {"type": "string", "enum": list(KNOBS)},
        "value": {"type": "string"},
        "reason": {"type": "string"},
        "scene_notes": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "scene": {"type": "integer"},
                    "note": {"type": "string"},
                },
                "required": ["scene", "note"],
                "additionalProperties": False,
            },
        },
    },
    "required": ["diagnosis", "knob", "value", "reason", "scene_notes"],
    "additionalProperties": False,
}
SYSTEM = f"""You improve a trained robot policy by changing how it is fine-tuned. The policy is a small vision-language-action
model for a simulated SO-101 arm. It was scored on a fixed set of simulated scenes (SELECT); you read the failures and
propose exactly ONE change to the next fine-tuning round. The change is kept only if success on SELECT rises. A separate
TEST set that you never see decides whether the whole process helped, so do not chase noise on SELECT.

You get, per failed scene: the failure tag, where the target block started (x forward, y left, metres), the goal
position when there is one, and contact statistics (closest gripper approach, highest the block was lifted, how far it
moved, distance to the goal at the end), plus per-tag averages, failure rates by region, the settings now in force, the
full history of earlier changes with their scores, and a picture: rows are failed scenes, columns are early, middle and
late moments of the rollout, rendered in Blender from the recorded trajectory.

The full menu of knobs, with ranges (value is a number written as text, except scope):
{json.dumps({k: {"low": v[0], "high": v[1], "meaning": v[2]} for k, v in KNOBS.items()})}
scope must be one of {list(SCOPES)}. Demonstration counts are whole numbers; batch is 64, 128, 256 or 512.

How to decide: first name the dominant failure pattern and what it says about the policy (is it where the block is,
how the grasp closes, how the block meets the socket, too little training, overfitting to the new data, forgetting).
Then pick the knob that addresses that cause, not the knob that is easiest. Demonstrations are one family of knob among
eight; training length, learning rate, noise, replay, batch size, weight decay and scope act on different causes. If
earlier rounds already tried a knob, say what you learned and move to a different cause when it stopped helping.
Never repeat a change that was reverted (same knob and same value); a rejected list in the input shows what is barred.
diagnosis: two or three plain sentences naming the pattern and the cause. reason: one plain sentence under 140
characters, no dashes, no claims beyond the numbers.
scene_notes: the picture shows the scenes listed in evidence_scenes, in that order from the top row. For each one write
what you see in a single plain sentence under 140 characters (where the block ended up, what the gripper did), using the
scene number from the list. Only describe what the picture and the numbers show."""


def wilson(k: int, n: int, z: float = 1.96) -> list[float]:
    if n == 0:
        return [0.0, 0.0]
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [round(max(0.0, c - h), 3), round(min(1.0, c + h), 3)]


def split(
    n_select: int | None = None,
    n_test: int | None = None,
    max_rounds: int = MAX_ROUNDS,
) -> dict[str, list[int]]:
    """The three scene sets as seed lists. Scene i of a set always has the same seed, so it is the same scene every time."""
    n_select = N_SCENES if n_select is None else n_select
    n_test = N_TEST if n_test is None else n_test
    train = list(range(START_SEED0, START_SEED0 + START_DEMOS)) + [
        ROUND_SEED0 + 1000 * r + i
        for r in range(1, max_rounds + 1)
        for i in range(KNOBS["extra_demos_near_failures"][1])
    ]
    sets = {
        "train": train,
        "select": list(range(SEED0, SEED0 + n_select)),
        "test": list(range(TEST_SEED0, TEST_SEED0 + n_test)),
    }
    check_disjoint(sets)
    return sets


def check_disjoint(sets: dict[str, list[int]]) -> None:
    names = list(sets)
    for i, a in enumerate(names):
        if len(set(sets[a])) != len(sets[a]):
            raise ValueError(f"{a} repeats a seed")
        for b in names[i + 1 :]:
            both = set(sets[a]) & set(sets[b])
            if both:
                raise ValueError(f"{a} and {b} share {len(both)} seeds")


def mcnemar_exact(before: list[int], after: list[int]) -> dict:
    """Paired comparison on the same scenes: b = fixed (0 -> 1), c = broken (1 -> 0); two-sided exact binomial p."""
    b = sum(1 for x, y in zip(before, after, strict=True) if not x and y)
    c = sum(1 for x, y in zip(before, after, strict=True) if x and not y)
    n = b + c
    if n == 0:
        return {"fixed": b, "broken": c, "p": 1.0}
    k = min(b, c)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / 2**n
    return {"fixed": b, "broken": c, "p": round(min(1.0, 2 * tail), 4)}


def paired_bootstrap(
    before: list[int], after: list[int], reps: int = 10_000, seed: int = 0
) -> list[float]:
    """95% interval of the change in success rate, resampling scenes (pairs stay together)."""
    d = np.asarray(after, float) - np.asarray(before, float)
    rng = np.random.default_rng(seed)
    means = d[rng.integers(0, len(d), (reps, len(d)))].mean(axis=1)
    lo, hi = np.quantile(means, [0.025, 0.975])
    return [round(float(lo), 3), round(float(hi), 3)]


def compare(before: list[int], after: list[int]) -> dict:
    n = len(before)
    kb, ka = sum(before), sum(after)
    return {
        "n": n,
        "before": {"k": kb, "rate": round(kb / n, 3), "ci": wilson(kb, n)},
        "after": {"k": ka, "rate": round(ka / n, 3), "ci": wilson(ka, n)},
        "delta": round((ka - kb) / n, 3),
        "delta_ci": paired_bootstrap(before, after),
        "mcnemar": mcnemar_exact(before, after),
    }


def clean_change(raw: dict) -> dict:
    """Validate a proposed change: a known knob, a value clamped into its range, a short plain reason."""
    knob = raw.get("knob")
    if knob not in KNOBS:
        raise ValueError(f"unknown knob {knob!r}")
    val = str(raw.get("value", "")).strip().lower()
    lo, hi, _ = KNOBS[knob]
    if knob == "scope":
        if val not in SCOPES:
            raise ValueError(f"unknown scope {val!r}")
        v: float | str = val
    else:
        x = float(val)
        if knob == "batch":
            v = min((64, 128, 256, 512), key=lambda b: abs(b - x))
        elif knob.startswith("extra_demos") or knob == "train_seconds":
            v = int(min(max(round(x), lo), hi))
        else:
            v = float(min(max(x, lo), hi))
    reason = " ".join(str(raw.get("reason", "")).replace("—", ",").split())[:140]
    out = {"knob": knob, "value": v, "reason": reason}
    diag = " ".join(str(raw.get("diagnosis", "")).replace("—", ",").split())[:400]
    if diag:
        out["diagnosis"] = diag
    notes = []
    for n in raw.get("scene_notes") or []:
        try:
            sc, txt = (
                int(n["scene"]),
                " ".join(str(n["note"]).replace("—", ",").split())[:160],
            )
        except (KeyError, TypeError, ValueError):
            continue
        if txt and sc >= 0 and len(notes) < 6:
            notes.append({"scene": sc, "note": txt})
    if notes:
        out["scene_notes"] = notes
    return out


def describe(ch: dict) -> str:
    k, v = ch["knob"], ch["value"]
    if k.startswith("extra_demos"):
        where = "at the failed starts" if k.endswith("failures") else "in fresh scenes"
        return f"add {v} demonstrations {where}"
    if k == "scope":
        return (
            "train all layers"
            if v == "all"
            else f"train only the {v.replace('_', ' ')}"
        )
    return {
        "lr": f"learning rate {v:.2g}",
        "train_seconds": f"train {v} s per round",
        "state_noise": f"joint noise {v:.3f}",
        "replay_frac": f"replay {round(v * 100)}% of each batch from base data",
        "batch": f"batch size {v}",
        "weight_decay": f"weight decay {v:.2g}",
    }[k]


STAT_KEYS = (
    "min_grip_dist",
    "target_max_z",
    "target_moved",
    "final_goal_dist",
    "distractor_lift",
)


def failure_report(scored: list[dict], cfg: dict) -> dict:
    fails = [s for s in scored if not s["success"]]
    tags: dict[str, int] = {}
    for s in fails:
        tags[s["failure"]] = tags.get(s["failure"], 0) + 1
    r = np.array([np.hypot(*s["target_xy"]) for s in scored])
    side = np.array([s["target_xy"][1] for s in scored])
    ok = np.array([s["success"] for s in scored])

    def rate(mask) -> dict:
        n = int(mask.sum())
        return {
            "scenes": n,
            "failure_rate": round(float((~ok[mask]).mean()), 2) if n else None,
        }

    def mean_stats(rows: list[dict]) -> dict:
        return {
            k: round(float(np.mean([x["stats"].get(k, 0.0) for x in rows])), 3)
            for k in STAT_KEYS
        }

    q1, q2 = np.quantile(r, [1 / 3, 2 / 3])
    return {
        "scenes": len(scored),
        "succeeded": int(ok.sum()),
        "failure_tags": tags,
        "per_tag_mean_stats": {
            t: mean_stats([s for s in fails if s["failure"] == t]) for t in tags
        },
        "success_mean_stats": mean_stats([s for s in scored if s["success"]])
        if ok.any()
        else None,
        "failures": [
            {
                "scene": s["index"],
                "tag": s["failure"],
                "target_xy": [round(v, 3) for v in s["target_xy"]],
                "goal_xy": [round(v, 3) for v in s["goal_xy"]]
                if s["goal_xy"]
                else None,
                "steps": s["steps"],
                **{k: round(float(s["stats"].get(k, 0.0)), 3) for k in STAT_KEYS},
            }
            for s in fails[:40]
        ],
        "by_distance_from_base": {
            "near": rate(r <= q1),
            "middle": rate((r > q1) & (r <= q2)),
            "far": rate(r > q2),
        },
        "by_side": {
            "left": rate(side > 0.04),
            "centre": rate(np.abs(side) <= 0.04),
            "right": rate(side < -0.04),
        },
        "median_closest_gripper_distance_m_in_failures": round(
            float(np.median([s["stats"]["min_grip_dist"] for s in fails])), 3
        )
        if fails
        else None,
        "current_settings": cfg,
    }


def pick_evidence(scored: list[dict], rnd: int, rows: int = 4) -> list[int]:
    """Which failed scenes the model is shown this round: evenly spread over the failures, shifted each round so that
    different scenes get looked at (and described) over a run."""
    fails = [s["index"] for s in scored if not s["success"]]
    if len(fails) <= rows:
        return fails
    step = len(fails) / rows
    return sorted({fails[int(step * k + rnd) % len(fails)] for k in range(rows)})


def evidence_png(sheets: Path, fails: list[int]) -> bytes | None:
    """Rendered frames of the chosen failures (early, middle, late moment), cropped from a round's Blender sheets."""
    from PIL import Image

    cols = (2, 4, 7)
    if not fails or not all((sheets / f"m{k}.jpg").exists() for k in cols):
        return None
    tw, th = montage.TILE_W, montage.TILE_H
    sh = {k: Image.open(sheets / f"m{k}.jpg") for k in cols}
    out = Image.new("RGB", (len(cols) * tw * 2, len(fails) * th * 2))
    for ri, i in enumerate(fails):
        x, y = (i % montage.COLS) * tw, (i // montage.COLS) * th
        for ci, k in enumerate(cols):
            tile = sh[k].crop((x, y, x + tw, y + th)).resize((tw * 2, th * 2))
            out.paste(tile, (ci * tw * 2, ri * th * 2))
    buf = io.BytesIO()
    out.save(buf, "PNG")
    return buf.getvalue()


def _same(a: dict, b: dict) -> bool:
    if a["knob"] != b["knob"]:
        return False
    if isinstance(a["value"], str) or isinstance(b["value"], str):
        return a["value"] == b["value"]
    return math.isclose(float(a["value"]), float(b["value"]), rel_tol=1e-6)


def barred(history: list[dict]) -> list[dict]:
    """Changes that were tried and reverted: the model may not propose these again."""
    return [
        {"knob": h["change"]["knob"], "value": h["change"]["value"]}
        for h in history
        if not h["kept"]
    ]


FALLBACK = [
    {"knob": "extra_demos_near_failures", "value": "30"},
    {"knob": "lr", "value": "0.0001"},
    {"knob": "train_seconds", "value": "60"},
    {"knob": "replay_frac", "value": "0.3"},
    {"knob": "state_noise", "value": "0.05"},
    {"knob": "scope", "value": "action_layers"},
    {"knob": "weight_decay", "value": "0.001"},
    {"knob": "batch", "value": "128"},
]


def fallback_change(rnd: int, history: list[dict], why: str) -> dict:
    ban = barred(history)
    for k in range(len(FALLBACK)):
        pick = clean_change(
            {**FALLBACK[(rnd - 1 + k) % len(FALLBACK)], "reason": f"fixed menu: {why}"}
        )
        if not any(_same(pick, b) for b in ban):
            return pick
    return pick


def random_change(rng: np.random.Generator, history: list[dict]) -> dict:
    """The control arm's change: a random valid knob and value, never a reverted repeat."""
    ban = barred(history)
    for _ in range(50):
        knob = str(rng.choice(list(KNOBS)))
        lo, hi, _ = KNOBS[knob]
        if knob == "scope":
            val = str(rng.choice(list(SCOPES)))
        elif knob in ("lr", "weight_decay"):
            val = str(float(np.exp(rng.uniform(np.log(lo), np.log(hi)))))
        else:
            val = str(float(rng.uniform(lo, hi)))
        ch = clean_change(
            {"knob": knob, "value": val, "reason": "control: random valid change"}
        )
        if not any(_same(ch, b) for b in ban):
            return ch
    return ch


def propose(
    report: dict,
    history: list[dict],
    rnd: int,
    images: list[bytes] | None = None,
    extra: dict | None = None,
) -> tuple[dict, str]:
    """One change from the language model, validated. Returns (change, backend that answered; "rules" if none did).

    A change identical to one that was reverted is sent back once with the barred list; a second repeat is dropped for
    the fixed menu."""
    ban = barred(history)
    payload = {
        "round": rnd,
        "failure_report": report,
        "history": history,
        "rejected": ban,
        "pictures": "rows are failed scenes, columns are early, middle and late moments"
        if images
        else "none this round",
        **(extra or {}),
    }
    why = ""
    for attempt in range(2):
        try:
            raw, _model, backend = llama.chat_json_ex(
                "optimizer", SYSTEM, json.dumps(payload), SCHEMA, images
            )
            ch = clean_change(raw)
        except (RuntimeError, ValueError, TypeError) as e:
            why = f"no model answered: {e}"
            break
        if not any(_same(ch, b) for b in ban):
            return ch, backend
        why = "the model repeated a reverted change"
        payload["rejected_this_round"] = {"knob": ch["knob"], "value": ch["value"]}
    return fallback_change(rnd, history, why), "rules"


# SubQ picks from the menu enumerated on this grid (each value inside its knob's range)
GRID = {
    "extra_demos_near_failures": (10, 20, 30, 40, 50, 60),
    "extra_demos_random": (10, 30, 60),
    "lr": (5e-5, 1e-4, 2e-4, 4e-4, 6e-4),
    "train_seconds": (20, 40, 60, 75),
    "state_noise": (0.0, 0.02, 0.05, 0.08),
    "replay_frac": (0.0, 0.2, 0.4, 0.7),
    "batch": (64, 128, 256, 512),
    "weight_decay": (1e-5, 1e-4, 1e-3, 1e-2),
    "scope": tuple(SCOPES),
}
SUBQ_ASK = (
    "Pick the one change to how the policy is fine-tuned that will most raise success on the same SELECT scenes next "
    "round. The state holds every past round (its change, its SELECT score, kept or reverted), the current parent's "
    "settings and failure report, every scored scene of every round, and the recorded joint and object trajectories."
)


def menu(cfg: dict, history: list[dict]) -> dict[str, dict]:
    """Every valid change on the grid, minus reverted ones and minus no-ops on the parent's settings."""
    ban = barred(history)
    out: dict[str, dict] = {}
    for knob, vals in GRID.items():
        for v in vals:
            ch = clean_change({"knob": knob, "value": str(v), "reason": ""})
            if any(_same(ch, b) for b in ban):
                continue
            noop = knob in cfg and not knob.startswith("extra_demos")
            if noop and _same(ch, {"knob": knob, "value": cfg[knob]}):
                continue
            out[f"opt_{len(out)}"] = ch
    return out


def subq_state(
    report: dict, cfg: dict, rounds: list[dict], scenes: dict, traj: dict
) -> dict:
    """What SubQ reads, most decisive first (subq.fit cuts from the end when it is too long)."""
    return {
        "parent_settings": cfg,
        "parent_failure_report": report,
        "rounds": rounds,
        "knobs": {k: KNOBS[k][2] for k in KNOBS},
        "scenes_by_round": scenes,
        "trajectories": traj,
    }


def subq_change(state: dict, cfg: dict, history: list[dict]) -> tuple[dict, float]:
    """SubQ's change: a `choice` over the enumerated menu. Returns (change, its probability). Raises subq.Unavailable."""
    opts = menu(cfg, history)
    crit = {name: describe(ch) for name, ch in opts.items()}
    ans = subq.decide(
        state,
        {"change": {"type": "choice", "instructions": SUBQ_ASK, "criteria": crit}},
        "optimizer",
    )
    pick, p = subq.choice(ans["change"])
    ch = dict(opts[pick])
    probs = sorted(
        (ans["change"].get("probabilities") or {}).items(), key=lambda kv: -kv[1]
    )
    ch["reason"] = f"picked with p {p:.2f} of {len(opts)} options"
    ch["runners_up"] = [
        {"text": crit[n], "p": round(float(q), 3)} for n, q in probs[1:4] if n in crit
    ]
    return ch, p


def propose_pair(
    report: dict,
    history: list[dict],
    rnd: int,
    images: list[bytes] | None,
    extra: dict | None,
    state: dict,
    cfg: dict,
    run: str = "",
) -> tuple[tuple[dict, str], tuple[dict | None, float | None, str]]:
    """Muse Wonder and SubQ at the same moment. ((meta change, backend), (subq change or None, p, error))."""
    with lanes.about(run, f"round {rnd} proposal"):
        m, q = lanes.both(
            lambda: propose(report, history, rnd, images, extra),
            lambda: subq_change(state, cfg, history),
        )
    if isinstance(
        m, Exception
    ):  # propose never raises for model errors; this is a bug, surface it
        raise m
    if isinstance(q, Exception):
        return m, (None, None, str(q) or type(q).__name__)
    return m, (q[0], q[1], "")


def same_change(a: dict | None, b: dict | None) -> bool:
    return bool(a and b) and _same(a, b)


def saved_duels(limit: int = 40) -> list[dict]:
    """Head to head rows from saved optimizer runs, newest run first, for the Muse + SubQ view after a restart."""
    out: list[dict] = []
    for d in sorted(RUNS.glob("opt-*/run.json"), key=lambda p: -p.stat().st_mtime):
        try:
            rows = json.loads(d.read_text()).get("rounds") or []
        except (OSError, ValueError):
            continue
        out += [r["duel"] for r in reversed(rows) if r.get("duel")]
        if len(out) >= limit:
            break
    return list(reversed(out[:limit]))


@dataclass
class Opt:
    id: str
    skill: dict
    n_rounds: int
    started: float = field(default_factory=time.time)
    session_start: float = field(default_factory=time.time, repr=False)
    status: str = "starting"
    phase: str = ""
    llm: dict = field(default_factory=dict)
    rounds: list = field(default_factory=list)
    best_round: int = 0
    hero: dict | None = None
    error: str = ""
    n_scenes: int = N_SCENES
    seconds: float = 0.0
    split: dict = field(default_factory=dict)
    test: dict = field(default_factory=dict)
    control: dict = field(default_factory=dict)
    blender: dict = field(default_factory=dict)
    mode: str = "bounded"  # "perpetual" runs until stopped
    engine: dict = field(
        default_factory=dict
    )  # the framework and its population, see optimizer_evo
    prior_seconds: float = 0.0  # time spent in earlier sessions of a resumed run

    def to_json(self) -> dict:
        d = {k: v for k, v in self.__dict__.items() if k != "session_start"}
        live = self.status in ("starting", "running")
        d["seconds"] = round(
            (time.time() - self.session_start + self.prior_seconds)
            if live
            else self.seconds,
            1,
        )
        return d


def _save(o: Opt) -> None:
    d = RUNS / f"opt-{o.id}"
    d.mkdir(parents=True, exist_ok=True)
    tmp = d / "run.json.tmp"
    tmp.write_text(json.dumps(o.to_json(), indent=1))
    tmp.replace(d / "run.json")


def _compact(r, nbody_frames: int = KEEP_FRAMES) -> dict:
    idx = montage.moment_index(len(r.frames), nbody_frames)
    return {
        "index": r.index,
        "success": bool(r.success),
        "failure": r.failure,
        "steps": r.steps,
        "stats": r.stats,
        "layout": r.layout,
        "target_xy": [float(v) for v in r.layout["blocks"]["target"][1:3]],
        "goal_xy": _goal_xy(r.layout),
        "frames": np.array([r.frames[i] for i in idx], np.float32),
    }


def score(
    model: VLA,
    spec,
    layouts: list[dict],
    robot: str,
    progress: Callable[[int], None],
    first: int = 0,
    traj: dict[int, np.ndarray] | None = None,
) -> list[dict]:
    """The same scenes every time: scene i always has seed SEED0 + i (so the same lighting and the same layout)."""
    out: list[dict] = []
    for c in range(0, len(layouts), CHUNK):
        part = layouts[c : c + CHUNK]
        rolls = evaluate(
            model,
            spec,
            len(part),
            seed0=SEED0 + first + c,
            layouts=part,
            record=True,
            robot=robot,
        )
        for r in rolls:
            r.index += c
            if traj is not None:
                traj[r.index] = np.asarray(r.frames, np.float32)
            out.append(_compact(r))
        progress(len(out))
    return out


def collect(
    spec,
    n: int,
    near: list[dict] | None,
    seed_base: int,
    tmp: Path,
    robot: str,
    data: Demos,
) -> int:
    jobs = []
    for i in range(n):
        j = {
            "spec": spec.to_json(),
            "seed": seed_base + i,
            "out": str(tmp / f"ep{seed_base + i}.npz"),
            "noise": 0.05,
            "robot": robot,
        }
        if near:
            j["near"] = near[i % len(near)]
        jobs.append(j)
    ok = 0
    p = pool()
    for f in as_completed([p.submit(demo_job, j) for j in jobs]):
        r = f.result()
        if r["ok"]:
            data.add_episode(r["path"], spec.instruction)
            Path(r["path"]).unlink()
            ok += 1
    return ok


def eligible(skills: dict[str, Skill]) -> list[Skill]:
    """Single-arm skills with headroom: the optimizer can only help where the policy still fails."""
    return sorted(
        (
            s
            for s in skills.values()
            if not s.composite
            and s.robot == "so101"
            and s.model
            and s.success_rate < 0.9
        ),
        key=lambda s: s.success_rate,
    )


def score_outcomes(
    model: VLA, spec, layouts: list[dict], robot: str, seed0: int
) -> list[int]:
    """Success flags only (no recording): how TEST is scored."""
    out: list[int] = []
    for c in range(0, len(layouts), CHUNK):
        part = layouts[c : c + CHUNK]
        rolls = evaluate(
            model,
            spec,
            len(part),
            seed0=seed0 + c,
            layouts=part,
            record=False,
            robot=robot,
        )
        out += [int(r.success) for r in rolls]
    return out


def _copy_demos(data: Demos) -> Demos:
    d = copy.copy(data)
    d.vis, d.states, d.actions = list(data.vis), list(data.states), list(data.actions)
    d.aux, d.lang_idx = list(data.aux), list(data.lang_idx)
    d.instructions, d._packed = list(data.instructions), None
    return d


def run(
    skill: Skill,
    n_rounds: int,
    opt: Opt,
    replay: Demos | None,
    compute: contextlib.AbstractContextManager | None = None,
    emit: Callable[..., None] | None = None,
    train_seconds: float = DEFAULTS["train_seconds"],
    control: str | None = "random",
    **kw,
) -> Opt:
    """Run the optimizer on `skill` for `n_rounds` rounds (0 or less: until stopped), or continue a saved run.

    The loop itself lives in optimizer_evo.py (population database from OpenEvolve, the model as the proposal operator);
    the call signature is the one the web API and the Fetch agent have always used. Extra keywords: test_every,
    round_budget_s, resume (see optimizer_evo.run)."""
    from . import optimizer_evo

    return optimizer_evo.run(
        skill, n_rounds, opt, replay, compute, emit, train_seconds, control, **kw
    )


def _hero(
    opt: Opt,
    stage: montage.Stage,
    scored: dict[int, list[dict]],
    out: Path,
    blender: bool = False,
    bl: dict | None = None,
) -> None:
    """One scene, the first try against the last kept policy: prefer a scene that failed then and passes now."""
    a, b = scored[0], scored[opt.best_round]
    pick = next(
        (i for i in range(N_SCENES) if not a[i]["success"] and b[i]["success"]), None
    )
    if pick is None:
        pick = next((i for i in range(N_SCENES) if not a[i]["success"]), 0)
    for sc, name in ((a[pick], "hero_r0.jpg"), (b[pick], "hero_final.jpg")):
        if blender:
            st = blender_montage.render_hero(stage, sc, out / name)
            if bl is not None:
                bl["renders"] += st["renders"]
                bl["seconds"] = round(bl["seconds"] + st["seconds"], 1)
                bl["hero_renders"] = bl.get("hero_renders", 0) + st["renders"]
        else:
            montage.write_hero(stage, sc, out / name)
    opt.hero = {
        "scene": pick,
        "round_a": 0,
        "round_b": opt.best_round,
        "frames": montage.HERO_FRAMES,
        "w": montage.HERO_W,
        "h": montage.HERO_H,
        "a_success": a[pick]["success"],
        "b_success": b[pick]["success"],
        "a_failure": a[pick]["failure"],
        "b_failure": b[pick]["failure"],
        "engine": "blender-eevee" if blender else "mujoco",
    }


def load_opt(oid: str) -> Opt:
    """A saved run as an Opt again, to continue it."""
    d = load_run(oid)
    if d is None:
        raise ValueError(f"no run {oid}")
    names = Opt.__dataclass_fields__
    o = Opt(**{k: v for k, v in d.items() if k in names and k != "session_start"})
    o.prior_seconds = float(d.get("seconds", 0.0))
    return o


def new_opt(skill: Skill, n_rounds: int) -> Opt:
    return Opt(
        mode="perpetual" if n_rounds <= 0 else "bounded",
        id=time.strftime("%H%M%S") + "-" + uuid4().hex[:4],
        skill={
            "id": skill.id,
            "instruction": skill.instruction,
            "success_rate": skill.success_rate,
            "n_eval": skill.n_eval,
        },
        n_rounds=n_rounds,
    )


def list_runs() -> list[dict]:
    out = []
    for p in RUNS.glob("opt-*/run.json"):
        d = json.loads(p.read_text())
        out.append(
            {
                k: d.get(k)
                for k in ("id", "skill", "status", "n_rounds", "started", "best_round")
            }
            | {"default": d.get("id") == DEFAULT_RUN}
            | {
                "rounds_done": sum(
                    1 for r in d["rounds"] if r.get("success") is not None
                )
            }
        )
    return sorted(out, key=lambda r: (not r["default"], -(r["started"] or 0)))


def load_run(oid: str) -> dict | None:
    p = RUNS / f"opt-{oid}" / "run.json"
    return (
        json.loads(p.read_text())
        if p.exists() and oid.replace("-", "").isalnum()
        else None
    )


def asset(oid: str, rel: str) -> Path | None:
    base = (RUNS / f"opt-{oid}").resolve()
    p = (base / rel).resolve()
    ok = (
        oid.replace("-", "").isalnum()
        and p.is_file()
        and base in p.parents
        and p.suffix == ".jpg"
    )
    return p if ok else None


def rerender_hero(oid: str) -> dict:
    """Replay the saved run's hero scene again (the first-try skill and final.pt) and rewrite the two filmstrips."""
    out = RUNS / f"opt-{oid}"
    d = json.loads((out / "run.json").read_text())
    from .registry import Registry

    sk = Registry().skills[d["skill"]["id"]]
    spec, dev = sk.task(), device()
    i = d["hero"]["scene"]
    lay = sample_layout(
        spec, np.random.default_rng(SEED0 + i), robot=sk.robot
    ).to_json()
    stage = montage.Stage(sk.robot)
    res = {}
    for tag, path in (("a", ROOT / sk.model), ("b", out / "final.pt")):
        m, _ = VLA.load(path, dev)
        r = score(m, spec, [lay], sk.robot, lambda n: None, first=i)[0]
        res[tag] = r
        montage.write_hero(
            stage, r, out / ("hero_r0.jpg" if tag == "a" else "hero_final.jpg")
        )
    stage.close()
    d["hero"].update(
        a_success=res["a"]["success"],
        b_success=res["b"]["success"],
        a_failure=res["a"]["failure"],
        b_failure=res["b"]["failure"],
    )
    (out / "run.json").write_text(json.dumps(d, indent=1))
    return d["hero"]
