"""The router: before anything trains, decide what a prompt needs and say why.

Decisions, in the order they are checked:

  not_feasible  the scene or the arm cannot do it: an object that is not on the table, a motion the arm has no
                expert or base policy for, a point outside the arm's reach, or the parser found no task at all
  too_complex   it can be done in principle but not by one chain: more steps than the planner chains, a tower
                taller than two, or more new skills to train than fit the chain budget
  reuse         a tile in the library already does exactly this, measured at REUSE_MIN or better
  chain         several steps; the planner calls existing skill agents and trains the missing ones
  train         one new skill, fine-tuned from the nearest tile (or the base policy)

Every decision carries the checks it ran (each one ok or not, with what it measured), an estimate of the time
it will take from the library's own history, and for a refusal a suggestion the page can run in one click.
"""

from __future__ import annotations

import json
import math
import re
import statistics
from dataclasses import asdict, dataclass, field
from functools import lru_cache

import numpy as np

from .compose import MAX_STEPS, Step
from .compose import sample_layout as compose_layout
from .registry import BASE_PATH, ROBOT, Registry
from .sim.robots import is_default, robot_feasibility
from .sim.robots import label as robot_label
from .sim.scene import BLOCK_COLORS, BOWL_COLORS, PAD_COLORS, Layout, Scene
from .sim.tasks import BLOCK_ANG, BLOCK_R, SKILLS, TaskSpec, sample_layout

REUSE_MIN = 0.75  # a tile this good is reused instead of trained again
DEFAULT_TRAIN_S = (
    172.0  # the one-prompt budget, used when the library has no history for a motion
)
CHAIN_EXEC_S = 30.0  # running a chain's steps in 20 held-out scenes
CHAIN_BUDGET_S = 400.0  # a chain that needs longer than this is split into tiles first
REACH_LAYOUTS = 4  # sampled scenes the reachability check solves IK in
REACH_TOL = 0.008  # metres: IK residual that still counts as reached

PALETTE = {"block": BLOCK_COLORS, "bowl": BOWL_COLORS, "square": PAD_COLORS}
# words the offline parser recognises as colours or as the scene's objects (the LLM path reports them itself)
COLOR_WORDS = set(BLOCK_COLORS) | set(BOWL_COLORS) | set(PAD_COLORS) | {
    "brown", "grey", "gray", "cyan", "magenta", "violet", "teal", "gold", "silver", "beige", "navy",
    "maroon", "turquoise", "lime", "indigo",
}  # fmt: skip
NOUNS = {
    "block": "block", "blocks": "block", "cube": "block", "cubes": "block", "brick": "block",
    "bowl": "bowl", "bowls": "bowl", "square": "square", "squares": "square", "pad": "square",
    "pads": "square", "tape": "square",
}  # fmt: skip
FILLER = {
    "and",
    "or",
    "one",
    "ones",
    "on",
    "in",
    "into",
    "onto",
    "to",
    "then",
    "with",
    "of",
    "it",
    "up",
    "down",
}
ALWAYS_THERE = ("table", "tabletop", "arm", "robot", "gripper", "itself", "air")
QTY = re.compile(
    r"(\d+(?:\.\d+)?)\s*(mm|millimet(?:er|re)s?|cm|centimet(?:er|re)s?|met(?:er|re)s?|m|inch(?:es)?|in|ft|foot|feet)\b"
)
UP_WORDS = ("up", "high", "above", "height", "lift", "raise", "tall", "air")


@dataclass
class Check:
    name: str
    ok: bool
    detail: str


@dataclass
class Route:
    decision: str
    reason: str
    checks: list[Check] = field(default_factory=list)
    suggestion: str = ""
    estimate_s: float = 0.0
    expected: float | None = None  # success rate the library's history points to
    skill_id: str = ""  # reuse: the tile; remix: the tile it starts from
    steps: list[dict] = field(default_factory=list)  # chain: per step, reuse or train

    @property
    def refused(self) -> bool:
        return self.decision in ("not_feasible", "too_complex")

    def to_json(self) -> dict:
        return {**asdict(self), "refused": self.refused}


# ------------------------------------------------------------------ the arm's workspace


@lru_cache(maxsize=1)
def workspace() -> dict:
    """Measured from the MJCF: how far and how high the jaws get (forward kinematics over the joint limits),
    and how far out they can still point straight down at table height (IK along the arm's forward axis)."""
    from .sim.expert import HOVER_Z, IK

    sc = Scene(Layout())
    ik = IK(sc)
    rng = np.random.default_rng(0)
    pts = np.array([ik.fk(rng.uniform(ik.lo, ik.hi))[0] for _ in range(3000)])
    r_down = 0.0
    for r in np.arange(0.10, 0.50, 0.01):
        _q, err = ik.solve(np.array([r, 0.0, HOVER_Z]), sc.joint_pos()[:5], iters=200)
        if err < REACH_TOL:
            r_down = float(r)
    sc.close()
    return {
        "max_radius": round(float(np.linalg.norm(pts[:, :2], axis=1).max()), 3),
        "max_height": round(float(pts[:, 2].max()), 3),
        "reach_down": round(r_down, 3),
        # the table sector the skills are trained in (sim/tasks.py): its widest chord
        "sector_span": round(2 * BLOCK_R[1] * math.sin(BLOCK_ANG), 3),
    }


def quantities(prompt: str) -> list[tuple[float, str, bool]]:
    """Distances in the sentence: (metres, as written, is it a height)."""
    out = []
    low = prompt.lower()
    for m in QTY.finditer(low):
        u = m.group(2)
        scale = (
            0.001 if u.startswith(("mm", "millimet"))
            else 0.01 if u.startswith(("cm", "centimet"))
            else 0.0254 if u.startswith("in")
            else 0.3048 if u in ("ft", "foot", "feet")
            else 1.0
        )  # fmt: skip
        around = low[max(0, m.start() - 24) : m.end() + 24]
        up = any(re.search(rf"\b{w}\b", around) for w in UP_WORDS)
        out.append((float(m.group(1)) * scale, m.group(0), up))
    return out


def check_quantities(prompt: str) -> Check | None:
    qs = quantities(prompt)
    if not qs:
        return None
    ws = workspace()
    for metres, text, up in qs:
        if up and metres > ws["max_height"]:
            return Check(
                "reach",
                False,
                f"{text} up is out of reach: the SO-101's jaws get at most {ws['max_height'] * 100:.0f} cm above "
                "the table (forward kinematics over its joint limits)",
            )
        if not up and metres > ws["sector_span"]:
            return Check(
                "reach",
                False,
                f"{text} is wider than the arm's workspace: it works {BLOCK_R[0] * 100:.0f}-{BLOCK_R[1] * 100:.0f} cm "
                f"from its base, at most {ws['sector_span'] * 100:.0f} cm across "
                f"(physical reach {ws['max_radius'] * 100:.0f} cm)",
            )
    said = ", ".join(t for _m, t, _u in qs)
    return Check("reach", True, f"{said} fits inside the arm's workspace")


def check_reach(parsed: TaskSpec | list[Step], seed: int = 0) -> Check:
    """Sample the scenes this task would train and test in, and solve IK, jaws pointing down, for the heights
    the scripted expert works at over every object in them: hover over blocks, release over bowls, push height
    over squares."""
    from .sim.expert import HOVER_Z, IK, PUSH_Z, RELEASE_Z

    rng = np.random.default_rng(seed)
    try:
        lays = [
            compose_layout(parsed, rng)
            if isinstance(parsed, list)
            else sample_layout(parsed, rng)
            for _ in range(REACH_LAYOUTS)
        ]
    except RuntimeError:
        return Check(
            "reach",
            False,
            "no table layout can hold every object of this task inside the arm's reach",
        )
    worst, n_obj, missed = 0.0, 0, 0
    sc = Scene(lays[0])
    ik = IK(sc)
    q0 = sc.joint_pos()[:5]
    for lay in lays:
        pts = [(x, y, HOVER_Z) for _c, x, y, _yaw in lay.blocks.values()]
        pts += [(x, y, RELEASE_Z) for _c, x, y in {**lay.bowls, **lay.bins}.values()]
        pts += [(x, y, PUSH_Z) for _c, x, y in lay.pads.values()]
        for x, y, z in pts:
            _q, err = ik.solve(np.array([x, y, z]), q0, iters=150)
            worst = max(worst, err)
            n_obj += 1
            missed += err > REACH_TOL
    sc.close()
    if missed:
        return Check(
            "reach",
            False,
            f"{missed} of {n_obj} object positions in {REACH_LAYOUTS} sampled scenes are out of the jaws' reach",
        )
    return Check(
        "reach",
        True,
        f"IK reaches all {n_obj} object positions in {REACH_LAYOUTS} sampled scenes (worst miss {worst * 1000:.1f} mm)",
    )


# ------------------------------------------------------------------ objects and motions


def lexicon_mentions(prompt: str) -> list[dict]:
    """Offline stand-in for the parser's `mentions`: the scene nouns, plus any coloured thing that is not one
    ("the red mug"), which the object check then reports as missing from the table."""
    toks = re.findall(r"[a-z]+", prompt.lower())
    out = []
    for i, t in enumerate(toks):
        before = toks[max(0, i - 2) : i]
        color = next((w for w in reversed(before) if w in COLOR_WORDS), None)
        if t in NOUNS:
            out.append(
                {"phrase": " ".join([*before, t]), "kind": NOUNS[t], "color": color}
            )
        elif (
            i
            and toks[i - 1] in COLOR_WORDS
            and t not in COLOR_WORDS
            and t not in FILLER
        ):
            out.append(
                {"phrase": f"{toks[i - 1]} {t}", "kind": "other", "color": toks[i - 1]}
            )
    return out


def check_objects(mentions: list[dict]) -> Check:
    missing = []
    for m in mentions:
        phrase = (m.get("phrase") or "").lower()
        kind, color = m.get("kind"), (m.get("color") or "").lower() or None
        if any(w in phrase for w in ALWAYS_THERE):
            continue
        if kind not in PALETTE:
            missing.append(
                f'"{m.get("phrase")}" (the table only has blocks, bowls and taped squares)'
            )
        elif color and color not in PALETTE[kind]:
            missing.append(
                f"a {color} {kind} ({kind}s come in {', '.join(PALETTE[kind])})"
            )
    if missing:
        return Check("objects", False, "not in the scene: " + "; ".join(missing))
    named = ", ".join(m.get("phrase", "") for m in mentions) or "nothing specific"
    return Check("objects", True, f"on the table: {named}")


@lru_cache(maxsize=1)
def base_rates() -> dict[str, tuple[int, int]]:
    """Per motion, the base policy's measured success (models/base.json, written by scripts/pretrain_base.py)."""
    from .llm import rule_motion

    path = BASE_PATH.with_suffix(".json")
    if not path.exists():
        return {}
    out: dict[str, tuple[int, int]] = {}
    for ins, r in json.loads(path.read_text()).get("skills", {}).items():
        out[rule_motion(ins)[0]] = (r["succeeded"], r["n"])
    return out


def check_primitives(parsed: TaskSpec | list[Step], info: dict) -> Check:
    if info.get("model") == "rules" and not info.get("motion_word"):
        return Check(
            "primitives",
            False,
            "no motion this arm knows is named (touch, push, lift, put in a bowl, stack)",
        )
    skills = [s.skill for s in parsed] if isinstance(parsed, list) else [parsed.skill]
    rates = base_rates()
    parts = []
    for sk in dict.fromkeys(skills):
        if sk not in SKILLS:
            return Check(
                "primitives", False, f"no scripted expert or base policy for '{sk}'"
            )
        k, n = rates.get(sk, (0, 0))
        parts.append(f"{sk} (base {k}/{n})" if n else f"{sk} (no base measurement)")
    return Check("primitives", True, "known motions: " + ", ".join(parts))


# ------------------------------------------------------------------ estimates


def train_estimate(registry: Registry, skill: str) -> tuple[float, str]:
    past = [
        s.seconds
        for s in registry.skills.values()
        if not s.composite and s.seconds > 0 and s.task().skill == skill
    ]
    if past:
        return float(
            statistics.median(past)
        ), f"median of {len(past)} earlier {skill} tile(s)"
    return DEFAULT_TRAIN_S, "the one-prompt budget (no earlier tile of this motion)"


def expected_rate(registry: Registry, skill: str) -> float | None:
    rates = [
        s.success_rate
        for s in registry.skills.values()
        if not s.composite and s.task().skill == skill
    ]
    if rates:
        return round(float(statistics.median(rates)), 3)
    k, n = base_rates().get(skill, (0, 0))
    return round(k / n, 3) if n else None


def _suggest(parsed, registry: Registry) -> str:
    if isinstance(parsed, TaskSpec):
        return parsed.instruction
    if isinstance(parsed, list) and parsed:
        return parsed[0].instruction
    best = max(registry.skills.values(), key=lambda s: s.success_rate, default=None)
    return best.instruction if best else "touch the red block"


SKILL_VERB = {
    "reach": "reach a block",
    "push": "push a block",
    "lift": "pick up a block",
    "pick_place": "pick and place a block",
    "stack": "stack blocks",
}


def check_robot(parsed: TaskSpec | list[Step], robot: str) -> Check:
    """Can the active robot do every skill the prompt needs? (sim/robots.py: fingers, reach, torque)"""
    skills = (
        list(dict.fromkeys(s.skill for s in parsed))
        if isinstance(parsed, list)
        else [parsed.skill]
    )
    name = robot_label(robot)
    for sk in skills:
        ok, why = robot_feasibility(robot, sk)
        if not ok:
            return Check("robot", False, f"the {name} cannot {SKILL_VERB[sk]}: {why}")
    return Check("robot", True, f"{name}: {why}")


# ------------------------------------------------------------------ the decision


def route(
    prompt: str,
    parsed: TaskSpec | list[Step] | None,
    info: dict,
    registry: Registry,
    mode: str = "auto",
    from_skill: str = "",
    robot: str = ROBOT,
) -> Route:
    checks: list[Check] = []
    mentions = info.get("mentions")
    if mentions is None:
        mentions = lexicon_mentions(prompt)
    objects = check_objects(mentions)
    checks.append(objects)

    if parsed is None:
        why = (
            info.get("note") or "no tabletop task this arm can learn is in the sentence"
        )
        return Route("not_feasible", why, checks, suggestion=_suggest(None, registry))
    if not objects.ok:
        return Route(
            "not_feasible",
            objects.detail,
            checks,
            suggestion=_suggest(parsed, registry),
        )
    prim = check_primitives(parsed, info)
    checks.append(prim)
    if not prim.ok:
        return Route(
            "not_feasible", prim.detail, checks, suggestion=_suggest(None, registry)
        )
    qty = check_quantities(prompt)
    if qty is not None:
        checks.append(qty)
        if not qty.ok:
            return Route(
                "not_feasible",
                qty.detail,
                checks,
                suggestion=_suggest(parsed, registry),
            )
    if is_default(
        robot
    ):  # the SO-101's IK; another robot's reach is part of its robot check below
        reach = check_reach(parsed)
        checks.append(reach)
        if not reach.ok:
            return Route(
                "not_feasible",
                reach.detail,
                checks,
                suggestion=_suggest(None, registry),
            )

    fit = check_robot(parsed, robot)
    checks.append(fit)
    if not fit.ok:
        return Route(
            "not_feasible", fit.detail, checks, suggestion=_suggest(None, registry)
        )

    if isinstance(parsed, list):
        return _route_chain(parsed, info, registry, checks, mode, robot)

    spec = parsed
    est, est_why = train_estimate(registry, spec.skill)
    checks.append(Check("estimate", True, f"~{est:.0f} s to train ({est_why})"))
    if mode == "remix":
        src = registry.skills.get(from_skill)
        if src is None or src.composite:
            return Route(
                "not_feasible", "the tile to remix has no checkpoint of its own", checks
            )
        if src.robot != robot:
            return Route(
                "not_feasible",
                f"that tile was trained on {src.robot}; the active robot is {robot}, "
                "and a policy only runs on the joints it learned",
                checks,
            )
        note = (
            ""
            if src.task().skill == spec.skill
            else f" (a {src.task().skill} skill learning {spec.skill})"
        )
        return Route(
            "train",
            f"remix: fine-tuning from '{src.name or src.instruction}'{note}",
            checks,
            estimate_s=est,
            expected=expected_rate(registry, spec.skill),
            skill_id=src.id,
        )
    if mode == "scratch":
        return Route(
            "train",
            "new skill from scratch: starting from the multi-task base policy, not the nearest tile",
            checks,
            estimate_s=est,
            expected=expected_rate(registry, spec.skill),
        )
    have = registry.exact(spec.instruction, robot)
    if (
        mode == "auto"
        and have
        and have.robot == robot
        and have.success_rate >= REUSE_MIN
    ):
        checks.append(
            Check(
                "library",
                True,
                f"'{have.name or have.instruction}' already measures {have.success_rate:.0%}",
            )
        )
        return Route(
            "reuse",
            f"already in the library at {have.success_rate:.0%} on {have.n_eval} held-out scenes; no training needed",
            checks,
            estimate_s=0.0,
            expected=have.success_rate,
            skill_id=have.id,
        )
    if have:
        checks.append(
            Check(
                "library",
                True,
                f"'{have.instruction}' exists at {have.success_rate:.0%} (below {REUSE_MIN:.0%}): training a better one",
            )
        )
    return Route(
        "train",
        f"one new {spec.skill} skill, about {est:.0f} s",
        checks,
        estimate_s=est,
        expected=expected_rate(registry, spec.skill),
    )


def _route_chain(
    steps: list[Step],
    info: dict,
    registry: Registry,
    checks: list[Check],
    mode: str,
    robot: str,
) -> Route:
    dropped = info.get("dropped") or ""
    raw = int(info.get("raw_steps") or len(steps))
    if raw > MAX_STEPS:
        return Route(
            "too_complex",
            f"needs {raw} steps; the planner chains at most {MAX_STEPS}. Split it: run the first {MAX_STEPS} as one "
            "prompt, then the rest",
            checks,
            suggestion=" and ".join(s.instruction for s in steps[:MAX_STEPS]),
        )
    if (
        "towers stop" in dropped
        or "moved only once" in dropped
        or "already holds" in dropped
    ):
        return Route(
            "too_complex",
            f"the scene cannot hold this chain as asked ({dropped})",
            checks,
            suggestion=" and ".join(s.instruction for s in steps),
        )
    plan, total, missing = [], CHAIN_EXEC_S, []
    for s in steps:
        have = registry.exact(s.instruction, robot)
        if have and have.robot == robot:
            plan.append(
                {"instruction": s.instruction, "source": "reuse", "skill_id": have.id}
            )
        else:
            est, _ = train_estimate(registry, s.skill)
            total += est
            missing.append(s.instruction)
            plan.append(
                {"instruction": s.instruction, "source": "train", "estimate_s": est}
            )
    checks.append(
        Check(
            "estimate",
            total <= CHAIN_BUDGET_S,
            f"{len(steps) - len(missing)} step(s) reuse a tile, {len(missing)} to train: ~{total:.0f} s "
            f"(chain budget {CHAIN_BUDGET_S:.0f} s)",
        )
    )
    if total > CHAIN_BUDGET_S:
        return Route(
            "too_complex",
            f"{len(missing)} new skills to train first (~{total / 60:.0f} min). Train them as tiles one at a time, "
            "then the chain reuses them",
            checks,
            suggestion=missing[0],
            estimate_s=total,
            steps=plan,
        )
    same = [
        s
        for s in registry.skills.values()
        if s.composite
        and [st["instruction"] for st in s.steps] == [st.instruction for st in steps]
    ]
    if mode == "auto" and same:
        best = max(same, key=lambda s: s.success_rate)
        return Route(
            "reuse",
            f"this chain is already a tile ({best.success_rate:.0%} end to end)",
            checks,
            expected=best.success_rate,
            skill_id=best.id,
            steps=plan,
        )
    return Route(
        "chain",
        f"{len(steps)} steps: {len(steps) - len(missing)} reuse a skill agent, {len(missing)} to train, ~{total:.0f} s",
        checks,
        estimate_s=total,
        steps=plan,
    )
