"""Task specs, layouts, and the programmatic success checks.

A TaskSpec is what the task agent produces from a sentence. Its success check is a small list of predicates
(a closed DSL, never code from the language model) evaluated on the simulator state.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field

import numpy as np

from .scene import (
    BLOCK_COLORS,
    BOWL_COLORS,
    BOWL_R,
    BOWL_WALL_H,
    CUBE_HALF,
    PAD_COLORS,
    PAD_HALF,
    Layout,
    Scene,
)

SKILLS = ("reach", "push", "lift", "pick_place", "stack")
# the complexity ladder: tier 1 single motions, tier 2 pick-and-carry; tier 3 composes published skills
TIER = {"reach": 1, "push": 1, "lift": 1, "pick_place": 2, "stack": 2}

# predicates the success DSL understands
PREDICATES = {
    "in_bowl": "block centre inside the bowl's inner radius and resting low",
    "in_bin": "block centre inside the tote's opening and below its rim",
    "on_pad": "block centre inside the taped square, on the table",
    "lifted": "block held above min_z metres",
    "touching": "closed gripper hovering just above the block (within 2.5 cm sideways)",
    "on_block": "block resting on top of the other block",
    "still": "block not moving (speed below 2 cm/s)",
    "undisturbed": "every distractor within 3 cm of where it started",
}


@dataclass
class TaskSpec:
    skill: str
    obj: str  # block colour
    target: str | None = None  # bowl / pad / base-block colour
    distractors: int = 2
    instruction: str = ""
    success: list[dict] = field(default_factory=list)

    def __post_init__(self):
        assert self.skill in SKILLS, self.skill
        assert self.obj in BLOCK_COLORS, self.obj
        if not self.instruction:
            self.instruction = canonical_instruction(self.skill, self.obj, self.target)
        if not self.success:
            self.success = default_success(self.skill)

    def to_json(self) -> dict:
        return asdict(self)

    @staticmethod
    def from_json(d: dict) -> TaskSpec:
        return TaskSpec(
            **{
                k: d[k]
                for k in (
                    "skill",
                    "obj",
                    "target",
                    "distractors",
                    "instruction",
                    "success",
                )
                if k in d
            }
        )


def canonical_instruction(skill: str, obj: str, target: str | None) -> str:
    if skill == "pick_place":
        return f"put the {obj} block in the {target} bowl"
    if skill == "push":
        return f"push the {obj} block onto the {target} square"
    if skill == "lift":
        return f"pick up the {obj} block and hold it up"
    if skill == "reach":
        return f"touch the {obj} block"
    return f"stack the {obj} block on the {target} block"


def default_success(skill: str) -> list[dict]:
    if skill == "pick_place":
        return [
            {"op": "in_bowl", "block": "target", "bowl": "goal"},
            {"op": "still", "block": "target"},
        ]
    if skill == "push":
        return [{"op": "on_pad", "block": "target", "pad": "goal"}]
    if skill == "lift":
        return [{"op": "lifted", "block": "target", "min_z": 0.05}]
    if skill == "reach":
        return [{"op": "touching", "block": "target"}]
    return [
        {"op": "on_block", "block": "target", "base": "goal"},
        {"op": "still", "block": "target"},
    ]


def valid_targets(skill: str) -> list[str]:
    if skill == "pick_place":
        return list(BOWL_COLORS)
    if skill == "push":
        return list(PAD_COLORS)
    if skill == "stack":
        return list(BLOCK_COLORS)
    return []


def check(spec: TaskSpec, scene: Scene, start: Layout | None = None) -> bool:
    """Evaluate every predicate of the spec's success check on the current simulator state."""
    return all(_pred(p, scene, start) for p in spec.success)


def _pred(p: dict, scene: Scene, start: Layout | None) -> bool:
    op = p["op"]
    blk = p.get("block", "target")
    if op == "undisturbed":
        if start is None:
            return True
        for name, (_c, x, y, _yaw) in start.blocks.items():
            if (
                name.startswith("distractor")
                and np.linalg.norm(scene.body_pos(name)[:2] - [x, y]) > 0.03
            ):
                return False
        return True
    c = scene.body_pos(blk)
    if op == "in_bowl":
        b = scene.body_pos(p.get("bowl", "goal"))
        return bool(
            np.linalg.norm(c[:2] - b[:2]) < BOWL_R and c[2] < BOWL_WALL_H + CUBE_HALF
        )
    if op == "in_bin":
        from .line import in_bin

        return in_bin(c, scene.body_pos(p.get("bin", "goal")))
    if op == "on_pad":
        b = scene.body_pos(p.get("pad", "goal"))
        return bool(
            np.all(np.abs(c[:2] - b[:2]) < PAD_HALF) and c[2] < CUBE_HALF + 0.005
        )
    if op == "lifted":
        return bool(c[2] > float(p.get("min_z", 0.05)))
    if op == "touching":
        e = scene.ee_pos()
        dz = e[2] - (c[2] + CUBE_HALF)
        return bool(
            np.linalg.norm(e[:2] - c[:2]) < TOUCH_XY and TOUCH_Z[0] < dz < TOUCH_Z[1]
        )
    if op == "on_block":
        b = scene.body_pos(p.get("base", "goal"))
        return bool(
            np.all(np.abs(c[:2] - b[:2]) < CUBE_HALF * 1.2)
            and abs(c[2] - (b[2] + 2 * CUBE_HALF)) < 0.008
        )
    if op == "still":
        m = scene.model
        adr = m.joint(f"{blk}_free").dofadr[0]
        return bool(np.linalg.norm(scene.data.qvel[adr : adr + 3]) < 0.02)
    raise ValueError(f"unknown predicate {op}")


# ---------------------------------------------------------------- layouts

TOUCH_XY = 0.025
TOUCH_Z = (-0.01, 0.035)  # gripper frame height above the block's top face

BLOCK_R = (0.14, 0.27)
BLOCK_ANG = math.radians(36)
GOAL_R = (0.18, 0.26)
GOAL_ANG = math.radians(30)


def _polar(rng, rr, ang):
    r = rng.uniform(*rr)
    a = rng.uniform(-ang, ang)
    return np.array([r * math.cos(a), r * math.sin(a)])


def _seg_dist(p, a, b) -> float:
    ab = b - a
    t = float(np.clip(np.dot(p - a, ab) / max(np.dot(ab, ab), 1e-9), 0, 1))
    return float(np.linalg.norm(p - (a + t * ab)))


EXTRA_P = (
    0.6  # chance of each extra fixture (a second bowl, a second square) on the table
)


def sample_layout(
    spec: TaskSpec,
    rng: np.random.Generator,
    near: dict | None = None,
    tries: int = 400,
    extra_p: float = EXTRA_P,
) -> Layout:
    """Random layout for a spec. `near` pins the target block / goal near given xy (the critic's focus).

    With probability extra_p each, a distractor bowl and a distractor square of other colours share the table,
    so the policy has to read the goal's colour from the instruction, and scenes with several fixtures (the
    composed tasks) stay inside the training distribution."""
    others = [
        c
        for c in BLOCK_COLORS
        if c not in (spec.obj, spec.target if spec.skill == "stack" else None)
    ]
    for _ in range(tries):
        lay = Layout()
        if spec.skill in ("pick_place", "push", "stack"):
            goal = _polar(rng, GOAL_R, GOAL_ANG)
            if near and "goal" in near:
                goal = np.asarray(near["goal"]) + rng.uniform(-0.015, 0.015, 2)
        else:
            goal = None
        blk = _polar(rng, BLOCK_R, BLOCK_ANG)
        if near and "target" in near:
            blk = np.asarray(near["target"]) + rng.uniform(-0.015, 0.015, 2)
        r = np.linalg.norm(blk)
        if (
            not (0.125 < r < 0.285)
            or abs(math.atan2(blk[1], blk[0])) > BLOCK_ANG + 0.05
        ):
            continue
        if goal is not None:
            gap = np.linalg.norm(blk - goal)
            if spec.skill == "pick_place" and gap < BOWL_R + 0.06:
                continue
            if spec.skill == "push" and not (0.08 < gap < 0.17):
                continue
            if spec.skill == "stack" and gap < 0.08:
                continue
            if spec.skill == "push":
                # the pusher stands behind the block on the block->goal line; keep that point reachable
                u = (goal - blk) / gap
                behind = blk - u * 0.05
                if not (0.11 < np.linalg.norm(behind) < 0.29):
                    continue
        # target yaw within +-34 deg of the radial direction: the expert's grasp face (and so the wrist roll) is
        # then a smooth function of the scene, never a 90 degree flip a policy would have to guess
        yaw = math.atan2(blk[1], blk[0]) + float(rng.uniform(-0.6, 0.6))
        if spec.skill == "pick_place":
            jaw = blk + 0.05 * np.array([math.cos(yaw), math.sin(yaw)])
            if np.linalg.norm(jaw - goal) < BOWL_R + 0.02:
                continue
        lay.blocks["target"] = (spec.obj, float(blk[0]), float(blk[1]), yaw)
        placed = [blk]
        if spec.skill == "pick_place":
            lay.bowls["goal"] = (spec.target, float(goal[0]), float(goal[1]))
        elif spec.skill == "push":
            lay.pads["goal"] = (spec.target, float(goal[0]), float(goal[1]))
        elif spec.skill == "stack":
            lay.blocks["goal"] = (
                spec.target,
                float(goal[0]),
                float(goal[1]),
                float(rng.uniform(-0.2, 0.2)),
            )
            placed.append(goal)
        fixtures = [] if goal is None else [goal]
        ok = True
        for kind, colors, half in (
            ("bowls", BOWL_COLORS, BOWL_R),
            ("pads", PAD_COLORS, PAD_HALF),
        ):
            if rng.random() >= extra_p:
                continue
            goal_kind = {"pick_place": "bowls", "push": "pads"}.get(spec.skill)
            pool_c = [c for c in colors if kind != goal_kind or c != spec.target]
            for _ in range(60):
                p = _polar(rng, (0.16, 0.27), math.radians(42))
                if any(np.linalg.norm(p - q) < half + CUBE_HALF + 0.04 for q in placed):
                    continue
                if any(np.linalg.norm(p - f) < half + BOWL_R + 0.03 for f in fixtures):
                    continue
                if goal is not None and _seg_dist(p, blk, goal) < half + 0.03:
                    continue
                if _seg_dist(p, np.zeros(2), blk) < half + 0.02:
                    continue
                lay_kind = getattr(lay, kind)
                lay_kind["x" + kind[:-1] + "0"] = (
                    pool_c[int(rng.integers(len(pool_c)))],
                    float(p[0]),
                    float(p[1]),
                )
                fixtures.append(p)
                break
        for i in range(spec.distractors):
            for _ in range(60):
                p = _polar(rng, BLOCK_R, BLOCK_ANG)
                if any(np.linalg.norm(p - q) < 0.065 for q in placed):
                    continue
                if any(
                    np.linalg.norm(p - f) < BOWL_R + 0.05
                    for f in (fixtures if spec.skill != "stack" else fixtures[1:])
                ):
                    continue
                if goal is not None and _seg_dist(p, blk, goal) < 0.055:
                    continue
                if _seg_dist(p, np.zeros(2), blk) < 0.04:
                    continue
                color = others[int(rng.integers(len(others)))]
                lay.blocks[f"distractor{i}"] = (
                    color,
                    float(p[0]),
                    float(p[1]),
                    float(rng.uniform(-0.8, 0.8)),
                )
                placed.append(p)
                break
            else:
                ok = False
        if ok:
            return lay
    raise RuntimeError("could not sample a layout")


def structure_layout(spec: TaskSpec) -> Layout:
    """A layout with every entity of the spec but arbitrary positions (fixes the compiled model's structure)."""
    return sample_layout(spec, np.random.default_rng(0))
