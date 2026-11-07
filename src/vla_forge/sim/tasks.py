"""Task specs, layouts, and the programmatic success checks.

A TaskSpec is what the task agent produces from a sentence. Its success check is a small list of predicates
(a closed DSL, never code from the language model) evaluated on the simulator state.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field

import numpy as np

from .props import (
    KIT_SLOTS,
    SLOT_INNER,
    SOCKET_CLEAR,
    TRAY_BASE,
    slot_offset,
)
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

SKILLS = ("reach", "push", "lift", "pick_place", "stack", "kit", "insert", "rotate", "pack")
# the complexity ladder: tier 1 single motions, tier 2 pick-and-carry (kit: into a named tray slot); tier 3
# composes published skills, and also holds the precision insertions (3.5 mm clearance, yaw-aligned)
TIER = {
    "reach": 1,
    "push": 1,
    "lift": 1,
    "pick_place": 2,
    "stack": 2,
    "kit": 2,
    "insert": 3,
    "rotate": 3,
    "pack": 3,
}

# predicates the success DSL understands
PREDICATES = {
    "in_bowl": "block centre inside the bowl's inner radius and resting low",
    "in_bin": "block centre inside the tote's opening and below its rim",
    "on_pad": "block centre inside the taped square, on the table",
    "lifted": "block held above min_z metres",
    "touching": "closed gripper hovering just above the block (within 2.5 cm sideways)",
    "on_block": "block resting on top of the other block",
    "in_slot": "block resting low inside the tray slot marked by the goal point",
    "seated": "block sunk to the bottom of the socket, centred within its clearance",
    "aligned": "block yaw matches the socket's yaw (modulo a quarter turn) within 0.17 rad",
    "still": "block not moving (speed below 2 cm/s)",
    "packed": "every library tool inside the toolbox's measured cavity, below the rim, and resting",
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
    feasibility: dict = field(default_factory=dict)  # sim/feasibility.py: included / excluded objects and the numbers

    def __post_init__(self):
        assert self.skill in SKILLS, self.skill
        assert self.obj in BLOCK_COLORS or self.skill == "pack", self.obj
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
                    "feasibility",
                )
                if k in d
            }
        )


def canonical_instruction(skill: str, obj: str, target: str | None) -> str:
    if skill == "pack":
        return "pack the ratchet wrench, pliers and screwdriver into the toolbox"
    if skill == "pick_place":
        return f"put the {obj} block in the {target} bowl"
    if skill == "push":
        return f"push the {obj} block onto the {target} square"
    if skill == "lift":
        return f"pick up the {obj} block and hold it up"
    if skill == "reach":
        return f"touch the {obj} block"
    if skill == "kit":
        return f"put the {obj} block in the {target} slot of the tray"
    if skill == "insert":
        return f"insert the {obj} block into the socket"
    if skill == "rotate":
        return f"pick up the {obj} block, rotate it and set it in the diamond socket"
    return f"stack the {obj} block on the {target} block"


def default_success(skill: str) -> list[dict]:
    if skill == "pack":
        return [{"op": "packed"}]
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
    if skill == "kit":
        return [
            {"op": "in_slot", "block": "target", "slot": "goal"},
            {"op": "still", "block": "target"},
        ]
    if skill == "insert":
        return [
            {"op": "seated", "block": "target", "socket": "goal"},
            {"op": "still", "block": "target"},
        ]
    if skill == "rotate":
        return [
            {"op": "seated", "block": "target", "socket": "goal"},
            {"op": "aligned", "block": "target", "socket": "goal"},
            {"op": "still", "block": "target"},
        ]
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
    if skill == "kit":
        return list(KIT_SLOTS)
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
    if op == "packed":
        from .toolbox import packed

        return packed(scene)
    c = scene.body_pos(blk)
    if op == "in_bowl":
        b = scene.body_pos(p.get("bowl", "goal"))
        return bool(
            np.linalg.norm(c[:2] - b[:2]) < BOWL_R and c[2] < BOWL_WALL_H + CUBE_HALF
        )
    if op == "in_slot":
        g = scene.body_pos(p.get("slot", "goal"))
        yaw = scene.body_yaw(p.get("slot", "goal"))
        d = c[:2] - g[:2]
        local = np.array(
            [
                d[0] * math.cos(yaw) + d[1] * math.sin(yaw),
                -d[0] * math.sin(yaw) + d[1] * math.cos(yaw),
            ]
        )
        return bool(
            np.all(np.abs(local) < SLOT_INNER / 2 - CUBE_HALF + 0.005)
            and c[2] < TRAY_BASE + CUBE_HALF + 0.004
        )
    if op in ("seated", "aligned"):
        g = scene.body_pos(p.get("socket", "goal"))
        if op == "seated":
            return bool(
                np.linalg.norm(c[:2] - g[:2]) < SOCKET_CLEAR + 0.0015
                and c[2] < CUBE_HALF + 0.004
            )
        err = (scene.body_yaw(blk) - scene.body_yaw(p.get("socket", "goal"))) % (
            math.pi / 2
        )
        return bool(min(err, math.pi / 2 - err) < 0.17)
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
    if spec.skill == "pack":
        from .toolbox import sample

        box, tools = sample(rng)
        return Layout(props=box, tools=tools)
    if spec.skill in HARD_SKILLS:
        return _sample_hard(spec, rng, near, tries)
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


HARD_SKILLS = ("kit", "insert", "rotate")
HARD_BLOCK_R = (
    0.165,
    0.26,
)  # the grasp is least reliable right at the base and at full reach
TRAY_R = (0.19, 0.25)
TRAY_ANG = math.radians(22)
SOCKET_FOOT = 0.057  # the socket plate's corner radius
TRAY_FOOT = 0.091  # the tray's corner radius
JAW_BACK = 0.05  # the gripper body sits this far inboard of the block while it grasps


def _hard_goal(spec, rng, near) -> tuple[dict, np.ndarray, np.ndarray, float, float]:
    """(props, goal xy, fixture centre xy, fixture radius, fixture yaw) for kit / insert / rotate."""
    if spec.skill == "kit":
        yaw = float(rng.uniform(-0.1, 0.1))
        off = np.array(slot_offset(spec.target or KIT_SLOTS[0]))
        rot = np.array(
            [[math.cos(yaw), -math.sin(yaw)], [math.sin(yaw), math.cos(yaw)]]
        )
        if near and "goal" in near:
            goal = np.asarray(near["goal"]) + rng.uniform(-0.015, 0.015, 2)
            centre = goal - rot @ off
        else:
            centre = _polar(rng, TRAY_R, TRAY_ANG)
            goal = centre + rot @ off
        props = {
            "tray": ("tray", "gray", float(centre[0]), float(centre[1]), yaw),
            "goal": ("marker", "gray", float(goal[0]), float(goal[1]), yaw),
        }
        return props, goal, centre, TRAY_FOOT, yaw
    centre = _polar(rng, GOAL_R, GOAL_ANG)
    if near and "goal" in near:
        centre = np.asarray(near["goal"]) + rng.uniform(-0.015, 0.015, 2)
    yaw = math.atan2(centre[1], centre[0]) + (
        math.pi / 4 if spec.skill == "rotate" else 0.0
    )
    props = {"goal": ("socket", "gray", float(centre[0]), float(centre[1]), yaw)}
    return props, centre, centre, SOCKET_FOOT, yaw


def _sample_hard(spec, rng, near, tries) -> Layout:
    """kit / insert / rotate: one fixture on the table (a tray with a goal point, or a socket) and the block."""
    others = [c for c in BLOCK_COLORS if c != spec.obj]
    for _ in range(tries):
        props, _goal, centre, foot, _fy = _hard_goal(spec, rng, near)
        r_c = np.linalg.norm(centre)
        if not (0.15 < r_c < 0.28) or abs(math.atan2(centre[1], centre[0])) > 0.55:
            continue
        blk = _polar(rng, HARD_BLOCK_R, BLOCK_ANG)
        if near and "target" in near:
            blk = np.asarray(near["target"]) + rng.uniform(-0.015, 0.015, 2)
        r = np.linalg.norm(blk)
        if (
            not (HARD_BLOCK_R[0] - 0.01 < r < HARD_BLOCK_R[1] + 0.015)
            or abs(math.atan2(blk[1], blk[0])) > BLOCK_ANG + 0.05
        ):
            continue
        jaw = blk * (1 - JAW_BACK / r)
        if np.linalg.norm(blk - centre) < foot + 0.03:
            continue
        if np.linalg.norm(jaw - centre) < foot + 0.02:
            continue
        yaw = math.atan2(blk[1], blk[0]) + float(rng.uniform(-0.6, 0.6))
        if spec.skill == "rotate":
            # the cube fits every quarter turn, so the turn to make is the nearest one: at least 17 degrees,
            # or "rotate" would be a skill of doing nothing
            need = (props["goal"][4] - yaw + math.pi / 4) % (math.pi / 2) - math.pi / 4
            if abs(need) < 0.3:
                continue
        lay = Layout(props=props)
        lay.blocks["target"] = (spec.obj, float(blk[0]), float(blk[1]), yaw)
        placed = [blk]
        ok = True
        for i in range(spec.distractors):
            for _t in range(60):
                p = _polar(rng, BLOCK_R, BLOCK_ANG)
                if any(np.linalg.norm(p - q) < 0.065 for q in placed):
                    continue
                if np.linalg.norm(p - centre) < foot + 0.04:
                    continue
                if _seg_dist(p, blk, centre) < 0.055:
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
