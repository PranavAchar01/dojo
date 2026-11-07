"""Tier 3: tasks composed from published skills.

A composed task is an ordered list of steps, each one a tier 1-2 skill (push, put in a bowl, stack). The steps
share one scene: every block and fixture gets a name ("blk_red", "bowl_white", "sq_blue"), each step's success
check is the skill's own predicate on those names, and the whole task succeeds when every step's predicate
holds at the end. Each step is executed by the published skill agent for that exact instruction; a step with
no skill yet is trained on the fly first, through the normal one-prompt pipeline.

This module is torch-free so the simulator workers can import it.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import numpy as np

from .sim.scene import (
    BLOCK_COLORS,
    BOWL_COLORS,
    BOWL_R,
    CUBE_HALF,
    PAD_COLORS,
    PAD_HALF,
    Layout,
)
from .sim.tasks import (
    TaskSpec,
    _polar,
    _seg_dist,
    bands,
    canonical_instruction,
    workspace_of,
)

COMPOSABLE = ("push", "pick_place", "stack")
MAX_STEPS = 3


@dataclass
class Step:
    skill: str
    obj: str
    target: str

    @property
    def instruction(self) -> str:
        return canonical_instruction(self.skill, self.obj, self.target)

    def single(self) -> TaskSpec:
        """The step as a stand-alone tier 1-2 task (what a missing skill is trained on)."""
        return TaskSpec(self.skill, self.obj, self.target)

    def fixture(self) -> str:
        return {"pick_place": "bowl_", "push": "sq_", "stack": "blk_"}[
            self.skill
        ] + self.target

    def in_scene(self) -> TaskSpec:
        """The step inside the composed scene: same instruction, predicates on the scene's own names."""
        blk, fx = f"blk_{self.obj}", self.fixture()
        core = {
            "pick_place": [
                {"op": "in_bowl", "block": blk, "bowl": fx},
                {"op": "still", "block": blk},
            ],
            "push": [{"op": "on_pad", "block": blk, "pad": fx}],
            "stack": [
                {"op": "on_block", "block": blk, "base": fx},
                {"op": "still", "block": blk},
            ],
        }[self.skill]
        return TaskSpec(self.skill, self.obj, self.target, success=core)

    def to_json(self) -> dict:
        return {**asdict(self), "instruction": self.instruction}


def validate(raw: list[dict]) -> tuple[list[Step], str]:
    """Model output -> steps the scene can hold. Returns (steps, note about anything that was dropped)."""
    steps: list[Step] = []
    notes = []
    moved: set[str] = set()
    used: set[str] = set()
    for r in raw[: MAX_STEPS + 2]:
        sk, obj, tg = r.get("skill"), r.get("object_color"), r.get("target_color")
        table = {
            "pick_place": BOWL_COLORS,
            "push": PAD_COLORS,
            "stack": BLOCK_COLORS,
        }.get(sk)
        if table is None or obj not in BLOCK_COLORS or tg not in table:
            notes.append(f"dropped a step this arm cannot learn ({sk} {obj} -> {tg})")
            continue
        st = Step(sk, obj, tg)
        if sk == "stack" and tg == obj:
            notes.append("dropped stacking a block on itself")
            continue
        if obj in moved:
            notes.append(f"the {obj} block is moved only once")
            continue
        if st.fixture() in used:
            notes.append(
                f"the {tg} {'bowl' if sk == 'pick_place' else 'square' if sk == 'push' else 'block'} "
                "already holds a block"
            )
            continue
        if (
            sk == "stack"
            and tg in moved
            and any(s.obj == tg and s.skill == "stack" for s in steps)
        ):
            notes.append("towers stop at two blocks")
            continue
        if len(steps) == MAX_STEPS:
            notes.append(f"kept the first {MAX_STEPS} steps")
            break
        steps.append(st)
        moved.add(obj)
        used.add(st.fixture())
    return steps, "; ".join(notes)


def entities(steps: list[Step]) -> tuple[list[str], list[str], list[str]]:
    """Block, bowl and square colours the scene needs."""
    blocks, bowls, pads = [], [], []
    for s in steps:
        if s.obj not in blocks:
            blocks.append(s.obj)
        if s.skill == "stack" and s.target not in blocks:
            blocks.append(s.target)
        if s.skill == "pick_place":
            bowls.append(s.target)
        if s.skill == "push":
            pads.append(s.target)
    return blocks, bowls, pads


def sample_layout(
    steps: list[Step],
    rng: np.random.Generator,
    distractors: int = 1,
    tries: int = 4000,
    robot: str | None = None,
) -> Layout:
    """One composed scene. Every step keeps the geometry its skill was trained on: the block a radial-ish
    grasp away from its goal, a push lane 8-17 cm long with the pusher's standoff reachable, no other object
    in a step's path, and grasp faces within 34 degrees of radial."""
    bd = bands(workspace_of(robot))
    fix_r = (0.17, 0.27) if robot is None else (max(0.17, bd["fix_r"][0]), bd["fix_r"][1])
    blocks, bowls, pads = entities(steps)
    spare = [c for c in BLOCK_COLORS if c not in blocks]
    for _ in range(tries):
        fx: dict[str, np.ndarray] = {}
        ok = True
        for name, half in [(f"bowl_{c}", BOWL_R) for c in bowls] + [
            (f"sq_{c}", PAD_HALF) for c in pads
        ]:
            for _ in range(80):
                p = _polar(rng, fix_r, bd["fix_ang"])
                if all(
                    np.linalg.norm(p - q) > half + _half(n) + 0.03
                    for n, q in fx.items()
                ):
                    fx[name] = p
                    break
            else:
                ok = False
        if not ok:
            continue
        pos: dict[str, np.ndarray] = {}
        yaw: dict[str, float] = {}
        for s in steps:  # pushed blocks go first: their lane fixes where they can start
            for c in [s.obj] + ([s.target] if s.skill == "stack" else []):
                if c in pos:
                    continue
                p = _block_for(s, c, fx, pos, rng, bd)
                if p is None:
                    ok = False
                    break
                pos[c] = p
                yaw[c] = math.atan2(p[1], p[0]) + float(rng.uniform(-0.6, 0.6))
            if not ok:
                break
        if not ok or not _paths_clear(steps, fx, pos):
            continue
        for k in range(distractors):
            for _ in range(80):
                p = _polar(rng, bd["block_r"], bd["block_ang"])
                if _free(p, fx, pos, bd) and all(
                    _seg_dist(p, pos[s.obj], _goal(s, fx, pos)) > 0.055 for s in steps
                ):
                    c = spare[k % len(spare)]
                    pos[c] = p
                    yaw[c] = float(rng.uniform(-0.8, 0.8))
                    break
        lay = Layout()
        for c, p in pos.items():
            lay.blocks[f"blk_{c}"] = (c, float(p[0]), float(p[1]), yaw[c])
        for n, p in fx.items():
            (lay.bowls if n.startswith("bowl_") else lay.pads)[n] = (
                n.split("_", 1)[1],
                float(p[0]),
                float(p[1]),
            )
        return lay
    raise RuntimeError("could not fit the composed scene on the table")


def _half(name: str) -> float:
    return BOWL_R if name.startswith("bowl_") else PAD_HALF


def _goal(s: Step, fx: dict, pos: dict) -> np.ndarray:
    return pos[s.target] if s.skill == "stack" else fx[s.fixture()]


def _free(p, fx, pos, bd) -> bool:
    r = np.linalg.norm(p)
    return (
        bd["r_ok"][0] < r < bd["r_ok"][1]
        and all(np.linalg.norm(p - q) > 0.065 for q in pos.values())
        and all(
            np.linalg.norm(p - q) > _half(n) + CUBE_HALF + 0.04 for n, q in fx.items()
        )
        and all(_seg_dist(q, np.zeros(2), p) > 0.04 for q in pos.values())
    )


def _block_for(s: Step, c: str, fx, pos, rng, bd):
    for _ in range(120):
        p = _polar(rng, bd["block_r"], bd["block_ang"])
        if not _free(p, fx, pos, bd):
            continue
        if (
            s.skill == "stack"
        ):  # the two blocks start at least 8 cm apart, as in training
            other = s.target if c == s.obj else s.obj
            if other in pos and np.linalg.norm(p - pos[other]) < 0.08:
                continue
            return p
        if s.skill == "pick_place":
            if np.linalg.norm(p - fx[s.fixture()]) < BOWL_R + 0.06:
                continue
        elif s.skill == "push":
            g = fx[s.fixture()]
            gap = np.linalg.norm(p - g)
            if not 0.08 < gap < 0.17:
                continue
            if not bd["behind"][0] < np.linalg.norm(p - (g - p) / gap * 0.05) < bd["behind"][1]:
                continue
        return p
    return None


def _paths_clear(steps: list[Step], fx: dict, pos: dict) -> bool:
    """Nothing else lies on a step's carry or push line (the skill never saw that in training)."""
    for s in steps:
        a, b = pos[s.obj], _goal(s, fx, pos)
        for c, q in pos.items():
            if c not in (s.obj, s.target) and _seg_dist(q, a, b) < 0.055:
                return False
        for n, q in fx.items():
            if n != s.fixture() and _seg_dist(q, a, b) < _half(n) + 0.03:
                return False
    return True


def step_layout(
    steps: list[Step],
    k: int,
    rng: np.random.Generator,
    done_p: float = 0.5,
    robot: str | None = None,
) -> Layout:
    """A scene for training step k on its own, drawn from the composed task's scenes.

    A missing skill trained only on single-skill tables met a second bowl, square or finished block for the
    first time at test time; drawn from here, its demos and its held-out checks already look like the composed
    task. With probability done_p the earlier steps are shown finished (their block already in its bowl or on
    its square). The step's own block and fixture are renamed "target" and "goal", the names the expert and the
    single-skill success check use; every other block becomes a distractor that must stay where it is."""
    lay = sample_layout(steps, rng, robot=robot)
    if rng.random() < done_p:
        for s in steps[:k]:
            name = f"blk_{s.obj}"
            if s.skill == "stack" or name not in lay.blocks:
                continue
            fx = s.fixture()
            _c, x, y = (lay.bowls if s.skill == "pick_place" else lay.pads)[fx]
            c, _x, _y, yaw = lay.blocks[name]
            lay.blocks[name] = (c, x, y, yaw)
    s = steps[k]
    out = Layout()
    goal_blk = f"blk_{s.target}" if s.skill == "stack" else None
    n = 0
    for name, v in lay.blocks.items():
        if name == f"blk_{s.obj}":
            out.blocks["target"] = v
        elif name == goal_blk:
            out.blocks["goal"] = v
        else:
            out.blocks[f"distractor{n}"] = v
            n += 1
    for src, dst in ((lay.bowls, out.bowls), (lay.pads, out.pads)):
        for name, v in src.items():
            dst["goal" if name == s.fixture() else name] = v
    return out
