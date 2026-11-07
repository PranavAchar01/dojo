"""The bridge as checkable rules. Each rule returns pass or fail with a measured detail; nothing is judged by eye.

Grid rules run on the stud/plate grid (exact integers). The last two run a physics settle: every brick is placed at
its target pose, dropped under gravity for two seconds, and the largest displacement and the deck's tilt are measured."""

from __future__ import annotations

import itertools
from dataclasses import asdict, dataclass

import mujoco
import numpy as np

from . import hero_bricks as hb

SETTLE_S = 2.0
SETTLE_MAX_MM = 1.0  # no brick may move more than this over the settle
DECK_TILT_MAX_DEG = 1.0


@dataclass
class Result:
    id: str
    label: str
    ok: bool
    detail: str


def _roles(bricks) -> tuple[list[list[hb.Brick]], hb.Brick | None, list[hb.Brick]]:
    """Split into pillar stacks (grouped by footprint column), the deck (longest brick) and what sits above it."""
    bricks = list(bricks)
    if not bricks:
        return [], None, []
    deck = max(bricks, key=lambda b: (max(b.footprint), b.lvl))
    rest = [b for b in bricks if b is not deck]
    above = [b for b in rest if b.lvl >= deck.top]
    stacks: dict[tuple, list[hb.Brick]] = {}
    for b in rest:
        if b in above:
            continue
        stacks.setdefault((b.col, b.row, b.yaw_deg), []).append(b)
    return (
        [sorted(s, key=lambda b: b.lvl) for _, s in sorted(stacks.items())],
        deck,
        above,
    )


def _x_span(b: hb.Brick) -> range:
    return range(b.col, b.col + b.footprint[0])


def grid_rules(bricks, rules: hb.Rules) -> list[Result]:
    pillars, deck, _ = _roles(bricks)
    out: list[Result] = []

    # 1 the shape asked for
    ok = len(pillars) == 2 and deck is not None
    detail = f"{len(pillars)} pillars" + ("" if deck else ", no deck")
    if ok:
        left, right = pillars
        gap = right[0].col - (left[0].col + left[0].footprint[0])
        heights = [len(left), len(right)]
        ok = (
            gap == rules.gap
            and heights == [rules.pillar] * 2
            and max(deck.footprint) == rules.deck
        )
        detail = f"gap {gap} studs (want {rules.gap}), pillars {heights[0]} and {heights[1]} bricks (want {rules.pillar}), deck {max(deck.footprint)} studs (want {rules.deck})"
    out.append(
        Result(
            "shape", "Gap, pillar height and deck length match the rules", ok, detail
        )
    )
    if len(pillars) != 2 or deck is None:
        for rid, label in (
            ("stacked", "Each pillar's bricks are stud-aligned and directly stacked"),
            ("tops", "The pillar tops are level"),
            ("deck_rests", "The deck rests on both pillars, at least 1 stud on each"),
            ("deck_level", "The deck is level"),
            ("gap_empty", "The gap under the deck is empty"),
        ):
            out.append(Result(rid, label, False, "needs two pillars and a deck"))
        return out

    # 2 stud-aligned and directly stacked
    bad = []
    for name, p in zip(("left", "right"), pillars):
        if p[0].lvl != 0:
            bad.append(f"{name} pillar starts at level {p[0].lvl}, not on the table")
        for lo, hi in itertools.pairwise(p):
            if hi.lvl != lo.top:
                bad.append(f"{hi.name} is {hi.lvl - lo.top} plates off the brick below")
            if (hi.col, hi.row, hi.yaw_deg) != (lo.col, lo.row, lo.yaw_deg):
                bad.append(f"{hi.name} is not stud-aligned with {lo.name}")
    out.append(
        Result(
            "stacked",
            "Each pillar's bricks are stud-aligned and directly stacked",
            not bad,
            "; ".join(bad) or "all pillar bricks share one stud column and touch",
        )
    )

    # 3 pillar tops level
    tops = [p[-1].top for p in pillars]
    out.append(
        Result(
            "tops",
            "The pillar tops are level",
            tops[0] == tops[1],
            f"tops at plate level {tops[0]} and {tops[1]}",
        )
    )

    # 4 deck rests on both pillars
    rest = []
    for name, p in zip(("left", "right"), pillars):
        ov = len(set(_x_span(deck)) & set(_x_span(p[-1])))
        rows = len(
            set(range(deck.row, deck.row + deck.footprint[1]))
            & set(range(p[-1].row, p[-1].row + p[-1].footprint[1]))
        )
        rest.append((name, ov if rows else 0))
    seated = deck.lvl == max(tops) and deck.lvl == min(tops)
    ok = seated and all(ov >= 1 for _, ov in rest)
    out.append(
        Result(
            "deck_rests",
            "The deck rests on both pillars, at least 1 stud on each",
            ok,
            f"deck bottom at level {deck.lvl}, pillar tops {tops}; overlap "
            + ", ".join(f"{n} {ov} stud" + ("" if ov == 1 else "s") for n, ov in rest),
        )
    )

    # 5 deck level (grid: whole brick, yaw a multiple of 90)
    ok = deck.yaw_deg % 90 == 0
    out.append(
        Result(
            "deck_level",
            "The deck is level",
            ok,
            "deck lies flat on the stud grid" if ok else f"deck yaw {deck.yaw_deg} deg",
        )
    )

    # 6 gap empty: no cell between the pillars below the deck
    left, right = pillars
    gx0, gx1 = left[0].col + left[0].footprint[0], right[0].col
    rows = range(deck.row, deck.row + deck.footprint[1])
    intruders = sorted(
        {
            b.name
            for b in bricks
            if b is not deck
            and any(
                gx0 <= x < gx1 and y in rows and z < deck.lvl for x, y, z in hb.cells(b)
            )
        }
    )
    if not (gx0 < gx1):
        intruders = ["no gap between pillars"]
    out.append(
        Result(
            "gap_empty",
            "The gap under the deck is empty",
            not intruders,
            ("in the way: " + ", ".join(intruders))
            if intruders
            else f"{gx1 - gx0} studs wide, empty from the table to the deck",
        )
    )
    return out


def settle(bricks) -> dict:
    """Place every brick at its target pose, run gravity for SETTLE_S, report movement and the deck's tilt."""
    spec = mujoco.MjSpec()
    spec.option.timestep = 0.002
    spec.worldbody.add_geom(
        type=mujoco.mjtGeom.mjGEOM_PLANE,
        size=[1, 1, 0.01],
        friction=[0.8, 0.005, 0.0001],
    )
    for b in bricks:
        nx, ny = b.footprint
        h = hb.PART_PLATES[b.part] * hb.PLATE
        body = spec.worldbody.add_body(
            name=b.name, pos=list(b.target_pos() + np.array([0, 0, h / 2]))
        )
        body.add_freejoint()
        body.add_geom(
            type=mujoco.mjtGeom.mjGEOM_BOX,
            size=[nx * hb.STUD / 2 - 1e-5, ny * hb.STUD / 2 - 1e-5, h / 2],
            density=1050,
            friction=[0.8, 0.005, 0.0001],
            condim=4,
        )
    m = spec.compile()
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    start = {b.name: d.xpos[m.body(b.name).id].copy() for b in bricks}
    for _ in range(int(SETTLE_S / m.opt.timestep)):
        mujoco.mj_step(m, d)
    moved = {
        n: float(np.linalg.norm(d.xpos[m.body(n).id] - p)) * 1e3
        for n, p in start.items()
    }
    tilt = 0.0
    deck = _roles(bricks)[1]
    if deck is not None:
        z = d.xmat[m.body(deck.name).id].reshape(3, 3)[:, 2]
        tilt = float(np.degrees(np.arccos(np.clip(z[2], -1, 1))))
    return {
        "moved_mm": moved,
        "max_mm": max(moved.values(), default=0.0),
        "deck_tilt_deg": tilt,
    }


def check(bricks, rules: hb.Rules, physics: bool = True) -> list[Result]:
    bricks = tuple(bricks)
    out = grid_rules(bricks, rules)
    problems = hb.check_assembly(bricks)
    out.append(
        Result(
            "supported",
            "No brick overlaps another and every brick rests on studs",
            not problems,
            "; ".join(problems) or "every brick engages a stud below",
        )
    )
    if physics:
        s = settle(bricks)
        worst = max(s["moved_mm"], key=s["moved_mm"].get) if s["moved_mm"] else "-"
        out.append(
            Result(
                "settle",
                f"The structure stands: nothing moves more than {SETTLE_MAX_MM:g} mm in {SETTLE_S:g} s",
                s["max_mm"] <= SETTLE_MAX_MM,
                f"largest move {s['max_mm']:.3f} mm ({worst})",
            )
        )
        # deck level, measured: the settled deck stays flat
        for r in out:
            if r.id == "deck_level" and r.ok:
                r.ok = s["deck_tilt_deg"] <= DECK_TILT_MAX_DEG
                r.detail += f"; settled tilt {s['deck_tilt_deg']:.3f} deg (limit {DECK_TILT_MAX_DEG:g})"
    return out


def to_json(results: list[Result]) -> list[dict]:
    return [asdict(r) for r in results]
