"""The brick bridge as a run of the normal pipeline: its task and rules, a route (two SO-101s, scripted), the scene,
recorded builds, the rule counts, and a library tile that says "scripted".

Nothing is trained. `hero_build.run` drives both arms with the scripted expert on randomized brick placements; the
rules are `hero_rules.check` on the bricks that really ended up snapped. Every number is read off those runs.
"""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import get_context

import numpy as np

from . import hero_bricks as hb
from . import hero_replay as hr_
from . import hero_rules as hr

INSTRUCTION = "Build the brick bridge from the plan"
NAME = "Brick bridge"
SEEDS = tuple(range(7101, 7121))  # placements nobody tuned the expert on
WORKERS = 2


def plan() -> tuple[hb.Rules, tuple[hb.Brick, ...]]:
    """The plan in use (set by the prompt window), else the one the rules generate."""
    active = hb.load_active()
    return active if active else (hb.Rules(), hb.generate())


def check_plan(bricks, rules: hb.Rules) -> list[hr.Result]:
    return hr.check(bricks, rules)


def spec_json(rules: hb.Rules, bricks, results: list[hr.Result]) -> dict:
    """The task as the rail's Task card reads it: the instruction and one row per rule, checked on the plan itself."""
    return {
        "instruction": INSTRUCTION,
        "skill": "build",
        "obj": f"a {len(bricks)}-brick bridge, gap {rules.gap}, pillars {rules.pillar}, deck {rules.deck}",
        "target": "",
        "distractors": 0,
        "success": [
            {"op": r.id, "rule": r.label, "on the plan": "pass" if r.ok else "fail"} for r in results
        ],
    }


def route_json(rules_total: int) -> dict:
    return {
        "decision": "scripted_build",
        "reason": "A scripted expert drives two SO-101s; no policy is trained, so nothing is reused or fine-tuned.",
        "checks": [
            {"name": "arms", "ok": True, "detail": "two SO-101s, one on each side of the table"},
            {"name": "controller", "ok": True, "detail": "scripted pick-and-place with a snap-on-contact clutch"},
            {"name": "success", "ok": True, "detail": f"{rules_total} rules checked on the bricks that end up seated"},
        ],
        "suggestion": "",
        "estimate_s": 0.0,
        "expected": None,
        "skill_id": "",
        "steps": [],
        "refused": False,
    }


def build_one(seed: int, bricks, rules: hb.Rules) -> dict:
    """One recorded build, in a worker process (the table is not picklable, so it stays here)."""
    from . import hero_build as hbd

    out = hbd.run(seed, record=True, bricks=bricks, rules=rules)
    out.pop("table")
    return out


def build_all(bricks, rules: hb.Rules, on_done=None) -> list[dict]:
    """The builds for every seed, WORKERS at a time, in seed order."""
    res: dict[int, dict] = {}
    with ProcessPoolExecutor(WORKERS, mp_context=get_context("spawn")) as pool:
        futs = {pool.submit(build_one, s, tuple(bricks), rules): s for s in SEEDS}
        for f in as_completed(futs):
            res[futs[f]] = f.result()
            if on_done:
                on_done(len(res), res[futs[f]])
    return [res[s] for s in SEEDS]


def success(b: dict) -> bool:
    return b["seated"] == b["total"] and b["rules_passed"] == b["rules_total"]


def best(builds: list[dict]) -> dict:
    """The build the card shows: a successful one (else the most bricks seated) with the smallest worst placement error."""
    def miss(b: dict) -> float:
        return max((x["pos_err_mm"] or 0.0) for x in b["bricks"])

    return min(builds, key=lambda b: (not success(b), -b["seated"], miss(b)))


def failures(builds: list[dict]) -> dict[str, int]:
    """Rule id -> how many builds failed it; builds that left bricks unseated count under 'bricks seated'."""
    out: dict[str, int] = {}
    for b in builds:
        if b["seated"] < b["total"]:
            out["bricks seated"] = out.get("bricks seated", 0) + 1
        for r in b["rules"]:
            if not r["ok"]:
                out[r["id"]] = out.get(r["id"], 0) + 1
    return out


def detail(builds: list[dict], seconds: float) -> str:
    """What the card says under the name: measured counts and exactly which rules failed."""
    n = len(builds)
    seated = ", ".join(f"{b['seated']}/{b['total']}" for b in builds)
    bad = failures(builds)
    ok = sum(success(b) for b in builds)
    txt = f"Scripted build on {n} fresh placements, not a trained policy. Bricks seated per build: {seated}."
    txt += f" {ok}/{n} builds pass every rule."
    if bad:
        txt += " Failed: " + ", ".join(f"{k} ({v}/{n})" for k, v in bad.items()) + "."
    return txt


def scene_and_start(bricks) -> tuple[dict, list[float]]:
    """The player's scene and the table's first frame (arms at home, bricks loose), for the plan in use."""
    from . import hero_build as hbd
    from . import hero_sim as ds

    keep = hb.DUCK
    try:
        hb.set_assembly(bricks)
        t = ds.Table.new(480)
        t.reset(np.random.default_rng(SEEDS[0]), jitter=0.02)
        return hr_.scene(t.m), hbd.pose_frame(t)
    finally:
        hb.set_assembly(keep)

