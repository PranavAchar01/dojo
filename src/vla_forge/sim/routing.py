"""Which arm does the task? Decided from each arm's measured answer to `feasibility()`, never from names.

Every available arm is asked the same question about the same objects (fits, jaw width, payload at the carry
pose, reach). The arm that does the whole task wins; ties go to the cheapest arm, where "cheap" is the arm's total
joint torque budget (the sum of its arm actuators' force limits, read from the compiled model), and the default
SO-101 goes first when it can do everything. When no single arm can, the objects are split across arms (each arm
takes what it can do; the split is re-checked per arm so the packing still holds). An object no arm can do is
excluded with the engine's own reasons from the arm that came closest.
"""

from __future__ import annotations

from functools import lru_cache

from . import feasibility, robots


def available() -> list[str]:
    """Every arm the Build tab accepts (`robots.usable`), with its own jaw or any catalogue gripper swapped in, as robot ids; default first."""
    cands = [
        robots.make_id(a, effector=e)
        for a in robots.ARMS
        for e, d in robots.EFFECTORS.items()
        if e == "native" or (d is not None and a != "so101")
    ]
    ids = [r for r in cands if robots.usable(r)[0]]
    return sorted(ids, key=lambda r: (not robots.is_default(r), cost(r)))


@lru_cache(maxsize=32)
def cost(robot_id: str) -> float:
    """Total arm-joint torque budget in N m, the proxy for how big (and dear) the arm is."""
    r = robots.calibrate(robot_id)
    m = robots.robot_spec(robot_id).compile()
    total = 0.0
    for j in r.arm_joints:
        a = robots._act_for_joint(m, j)
        if m.actuator_forcelimited[a]:
            total += float(m.actuator_forcerange[a, 1])
    return round(total, 2)


@lru_cache(maxsize=256)
def _ask(robot_id: str, objects: tuple, container, container_xy) -> dict:
    return feasibility.feasibility(
        robot_id, list(objects), container, container_xy=container_xy
    )


def _reasons(res: dict, name: str) -> list[str]:
    return next((e["reasons"] for e in res["excluded"] if e["name"] == name), [])


def route(
    objects: list[str],
    container: str | None = None,
    container_xy=None,
    robot_ids: list[str] | None = None,
    pinned: str | None = None,
    known: dict | None = None,
) -> dict:
    """Choose the arm (or arms) for these library objects.

    `known` maps robot id to a feasibility result already computed for it, so it is not run twice.
    Returns {chosen, assignments, excluded, ranked, pinned}."""
    cxy = tuple(container_xy) if container_xy is not None else None
    objs = tuple(objects)
    ids = robot_ids or available()
    results = {r: (known or {}).get(r) or _ask(r, objs, container, cxy) for r in ids}
    order = sorted(ids, key=lambda r: (not robots.is_default(r), cost(r)))

    ranked = []
    for r in order:
        res = results[r]
        ranked.append(
            {
                "robot": r,
                "label": robots.label(r),
                "cost_nm": cost(r),
                "score": round(len(res["included"]) / max(1, len(objs)), 3),
                "passes": list(res["included"]),
                "fails": [
                    {"object": e["name"], "reasons": e["reasons"]}
                    for e in res["excluded"]
                ],
            }
        )
    ranked.sort(key=lambda x: (-x["score"], order.index(x["robot"])))

    whole = next((x for x in ranked if x["score"] == 1.0), None)
    assignments: list[dict] = []
    left = [o for o in objs]
    if whole:
        chosen = [whole["robot"]]
        assignments.append(
            {
                "robot": whole["robot"],
                "objects": list(objs),
                "reason": (
                    "the default arm does the whole task"
                    if robots.is_default(whole["robot"])
                    else "the cheapest arm that passes fit, jaw width, payload and reach for every object"
                ),
            }
        )
        left = []
    else:
        chosen = []
        remaining = list(objs)
        for x in ranked:
            take = [o for o in remaining if o in x["passes"]]
            if not take:
                continue
            # the share is checked again as a task of its own, so packing and reach hold for this arm alone
            sub = _ask(x["robot"], tuple(take), container, cxy)
            take = list(sub["included"])
            if not take:
                continue
            chosen.append(x["robot"])
            assignments.append(
                {
                    "robot": x["robot"],
                    "objects": take,
                    "reason": f"no single arm can do every object; this one passes all checks for {', '.join(take)}",
                }
            )
            remaining = [o for o in remaining if o not in take]
        left = remaining

    excluded = []
    for o in left:
        best = min(
            ids,
            key=lambda r: (len(_reasons(results[r], o)) or 99, order.index(r)),
        )
        excluded.append(
            {
                "name": o,
                "closest_robot": best,
                "reasons": _reasons(results[best], o)
                or ["no available arm passed the checks"],
            }
        )

    out = {
        "chosen": chosen,
        "assignments": assignments,
        "excluded": excluded,
        "ranked": ranked,
        "pinned": None,
    }
    if pinned:
        mine = next((x for x in ranked if x["robot"] == pinned), None)
        ok = bool(mine) and mine["score"] == 1.0
        if mine is None:
            msg = f"{pinned} was not among the arms checked."
        elif ok:
            msg = f"{robots.label(pinned)} can do the whole task."
        else:
            failed = ", ".join(f["object"] for f in mine["fails"])
            msg = f"{robots.label(pinned)} cannot do {failed}."
            if chosen and chosen != [pinned]:
                msg += " " + (
                    f"{robots.label(chosen[0])} could."
                    if len(chosen) == 1
                    else "It takes "
                    + " and ".join(robots.label(c) for c in chosen)
                    + " together."
                )
        out["pinned"] = {"robot": pinned, "ok": ok, "message": msg}
    return out


def for_spec(
    spec, pinned: str | None = None, robot_ids: list[str] | None = None
) -> dict:
    """The routing for a task spec that carries a feasibility result; {} when it has no library objects."""
    f = spec.feasibility
    if not f:
        return {}
    cont = f.get("container")
    cxy = None
    from . import toolbox

    if cont:
        cxy = toolbox.box_centre()
    return route(
        list(f["checks"]["objects"]),
        cont,
        cxy,
        robot_ids=robot_ids,
        pinned=pinned,
        known={f["robot"]: f},
    )
