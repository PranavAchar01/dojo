"""Can this robot do this task with these objects? Answered from engine data only.

Everything here is measured, never named: each object's collision-hull extents and mass come from the converted
library asset, the container's cavity from the voxel measurement in toolbox.cavity, the jaw opening from the
robot's fingertip stroke, the payload from the actuator force limits at the carry pose, and reach from a top-down
IK solve at the drop point. Objects that fail get plain sentences with the numbers; the rest are checked for
packing together (shelves on the cavity floor, then stacked layers up to the rim).
"""

from __future__ import annotations

import math
from functools import lru_cache

import mujoco
import numpy as np

from . import robots, toolbox

G = 9.81
JAW_MARGIN = (
    0.005  # m kept free of the full opening so the fingers close without scraping
)
FIT_MARGIN = 0.004  # m of clearance each way inside the cavity
REACH_FRAC = 0.9  # share of the table workspace the arm must reach at grasp height (as the Build tab requires)


def _mm(x: float) -> str:
    return f"{x * 1000:.0f} mm"


def _kg(x: float) -> str:
    return f"{x:.2f} kg" if x < 10 else f"{x:.1f} kg"


def geometry(model_id: str) -> dict:
    """Hull extents, longest to shortest, and the mass of a library asset.

    Resting on its smallest face the object stands `short` high; a top-down jaw then closes across the smaller of
    the two horizontal extents, `mid`."""
    v = np.vstack(toolbox.hulls(model_id))
    long_, mid, short = sorted((float(e) for e in v.max(0) - v.min(0)), reverse=True)
    return {
        "long": long_,
        "mid": mid,
        "short": short,
        "mass": float(toolbox.meta(model_id)["mass_kg"]),
    }


@lru_cache(maxsize=64)
def _carry(robot_id: str, radius: float, z: float) -> dict:
    """Payload at a carry pose: top-down at (radius, 0, z) with gravity on, per joint against the actuator limit."""
    r = robots.calibrate(robot_id)
    spec = robots.robot_spec(robot_id)
    for b in spec.bodies:
        b.gravcomp = 0.0
    m = spec.compile()
    d = mujoco.MjData(m)
    robots.fix_qpos0(m)
    site = m.site(r.tcp_site).id
    _q, ep, eo = robots.solve_topdown(
        m,
        d,
        site,
        r.arm_joints,
        r.home,
        np.array([radius, 0.0, z]),
        np.array([1.0, 0.0, 0.0]),
        robots.seeds_for(robot_id, m, site, r.arm_joints),
    )
    reached = bool(ep < robots.REACH_TOL and eo < robots.ORI_TOL["hover"])
    q = d.qpos.copy()
    mujoco.mj_resetData(m, d)
    d.qpos[:] = q
    mujoco.mj_forward(m, d)
    jp = np.zeros((3, m.nv))
    mujoco.mj_jacSite(m, d, jp, None, site)
    best, limiting = math.inf, None
    for j in r.arm_joints:
        a = robots._act_for_joint(m, j)
        if not m.actuator_forcelimited[a]:
            continue
        lim = float(m.actuator_forcerange[a, 1])
        dof = m.joint(j).dofadr[0]
        base, per_kg = float(d.qfrc_bias[dof]), G * float(jp[2, dof])
        if abs(base) >= lim:
            cap = 0.0
        elif abs(per_kg) < 1e-9:
            continue
        else:
            cap = (math.copysign(lim, per_kg) - base) / per_kg
        if cap < best:
            best, limiting = cap, (j, lim)
    return {
        "reached": reached,
        "payload_kg": None if math.isinf(best) else max(0.0, best),
        "joint": limiting[0] if limiting else None,
        "limit_nm": limiting[1] if limiting else None,
    }


def _reaches(robot_id: str, xy, z: float) -> bool:
    r = robots.calibrate(robot_id)
    m = robots.robot_spec(robot_id).compile()
    robots.fix_qpos0(m)
    d = mujoco.MjData(m)
    site = m.site(r.tcp_site).id
    xy = np.asarray(xy, float)
    radial = np.array([*(xy / np.linalg.norm(xy)), 0.0])
    _q, ep, eo = robots.solve_topdown(
        m,
        d,
        site,
        r.arm_joints,
        r.home,
        np.array([*xy, z]),
        radial,
        robots.seeds_for(robot_id, m, site, r.arm_joints),
        w_ori=0.02,
    )
    return bool(ep < robots.REACH_TOL and eo < robots.ORI_TOL["hover"])


def _pack(items: list[tuple[str, dict]], cav: dict) -> tuple[list[str], dict[str, str]]:
    """Shelf-pack footprints (long side along the cavity's longer side) into layers stacked up to the rim.

    Returns (names that fit, {name: why not}). Largest footprint first; a layer is as tall as its tallest item."""
    cx, cy, depth = cav["size"]
    along, across = max(cx, cy) - FIT_MARGIN, min(cx, cy) - FIT_MARGIN
    layers: list[dict] = []  # {"rows": [[used_len, width]], "h": height}
    fits, why = [], {}
    for name, g in sorted(items, key=lambda it: -it[1]["long"] * it[1]["mid"]):
        L, W, H = g["long"], g["mid"], g["short"]
        placed = False
        for layer in layers:
            used_h = sum(x["h"] for x in layers)
            if used_h - layer["h"] + max(layer["h"], H) > depth - FIT_MARGIN:
                continue
            for row in layer["rows"]:
                if row[0] + L <= along and W <= row[1]:
                    row[0] += L
                    layer["h"] = max(layer["h"], H)
                    placed = True
                    break
            else:
                if sum(r[1] for r in layer["rows"]) + W <= across:
                    layer["rows"].append([L, W])
                    layer["h"] = max(layer["h"], H)
                    placed = True
            if placed:
                break
        if not placed:
            used_h = sum(x["h"] for x in layers)
            if used_h + H <= depth - FIT_MARGIN:
                layers.append({"rows": [[L, W]], "h": H})
                placed = True
        if placed:
            fits.append(name)
        else:
            why[name] = (
                f"with the others in the container it runs out of room: the floor "
                f"({_mm(along)} x {_mm(across)}) and {_mm(depth)} of stacking height are used up"
            )
    return fits, why


def feasibility(
    robot_id: str, objects: list[str], container: str | None = None, container_xy=None
) -> dict:
    """Which of these library objects can this robot put in this container (or just pick up when container is None)?

    `container_xy` is where the container's cavity centre sits on the table; default is the middle of the
    workspace. Returns {"included": [...], "excluded": [{"name", "reasons": [...]}], "checks": {...}}."""
    rep = robots.report(robot_id)
    jaw_max = rep["stroke_mm"][1] / 1000.0
    jaw_open = jaw_max - JAW_MARGIN
    cav = toolbox.cavity(container) if container else None
    if container_xy is None:
        container_xy = (sum(robots.WS_R) / 2, 0.0)
    ws_frac = rep["reach"]["fraction"]["grasp"]
    spawn_ok = ws_frac >= REACH_FRAC

    checks: dict = {
        "robot": {
            "jaw_max_mm": round(jaw_max * 1000, 1),
            "jaw_usable_mm": round(jaw_open * 1000, 1),
            "workspace_grasp_reach": ws_frac,
        },
        "objects": {},
    }
    if cav:
        checks["container"] = {
            "name": container,
            "cavity_mm": [round(v * 1000) for v in cav["size"]],
            "rim_m": round(cav["rim"], 3),
        }
    geoms = {n: geometry(n) for n in objects}
    excluded: dict[str, list[str]] = {}
    for name, g in geoms.items():
        why: list[str] = []
        c: dict = {
            "long_mm": round(g["long"] * 1000, 1),
            "grasp_mm": round(g["mid"] * 1000, 1),
            "height_mm": round(g["short"] * 1000, 1),
            "mass_kg": round(g["mass"], 3),
        }
        # fits_container: the object lies flat; its long side along the cavity's longer side
        if cav:
            cx, cy, depth = cav["size"]
            along, across = max(cx, cy) - FIT_MARGIN, min(cx, cy) - FIT_MARGIN
            ok = g["long"] <= along and g["mid"] <= across and g["short"] <= depth
            c["fits_container"] = ok
            if g["long"] > along:
                why.append(
                    f"It is {_mm(g['long'])} long and the container's cavity is only {_mm(along)} long."
                )
            if g["mid"] > across:
                why.append(
                    f"It is {_mm(g['mid'])} wide and the cavity is only {_mm(across)} wide."
                )
            if g["short"] > depth:
                why.append(
                    f"It is {_mm(g['short'])} thick and the cavity is only {_mm(depth)} deep."
                )
        # graspable: top-down jaws close across the smaller horizontal extent of the resting object
        gr = g["mid"] <= jaw_open
        c["graspable"] = gr
        if not gr:
            why.append(
                f"It is {_mm(g['mid'])} across where the jaws close and the gripper opens "
                f"{_mm(jaw_max)} ({_mm(jaw_open)} with a {_mm(JAW_MARGIN)} margin)."
            )
        # liftable: payload at the carry pose over the drop point (or the far edge of the workspace)
        radius = float(np.hypot(*container_xy) if cav else robots.WS_R[1])
        # the jaw tips are lowered to the cavity floor plus the release gap, as the packing expert does
        z = (
            cav["floor_min"] + toolbox.DROP_GAP + toolbox.GRASP_SITE_Z + 0.01
            if cav
            else robots.HOVER_Z
        )
        z = max(z, robots.GRASP_Z)
        carry = _carry(robot_id, round(radius, 3), round(z, 3))
        cap = carry["payload_kg"]
        lf = cap is None or g["mass"] <= cap
        c["liftable"] = lf
        c["payload_kg"] = None if cap is None else round(cap, 3)
        if not lf:
            why.append(
                f"It weighs {_kg(g['mass'])} and the arm can hold {_kg(cap)} at "
                f"{radius * 100:.0f} cm out (the {carry['joint']} motor is limited to {carry['limit_nm']:.2f} N m)."
            )
        # reachable: the spawn area at grasp height and the drop point above the container
        drop_ok = (not cav) or _reaches(robot_id, container_xy, z)
        rc = spawn_ok and drop_ok
        c["reachable"] = rc
        if not spawn_ok:
            why.append(
                f"The arm reaches only {ws_frac:.0%} of the table workspace at grasp height."
            )
        if not drop_ok:
            why.append(
                f"The arm cannot reach the drop point {radius * 100:.0f} cm out at {z * 100:.0f} cm high."
            )
        checks["objects"][name] = c
        if why:
            excluded[name] = why

    if cav:
        ok_items = [(n, g) for n, g in geoms.items() if n not in excluded]
        fits, packing = _pack(ok_items, cav)
        for n, w in packing.items():
            excluded[n] = ["Together with the others, " + w]
            checks["objects"][n]["fits_container"] = False
        checks["packing"] = {"placed": fits, "unplaced": list(packing)}

    included = [n for n in objects if n not in excluded]
    return {
        "robot": robot_id,
        "container": container,
        "included": included,
        "excluded": [
            {"name": n, "reasons": excluded[n]} for n in objects if n in excluded
        ],
        "checks": checks,
    }


def for_spec(spec, robot_id: str) -> dict:
    """The feasibility result for a task spec, or {} when the task has no library objects to check."""
    if spec.skill != "pack":
        return {}
    return feasibility(
        robot_id,
        list(toolbox.ALL_TOOLS),
        toolbox.TOOLBOX,
        container_xy=toolbox.box_centre(),
    )
