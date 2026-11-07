"""The "pack the tools into the toolbox" task, built on the engine's own task / scene / expert / rule path.

Everything here is measured from the library models (assets/library/<id>/): the collision hulls the converter
wrote (the same ones MuJoCo collides with), the converter's mass, and the SO-101 pad span. Nothing is shrunk: the
toolbox and the tools keep their real scale. The pieces:

* `tool_info` / `cavity`: the measured geometry (hull bounding boxes in the lying pose; the toolbox's open
  cavity, found by voxelising the hulls and flood-filling from outside);
* `feasibility`: the numbers the docs quote (does each tool fit the cavity, does the SO-101 jaw span its grasp
  section, does the payload sit inside the actuator limits);
* `add_toolbox` / `add_tools` / `reset_tools`: scene hooks called from sim/scene.py and sim/props.py;
* `sample`: held-out scene sampling (tools lying on the table, toolbox where the cavity sits in the arm's reach);
* `packed`: the code-checked success rule (predicate op "packed" in sim/tasks.py);
* `run_pack`: the scripted expert (called from sim/expert.py), built on the stock Expert's IK and motion helpers.
"""

from __future__ import annotations

import json
import math
from functools import cache, lru_cache
from pathlib import Path

import mujoco
import numpy as np
from scipy import ndimage as ndi

REPO = Path(__file__).resolve().parents[3]
LIB = REPO / "assets" / "library"

TOOLBOX = "metal_toolbox"
# the pipe wrench is in the library too; `feasibility()` says why it is not in the default set
TOOLS = ("ratchet_wrench", "pliers", "screwdriver")
ALL_TOOLS = ("pipe_wrench", *TOOLS)
VOXEL = 0.005
TOOL_RGBA = {  # plain Dojo-light colours (the policy cameras and the browser draw the same flat materials)
    "pipe_wrench": (0.80, 0.30, 0.22),
    "ratchet_wrench": (0.18, 0.40, 0.78),
    "pliers": (0.92, 0.66, 0.16),
    "screwdriver": (0.20, 0.62, 0.40),
}
BOX_RGBA = (0.62, 0.66, 0.72)
SETTLE_SPEED = 0.02  # m/s, the engine's "still" threshold


# ------------------------------------------------------------------ measured geometry


def _obj_vertices(path: Path) -> np.ndarray:
    return np.array(
        [
            [float(v) for v in line.split()[1:4]]
            for line in path.read_text().splitlines()
            if line.startswith("v ")
        ]
    )


@cache
def hulls(model_id: str) -> tuple[np.ndarray, ...]:
    """Vertices of the converter's collision hulls, body frame."""
    return tuple(
        _obj_vertices(p)
        for p in sorted((LIB / model_id / "collision").glob("hull_*.obj"))
    )


@cache
def meta(model_id: str) -> dict:
    return json.loads((LIB / model_id / "meta.json").read_text())


# lying pose: the model's long axis (z) becomes horizontal along the heading, its thinnest horizontal-free axis
# (y for these four) is up. Columns are the images of the model's x, y, z axes.
LYING = np.array([[0.0, 0.0, 1.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])


def _lying_rot(yaw: float) -> np.ndarray:
    c, s = math.cos(yaw), math.sin(yaw)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]]) @ LYING


def _quat(R: np.ndarray) -> np.ndarray:
    q = np.zeros(4)
    mujoco.mju_mat2Quat(q, R.flatten())
    return q


@cache
def tool_info(tool: str) -> dict:
    """Measured size (length along the long axis, horizontal width, height) and the grasp point of a tool lying
    flat, from the collision hulls; the grasp point is the centre of mass along the long axis."""
    v = np.vstack(hulls(tool))
    lo, hi = v.min(0), v.max(0)
    ext = hi - lo  # model frame x, y, z
    # centre of mass from the compiled model (mass-weighted hull inertia)
    m = mujoco.MjModel.from_xml_path(str(LIB / tool / "model.xml"))
    com = m.body_ipos[1] if m.nbody > 1 else np.zeros(3)
    return {
        "length": float(ext[2]),
        "width": float(ext[0]),
        "height": float(ext[1]),
        "com_along": float(com[2] - lo[2]),
        "origin_to_end": float(-lo[2]),
        "mass": float(meta(tool)["mass_kg"]),
        "lo": lo,
        "hi": hi,
    }


def lying_pos(tool: str, xy, yaw: float) -> tuple[np.ndarray, np.ndarray]:
    """Body position + quaternion that lay the tool flat with its hull-bbox centre over xy."""
    v = np.vstack(hulls(tool))
    R = _lying_rot(yaw)
    w = v @ R.T
    centre = (w.min(0) + w.max(0)) / 2
    pos = np.array([xy[0], xy[1], 0.0]) - np.array([centre[0], centre[1], 0.0])
    pos[2] = -w[:, 2].min()
    return pos, _quat(R)


@lru_cache(maxsize=16)
def cavity(container: str = TOOLBOX) -> dict:
    """A container asset's open cavity in its body frame (the toolbox by default), measured from the collision hulls.

    The hulls are voxelised (5 mm); the cavity is the free space not reachable from outside by a flood fill in the
    horizontal slices below the rim. The rim height is the lowest slice at which the cavity leaks to the outside.
    """
    hs = hulls(container)
    from scipy.spatial import ConvexHull

    eqs = [ConvexHull(h).equations for h in hs]
    v = np.vstack(hs)
    lo = np.floor((v.min(0) - 0.02) / VOXEL) * VOXEL
    hi = np.ceil((v.max(0) + 0.02) / VOXEL) * VOXEL
    ax = [np.arange(lo[i], hi[i], VOXEL) + VOXEL / 2 for i in range(3)]
    G = np.stack(np.meshgrid(*ax, indexing="ij"), -1).reshape(-1, 3)
    occ = np.zeros(len(G), bool)
    for e in eqs:
        occ |= np.all(G @ e[:, :3].T + e[:, 3] <= 1e-9, axis=1)
    occ = occ.reshape(len(ax[0]), len(ax[1]), len(ax[2]))
    zs = ax[2]
    start = int(np.searchsorted(zs, 0.03))
    pocket = None
    rim_k = start
    for k in range(start, len(zs)):
        free = ~occ[:, :, k]
        lab, _ = ndi.label(free)
        border = set(np.unique(np.r_[lab[0], lab[-1], lab[:, 0], lab[:, -1]])) - {0}
        inner = free & ~np.isin(lab, list(border))
        if k == start:
            # the pocket: the largest enclosed free region at the first slice above the floor
            lab2, n = ndi.label(inner)
            if n == 0:
                raise RuntimeError("toolbox cavity not found")
            sizes = ndi.sum(inner, lab2, range(1, n + 1))
            pocket = lab2 == (1 + int(np.argmax(sizes)))
        else:
            if not np.any(~np.isin(lab, list(border)) & pocket & free) or (
                (pocket & free & np.isin(lab, list(border))).any()
            ):
                break
        rim_k = k
    rim = float(zs[rim_k] + VOXEL / 2)
    # horizontal bounds: intersect the pocket's bbox over the slices from the floor to the rim
    x0, x1, y0, y1 = -9.0, 9.0, -9.0, 9.0
    floor = 0.0
    for k in range(start, rim_k + 1):
        free = ~occ[:, :, k]
        m = pocket & free
        ii, jj = np.where(m)
        x0, x1 = max(x0, ax[0][ii.min()]), min(x1, ax[0][ii.max()])
        y0, y1 = max(y0, ax[1][jj.min()]), min(y1, ax[1][jj.max()])
    # the floor: the highest occupied voxel under the pocket's centre rows
    ci = np.where((ax[0] > x0) & (ax[0] < x1))[0]
    cj = np.where((ax[1] > y0) & (ax[1] < y1))[0]
    sub = occ[np.ix_(ci, cj)]  # [i, j, k]
    for k in range(len(zs)):
        if not sub[:, :, k].any():
            floor = float(zs[k] - VOXEL / 2)
            break
    # the floor can ramp up inside the box: take the highest floor under any cell of the footprint
    top = np.zeros(sub.shape[:2])
    for k in range(start):
        top = np.where(sub[:, :, k], zs[k] + VOXEL / 2, top)
    floor = float(top.max())
    floor_min = float(top.min())
    return {
        "floor_min": floor_min,
        "x": (float(x0), float(x1)),
        "y": (float(y0), float(y1)),
        "floor": floor,
        "rim": rim,
        "size": (float(x1 - x0), float(y1 - y0), float(rim - floor)),
        "centre": (float((x0 + x1) / 2), float((y0 + y1) / 2)),
        "visual_bbox": tuple(float(a) for a in meta(container)["size_m"]),
    }


# ------------------------------------------------------------------ scene hooks


def _add_meshes(
    spec: mujoco.MjSpec, body, model_id: str, rgba, density: float | None, name: str
):
    """Visual meshes (group 2, plain colour) and the converter's collision hulls (group 3) on `body`."""
    import xml.etree.ElementTree as ET

    root = ET.parse(LIB / model_id / "model.xml").getroot()
    asset = root.find("asset")
    files = {m.get("name"): m.get("file") for m in asset.findall("mesh")}
    spec.add_material(
        name=f"{name}_mat", rgba=[*rgba, 1.0], specular=0.25, shininess=0.4
    )
    for g in root.find("worldbody").iter("geom"):
        if g.get("type") != "mesh":
            continue
        mname, f = g.get("mesh"), files[g.get("mesh")]
        tag = f"{name}/{mname}"
        spec.add_mesh(name=tag, file=str(LIB / model_id / f))
        if g.get("group") == "2":
            body.add_geom(
                type=mujoco.mjtGeom.mjGEOM_MESH,
                meshname=tag,
                material=f"{name}_mat",
                group=2,
                contype=0,
                conaffinity=0,
                density=0,
            )
        else:
            kw = {"density": density} if density is not None else {}
            body.add_geom(
                name=f"{name}_hull_{mname}",
                type=mujoco.mjtGeom.mjGEOM_MESH,
                meshname=tag,
                group=3,
                friction=[0.8, 0.005, 0.0001],
                rgba=[0.7, 0.2, 0.2, 0.3],
                condim=4,
                **kw,
            )


def add_toolbox(spec: mujoco.MjSpec, name: str) -> None:
    """The toolbox as a static fixture (mocap body) with its real converted collision."""
    body = spec.worldbody.add_body(name=name, mocap=True, pos=[0.25, 0, 0])
    _add_meshes(spec, body, TOOLBOX, BOX_RGBA, None, name)
    for g in body.geoms:  # collides with everything the arm and the tools are
        if g.group == 3:
            g.contype = g.conaffinity = 3


def add_tools(spec: mujoco.MjSpec, tools: dict) -> None:
    """Dynamic tools (free bodies) with their real converted meshes, real scale and the converter's mass."""
    from .scene import CUBE_BIT, GRASP_SOLIMP, GRASP_SOLREF

    for name, (tool, *_p) in tools.items():
        body = spec.worldbody.add_body(name=name, pos=[0.2, 0.0, 0.05])
        body.add_freejoint(name=f"{name}_free")
        n0 = len(body.geoms)
        _add_meshes(
            spec, body, tool, TOOL_RGBA[tool], meta(tool)["settings"]["density"], name
        )
        for g in body.geoms[n0:]:
            if g.group == 3:
                g.contype = g.conaffinity = CUBE_BIT | 1
                g.friction = [1.0, 0.01, 0.001]
                g.solref, g.solimp = GRASP_SOLREF, GRASP_SOLIMP


def reset_tools(scene, layout) -> None:
    m, d = scene.model, scene.data
    for name, (tool, x, y, yaw) in layout.tools.items():
        pos, quat = lying_pos(tool, (x, y), yaw)
        adr = m.joint(f"{name}_free").qposadr[0]
        d.qpos[adr : adr + 3] = pos + np.array([0, 0, 0.001])
        d.qpos[adr + 3 : adr + 7] = quat


# ------------------------------------------------------------------ the rule


def _hull_world(scene, name: str, tool: str) -> np.ndarray:
    b = scene.model.body(name).id
    R = scene.data.xmat[b].reshape(3, 3)
    return np.vstack(hulls(tool)) @ R.T + scene.data.xpos[b]


def _to_box_frame(scene, pts: np.ndarray, box: str = "goal") -> np.ndarray:
    b = scene.model.body(box).id
    R = scene.data.xmat[b].reshape(3, 3)
    p = scene.data.mocap_pos[scene.model.body_mocapid[b]]
    return (pts - p) @ R


def packed(scene, tools: list[str] | None = None, box: str = "goal") -> bool:
    """Every tool's hull bounding box lies inside the toolbox's measured cavity (below the rim, so none rests on
    the lid or the rim) and the tool is resting."""
    cav = cavity()
    names = tools or list(scene.layout.tools)
    for n in names:
        tool = scene.layout.tools[n][0]
        p = _to_box_frame(scene, _hull_world(scene, n, tool), box)
        lo, hi = p.min(0), p.max(0)
        if not (
            lo[0] >= cav["x"][0] - 0.004
            and hi[0] <= cav["x"][1] + 0.004
            and lo[1] >= cav["y"][0] - 0.004
            and hi[1] <= cav["y"][1] + 0.004
            and hi[2] <= cav["rim"]
        ):
            return False
        adr = scene.model.joint(f"{n}_free").dofadr[0]
        if np.linalg.norm(scene.data.qvel[adr : adr + 6]) >= SETTLE_SPEED:
            return False
    return True


# ------------------------------------------------------------------ held-out sampling

BOX_YAW = math.pi  # the box's long axis lies radially, its open lid leans to -y, away from the tools
BOX_CENTRE = (0.25, -0.15)  # world xy of the cavity centre: the near half of the cavity is inside the arm's reach


def sample(rng: np.random.Generator, tools=TOOLS) -> tuple[dict, dict]:
    """(toolbox prop, tools) for one scene: the cavity centre jittered a little; the tools lie on the table on the
    +y side, along the radial direction (the jaws close across the width), apart from each other."""
    cav = cavity()
    cx, cy = cav["centre"]
    c, s = math.cos(BOX_YAW), math.sin(BOX_YAW)
    ctr = np.array(BOX_CENTRE) + rng.uniform(-0.01, 0.01, 2)
    origin = ctr - np.array([c * cx - s * cy, s * cx + c * cy])
    box = {"goal": ("toolbox", "gray", float(origin[0]), float(origin[1]), BOX_YAW)}
    placed: list[tuple[float, float, float, float]] = []
    out = {}
    for i, t in enumerate(tools):
        info = tool_info(t)
        for _ in range(2000):
            x = rng.uniform(0.17, 0.26)
            y = rng.uniform(0.04, 0.26)
            a = rng.uniform(-0.15, 0.15) + math.pi * rng.integers(0, 2)  # along x, head either way
            ex_ = abs(math.cos(a)) * info["length"] / 2 + abs(math.sin(a)) * info["width"] / 2
            ey_ = abs(math.sin(a)) * info["length"] / 2 + abs(math.cos(a)) * info["width"] / 2
            if all(abs(y - qy) > ey_ + qey + 0.012 or abs(x - qx) > ex_ + qex + 0.012 for qx, qy, qex, qey in placed):
                break
        else:
            raise RuntimeError("no free spot for a tool")
        placed.append((x, y, ex_, ey_))
        out["target" if i == 0 else f"tool{i}"] = (t, float(x), float(y), float(a))
    return box, out


# ------------------------------------------------------------------ the scripted expert


DROP_GAP = 0.008  # the tool is released this far over the floor, or over the tools already in the box
SLOT = ((0.0, -0.075), (0.0, -0.03), (0.0, -0.052))  # box-frame (x, y) the tools are lowered to
GRASP_SITE_Z = 0.005  # jaw tips just off the table, as for the 3 cm blocks
CARRY_UP = 0.03


def _tool_state(sc, name: str, tool: str):
    """(grasp point = centre of mass, hull bbox centre, long-axis yaw) of a tool, from the live simulation."""
    b = sc.model.body(name).id
    com = sc.data.xipos[b].copy()
    pts = _hull_world(sc, name, tool)
    centre = (pts.min(0) + pts.max(0)) / 2
    R = sc.data.xmat[b].reshape(3, 3)
    long_axis = R[:, 2]
    return com, centre, float(math.atan2(long_axis[1], long_axis[0])), pts


def run_pack(ex) -> None:
    """Pack every tool: grasp at the centre of mass with the jaws across the width, lift, carry over the cavity,
    lower to just over the rim and open; the tool falls the last centimetres and settles. Built on the stock
    Expert's IK and motion helpers (goto / grip)."""
    from .expert import APPROACH_SPEED, DOWN, HOVER_Z, LIFT_SPEED, TRANSIT_SPEED

    sc, r = ex.sc, ex.r
    cav = cavity()
    box = sc.model.body("goal").id
    bp = sc.data.mocap_pos[sc.model.body_mocapid[box]]
    Rb = sc.data.xmat[box].reshape(3, 3)
    tools = sc.layout.tools
    order = sorted(tools, key=lambda n: ["target", "tool1", "tool2"].index(n))
    rim_z = cav["rim"]
    for k, name in enumerate(order):
        tool = tools[name][0]
        com, centre, yaw, pts = _tool_state(sc, name, tool)
        width = float(
            np.ptp(
                pts @ np.array([-math.sin(yaw), math.cos(yaw), 0.0])
            )
        )
        hw = width / 2
        # jaws close across the tool's width: the face normal closest to the radial direction
        radial = com[:2] / max(np.linalg.norm(com[:2]), 1e-9)
        cands = [
            np.array([-math.sin(yaw), math.cos(yaw), 0.0]),
            np.array([math.sin(yaw), -math.cos(yaw), 0.0]),
        ]
        heading = max(cands, key=lambda n: float(n[:2] @ radial))
        ex.ik.heading = heading
        c_open = r.centre_open - 0.015 + hw
        c_grasp = r.centre_grasp - 0.015 + hw
        bottom = pts[:, 2].min()

        def site(xy, z, c):
            return np.array([*xy, z]) - c * heading

        ex.g_goal = r.g_open
        ex.goto([*site(com[:2], HOVER_Z, c_open)[:2], HOVER_Z], TRANSIT_SPEED)
        ex.goto(site(com[:2], bottom + GRASP_SITE_Z, c_open), APPROACH_SPEED)
        ex.calm = True
        ex.goto(site(com[:2], bottom + GRASP_SITE_Z, c_grasp), APPROACH_SPEED * 0.5, settle=0.05)
        ex.grip(r.g_squeeze, settle=0.5)
        ex.calm = False
        up = bottom + GRASP_SITE_Z + CARRY_UP
        ex.goto(site(com[:2], up, c_grasp), LIFT_SPEED, settle=0.1)
        # carry to over the cavity and lower the tool to just over what is already there
        com, centre, _yaw, _pts = _tool_state(sc, name, tool)
        grip_off = com[:2] - centre[:2]
        bx, by = SLOT[k % len(SLOT)]
        goal_xy = bp[:2] + Rb[:2, :2] @ np.array([bx, by])
        held = sc.ee_pos()[2] - _pts[:, 2].min()  # jaw-tip height above the tool's bottom
        rest = cav["floor_min"] + DROP_GAP
        for other in order[:k]:
            o = _hull_world(sc, other, tools[other][0])
            rest = max(rest, float(o[:, 2].max()) + DROP_GAP)
        release = rest + held
        safe = max(release, up) + 0.01
        ex.goto(site(com[:2], safe, c_grasp), LIFT_SPEED, settle=0.0)
        ex.calm = True
        ex.goto(site(goal_xy + grip_off, safe, c_grasp), TRANSIT_SPEED, settle=0.2)
        ex.goto(site(goal_xy + grip_off, release, c_grasp), LIFT_SPEED, settle=0.15)
        ex.grip(r.g_open, settle=0.5)
        ex.goto(site(goal_xy + grip_off, safe + 0.02, c_open), LIFT_SPEED, settle=0.0)
        ex.calm = False
    ex.home()
    ex.hold(3.0)  # the last tool settles in the cavity
