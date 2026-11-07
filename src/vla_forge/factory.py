"""Factory lines: a company describes its line, the planner splits it into stations along one conveyor, the router
checks each station, missing station skills are trained through the normal agent pipeline, and the whole line
runs part by part.

A station is one arm beside the conveyor with one tote. Its job is a skill: a trained VLA from the library
("vla") or the scripted expert with privileged state ("scripted"), and the page labels which. Every station
runs in its own MuJoCo cell (sim/line.py): the zone's belt brings the part in, stops at the photo-eye if the part
is this station's, the station moves it into its tote, the arm goes home, and parts for later stations ride
through to the next zone. The schedule across zones (one part per zone at a time, parts released at a fixed
interval) gives the line's makespan and throughput. Durations are simulated time, not wall clock.
"""

from __future__ import annotations

import asyncio
import hashlib
import itertools
import json
import math
import time
from dataclasses import asdict, dataclass, field

import mujoco
import numpy as np

from . import llm
from .router import Check, Route
from .sim import line as L
from .sim.props import KIT_SLOTS
from .sim.robots import is_default, label, robot_feasibility
from .sim.scene import BLOCK_COLORS, CUBE_HALF, Layout, Look, Scene
from .sim.tasks import TaskSpec, canonical_instruction, check

ROOT = L.REPO
LINES_DIR = ROOT / "runs" / "lines"
FPS = 15  # line playback frame rate (recorded at 30 Hz, every other frame kept)
RELEASE_EVERY = 3.0  # s between parts entering the line
HOME_S = 1.2  # scripted return to home after a station finishes (industrial arms home between cycles)
PART_COLORS = ("red", "blue", "green", "yellow", "orange", "purple")
ROBOTS = ("so101", "vx300s.native.native")
ANY = "any"  # a catch-all station takes every part that reaches it
# assembly fixtures: where the pallet (a 2 x 2 tray) and the socket stand in a station's cell, off the belt
TRAY_AT = (0.135, 0.11)
SOCKET_AT = (0.145, 0.095)
ASSEMBLY_SKILLS = ("kit", "insert")
# a station skill is only reused in the cell it was trained in; moving a fixture changes this tag
CELL_TAG = hashlib.sha1(repr((TRAY_AT, SOCKET_AT)).encode()).hexdigest()[:6]


@dataclass
class Station:
    index: int
    name: str
    part: str  # part colour it takes, or "any"
    bin: str  # tote colour
    robot: str = "so101"
    controller: str = "vla"  # vla | scripted
    skill_id: str = ""
    route: dict = field(default_factory=dict)
    status: str = (
        "planned"  # planned | reuse | train | training | ready | scripted | refused
    )
    task: str = "pick_place"  # pick_place (sorting), kit (seat in a pallet slot) or insert (into a socket)
    slot: str = ""  # kit only: which pallet slot, e.g. "front-left"

    @property
    def instruction(self) -> str:
        if self.task != "pick_place":
            return self.spec().instruction
        what = "the block" if self.part == ANY else f"the {self.part} block"
        return f"put {what} in the {self.bin} bin"

    def spec(self, part: str | None = None) -> TaskSpec:
        color = part or (self.part if self.part != ANY else "red")
        if self.task != "pick_place":
            target = (self.slot or "front-left") if self.task == "kit" else None
            return TaskSpec(
                self.task,
                color,
                target,
                distractors=0,
                instruction=canonical_instruction(self.task, color, target),
            )
        return TaskSpec(
            "pick_place",
            color,
            self.bin,
            distractors=0,
            instruction=self.instruction
            if self.part != ANY
            else f"put the {color} block in the {self.bin} bin",
            success=[
                {"op": "in_bin", "block": "target", "bin": "goal"},
                {"op": "still", "block": "target"},
            ],
        )

    def takes(self, color: str) -> bool:
        return self.part in (ANY, color)

    def to_json(self) -> dict:
        return {
            **asdict(self),
            "instruction": self.instruction,
            "robot_label": label(self.robot),
        }


@dataclass
class Line:
    id: str
    prompt: str
    name: str
    stations: list[Station]
    parts: list[str]  # the part colours fed in, in order
    plan: dict = field(default_factory=dict)  # which model planned it, notes
    created: float = field(default_factory=time.time)
    status: str = "planned"  # planned | building | running | done | failed
    result: dict = field(default_factory=dict)
    training: list[dict] = field(
        default_factory=list
    )  # [{station, run, seconds, success_rate}]
    assembly: bool = False  # every part visits every station in order (else: each part goes to one station)

    def to_json(self, full: bool = False) -> dict:
        d = {
            "id": self.id,
            "prompt": self.prompt,
            "name": self.name,
            "stations": [s.to_json() for s in self.stations],
            "parts": self.parts,
            "plan": self.plan,
            "created": self.created,
            "status": self.status,
            "training": self.training,
            "assembly": self.assembly,
            "result": {k: v for k, v in self.result.items() if full or k != "playback"},
        }
        return d


# ------------------------------------------------------------------ planning

PLAN_SYSTEM = f"""You turn a company's description of a factory line into stations along one conveyor for a
simulated cell. Parts are coloured cubes ({", ".join(PART_COLORS)}) that ride the conveyor one at a time.
Every station is one robot arm beside the belt with one tote; a station picks the parts of one colour off the
belt into its tote, or every part that reaches it (part = "any", only as the last station: the end of line).
Tote colours: {", ".join(L.BIN_COLORS)}. Robots: "so101" (SO-101 arm) or "vx300s.native.native" (ViperX 300).
controller: "vla" when the station should get its own trained vision-language-action policy, "scripted" when
the description asks for a fixed, programmed or scripted cell (use "scripted" for an end-of-line catch-all
unless the text asks otherwise, and "vla" otherwise). Keep stations in the order parts meet them. At most 4
stations. parts: the colours the line receives (2 to 4 colours). name: a short name for the line. note: one
short sentence on anything in the description you could not map (empty if none). If the description is not a
conveyor sorting/assembly line at all, return no stations and say why in note.

Assembly lines: when the description is about building or assembling something (a housing, a pallet, a part
inserted into a socket, a finished assembly into a crate), set assembly = true. Then every part visits every
station in order (parts: one colour, "red"), each station has a task: "kit" (seat the part in a slot of a
2 x 2 pallet; slot = front-left, front-right, back-left or back-right), "insert" (press it into a square
socket) or "pick_place" (move it into a tote; the crate is the tote "gray"). Usual shape: SO-101 "vla" kit
station, SO-101 "vla" insert station, then a "scripted" ViperX pick_place station that moves the finished
assembly into the crate (part = "any" there). Sorting lines keep assembly = false and task = "pick_place"."""

PLAN_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["name", "stations", "parts", "note", "assembly"],
    "properties": {
        "name": {"type": "string"},
        "assembly": {"type": "boolean"},
        "note": {"type": "string"},
        "parts": {
            "type": "array",
            "items": {"type": "string", "enum": list(PART_COLORS)},
        },
        "stations": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": [
                    "name",
                    "part",
                    "bin",
                    "robot",
                    "controller",
                    "task",
                    "slot",
                ],
                "properties": {
                    "name": {"type": "string"},
                    "task": {
                        "type": "string",
                        "enum": ["pick_place", "kit", "insert"],
                    },
                    "slot": {
                        "type": "string",
                        "enum": ["", *KIT_SLOTS],
                    },
                    "part": {"type": "string", "enum": [*PART_COLORS, ANY]},
                    "bin": {"type": "string", "enum": list(L.BIN_COLORS)},
                    "robot": {"type": "string", "enum": list(ROBOTS)},
                    "controller": {"type": "string", "enum": ["vla", "scripted"]},
                },
            },
        },
    },
}


def _rule_plan(text: str) -> dict:
    """Offline stand-in: '<colour> parts ... <colour> bin' pairs in order, plus an end-of-line catch-all."""
    import re

    t = text.lower()
    if re.search(r"\b(assembl\w*|housing|pallet|socket|insert\w*|crate)\b", t):
        return {**assembly_plan(), "note": ""}
    stations, parts = [], []
    for m in re.finditer(
        r"\b("
        + "|".join(PART_COLORS)
        + r")\b[^.,;]*?\b("
        + "|".join(L.BIN_COLORS)
        + r")\s+(?:bin|tote)",
        t,
    ):
        stations.append(
            {
                "name": f"Sort {m.group(1)}",
                "part": m.group(1),
                "bin": m.group(2),
                "robot": "so101",
                "controller": "vla",
            }
        )
        parts.append(m.group(1))
    if re.search(
        r"\b(rest|everything else|reject|leftover|end of (the )?line|anything else)\b",
        t,
    ):
        stations.append(
            {
                "name": "End of line",
                "part": ANY,
                "bin": "gray",
                "robot": "vx300s.native.native",
                "controller": "scripted",
            }
        )
        parts += [c for c in ("green", "yellow") if c not in parts][:1]
    return {
        "name": "Sort line",
        "stations": stations,
        "parts": parts,
        "note": "" if stations else "no stations found",
    }


def assembly_plan() -> dict:
    """The default assembly line: a VLA station seats the housing in a pallet nest, a VLA station inserts it into
    a socket, a scripted ViperX moves the finished assembly into the crate."""
    return {
        "name": "Housing assembly line",
        "assembly": True,
        "parts": ["red"],
        "note": "",
        "stations": [
            {
                "name": "Seat housing in pallet",
                "part": "red",
                "bin": "gray",
                "robot": "so101",
                "controller": "vla",
                "task": "kit",
                "slot": "front-left",
            },
            {
                "name": "Insert into socket",
                "part": "red",
                "bin": "gray",
                "robot": "so101",
                "controller": "vla",
                "task": "insert",
                "slot": "",
            },
            {
                "name": "Assembly to crate",
                "part": ANY,
                "bin": "gray",
                "robot": "vx300s.native.native",
                "controller": "scripted",
                "task": "pick_place",
                "slot": "",
            },
        ],
    }


def plan_line(text: str) -> Line:
    t0 = time.time()
    try:
        raw, model = llm._chat_json(PLAN_SYSTEM, text, PLAN_SCHEMA, "plan")
    except RuntimeError:
        raw, model = _rule_plan(text), "keyword planner (offline)"
    stations = []
    seen = set()
    assembly = bool(raw.get("assembly"))
    for i, s in enumerate(raw.get("stations", [])[:4]):
        part = s.get("part") if s.get("part") in (*PART_COLORS, ANY) else None
        if part is None or (part == ANY and i != len(raw["stations"][:4]) - 1):
            continue
        if part in seen and not assembly:
            continue
        seen.add(part)
        task = s.get("task") if s.get("task") in ("kit", "insert") else "pick_place"
        stations.append(
            Station(
                index=len(stations),
                name=str(s.get("name", ""))[:40] or f"Station {len(stations) + 1}",
                part="red" if assembly and part != ANY else part,
                bin=s["bin"] if s.get("bin") in L.BIN_COLORS else "gray",
                robot=s["robot"] if s.get("robot") in ROBOTS else "so101",
                controller="scripted" if s.get("controller") == "scripted" else "vla",
                task=task if assembly else "pick_place",
                slot=(s.get("slot") if s.get("slot") in KIT_SLOTS else "front-left")
                if task == "kit" and assembly
                else "",
            )
        )
    parts = [c for c in raw.get("parts", []) if c in PART_COLORS][:4] or [
        s.part for s in stations if s.part != ANY
    ]
    if assembly:
        parts = ["red"]
    elif stations and stations[-1].part == ANY and all(c in seen for c in parts):
        parts.append(
            next(c for c in PART_COLORS if c not in seen)
        )  # something for the end of line to catch
    lid = (
        time.strftime("%H%M%S")
        + "-"
        + hashlib.sha1(f"{text}{time.time()}".encode()).hexdigest()[:4]
    )
    return Line(
        lid,
        text,
        str(raw.get("name", "Line"))[:48],
        stations,
        parts,
        plan={
            "model": model,
            "seconds": round(time.time() - t0, 1),
            "note": raw.get("note", ""),
        },
        assembly=assembly,
    )


# ------------------------------------------------------------------ routing


def _reach_check(st: Station, n: int = 4) -> Check:
    """IK, jaws down, over where this station's parts stop and where its tote stands."""
    from .router import REACH_TOL
    from .sim.expert import HOVER_Z, IK, RELEASE_Z

    rng = np.random.default_rng(st.index)
    lays = [train_layout(st, rng) for _ in range(n)]
    sc = Scene(lays[0], st.robot)
    ik = IK(sc)
    q0 = sc.joint_pos()[: ik.n]
    worst, miss, tot = 0.0, 0, 0
    for lay in lays:
        pts = [(x, y, HOVER_Z) for _c, x, y, _yaw in lay.blocks.values()]
        pts += [(x, y, RELEASE_Z) for _c, x, y in lay.bins.values()]
        if "goal" in lay.props:  # the pallet slot or the socket
            pts.append((*lay.props["goal"][2:4], RELEASE_Z))
        for x, y, z in pts:
            _q, err = ik.solve(np.array([x, y, z]), q0, iters=150)
            worst, miss, tot = max(worst, err), miss + (err > REACH_TOL), tot + 1
    sc.close()
    if miss:
        return Check(
            "reach",
            False,
            f"{miss} of {tot} pick and drop points in {n} cells are out of reach",
        )
    return Check(
        "reach",
        True,
        f"IK reaches the photo-eye stop and the tote in {n} sampled cells (worst miss {worst * 1000:.1f} mm)",
    )


def route_station(st: Station | dict, registry) -> Route:
    """reuse a station skill from the library, train one, run the scripted cell, or refuse (with the reason)."""
    if isinstance(st, dict):
        st = Station(**{k: st[k] for k in Station.__dataclass_fields__ if k in st})
    checks = []
    ok, why = robot_feasibility(st.robot, st.task)
    checks.append(Check("robot", ok, f"{label(st.robot)}: {why}"))
    if not ok:
        return Route("not_feasible", why, checks)
    reach = _reach_check(st)
    checks.append(reach)
    if not reach.ok:
        return Route("not_feasible", reach.detail, checks)
    if st.controller == "scripted":
        checks.append(
            Check(
                "controller",
                True,
                "scripted cell: the expert with privileged state, no policy to train",
            )
        )
        return Route(
            "scripted", "runs the scripted pick and place; nothing to train", checks
        )
    if st.part == ANY:
        return Route(
            "not_feasible",
            "a catch-all station has to be scripted (one VLA per part colour)",
            checks,
        )
    if st.task == "pick_place":
        sk = registry.exact(st.instruction, st.robot)
    else:  # only a skill trained in this kind of cell (belt, pallet or socket where the line puts it)
        hits = [
            k
            for k in registry.skills.values()
            if k.origin == "station"
            and not k.composite
            and k.instruction == st.instruction
            and k.robot == st.robot
            and f"cell {CELL_TAG}" in k.prompt
        ]
        sk = max(hits, key=lambda k: (k.success_rate, k.created)) if hits else None
    if sk is not None:
        checks.append(
            Check(
                "library",
                True,
                f"tile '{sk.instruction}' on this robot, {sk.success_rate:.0%} on held-out cells",
            )
        )
        return Route(
            "reuse",
            f"reuse tile {sk.id}",
            checks,
            skill_id=sk.id,
            expected=sk.success_rate,
        )
    near_id, _p, _sim, reason = registry.nearest(st.spec(), st.robot)
    checks.append(Check("library", True, reason))
    est = _train_estimate(registry)
    return Route(
        "train",
        f"train a station skill, {reason}",
        checks,
        estimate_s=est,
        skill_id=near_id or "",
    )


def _train_estimate(registry) -> float:
    ts = [
        s.seconds
        for s in registry.skills.values()
        if s.origin == "station" and s.seconds
    ]
    if not ts:
        ts = [
            s.seconds
            for s in registry.skills.values()
            if not s.composite and s.seconds and s.task().skill == "pick_place"
        ]
    return float(np.median(ts)) if ts else 170.0


def route_line(line: Line, registry) -> None:
    for st in line.stations:
        r = route_station(st, registry)
        st.route = r.to_json()
        st.status = {"reuse": "ready", "train": "train", "scripted": "scripted"}.get(
            r.decision, "refused"
        )
        if r.decision == "reuse":
            st.skill_id = r.skill_id


# ------------------------------------------------------------------ building (training missing stations)


async def build_line(line: Line, agents, forge, on_event) -> None:
    """Train every station that needs a skill, one after another, through the normal agent pipeline."""
    line.status = "building"
    for st in line.stations:
        if st.status != "train":
            continue
        st.status = "training"
        on_event("station", station=st.to_json())
        sj = {
            **st.to_json(),
            "spec": st.spec().to_json(),
            "line_name": line.name,
            "line": line.id,
        }
        prompt = (
            st.instruction
            if st.task == "pick_place"
            else f"{st.instruction} (cell {CELL_TAG})"
        )
        run = await agents.submit(prompt, "train", "", station=sj)
        on_event("station_run", station=st.index, run=run.id)
        fut = agents.pending.get(run.id)
        t0 = time.time()
        while (
            run.status not in ("published", "rejected", "failed", "error")
            and time.time() - t0 < 900
        ):
            if fut is not None and fut.done():
                break
            await asyncio.sleep(0.5)
        if run.skill_id:
            st.skill_id, st.status = run.skill_id, "ready"
            sk = forge.registry.skills[run.skill_id]
            line.training.append(
                {
                    "station": st.index,
                    "run": run.id,
                    "seconds": run.seconds,
                    "success_rate": sk.success_rate,
                    "n_eval": sk.n_eval,
                    "from": run.init_from.get("skill"),
                }
            )
        else:
            st.status = "refused"
            st.route = {**st.route, "reason": run.error or "training did not publish"}
        on_event("station", station=st.to_json())


# ------------------------------------------------------------------ running


def _frames_30hz(frames10: list[list[float]]) -> list[list[float]]:
    """Expert ticks come at 10 Hz; interpolate positions and quaternions to 30 Hz like the policy recordings."""
    out = []
    for a, b in itertools.pairwise(frames10):
        fa, fb = np.asarray(a).reshape(-1, 7), np.asarray(b).reshape(-1, 7)
        for k in range(3):
            t = k / 3
            f = fa + (fb - fa) * t
            q = f[:, 3:]
            f[:, 3:] = q / np.linalg.norm(q, axis=1, keepdims=True)
            out.append(np.round(f, 5).flatten().tolist())
    out.append(frames10[-1])
    return out


def _belt_phase(sc: Scene, until_y: float, max_s: float = 12.0) -> list[list[float]]:
    """Run the zone's belt until the part's centre passes until_y, then stop it and let things settle."""
    m, d = sc.model, sc.data
    frames, k = [], 0
    while sc.body_pos("target")[1] < until_y and k * m.opt.timestep < max_s:
        L.drive_belt(m, d, L.BELT_SPEED)
        mujoco.mj_step(m, d)
        k += 1
        if k % 10 == 0:
            frames.append(sc.pose_frame())
    L.drive_belt(m, d, 0.0)
    for j in range(60):
        mujoco.mj_step(m, d)
        if j % 10 == 9:
            frames.append(sc.pose_frame())
    return frames


def _home_phase(sc: Scene, q_from: np.ndarray, frame: list[float]) -> list[list[float]]:
    """Put the cell back in the state of the last recorded frame and drive the arm home over HOME_S."""
    m, d = sc.model, sc.data
    f = np.asarray(frame).reshape(-1, 7)
    bid = m.body("target").id
    adr = m.joint("target_free").qposadr[0]
    d.qpos[adr : adr + 3] = f[bid - 1, :3]
    d.qpos[adr + 3 : adr + 7] = f[bid - 1, 3:]
    d.qvel[:] = 0
    d.qpos[sc.qadr] = q_from[: len(sc.qadr)]
    mujoco.mj_forward(m, d)
    target = np.append(sc.robot.home, q_from[-1])
    n = int(HOME_S / (m.opt.timestep * 6))
    frames = []
    for i in range(n):
        a = 0.5 - 0.5 * math.cos(math.pi * (i + 1) / n)
        d.ctrl[:] = sc.to_ctrl(q_from + (target - q_from) * a)
        for s in range(6):
            mujoco.mj_step(m, d)
            if (i * 6 + s) % 10 == 9:
                frames.append(sc.pose_frame())
    return frames


def _cell_layout(
    st: Station,
    color: str,
    x: float,
    y: float,
    yaw: float,
    rng: np.random.Generator | None = None,
) -> Layout:
    """A station's cell with the part at (x, y). Sorting: a tote. Kit: a 2 x 2 pallet beside the belt with a goal
    marker at the named slot. Insert: a socket beside the belt. rng jitters where the fixture stands."""
    from .sim.props import slot_offset

    lay = Layout(belt=True)
    lay.blocks["target"] = (color, x, y, yaw)
    if st.task == "pick_place":
        lay.bins["goal"] = (st.bin, *L.TOTE_XY)
        return lay
    j = (lambda a: float(rng.uniform(-a, a))) if rng is not None else (lambda a: 0.0)
    if st.task == "kit":
        fy = j(0.1)
        cx, cy = TRAY_AT[0] + j(0.008), TRAY_AT[1] + j(0.008)
        ox, oy = slot_offset(st.slot or "front-left")
        gx = cx + ox * math.cos(fy) - oy * math.sin(fy)
        gy = cy + ox * math.sin(fy) + oy * math.cos(fy)
        lay.props["tray"] = ("tray", "gray", cx, cy, fy)
        lay.props["goal"] = ("marker", "gray", gx, gy, fy)
    else:
        sx, sy = SOCKET_AT[0] + j(0.006), SOCKET_AT[1] + j(0.008)
        lay.props["goal"] = ("socket", "gray", sx, sy, math.atan2(sy, sx))
    return lay


def train_layout(st: Station | dict, rng: np.random.Generator) -> Layout:
    """A training or held-out scene for a station: the part stopped at the photo-eye, the station's fixtures."""
    if isinstance(st, dict):
        st = Station(**{k: st[k] for k in Station.__dataclass_fields__ if k in st})
    if st.task == "pick_place":
        return L.station_layout(st.spec().obj, st.bin, rng)
    lay = L.station_layout(st.spec().obj, "gray", rng)
    _c, x, y, yaw = lay.blocks["target"]
    return _cell_layout(st, st.spec().obj, x, y, yaw, rng)


def _yaw_of(frame: list[float], body: int) -> float:
    w, x, y, z = np.asarray(frame).reshape(-1, 7)[body - 1, 3:]
    return math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


def run_line(
    line: Line, registry, compute, n_parts: int = 12, seed: int = 7, on_progress=None
) -> dict:
    """Feed n_parts parts through every station, recording each zone's episodes; then schedule the zones."""
    from .encoders import device
    from .policy import VLA
    from .rollout import evaluate

    t_wall = time.time()
    rng = np.random.default_rng(seed)
    pal = line.parts or ["red", "blue"]
    parts = [pal[int(rng.integers(len(pal)))] for _ in range(n_parts)]
    load_yaw = [float(rng.uniform(-0.6, 0.6)) for _ in range(n_parts)]
    on_line = list(range(n_parts))  # parts still riding the conveyor
    episodes: list[dict] = []
    fate = {
        i: {
            "part": i,
            "color": parts[i],
            "station": None,
            "ok": False,
            "why": "reached the end of the line",
        }
        for i in range(n_parts)
    }
    per_station = []
    for st in line.stations:
        if line.assembly:  # every part that is still good goes through every station
            mine, passing = list(on_line), []
        else:
            mine = [i for i in on_line if st.takes(parts[i])]
            passing = [i for i in on_line if i not in mine]
        stats = {"station": st.index, "parts": len(mine), "succeeded": 0, "work_s": []}
        # belt phases (scripted line control)
        arrivals = {}
        for i in on_line:
            lay = _cell_layout(
                st,
                parts[i],
                L.BELT_X,
                L.ENTRY_Y,
                math.atan2(L.WINDOW_Y, L.BELT_X) + load_yaw[i],
            )
            sc = Scene(lay, st.robot)
            sc.reset(lay)
            stop = L.WINDOW_Y + 0.011 if i in mine else L.EXIT_Y
            fr = [sc.pose_frame(), *_belt_phase(sc, stop)]
            p = sc.body_pos("target")
            arrivals[i] = (
                fr,
                float(p[0]),
                float(p[1]),
                _yaw_of(fr[-1], sc.model.body("target").id),
            )
            sc.close()
        for i in passing:
            episodes.append(
                {
                    "station": st.index,
                    "part": i,
                    "color": parts[i],
                    "kind": "pass",
                    "frames": arrivals[i][0],
                    "ok": True,
                }
            )
        if mine:
            lays = [
                _cell_layout(
                    st,
                    parts[i],
                    *arrivals[i][1:],
                    rng=np.random.default_rng(seed * 7919 + st.index * 131 + i),
                ).to_json()
                for i in mine
            ]
            results = []
            if st.controller == "vla":
                sk = registry.skills[st.skill_id]
                model, _ = VLA.load(ROOT / sk.model, device())
                spec = st.spec()
                with compute:
                    rolls = evaluate(
                        model,
                        spec,
                        len(lays),
                        layouts=lays,
                        record=True,
                        robot=st.robot,
                        seed0=seed * 1000 + st.index * 100,
                    )
                for r in rolls:
                    results.append(
                        (
                            r.success,
                            r.frames,
                            np.asarray(r.final_q),
                            r.steps * 0.1,
                            r.failure,
                        )
                    )
            else:
                for i, lj in zip(mine, lays, strict=True):
                    from .sim.expert import Expert

                    spec = st.spec(parts[i])
                    lay = Layout.from_json(lj)
                    sc = Scene(lay, st.robot)
                    sc.reset(lay, Look.sample(np.random.default_rng(i)))
                    f10 = []
                    ex = Expert(
                        sc,
                        spec,
                        on_tick=lambda s, f=f10: f.append(s.pose_frame()),
                        rng=np.random.default_rng(i),
                    )
                    ex.run()
                    ok = check(spec, sc, lay)
                    results.append(
                        (
                            bool(ok),
                            _frames_30hz(f10),
                            sc.joint_pos(),
                            len(f10) * 0.1,
                            "" if ok else "missed the tote",
                        )
                    )
                    sc.close()
            for i, lj, (ok, frames, qf, work, why) in zip(
                mine, lays, results, strict=True
            ):
                home = []
                if st.controller == "vla" and is_default(st.robot) and frames:
                    sc = Scene(Layout.from_json(lj), st.robot)
                    sc.reset(Layout.from_json(lj))
                    home = _home_phase(sc, qf, frames[-1])
                    sc.close()
                episodes.append(
                    {
                        "station": st.index,
                        "part": i,
                        "color": parts[i],
                        "kind": "pick",
                        "frames": arrivals[i][0] + frames + home,
                        "work_s": round(work, 2),
                        "ok": bool(ok),
                        "failure": why,
                    }
                )
                stats["succeeded"] += bool(ok)
                stats["work_s"].append(round(work, 2))
                fate[i] = {
                    "part": i,
                    "color": parts[i],
                    "station": st.index,
                    "ok": bool(ok),
                    "why": "" if ok else why,
                }
        on_line = [i for i in mine if fate[i]["ok"]] if line.assembly else passing
        stats["mean_work_s"] = (
            round(float(np.mean(stats["work_s"])), 2) if stats["work_s"] else None
        )
        per_station.append(stats)
        if on_progress:
            on_progress(
                {
                    "station": st.index,
                    **{k: v for k, v in stats.items() if k != "work_s"},
                }
            )
    sched = _schedule(line, episodes, n_parts)
    done = sum(f["ok"] for f in fate.values())
    makespan = sched["makespan"]
    for e in episodes:  # keep playback light: 15 fps, 4 decimals
        e["frames"] = [[round(v, 4) for v in f] for f in e["frames"][::2]]
    result = {
        "n_parts": n_parts,
        "parts": parts,
        "completed": done,
        "success_rate": round(done / n_parts, 3),
        "makespan_s": round(makespan, 1),
        "throughput_per_min": round(60 * done / makespan, 2) if makespan else 0.0,
        "per_station": per_station,
        "fate": list(fate.values()),
        "release_every_s": RELEASE_EVERY,
        "wall_seconds": round(time.time() - t_wall, 1),
        "seed": seed,
        "playback": {
            "fps": FPS,
            "episodes": [
                {**e, **sched["times"][(e["station"], e["part"])]} for e in episodes
            ],
        },
    }
    return result


def _schedule(line: Line, episodes: list[dict], n_parts: int) -> dict:
    """One part per zone at a time; parts enter zone 1 every RELEASE_EVERY s, wait at a zone's entry if it is busy,
    and move on to the next zone when they leave this one."""
    by = {(e["station"], e["part"]): e for e in episodes}
    free = [0.0] * len(line.stations)
    times = {}
    makespan = 0.0
    for i in range(n_parts):
        t = i * RELEASE_EVERY
        for k in range(len(line.stations)):
            e = by.get((k, i))
            if e is None:
                break
            start = max(t, free[k])
            dur = len(e["frames"]) / 30.0
            times[(k, i)] = {"start": round(start, 2), "duration": round(dur, 2)}
            free[k] = start + dur
            t = start + dur
            makespan = max(makespan, t)
            if e["kind"] == "pick" and not line.assembly:
                break
    return {"times": times, "makespan": makespan}


# ------------------------------------------------------------------ persistence


def save(line: Line) -> None:
    LINES_DIR.mkdir(parents=True, exist_ok=True)
    (LINES_DIR / f"{line.id}.json").write_text(json.dumps(line.to_json(full=True)))


def load_all() -> dict[str, Line]:
    out = {}
    for p in sorted(LINES_DIR.glob("*.json")) if LINES_DIR.exists() else []:
        d = json.loads(p.read_text())
        sts = [
            Station(**{k: s[k] for k in Station.__dataclass_fields__ if k in s})
            for s in d["stations"]
        ]
        ln = Line(
            d["id"],
            d["prompt"],
            d["name"],
            sts,
            d["parts"],
            d.get("plan", {}),
            d.get("created", 0.0),
            d.get("status", "planned"),
            d.get("result", {}),
            d.get("training", []),
            d.get("assembly", False),
        )
        out[ln.id] = ln
    return out


def block_colors() -> dict:
    return {c: list(BLOCK_COLORS[c]) for c in PART_COLORS}


def cell_scenes(line: Line) -> list[dict]:
    """Each station's cell for the line view: robot meshes, belt, tote, and one part to recolour per episode."""
    from .viz import export_scene

    out = []
    for st in line.stations:
        lay = _cell_layout(st, "red", L.BELT_X, L.ENTRY_Y, 0.0)
        sc = Scene(lay, st.robot)
        sc.reset(lay)
        home = sc.pose_frame()
        sc.close()
        out.append(
            {**export_scene(lay, st.robot), "home": home, "cube_half": CUBE_HALF}
        )
    return out
