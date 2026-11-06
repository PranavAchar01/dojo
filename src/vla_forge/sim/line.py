"""Factory-line equipment for a station cell: a conveyor and totes, both from published sources.

* Conveyor. The construction is the one MuJoCo's maintainers gave for a constant-speed belt
  (google-deepmind/mujoco issue #547, Yuval Tassa): a long box on a slide joint with large damping, driven by a
  constant force, so it settles at force / damping m/s and carries whatever rests on it by friction. Here the
  force goes in through `qfrc_applied` instead of a biased `general` actuator, so the arm's control vector (what
  the policy outputs) keeps its size. The belt only touches the parts.
* Totes. The floor and four wall boxes of body `bin1` in robosuite's `bins_arena.xml` (vendor/robosuite, MIT),
  read from that file and scaled down to a tote the SO-101 can reach into.

A station cell is the normal table scene plus a belt running along y in front of the arm and one tote beside it.
Parts arrive on the belt from -y; the belt stops when a part crosses the station's photo-eye line (scripted
line control), the station's policy moves the part into its tote, and parts for other stations ride on through.
"""

from __future__ import annotations

import math
import xml.etree.ElementTree as ET
from functools import lru_cache

import mujoco
import numpy as np

from .scene import CUBE_BIT, CUBE_HALF, REPO, Layout

BINS_XML = REPO / "vendor" / "robosuite" / "bins_arena.xml"

# belt: along y, in front of the arm, top face just above the table so parts on it touch only the belt
BELT_X = 0.245
BELT_HALF_W = 0.035
BELT_HALF_L = 0.95
BELT_HALF_T = 0.003
BELT_TOP = 0.001
BELT_DAMPING = 1000.0  # N s / m, as in the issue's model
BELT_SPEED = 0.12  # m/s while running
ENTRY_Y = (
    -0.32
)  # where a part enters the cell (upstream, out of the policy camera's view)
EXIT_Y = 0.34
WINDOW_Y = -0.06  # the photo-eye: the belt stops when a part reaches this line

# tote: inner half-size and wall height after scaling robosuite's bin
TOTE_INNER = 0.045
TOTE_WALL_H = 0.028
TOTE_XY = (0.155, 0.085)

BIN_COLORS = {
    "red": (0.80, 0.16, 0.13),
    "blue": (0.16, 0.34, 0.78),
    "green": (0.18, 0.56, 0.30),
    "yellow": (0.92, 0.74, 0.12),
    "gray": (0.42, 0.43, 0.46),
}


@lru_cache(maxsize=1)
def bin_geoms() -> list[dict]:
    """robosuite's bin1 collision boxes (floor + 4 walls), scaled so the inner opening is 2 x TOTE_INNER square
    and the walls TOTE_WALL_H tall, with the floor's underside on z = 0."""
    root = ET.parse(BINS_XML).getroot()
    body = next(b for b in root.iter("body") if b.get("name") == "bin1")
    boxes = []
    for g in body.findall("geom"):
        if g.get("type") != "box" or g.get("group") != "0":
            continue
        pos = np.array([float(v) for v in g.get("pos", "0 0 0").split()])
        size = np.array([float(v) for v in g.get("size").split()])
        boxes.append((pos, size))
    floor = next(b for b in boxes if b[0][2] == 0.0)
    walls = [b for b in boxes if b[0][2] != 0.0]
    fx, fy, fz = floor[1]
    wall_h = (
        max(p[2] + s[2] for p, s in walls) - fz
    )  # wall height above the floor's top
    scale = np.array([TOTE_INNER / fx, TOTE_INNER / fy, TOTE_WALL_H / (wall_h + fz)])
    out = []
    for pos, size in boxes:
        p, s = pos * scale, size * scale
        p[2] += fz * scale[2]  # lift so the floor's bottom face sits on z = 0
        out.append({"pos": p.round(5).tolist(), "size": s.round(5).tolist()})
    return out


def tote_floor_top() -> float:
    f = next(
        g
        for g in bin_geoms()
        if g["size"][0] > TOTE_INNER * 0.9 and g["size"][1] > TOTE_INNER * 0.9
    )
    return f["pos"][2] + f["size"][2]


def add_belt(spec: mujoco.MjSpec) -> None:
    spec.add_material(name="belt_mat", rgba=[0.11, 0.115, 0.125, 1.0])
    belt = spec.worldbody.add_body(
        name="belt", pos=[BELT_X, 0.0, BELT_TOP - BELT_HALF_T]
    )
    belt.add_joint(
        name="belt_slide",
        type=mujoco.mjtJoint.mjJNT_SLIDE,
        axis=[0, 1, 0],
        damping=BELT_DAMPING,
    )
    belt.add_geom(
        name="belt_geom",
        type=mujoco.mjtGeom.mjGEOM_BOX,
        size=[BELT_HALF_W, BELT_HALF_L, BELT_HALF_T],
        material="belt_mat",
        friction=[0.9, 0.005, 0.0001],
        contype=CUBE_BIT,
        conaffinity=CUBE_BIT,
        mass=0.6,
    )
    spec.add_exclude(
        bodyname1="world", bodyname2="belt"
    )  # the belt slides over the table, not on it


def add_bin(spec: mujoco.MjSpec, name: str, color: str) -> None:
    spec.add_material(name=f"{name}_mat", rgba=[*BIN_COLORS[color], 1.0])
    body = spec.worldbody.add_body(name=name, mocap=True, pos=[*TOTE_XY, 0.0])
    for i, g in enumerate(bin_geoms()):
        body.add_geom(
            name=f"{name}_box{i}",
            type=mujoco.mjtGeom.mjGEOM_BOX,
            size=g["size"],
            pos=g["pos"],
            material=f"{name}_mat",
            friction=[1.0, 0.005, 0.0001],
        )


def on_belt(x: float) -> bool:
    return abs(x - BELT_X) < BELT_HALF_W


def in_bin(c: np.ndarray, b: np.ndarray) -> bool:
    """Part centre inside the tote's opening and below its rim."""
    return bool(
        np.all(np.abs(c[:2] - b[:2]) < TOTE_INNER) and c[2] < TOTE_WALL_H + CUBE_HALF
    )


def window_yaw() -> float:
    return math.atan2(WINDOW_Y, BELT_X)


def station_layout(
    color: str, bin_color: str, rng: np.random.Generator, at_window: bool = True
) -> Layout:
    """A station cell: the part on the belt at the photo-eye (or at the cell entry), and the station's tote.
    Jitter: where a part ends up after the belt stops (+-1.5 cm along, +-1.2 cm across) and how it was loaded
    (within +-34 deg of square to the arm), and +-8 mm on where the tote stands."""
    x = BELT_X + float(rng.uniform(-0.012, 0.012))
    y = (WINDOW_Y + float(rng.uniform(-0.015, 0.015))) if at_window else ENTRY_Y
    yaw = math.atan2(y, x) + float(rng.uniform(-0.6, 0.6))
    bx, by = TOTE_XY
    lay = Layout(belt=True)
    lay.blocks["target"] = (color, x, y, yaw)
    lay.bins["goal"] = (
        bin_color,
        bx + float(rng.uniform(-0.008, 0.008)),
        by + float(rng.uniform(-0.008, 0.008)),
    )
    return lay


def belt_dof(model: mujoco.MjModel) -> int:
    return int(model.joint("belt_slide").dofadr[0])


def drive_belt(model: mujoco.MjModel, data: mujoco.MjData, speed: float) -> None:
    """Constant force against the joint damping: the belt settles at `speed` m/s (0 stops it)."""
    data.qfrc_applied[belt_dof(model)] = BELT_DAMPING * speed
