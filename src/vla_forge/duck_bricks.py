"""The LEGO Serious Play duck, rebuilt from the real kit's six parts with the real LDraw meshes.

The kit (set 2000416 "Duck": 2x4, 2x2, 1x2, a printed 2x2 and two red 2x3 plates) ships with no instructions, so the
assembly here copies one duck from a published photo of builds from that kit (source in assets/lego/manifest.json).
Facing +x. Grid: 8 mm studs, 3.2 mm plate levels (a brick is 3 levels)."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np

from . import duck_ldraw as ld

ROOT = Path(__file__).resolve().parents[2]
LEGO = ROOT / "assets" / "lego"
LIB = LEGO / "_dl" / "x" / "ldraw"  # raw library download (git-ignored); the converted meshes are committed
MESHES = LEGO / "meshes"

STUD = ld.STUD  # 8 mm
PLATE = 8 * ld.LDU  # 3.2 mm
LAYER = ld.BRICK_H  # 9.6 mm
YELLOW = (1.0, 0.80, 0.02, 1.0)
RED = (0.80, 0.04, 0.05, 1.0)
PART_FOOTPRINT = {  # studs along the part's own x, y
    "3001": (4, 2),
    "3002": (3, 2),
    "3003": (2, 2),
    "3004": (2, 1),
    "3039": (2, 2),
    "3021": (3, 2),
}
PART_PLATES = {"3001": 3, "3002": 3, "3003": 3, "3004": 3, "3039": 3, "3021": 1}
PART_NAME = {
    "3001": "Brick 2 x 4",
    "3002": "Brick 2 x 3",
    "3003": "Brick 2 x 2",
    "3004": "Brick 1 x 2",
    "3039": "Slope 45 2 x 2",
    "3021": "Plate 2 x 3",
}


@dataclass(frozen=True)
class Brick:
    name: str
    part: str
    rgba: tuple
    col: int  # stud column of the footprint's low-x corner
    row: int
    lvl: int  # plate level of the bottom face (0 = table)
    yaw_deg: int  # 0 or 90
    role: str

    @property
    def footprint(self) -> tuple[int, int]:
        nx, ny = PART_FOOTPRINT[self.part]
        return (ny, nx) if self.yaw_deg % 180 else (nx, ny)

    def target_pos(self) -> np.ndarray:
        """Bottom centre of the brick when seated (metres, table top at z = 0)."""
        nx, ny = self.footprint
        return np.array([(self.col + nx / 2) * STUD, (self.row + ny / 2) * STUD, self.lvl * PLATE])

    def target_quat(self) -> np.ndarray:
        h = math.radians(self.yaw_deg) / 2
        return np.array([math.cos(h), 0.0, 0.0, math.sin(h)])


# Bottom-up: red foot plate, a 2x2 leg on it, the 2x4 body across the leg (rear overhangs as the tail end), the red
# beak plate on the body's front studs (one stud forward of the body), the head on the plate, a low 1x2 tail.
DUCK: tuple[Brick, ...] = (
    Brick("foot", "3021", RED, 1, 0, 0, 0, "red plate, the duck's foot"),
    Brick("leg", "3003", YELLOW, 2, 0, 1, 0, "2x2 leg on the foot"),
    Brick("body", "3001", YELLOW, 0, 0, 4, 0, "2x4 body across the leg, rear overhangs"),
    Brick("beak", "3021", RED, 2, 0, 7, 0, "red plate on the front studs, one stud past the body"),
    Brick("head", "3003", YELLOW, 2, 0, 8, 0, "2x2 head (printed eye) on the plate"),
    Brick("tail", "3004", YELLOW, 0, 0, 7, 0, "1x2 tail, lying on the rear studs (not visible in the source photo)"),
)

# Duck centre in the camera and exploded views, metres.
CENTRE = np.array([2.0 * STUD, STUD, 1.8 * LAYER])


def brick_map() -> dict[str, Brick]:
    return {b.name: b for b in DUCK}


def cells(b: Brick) -> set[tuple[int, int, int]]:
    nx, ny = b.footprint
    return {
        (b.col + i, b.row + j, b.lvl + k)
        for i in range(nx)
        for j in range(ny)
        for k in range(PART_PLATES[b.part])
    }


def check_assembly() -> list[str]:
    """Structural problems of the target: overlaps, floating bricks, bricks that do not engage a stud below."""
    problems = []
    seen: set = set()
    for b in DUCK:
        c = cells(b)
        if c & seen:
            problems.append(f"{b.name} overlaps an earlier brick")
        if b.lvl and not any((x, y, b.lvl - 1) in seen for x, y, z in c if z == b.lvl):
            problems.append(f"{b.name} rests on nothing")
        seen |= c
    return problems


def part_mesh_cached(part: str) -> tuple[np.ndarray, np.ndarray]:
    """Converted LDraw mesh (vertices in metres, bottom-centre origin). Cached as an .npz next to the manifest."""
    out = MESHES / f"{part}.npz"
    if out.exists():
        z = np.load(out)
        return z["v"], z["f"]
    v, f = ld.part_mesh(ld.Library(LIB), part)
    MESHES.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, v=v.astype(np.float32), f=f.astype(np.int32))
    return v, f


@lru_cache(maxsize=1)
def manifest() -> dict:
    return json.loads((LEGO / "manifest.json").read_text())
