"""The 6-brick duck: parts, colours, target poses, and the real LDraw meshes.

Facing +x. Grid cells are 8 mm studs and 9.6 mm brick layers. The assembly follows the part list Pranav picked
(2 x 3001, 3002, 3003, 3004, 3039); see docs/duck-demo.md for what is and is not checked against an official sheet."""

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
LAYER = ld.BRICK_H  # 9.6 mm
YELLOW = (1.0, 0.80, 0.02, 1.0)
ORANGE = (1.0, 0.38, 0.02, 1.0)
PART_FOOTPRINT = {  # studs along the part's own x, y
    "3001": (4, 2),
    "3002": (3, 2),
    "3003": (2, 2),
    "3004": (2, 1),
    "3039": (2, 2),
}
PART_NAME = {
    "3001": "Brick 2 x 4",
    "3002": "Brick 2 x 3",
    "3003": "Brick 2 x 2",
    "3004": "Brick 1 x 2",
    "3039": "Slope 45 2 x 2",
}


@dataclass(frozen=True)
class Brick:
    name: str
    part: str
    rgba: tuple
    col: int  # stud column of the footprint's low-x corner
    row: int
    layer: int
    yaw_deg: int  # 0 or 90
    role: str

    @property
    def footprint(self) -> tuple[int, int]:
        nx, ny = PART_FOOTPRINT[self.part]
        return (ny, nx) if self.yaw_deg % 180 else (nx, ny)

    def target_pos(self) -> np.ndarray:
        """Bottom centre of the brick when seated (metres, table top at z = 0)."""
        nx, ny = self.footprint
        return np.array([(self.col + nx / 2) * STUD, (self.row + ny / 2) * STUD, self.layer * LAYER])

    def target_quat(self) -> np.ndarray:
        h = math.radians(self.yaw_deg) / 2
        return np.array([math.cos(h), 0.0, 0.0, math.sin(h)])


# Build order is bottom-up. The head is the top brick, the beak rides the neck's front studs, the tail slope falls toward the head
# (its low edge faces +x).
DUCK: tuple[Brick, ...] = (
    Brick("base", "3001", YELLOW, 0, 0, 0, 0, "body, bottom"),
    Brick("body", "3001", YELLOW, -1, 0, 1, 0, "body, top (one stud back so the rear sticks out)"),
    Brick("neck", "3002", YELLOW, 1, 0, 2, 0, "neck, front of the body"),
    Brick("head", "3003", YELLOW, 1, 0, 3, 0, "head, the highest brick"),
    Brick("beak", "3004", ORANGE, 3, 0, 3, 90, "beak, on the neck's front studs, one stud past the head"),
    Brick("tail", "3039", YELLOW, -1, 0, 2, 90, "tail, slope rises to the rear"),
)

# Duck centre in the camera and exploded views, metres.
CENTRE = np.array([1.5 * STUD, STUD, 2.0 * LAYER])


def brick_map() -> dict[str, Brick]:
    return {b.name: b for b in DUCK}


def occupied(upto: int | None = None) -> set[tuple[int, int, int]]:
    cells = set()
    for b in DUCK[:upto]:
        nx, ny = b.footprint
        cells |= {(b.col + i, b.row + j, b.layer) for i in range(nx) for j in range(ny)}
    return cells


def check_assembly() -> list[str]:
    """Structural problems of the target: overlaps, floating bricks, bricks that do not engage a stud below."""
    problems = []
    seen: set = set()
    for k, b in enumerate(DUCK):
        nx, ny = b.footprint
        cells = {(b.col + i, b.row + j, b.layer) for i in range(nx) for j in range(ny)}
        if cells & seen:
            problems.append(f"{b.name} overlaps an earlier brick")
        if b.layer and not any((c, r, b.layer - 1) in seen for c, r, _ in cells):
            problems.append(f"{b.name} rests on nothing")
        seen |= cells
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
