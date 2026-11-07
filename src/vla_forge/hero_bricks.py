"""The hero build: a brick bridge generated from rules, with real LDraw part meshes.

Rules (the user can change the three numbers): two pillars of `pillar` stacked 2x2 bricks, a fixed `gap` of studs
between them, one deck brick spanning both pillars (overlapping each by at least a stud), and an optional 2x2 brick
centred on the deck. Nothing here is judged by eye: `hero_rules.check` turns each rule into a pass or fail.

Facing +x, 8 mm studs, 3.2 mm plate levels (a brick is 3 levels). Parts come from the LDraw Parts Library
(CC BY 4.0, see assets/lego/manifest.json)."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, replace
from functools import cache, lru_cache
from pathlib import Path

import numpy as np

from . import hero_ldraw as ld

ROOT = Path(__file__).resolve().parents[2]
LEGO = ROOT / "assets" / "lego"
LIB = (
    LEGO / "_dl" / "full" / "ldraw"
)  # raw library download (git-ignored); the converted meshes are committed
MESHES = LEGO / "meshes"

STUD = ld.STUD  # 8 mm
PLATE = 8 * ld.LDU  # 3.2 mm
LAYER = ld.BRICK_H  # 9.6 mm
RED = (0.80, 0.04, 0.05, 1.0)
YELLOW = (1.0, 0.80, 0.02, 1.0)
BLUE = (0.05, 0.28, 0.72, 1.0)
COLOURS = {"red": RED, "yellow": YELLOW, "blue": BLUE}
PART_FOOTPRINT = {  # studs along the part's own x, y
    "3003": (2, 2),
    "3001": (4, 2),
    "2456": (6, 2),
    "3007": (8, 2),
}
PART_PLATES = {p: 3 for p in PART_FOOTPRINT}
PART_NAME = {
    "3003": "Brick 2 x 2",
    "3001": "Brick 2 x 4",
    "2456": "Brick 2 x 6",
    "3007": "Brick 2 x 8",
}
DECKS = {4: "3001", 6: "2456", 8: "3007"}  # deck length in studs -> 2-wide brick


@dataclass(frozen=True)
class Brick:
    name: str
    part: str
    colour: str
    col: int  # stud column of the footprint's low-x corner
    row: int
    lvl: int  # plate level of the bottom face (0 = table)
    yaw_deg: int  # 0 or 90
    role: str
    source: str = "rules"  # "rules" for generated, "muse" or "user" once a person or the model changed it

    @property
    def rgba(self) -> tuple:
        return COLOURS[self.colour]

    @property
    def footprint(self) -> tuple[int, int]:
        nx, ny = PART_FOOTPRINT[self.part]
        return (ny, nx) if self.yaw_deg % 180 else (nx, ny)

    @property
    def top(self) -> int:
        return self.lvl + PART_PLATES[self.part]

    def target_pos(self) -> np.ndarray:
        """Bottom centre of the brick when seated (metres, table top at z = 0)."""
        nx, ny = self.footprint
        return np.array(
            [(self.col + nx / 2) * STUD, (self.row + ny / 2) * STUD, self.lvl * PLATE]
        )

    def target_quat(self) -> np.ndarray:
        h = math.radians(self.yaw_deg) / 2
        return np.array([math.cos(h), 0.0, 0.0, math.sin(h)])


@dataclass(frozen=True)
class Rules:
    gap: int = 4  # studs between the pillars
    pillar: int = 2  # bricks per pillar
    deck: int = 6  # deck length in studs
    tier: bool = True  # optional 2x2 on the deck

    def problems(self) -> list[str]:
        out = []
        if not 1 <= self.gap <= 6:
            out.append("gap must be 1 to 6 studs")
        if not 1 <= self.pillar <= 4:
            out.append("pillar must be 1 to 4 bricks")
        if self.deck not in DECKS:
            out.append(f"deck must be one of {sorted(DECKS)} studs")
        elif self.deck < self.gap + 2:
            out.append(
                "deck must reach at least one stud onto each pillar (deck >= gap + 2)"
            )
        elif self.deck > self.gap + 4:
            out.append("deck must end on the pillars (deck <= gap + 4)")
        return out

    def as_dict(self) -> dict:
        return {
            "gap": self.gap,
            "pillar": self.pillar,
            "deck": self.deck,
            "tier": self.tier,
        }


def generate(rules: Rules | None = None) -> tuple[Brick, ...]:
    """The bridge as stud-grid bricks, bottom up, one brick per step."""
    rules = rules or Rules()
    bad = rules.problems()
    if bad:
        raise ValueError("; ".join(bad))
    out: list[Brick] = []
    x2 = 2 + rules.gap  # column where the second pillar starts
    for k in range(rules.pillar):
        out.append(
            Brick(
                f"left{k + 1}",
                "3003",
                "red",
                0,
                0,
                3 * k,
                0,
                f"left pillar, brick {k + 1}",
            )
        )
        out.append(
            Brick(
                f"right{k + 1}",
                "3003",
                "red",
                x2,
                0,
                3 * k,
                0,
                f"right pillar, brick {k + 1}",
            )
        )
    span = x2 + 2
    c0 = (span - rules.deck) // 2
    out.append(
        Brick(
            "deck",
            DECKS[rules.deck],
            "yellow",
            c0,
            0,
            3 * rules.pillar,
            0,
            "deck across both pillars",
        )
    )
    if rules.tier:
        out.append(
            Brick(
                "tier",
                "3003",
                "blue",
                c0 + (rules.deck - 2) // 2,
                0,
                3 * rules.pillar + 3,
                0,
                "2x2 centred on the deck",
            )
        )
    order = {b.name: i for i, b in enumerate(sorted(out, key=lambda b: (b.lvl, b.col)))}
    return tuple(sorted(out, key=lambda b: order[b.name]))


DUCK: tuple[Brick, ...] = (
    generate()
)  # the active assembly; set_assembly swaps it (name kept for the table code)


def set_assembly(bricks) -> None:
    global DUCK
    DUCK = tuple(bricks)
    brick_map.cache_clear()


@cache
def brick_map() -> dict[str, Brick]:
    return {b.name: b for b in DUCK}


def centre() -> np.ndarray:
    """Middle of the assembly in the camera and exploded views, metres."""
    lo = np.array([min(b.col for b in DUCK), min(b.row for b in DUCK)], float) * STUD
    hi = (
        np.array(
            [
                max(b.col + b.footprint[0] for b in DUCK),
                max(b.row + b.footprint[1] for b in DUCK),
            ],
            float,
        )
        * STUD
    )
    top = max(b.top for b in DUCK) * PLATE
    return np.array([(lo[0] + hi[0]) / 2, (lo[1] + hi[1]) / 2, top / 2])


def cells(b: Brick) -> set[tuple[int, int, int]]:
    nx, ny = b.footprint
    return {
        (b.col + i, b.row + j, b.lvl + k)
        for i in range(nx)
        for j in range(ny)
        for k in range(PART_PLATES[b.part])
    }


def check_assembly(bricks=None) -> list[str]:
    """Structural problems: overlaps, floating bricks, bricks that do not engage a stud below (bottom-up order)."""
    problems = []
    seen: set = set()
    for b in sorted(bricks or DUCK, key=lambda b: b.lvl):
        c = cells(b)
        if c & seen:
            problems.append(f"{b.name} overlaps another brick")
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


def to_json(bricks) -> list[dict]:
    return [
        {
            "name": b.name,
            "part": b.part,
            "colour": b.colour,
            "col": b.col,
            "row": b.row,
            "lvl": b.lvl,
            "yaw": b.yaw_deg,
            "role": b.role,
            "source": b.source,
        }
        for b in bricks
    ]


def from_json(rows) -> tuple[Brick, ...]:
    return tuple(
        Brick(
            str(r["name"]),
            str(r["part"]),
            str(r["colour"]),
            int(r["col"]),
            int(r["row"]),
            int(r["lvl"]),
            int(r.get("yaw", 0)),
            str(r.get("role", "")),
            str(r.get("source", "user")),
        )
        for r in rows
    )


def replace_name(b: Brick, name: str) -> Brick:
    return replace(b, name=name)


ACTIVE = ROOT / "runs" / "hero_build.json"


def save_active(rules: Rules, bricks) -> None:
    """The build the build plan hands to the MuJoCo table ("Use this build")."""
    ACTIVE.parent.mkdir(parents=True, exist_ok=True)
    ACTIVE.write_text(
        json.dumps({"rules": rules.as_dict(), "bricks": to_json(bricks)}, indent=1)
    )


def load_active() -> tuple[Rules, tuple[Brick, ...]] | None:
    try:
        d = json.loads(ACTIVE.read_text())
        return Rules(**d["rules"]), from_json(d["bricks"])
    except (OSError, ValueError, KeyError, TypeError):
        return None
