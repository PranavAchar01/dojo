"""LDraw part reader: .dat files (official LDraw library, CC BY 4.0) to triangle meshes in metres, Z up.

Only the facts the duck needs are handled: line types 1 (sub-file), 3 and 4 (faces), the BFC winding statements, and
the Y-down LDraw frame. Lines 2 and 5 (edges) are dropped."""

from __future__ import annotations

from functools import cache
from pathlib import Path

import numpy as np

LDU = 0.0004  # one LDraw unit in metres (1 stud pitch = 20 LDU = 8 mm)
STUD = 20 * LDU
BRICK_H = 24 * LDU
# LDraw is right-handed with -Y up; MuJoCo is Z up. A proper rotation keeps the face winding.
TO_Z_UP = np.array([[1.0, 0, 0], [0, 0, 1.0], [0, -1.0, 0]])


class Library:
    def __init__(self, root: Path):
        self.root = Path(root)

    def find(self, name: str) -> Path:
        n = name.replace("\\", "/").lower()
        for sub in ("parts", "p", "parts/s", "p/48", "p/8"):
            p = self.root / sub / n
            if p.exists():
                return p
        raise FileNotFoundError(name)

    def triangles(self, name: str) -> np.ndarray:
        """All triangles of a part as an (n, 3, 3) array in LDraw units and frame, wound counter-clockwise."""
        return np.array(self._tris(name, self.find(name)), dtype=float).reshape(
            -1, 3, 3
        )

    def _tris(self, name: str, path: Path) -> list:
        return _parse(self, str(path))


@cache
def _parse_cached(libroot: str, path: str) -> tuple:
    lib = Library(Path(libroot))
    out: list = []
    ccw = True  # winding declared by the file (BFC CERTIFY CCW is the default)
    invert_next = False
    for raw in Path(path).read_text(errors="ignore").splitlines():
        t = raw.split()
        if not t:
            continue
        if t[0] == "0":
            u = " ".join(t[1:]).upper()
            if u.startswith("BFC"):
                if "INVERTNEXT" in u:
                    invert_next = True
                if "CW" in u.split() and "CCW" not in u.split():
                    ccw = False
                elif "CCW" in u.split():
                    ccw = True
            continue
        try:
            if t[0] == "1":
                v = [float(x) for x in t[2:14]]
                m = np.array(
                    [[v[3], v[4], v[5]], [v[6], v[7], v[8]], [v[9], v[10], v[11]]]
                )
                off = np.array(v[0:3])
                sub = lib.find(" ".join(t[14:]))
                tris = np.array(_parse(lib, str(sub))).reshape(-1, 3, 3)
                if len(tris):
                    if np.linalg.det(m) < 0:
                        tris = tris[:, ::-1]  # a mirrored sub-file flips its winding
                    if invert_next:
                        tris = tris[:, ::-1]
                    tris = tris @ m.T + off
                    out.extend(tris.tolist())
                invert_next = False
            elif t[0] in ("3", "4"):
                n = 3 if t[0] == "3" else 4
                p = np.array([float(x) for x in t[2 : 2 + 3 * n]]).reshape(n, 3)
                if not ccw:
                    p = p[::-1]
                out.append(p[[0, 1, 2]].tolist())
                if n == 4:
                    out.append(p[[0, 2, 3]].tolist())
                invert_next = False
        except FileNotFoundError:
            continue
    return tuple(tuple(map(tuple, x)) for x in out)


def _parse(lib: Library, path: str):
    return _parse_cached(str(lib.root), path)


def part_mesh(lib: Library, part: str) -> tuple[np.ndarray, np.ndarray]:
    """(vertices (n,3) in metres, Z up, origin at the bottom centre of the footprint; faces (m,3) int)."""
    tris = lib.triangles(f"{part}.dat") @ TO_Z_UP.T * LDU
    lo, hi = tris.reshape(-1, 3).min(0), tris.reshape(-1, 3).max(0)
    origin = np.array([(lo[0] + hi[0]) / 2, (lo[1] + hi[1]) / 2, lo[2]])
    tris = tris - origin
    verts, inv = np.unique(
        np.round(tris.reshape(-1, 3), 7), axis=0, return_inverse=True
    )
    faces = inv.reshape(-1, 3)
    keep = (
        (faces[:, 0] != faces[:, 1])
        & (faces[:, 1] != faces[:, 2])
        & (faces[:, 0] != faces[:, 2])
    )
    return verts, faces[keep]
