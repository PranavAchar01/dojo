"""Robot builder payloads cached by a hash of the parts that make them.

Assembling a robot is cheap next to checking it: the reach map solves inverse kinematics at 189 targets and takes
seconds. The view (meshes, tool point, reach map) and the report (checks, skills) of one configuration therefore
live on disk, keyed by a hash of the arm and gripper model files plus the code that reads them, and the server
hands out the stored bytes. A warm-up process fills the cache, and the card thumbnails, for every arm and wrist
choice so the first visit finds them ready.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import threading
from functools import cache
from pathlib import Path

from .sim import robots as R

ROOT = R.REPO / ".cache" / "robots"
_CODE = (Path(R.__file__), Path(__file__).with_name("viz.py"))
_locks: dict[str, threading.Lock] = {}
_guard = threading.Lock()
_mem: dict[str, bytes] = {}


@cache
def _tree(rel: str) -> str:
    """Name and size of every file beside a model file (its meshes and includes)."""
    base = (R.REPO / rel).parent
    h = hashlib.sha256((R.REPO / rel).read_bytes())
    for f in sorted(base.rglob("*")):
        if f.is_file():
            h.update(f"{f.relative_to(base)}:{f.stat().st_size}".encode())
    return h.hexdigest()


@cache
def _code() -> str:
    return hashlib.sha256(b"".join(p.read_bytes() for p in _CODE)).hexdigest()


def part_key(slot: str, part: str) -> str:
    """Hash of one catalog part's model and the code that turns it into a payload."""
    src = {"arm": R.ARMS, "effector": R.EFFECTORS}[slot][part].source.path
    return hashlib.sha256(f"{slot}:{part}:{_tree(src)}:{_code()}".encode()).hexdigest()[
        :20
    ]


def config_key(robot_id: str) -> str:
    """Hash of the arm, wrist choice and gripper that make a robot id, and the code that assembles them."""
    arm, wrist, ee = R.parse_id(robot_id)
    paths = [R.ARMS[arm].source.path]
    if R.EFFECTORS.get(ee) is not None:
        paths.append(R.EFFECTORS[ee].source.path)
    trees = ":".join(_tree(p) for p in paths)
    return hashlib.sha256(f"{arm}.{wrist}.{ee}:{trees}:{_code()}".encode()).hexdigest()[
        :20
    ]


def _once(key: str, name: str, make) -> bytes:
    """Bytes for one cached payload: memory, then disk, then built exactly once however many ask at the same time."""
    ck = f"{key}.{name}"
    if ck in _mem:
        return _mem[ck]
    with _guard:
        lock = _locks.setdefault(ck, threading.Lock())
    with lock:
        if ck in _mem:
            return _mem[ck]
        f = ROOT / f"{ck}.json.gz"
        if f.exists():
            data = f.read_bytes()
        else:
            data = gzip.compress(
                json.dumps(make(), separators=(",", ":")).encode(), compresslevel=6
            )
            ROOT.mkdir(parents=True, exist_ok=True)
            tmp = f.with_suffix(f".{threading.get_ident()}.tmp")
            tmp.write_bytes(data)
            tmp.replace(f)
        _mem[ck] = data
        return data


def view_blob(robot_id: str) -> tuple[bytes, str]:
    from .viz import robot_view

    key = config_key(robot_id)
    return _once(key, "view", lambda: robot_view(robot_id)), key


def report_blob(robot_id: str) -> tuple[bytes, str]:
    key = config_key(robot_id)

    def make() -> dict:
        ok, why, feas = R.usable(robot_id)
        return {
            **R.report(robot_id),
            "skills": feas,
            "usable": ok,
            "usable_reason": why,
        }

    return _once(key, "report", make), key


def part_blob(slot: str, part: str) -> tuple[bytes, str]:
    from .viz import part_view

    key = part_key(slot, part)
    return _once(key, "part", lambda: part_view(slot, part)), key


def configs() -> list[str]:
    """Every robot id the arm cards and the wrist cards can ask for: each arm with its own wrist and a locked one."""
    return [R.make_id(a, w) for a in R.ARMS for w in R.WRISTS]
