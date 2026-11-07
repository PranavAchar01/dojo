"""The bridge build as a library tile: the scene the player draws, the recorded poses, and the card's picture.

The player (web/app.js) draws a scene as visual mesh geoms per body, so the two arms and every LDraw brick go out as
that; their mesh buffers are served under the robot id `ROBOT_ID` at /api/meshes.
"""

from __future__ import annotations

import gzip
import json
from functools import lru_cache
from pathlib import Path

import mujoco
import numpy as np
from PIL import Image

from . import hero_build as hbd
from . import hero_sim as ds
from .viz import VISUAL_GROUP, _b64

THUMB_W, THUMB_H = 480, 360
ROBOT_ID = "so101-duo"


def _drawn(m: mujoco.MjModel, g: int) -> bool:
    if m.geom_type[g] != mujoco.mjtGeom.mjGEOM_MESH:
        return False
    name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g) or ""
    return m.geom_group[g] == VISUAL_GROUP or name.endswith("_vis")


def _canon(m: mujoco.MjModel) -> dict[int, int]:
    """Mesh id -> the first mesh with the same vertices: the two arms share one set of buffers."""
    first: dict[bytes, int] = {}
    out = {}
    for mid in range(m.nmesh):
        va, vn = m.mesh_vertadr[mid], m.mesh_vertnum[mid]
        key = m.mesh_vert[va : va + vn].tobytes()
        out[mid] = first.setdefault(key, mid)
    return out


@lru_cache(maxsize=1)
def meshes() -> dict:
    """Mesh buffers of both arms and every brick part, by mesh id (served once per page at /api/meshes)."""
    m = ds.Table.new(480).m
    out, canon = {}, _canon(m)
    for g in range(m.ngeom):
        if not _drawn(m, g):
            continue
        mid = canon[int(m.geom_dataid[g])]
        if str(mid) in out:
            continue
        va, vn = m.mesh_vertadr[mid], m.mesh_vertnum[mid]
        fa, fn = m.mesh_faceadr[mid], m.mesh_facenum[mid]
        out[str(mid)] = {
            "v": _b64(m.mesh_vert[va : va + vn].astype(np.float32)),
            "f": _b64(m.mesh_face[fa : fa + fn].astype(np.uint32)),
        }
    return out


def scene(m: mujoco.MjModel) -> dict:
    geoms, canon = [], _canon(m)
    for g in range(m.ngeom):
        if not _drawn(m, g):
            continue
        mid = canon[int(m.geom_dataid[g])]
        mat = m.geom_matid[g]
        rgba = m.mat_rgba[mat] if mat >= 0 else m.geom_rgba[g]
        geoms.append(
            {
                "body": int(m.geom_bodyid[g]) - 1,
                "mesh": mid,
                "name": mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_MESH, mid) or "",
                "pos": m.geom_pos[g].round(6).tolist(),
                "quat": m.geom_quat[g].round(6).tolist(),
                "rgba": rgba.round(4).tolist(),
            }
        )
    return {
        "robot_id": ROBOT_ID,
        "bodies": [m.body(i).name for i in range(1, m.nbody)],
        "robot": geoms,
        "blocks": [],
        "bowls": [],
        "pads": [],
        "props": [],
    }


def payload(index: int, res: dict, scene_json: dict, rnd: int = 0) -> dict:
    """One recorded build, shaped like a tile's rollout."""
    return {
        "round": rnd,
        "index": index,
        "success": res["rules_passed"] == res["rules_total"] and res["seated"] == res["total"],
        "failure": failure_text(res),
        "fps": round(hbd.FPS, 2),
        "scene": scene_json,
        "frames": res["frames"],
        "built": f"{res['seated']}/{res['total']} bricks seated",
        "rules": res["rules"],
    }


def failure_text(res: dict) -> str:
    bad = [r["id"] for r in res["rules"] if not r["ok"]]
    if res["seated"] == res["total"] and not bad:
        return ""
    return f"{res['seated']}/{res['total']} bricks seated; failed: {', '.join(bad) or 'none'}"


def save(skill_id: str, root: Path, thumb: np.ndarray, rolls: list[dict]) -> dict:
    d = root / "skills" / "tiles" / skill_id
    d.mkdir(parents=True, exist_ok=True)
    png = d / "thumb.png"
    Image.fromarray(thumb).resize((THUMB_W * 2, THUMB_H * 2), Image.LANCZOS).save(png, optimize=True)
    with gzip.open(d / "replay.json.gz", "wt") as fh:
        json.dump(rolls, fh, separators=(",", ":"))
    return {
        "thumb_png": str(png.relative_to(root)),
        "replay": str((d / "replay.json.gz").relative_to(root)),
        "replay_source": "scripted build",
    }
