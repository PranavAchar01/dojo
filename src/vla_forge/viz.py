"""Scene export for the browser renderer: the real SO-101 meshes from the compiled MuJoCo model, plus the
semantic objects (blocks, bowls, tape squares) so three.js can draw them as finished objects.

Body poses come from Scene.pose_frame() (world xpos + xquat of bodies 1..nbody-1), recorded during the policy's
own rollouts; the browser only interpolates and draws them.
"""

from __future__ import annotations

import base64
from functools import lru_cache

import mujoco
import numpy as np

from .sim.scene import (
    BLOCK_COLORS,
    BOWL_COLORS,
    BOWL_R,
    BOWL_WALL_H,
    CUBE_HALF,
    PAD_COLORS,
    PAD_HALF,
    Layout,
    build_spec,
)

VISUAL_GROUP = 2


def _b64(a: np.ndarray) -> str:
    return base64.b64encode(np.ascontiguousarray(a).tobytes()).decode()


@lru_cache(maxsize=1)
def _robot_meshes() -> dict:
    """Mesh vertex/face buffers of the SO-101 visual geoms (identical for every layout)."""
    m = build_spec(Layout()).compile()
    out = {}
    for g in range(m.ngeom):
        if (
            m.geom_group[g] != VISUAL_GROUP
            or m.geom_type[g] != mujoco.mjtGeom.mjGEOM_MESH
        ):
            continue
        mid = int(m.geom_dataid[g])
        if mid in out:
            continue
        va, vn = m.mesh_vertadr[mid], m.mesh_vertnum[mid]
        fa, fn = m.mesh_faceadr[mid], m.mesh_facenum[mid]
        out[mid] = {
            "v": _b64(m.mesh_vert[va : va + vn].astype(np.float32)),
            "f": _b64(m.mesh_face[fa : fa + fn].astype(np.uint32)),
        }
    return out


def export_scene(layout: Layout) -> dict:
    m = build_spec(layout).compile()
    bodies = [m.body(i).name for i in range(1, m.nbody)]
    robot = []
    for g in range(m.ngeom):
        if (
            m.geom_group[g] != VISUAL_GROUP
            or m.geom_type[g] != mujoco.mjtGeom.mjGEOM_MESH
        ):
            continue
        mat = m.geom_matid[g]
        rgba = m.mat_rgba[mat] if mat >= 0 else m.geom_rgba[g]
        robot.append(
            {
                "body": int(m.geom_bodyid[g]) - 1,
                "mesh": int(m.geom_dataid[g]),
                "pos": m.geom_pos[g].round(6).tolist(),
                "quat": m.geom_quat[g].round(6).tolist(),
                "rgba": rgba.round(4).tolist(),
            }
        )
    idx = {n: i for i, n in enumerate(bodies)}
    return {
        "bodies": bodies,
        "meshes": {str(k): v for k, v in _robot_meshes().items()},
        "robot": robot,
        "blocks": [
            {
                "body": idx[n],
                "name": n,
                "color": c,
                "rgb": list(BLOCK_COLORS[c]),
                "half": CUBE_HALF,
            }
            for n, (c, *_r) in layout.blocks.items()
        ],
        "bowls": [
            {
                "body": idx[n],
                "name": n,
                "color": c,
                "rgb": list(BOWL_COLORS[c]),
                "r": BOWL_R,
                "h": BOWL_WALL_H,
            }
            for n, (c, *_r) in layout.bowls.items()
        ],
        "pads": [
            {
                "body": idx[n],
                "name": n,
                "color": c,
                "rgb": list(PAD_COLORS[c]),
                "half": PAD_HALF,
            }
            for n, (c, *_r) in layout.pads.items()
        ],
    }
