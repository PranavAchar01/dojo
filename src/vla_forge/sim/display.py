"""Display-only restyle of a compiled scene, for library thumbnails.

The policy cameras render the unmodified model, so nothing here may run on a model a policy will see. The browser
studio (web/studio.js) uses the same role rules: white printed shells, black servos and joints, graphite steel, on a
sealed concrete floor with no table.
"""

from __future__ import annotations

import re

import mujoco
import numpy as np

SHELL = (0.945, 0.949, 0.953, 1.0)
SERVO = (0.08, 0.082, 0.094, 1.0)
STEEL = (0.44, 0.455, 0.48, 1.0)
CONCRETE = (0.56, 0.57, 0.59, 1.0)
FLOOR = (0.5, 0.51, 0.53, 1.0)

_SERVO_NAME = re.compile(
    r"motor|sts3215|horn|prop|gripper_bar|finger|_7_gripper|_6_wrist|_2_shoulder|hand_"
    r"|robotiq.*(base|driver|coupler|follower|spring)",
    re.IGNORECASE,
)
_KEEP_WHITE = re.compile(
    r"moving_jaw|gripper_moving_finger|static_finger$", re.IGNORECASE
)


def robot_role(rgba, name: str = "") -> str:
    if _SERVO_NAME.search(name) and not _KEEP_WHITE.search(name):
        return "servo"
    lum = 0.299 * rgba[0] + 0.587 * rgba[1] + 0.114 * rgba[2]
    if lum > 0.62:
        return "shell"
    if lum < 0.3:
        return "servo"
    return "steel"


_COLORS = {"shell": SHELL, "servo": SERVO, "steel": STEEL}


def restyle(m: mujoco.MjModel) -> None:
    """Black and white arm, concrete under it; geometry and physics untouched."""
    for gid in range(m.ngeom):
        name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, gid) or ""
        if name == "table":
            m.geom_rgba[gid] = CONCRETE
            m.geom_matid[gid] = -1
        elif name == "floor":
            m.geom_rgba[gid] = FLOOR
            m.geom_matid[gid] = -1
        elif m.geom_type[gid] == mujoco.mjtGeom.mjGEOM_MESH and m.geom_dataid[gid] >= 0:
            mesh = (
                mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_MESH, m.geom_dataid[gid]) or ""
            )
            if name.startswith(("block", "cube")):
                continue
            rgba = m.geom_rgba[gid].copy()
            if m.geom_matid[gid] >= 0:
                rgba = m.mat_rgba[m.geom_matid[gid]].copy()
            m.geom_rgba[gid] = np.asarray(_COLORS[robot_role(rgba, mesh or name)])
            m.geom_matid[gid] = -1
