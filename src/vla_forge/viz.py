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


def _mesh_buffers(m: mujoco.MjModel) -> dict:
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


@lru_cache(maxsize=16)
def _robot_meshes(robot: str = "so101") -> dict:
    """Mesh vertex/face buffers of the robot's visual geoms (identical for every layout)."""
    return _mesh_buffers(build_spec(Layout(), robot=robot).compile())


def export_scene(layout: Layout, robot: str = "so101") -> dict:
    robot_id = robot
    m = build_spec(layout, robot=robot_id).compile()
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
        "robot_id": robot_id,
        "bodies": bodies,
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


def preview(layout: Layout, robot: str = "so101") -> dict:
    """The scene plus one pose frame of it at rest (arm at home), for showing a layout before any rollout."""
    from .sim.scene import Scene

    sc = Scene(layout, robot)
    sc.reset(layout)
    frame = sc.pose_frame()
    sc.close()
    return {**export_scene(layout, robot), "frame": frame}


def robot_meshes(robot: str = "so101") -> dict:
    """A robot's mesh buffers, served once per robot per page load (/api/meshes?robot=)."""
    return {str(k): v for k, v in _robot_meshes(robot).items()}


def _static_view(m: mujoco.MjModel, d: mujoco.MjData) -> dict:
    """Visual meshes posed in the world frame (no bodies to animate): what a Build-tab preview draws."""
    geoms = []
    for g in range(m.ngeom):
        if (
            m.geom_group[g] != VISUAL_GROUP
            or m.geom_type[g] != mujoco.mjtGeom.mjGEOM_MESH
        ):
            continue
        mat = m.geom_matid[g]
        rgba = m.mat_rgba[mat] if mat >= 0 else m.geom_rgba[g]
        q = np.zeros(4)
        mujoco.mju_mat2Quat(q, d.geom_xmat[g])
        geoms.append(
            {
                "mesh": int(m.geom_dataid[g]),
                "pos": d.geom_xpos[g].round(5).tolist(),
                "quat": q.round(5).tolist(),
                "rgba": rgba.round(4).tolist(),
            }
        )
    # framing box: each visual mesh's vertices in the world frame (not just its origin)
    lo, hi = np.full(3, np.inf), np.full(3, -np.inf)
    for g in range(m.ngeom):
        if (
            m.geom_group[g] != VISUAL_GROUP
            or m.geom_type[g] != mujoco.mjtGeom.mjGEOM_MESH
        ):
            continue
        mid = m.geom_dataid[g]
        a, n = m.mesh_vertadr[mid], m.mesh_vertnum[mid]
        w = m.mesh_vert[a : a + n] @ d.geom_xmat[g].reshape(3, 3).T + d.geom_xpos[g]
        lo, hi = np.minimum(lo, w.min(0)), np.maximum(hi, w.max(0))
    if not np.isfinite(lo).all():
        lo, hi = np.zeros(3), np.zeros(3)
    return {
        "meshes": {str(k): v for k, v in _mesh_buffers(m).items()},
        "geoms": geoms,
        "bbox": [lo.round(4).tolist(), hi.round(4).tolist()],
    }


@lru_cache(maxsize=32)
def robot_view(robot_id: str) -> dict:
    """The assembled robot at its home pose with the jaws open, its tool point and the reach map."""
    from .sim import robots as R

    r = R.calibrate(robot_id)
    m = R.robot_spec(robot_id).compile()
    R.fix_qpos0(m)
    d = mujoco.MjData(m)
    d.qpos[[m.joint(j).qposadr[0] for j in r.arm_joints]] = r.home
    d.ctrl[m.actuator(r.grip_actuator).id] = r.g_open * r.grip_scale
    for a in range(m.nu):
        if a != m.actuator(r.grip_actuator).id:
            jid = m.actuator_trnid[a, 0]
            d.ctrl[a] = d.qpos[m.jnt_qposadr[jid]]
    m.opt.gravity[:] = 0
    for _ in range(
        300
    ):  # let the fingers open (they are driven, so step rather than set)
        mujoco.mj_step(m, d)
    mujoco.mj_forward(m, d)
    s = m.site(r.tcp_site).id
    return {
        "id": robot_id,
        **_static_view(m, d),
        "tcp": d.site_xpos[s].round(4).tolist(),
        "reach": R.reachability(robot_id),
    }


@lru_cache(maxsize=32)
def part_view(slot: str, part: str) -> dict:
    """One catalog part on its own, straight from its source model (an arm with its own jaw, or a gripper)."""
    from .sim import robots as R

    if slot == "arm":
        m = R._load_arm(R.ARMS[part]).compile()
    elif slot == "effector" and part != "native":
        m = mujoco.MjModel.from_xml_path(str(R.REPO / R.EFFECTORS[part].source.path))
    else:  # the wrist options and the native jaw are shown on the stock SO-101
        return robot_view(R.part_robot_id(slot, part))
    R.fix_qpos0(m)
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    return {"id": f"{slot}:{part}", **_static_view(m, d)}
