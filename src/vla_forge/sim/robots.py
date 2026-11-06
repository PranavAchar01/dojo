"""Custom robots: an arm, a wrist setting and an end effector, composed into one MuJoCo model from sourced parts.

Every part is an open-source MJCF used as published (vendor/so101 from TheRobotStudio, vendor/menagerie from
google-deepmind/mujoco_menagerie; licences and URLs in SOURCES.md). Nothing is modelled here: composition only
deletes elements (a native jaw, a locked roll joint) and attaches one sourced model to another at a frame that
is computed from the parts' own geometry.

A robot id names the three slots, `<arm>.<wrist>.<effector>`, so any process can rebuild the same model from
the id alone. "so101" is the stock SO-101 (the same model as `so101.native.native`, kept on its original code
path so its measured numbers stay exactly as they were).

What is computed from the parts, once per robot id:

* the tool frame (`tcp` site) at the fingertips, x along the approach direction and z along the closing
  direction, which is the convention of the SO-101's own `gripperframe` site that the scripted expert plans in;
* how wide the fingers open at each gripper command (measured in simulation, gravity off), from which the open,
  squeeze and closed commands for a 3 cm block follow;
* for a swapped effector, the mount frame: on the wrist's roll axis, at the end of the parts that stay, with the
  effector's closing axis lined up with the native jaw's. No adapter plate is modelled (no sourced part exists);
* a home pose found by IK (gripper above the table centre, pointing down) for arms other than the SO-101.

`robot_feasibility(robot_id, task)` answers whether a robot can do a task, with the reason, and `report(robot_id)`
holds the checks the Build tab shows (model loads, joint limits, reach over the table workspace, finger opening,
holding torque against the model's own actuator limits).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

import mujoco
import numpy as np

REPO = Path(__file__).resolve().parents[3]
MENAGERIE = REPO / "vendor" / "menagerie"
MENAGERIE_URL = "https://github.com/google-deepmind/mujoco_menagerie"
MENAGERIE_COMMIT = "98a70c569c4c1b6da467a20012f766ee1f8dc521"

DEFAULT = "so101"
BLOCK_W = 0.03  # the scene's blocks are 3 cm cubes (scene.CUBE_HALF * 2)
TIP_ZONE = (
    0.02  # fingertip region used to measure the opening, metres back from the tip
)
SQUEEZE_BELOW = (
    0.012  # squeeze command: fingers driven to this much narrower than the block
)
HOME_TCP = np.array(
    [0.20, 0.0, 0.15]
)  # where non-SO-101 arms rest: above the table centre, pointing down


# ------------------------------------------------------------------ the catalog


@dataclass(frozen=True)
class Source:
    name: str
    maker: str
    licence: str
    url: str
    path: str  # file under the repo
    notes: str = ""


@dataclass(frozen=True)
class ArmDef:
    id: str
    source: Source
    joints: tuple[str, ...]  # arm joints, base to wrist; the last one is the wrist roll
    grip_actuator: str  # the native jaw's actuator
    flange: str  # body that carries the native jaw (child of the roll joint)
    jaw_a: tuple[str, ...]  # bodies of the fixed / left finger
    jaw_b: tuple[str, ...]  # bodies of the moving / right finger
    strip_bodies: tuple[str, ...]  # bodies removed when the effector is swapped
    strip_meshes: tuple[
        str, ...
    ]  # meshes (geoms using them) removed from the flange when swapped
    pad_hint: str = (
        "pad"  # collision geoms with this in the name are the finger pads, if any exist
    )
    kp: float | None = (
        None  # position gain override (the SO-101 uses the STS3215 value, as in the stock scene)
    )
    reach_m: float = 0.0  # nominal reach from the maker, for the catalog card


@dataclass(frozen=True)
class EffectorDef:
    id: str
    source: Source
    root: str  # root body of the effector model
    actuator: str
    jaw_a: tuple[str, ...]
    jaw_b: tuple[str, ...]
    kind: str = "parallel"


SO101_SRC = Source(
    "SO-101 follower arm",
    "TheRobotStudio (SO-ARM100 repo), used by Hugging Face LeRobot",
    "Apache-2.0",
    "https://github.com/TheRobotStudio/SO-ARM100",
    "vendor/so101/so101_new_calib.xml",
    "MJCF generated from the Onshape CAD by the SO-ARM100 project",
)

ARMS: dict[str, ArmDef] = {
    "so101": ArmDef(
        "so101",
        SO101_SRC,
        ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"),
        "gripper",
        "gripper",
        ("gripper",),
        ("moving_jaw_so101_v1",),
        ("moving_jaw_so101_v1",),
        ("wrist_roll_follower_so101_v1", "sts3215_03a_v1"),
        kp=17.8,
        reach_m=0.38,
    ),
    "so100": ArmDef(
        "so100",
        Source(
            "SO-100 (SO-ARM100) arm",
            "TheRobotStudio; MJCF by MuJoCo Menagerie (trs_so_arm100)",
            "Apache-2.0",
            f"{MENAGERIE_URL}/tree/{MENAGERIE_COMMIT}/trs_so_arm100",
            "vendor/menagerie/trs_so_arm100/so_arm100.xml",
            "derived from the SO-ARM100 URDF",
        ),
        ("Rotation", "Pitch", "Elbow", "Wrist_Pitch", "Wrist_Roll"),
        "Jaw",
        "Fixed_Jaw",
        ("Fixed_Jaw",),
        ("Moving_Jaw",),
        ("Moving_Jaw",),
        (
            "Fixed_Jaw",
            "Fixed_Jaw_Motor",
            "Fixed_Jaw_Collision_1",
            "Fixed_Jaw_Collision_2",
        ),
        reach_m=0.38,
    ),
    "koch": ArmDef(
        "koch",
        Source(
            "Koch v1.1 low-cost arm",
            "Alexander Koch, updated by Hugging Face LeRobot; MJCF by MuJoCo Menagerie (low_cost_robot_arm)",
            "Apache-2.0",
            f"{MENAGERIE_URL}/tree/{MENAGERIE_COMMIT}/low_cost_robot_arm",
            "vendor/menagerie/low_cost_robot_arm/low_cost_robot_arm.xml",
            "derived from the koch-v1-1 follower URDF",
        ),
        ("base_rotation", "pitch", "elbow", "wrist_pitch", "wrist_roll"),
        "gripper",
        "gripper_static_finger",
        ("gripper_static_finger",),
        ("gripper_moving_finger",),
        ("gripper_moving_finger",),
        (
            "gripper_static_finger",
            "gripper_static_finger_motor",
            "gripper_static_finger_collision_1",
            "gripper_static_finger_collision_2",
        ),
        reach_m=0.30,
    ),
    "wx250s": ArmDef(
        "wx250s",
        Source(
            "WidowX 250 6DOF",
            "Trossen Robotics (Interbotix); MJCF by MuJoCo Menagerie (trossen_wx250s)",
            "BSD-3-Clause",
            f"{MENAGERIE_URL}/tree/{MENAGERIE_COMMIT}/trossen_wx250s",
            "vendor/menagerie/trossen_wx250s/wx250s.xml",
            "derived from the interbotix_xsarm_descriptions URDF",
        ),
        ("waist", "shoulder", "elbow", "forearm_roll", "wrist_angle", "wrist_rotate"),
        "gripper",
        "wx250s/gripper_link",
        ("wx250s/left_finger_link",),
        ("wx250s/right_finger_link",),
        ("wx250s/left_finger_link", "wx250s/right_finger_link"),
        ("wx250s_9_gripper_bar",),
        pad_hint="",
        reach_m=0.65,
    ),
    "vx300s": ArmDef(
        "vx300s",
        Source(
            "ViperX 300 6DOF",
            "Trossen Robotics (Interbotix); MJCF by MuJoCo Menagerie (trossen_vx300s)",
            "BSD-3-Clause",
            f"{MENAGERIE_URL}/tree/{MENAGERIE_COMMIT}/trossen_vx300s",
            "vendor/menagerie/trossen_vx300s/vx300s.xml",
            "derived from the interbotix_xsarm_descriptions URDF",
        ),
        ("waist", "shoulder", "elbow", "forearm_roll", "wrist_angle", "wrist_rotate"),
        "gripper",
        "gripper_link",
        ("left_finger_link",),
        ("right_finger_link",),
        ("left_finger_link", "right_finger_link"),
        ("vx300s_9_gripper_bar",),
        pad_hint="",
        reach_m=0.75,
    ),
}

WRISTS = {
    "native": {
        "name": "Native wrist (pitch + roll)",
        "notes": "the arm's own wrist, every joint driven",
    },
    "locked": {
        "name": "Roll locked",
        "notes": "the wrist roll joint is fixed at zero and its actuator removed: one joint fewer, so the action "
        "space shrinks by one. Same sourced links, no new part",
    },
}

EFFECTORS: dict[str, EffectorDef | None] = {
    "native": None,  # the arm's own jaw
    "robotiq_2f85": EffectorDef(
        "robotiq_2f85",
        Source(
            "Robotiq 2F-85 gripper",
            "Robotiq; MJCF by MuJoCo Menagerie (robotiq_2f85)",
            "BSD-2-Clause",
            f"{MENAGERIE_URL}/tree/{MENAGERIE_COMMIT}/robotiq_2f85",
            "vendor/menagerie/robotiq_2f85/2f85.xml",
            "derived from the ROS-Industrial robotiq description; 85 mm adaptive parallel gripper",
        ),
        "base_mount",
        "fingers_actuator",
        ("left_pad", "left_silicone_pad"),
        ("right_pad", "right_silicone_pad"),
    ),
    "franka_hand": EffectorDef(
        "franka_hand",
        Source(
            "Franka Hand",
            "Franka Robotics (Franka Emika); MJCF by MuJoCo Menagerie (franka_emika_panda/hand.xml)",
            "Apache-2.0",
            f"{MENAGERIE_URL}/tree/{MENAGERIE_COMMIT}/franka_emika_panda",
            "vendor/menagerie/franka_hand/hand.xml",
            "the Panda's parallel gripper, 80 mm stroke",
        ),
        "hand",
        "actuator8",
        ("left_finger",),
        ("right_finger",),
    ),
}

# effectors considered and not offered, so the page can say why
NOT_OFFERED = [
    {
        "slot": "effector",
        "name": "Suction cup",
        "reason": "no open-source MuJoCo model of a suction gripper was found in the sources used; "
        "MuJoCo has no suction primitive, so one would have to be modelled here, which the parts rule forbids",
    },
    {
        "slot": "effector",
        "name": "UMI gripper",
        "reason": "the Menagerie UMI model is the hand-held rig (1.56 kg body with a free-floating base), "
        "not a wrist-mountable finger set",
    },
]


def parse_id(robot_id: str) -> tuple[str, str, str]:
    if robot_id == DEFAULT:
        return "so101", "native", "native"
    parts = robot_id.split(".")
    if len(parts) != 3:
        raise ValueError(f"robot id must be <arm>.<wrist>.<effector>: {robot_id!r}")
    arm, wrist, ee = parts
    if arm not in ARMS or wrist not in WRISTS or ee not in EFFECTORS:
        raise ValueError(f"unknown part in robot id {robot_id!r}")
    return arm, wrist, ee


def make_id(arm: str, wrist: str = "native", effector: str = "native") -> str:
    rid = f"{arm}.{wrist}.{effector}"
    parse_id(rid)
    return DEFAULT if rid == "so101.native.native" else rid


def is_default(robot_id: str) -> bool:
    return parse_id(robot_id) == ("so101", "native", "native")


# ------------------------------------------------------------------ geometry helpers


def _geom_points(m: mujoco.MjModel, d: mujoco.MjData, g: int) -> np.ndarray:
    """World-space points that bound a geom (mesh vertices, box corners, sphere/capsule extremes)."""
    pos, mat = d.geom_xpos[g], d.geom_xmat[g].reshape(3, 3)
    t = m.geom_type[g]
    s = m.geom_size[g]
    if t == mujoco.mjtGeom.mjGEOM_MESH:
        mid = m.geom_dataid[g]
        a, n = m.mesh_vertadr[mid], m.mesh_vertnum[mid]
        local = m.mesh_vert[a : a + n]
    elif t == mujoco.mjtGeom.mjGEOM_BOX:
        local = np.array(
            [
                [sx * s[0], sy * s[1], sz * s[2]]
                for sx in (-1, 1)
                for sy in (-1, 1)
                for sz in (-1, 1)
            ]
        )
    elif t == mujoco.mjtGeom.mjGEOM_SPHERE:
        local = np.vstack([np.eye(3) * s[0], -np.eye(3) * s[0]])
    elif t in (mujoco.mjtGeom.mjGEOM_CAPSULE, mujoco.mjtGeom.mjGEOM_CYLINDER):
        r, h = s[0], s[1]
        local = np.array(
            [
                [x, y, z]
                for z in (-h - r, h + r)
                for x, y in ((r, 0), (-r, 0), (0, r), (0, -r))
            ]
        )
    else:
        local = np.zeros((1, 3))
    return pos + local @ mat.T


def _collision_geoms(m: mujoco.MjModel, bodies: list[int], hint: str) -> list[int]:
    gs = [
        g
        for g in range(m.ngeom)
        if m.geom_bodyid[g] in bodies and (m.geom_contype[g] or m.geom_conaffinity[g])
    ]
    if hint:
        pads = [g for g in gs if hint in (m.geom(g).name or "")]
        if pads:
            return pads
    return gs


def _frame(x: np.ndarray, z: np.ndarray) -> np.ndarray:
    """Rotation matrix with columns x, y = z cross x, z (x and z orthonormalised, z kept, x adjusted)."""
    z = z / np.linalg.norm(z)
    x = x - z * (x @ z)
    x = x / np.linalg.norm(x)
    return np.column_stack([x, np.cross(z, x), z])


def _quat(rot: np.ndarray) -> list[float]:
    q = np.zeros(4)
    mujoco.mju_mat2Quat(q, rot.flatten())
    return q.tolist()


# ------------------------------------------------------------------ composition


def _load_arm(arm: ArmDef) -> mujoco.MjSpec:
    spec = mujoco.MjSpec.from_file(str(REPO / arm.source.path))
    from .scene import TIMESTEP

    spec.option.timestep = TIMESTEP
    spec.option.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
    for k in list(spec.keys):
        spec.delete(k)
    if arm.id != "so101":
        # turn the base so that zero pan faces +x, like the SO-101 in the stock scene
        root = spec.worldbody.first_body()
        yaw = -_pan_offset(arm.id)
        q = np.zeros(4)
        mujoco.mju_mulQuat(
            q,
            [math.cos(yaw / 2), 0, 0, math.sin(yaw / 2)],
            np.asarray(root.quat, float),
        )
        root.quat = q.tolist()
        root.pos = [
            math.cos(yaw) * root.pos[0] - math.sin(yaw) * root.pos[1],
            math.sin(yaw) * root.pos[0] + math.cos(yaw) * root.pos[1],
            root.pos[2],
        ]
    return spec


def fix_qpos0(m: mujoco.MjModel) -> None:
    """Joints whose zero lies outside their limits (the Trossen fingers) start at the nearest limit."""
    for j in range(m.njnt):
        if m.jnt_limited[j] and int(m.jnt_type[j]) in (
            int(mujoco.mjtJoint.mjJNT_HINGE),
            int(mujoco.mjtJoint.mjJNT_SLIDE),
        ):
            a = m.jnt_qposadr[j]
            m.qpos0[a] = np.clip(m.qpos0[a], *m.jnt_range[j])


@lru_cache(maxsize=16)
def _pan_offset(arm_id: str) -> float:
    """Heading of the arm's reach at zero pan, from random poses of the bare source model."""
    arm = ARMS[arm_id]
    m = mujoco.MjModel.from_xml_path(str(REPO / arm.source.path))
    d = mujoco.MjData(m)
    pan = m.joint(arm.joints[0])
    sign = float(np.sign(m.jnt_axis[pan.id][2]) or 1.0)
    tip = m.body(arm.flange).id
    rng = np.random.default_rng(0)
    ang = []
    jid = [m.joint(j).id for j in arm.joints]
    lo, hi = m.jnt_range[jid, 0], m.jnt_range[jid, 1]
    for _ in range(400):
        q = rng.uniform(lo, hi)
        q[0] = 0.0
        d.qpos[[m.joint(j).qposadr[0] for j in arm.joints]] = q
        mujoco.mj_kinematics(m, d)
        p = d.xpos[tip]
        ang.append((np.hypot(p[0], p[1]), math.atan2(p[1], p[0])))
    r_a = np.array(ang)
    a = r_a[
        r_a[:, 0] >= np.quantile(r_a[:, 0], 0.85), 1
    ]  # the stretched-out poses point forward
    off = math.atan2(np.sin(a).mean(), np.cos(a).mean()) * sign
    return round(off / (math.pi / 2)) * (
        math.pi / 2
    )  # arm models face along an axis; drop sampling noise


def _find_body(spec: mujoco.MjSpec, name: str):
    b = spec.body(name)
    if b is None:
        raise KeyError(name)
    return b


def _delete_actuator(spec: mujoco.MjSpec, name: str) -> None:
    a = spec.actuator(name)
    if a is not None:
        spec.delete(a)


def _strip_native_jaw(spec: mujoco.MjSpec, arm: ArmDef) -> None:
    """Remove the native jaw: its actuator, any equality and exclude on its joints, its bodies and meshes."""
    moving_joints = set()
    for bname in arm.strip_bodies:
        b = _find_body(spec, bname)
        for j in b.find_all(mujoco.mjtObj.mjOBJ_JOINT):
            moving_joints.add(j.name)
    _delete_actuator(spec, arm.grip_actuator)
    for eq in list(spec.equalities):
        if eq.name1 in moving_joints or eq.name2 in moving_joints:
            spec.delete(eq)
    for t in list(spec.tendons):
        spec.delete(t)
    stripped = set(arm.strip_bodies)
    for ex in list(spec.excludes):
        if ex.bodyname1 in stripped or ex.bodyname2 in stripped:
            spec.delete(ex)
    for bname in arm.strip_bodies:
        spec.delete(_find_body(spec, bname))
    flange = _find_body(spec, arm.flange)
    for g in list(flange.geoms):
        if g.meshname in arm.strip_meshes:
            spec.delete(g)


def _lock_roll(spec: mujoco.MjSpec, arm: ArmDef) -> None:
    roll = arm.joints[-1]
    for a in list(spec.actuators):
        if a.target == roll:
            spec.delete(a)
    spec.delete(spec.joint(roll))


def _base_spec(robot_id: str, with_effector: bool = True) -> mujoco.MjSpec:
    """The composed robot without the tool site (geometry is measured on this)."""
    arm_id, wrist, ee_id = parse_id(robot_id)
    arm = ARMS[arm_id]
    spec = _load_arm(arm)
    ee = EFFECTORS[ee_id]
    if ee is not None:
        _strip_native_jaw(spec, arm)
        if with_effector:
            mount = _mount(arm_id, ee_id)
            child = mujoco.MjSpec.from_file(str(REPO / ee.source.path))
            for k in list(child.keys):
                child.delete(k)
            child.option.impratio = spec.option.impratio
            child.option.cone = spec.option.cone
            flange = _find_body(spec, arm.flange)
            frame = flange.add_frame(pos=mount["pos"], quat=mount["quat"])
            root = child.body(ee.root)
            frame.attach_body(root, "ee_", "")
            fb = _find_body(spec, arm.flange)
            parent = fb.parent
            for other in (arm.flange, parent.name):
                spec.add_exclude(bodyname1="ee_" + ee.root, bodyname2=other)
    elif arm_id != "so101":
        # the two native fingers never need to touch each other; their hulls overlap when closed (Trossen)
        for a in arm.jaw_a:
            for b in arm.jaw_b:
                if a != arm.flange and b != arm.flange:
                    spec.add_exclude(bodyname1=a, bodyname2=b)
    if wrist == "locked":
        _lock_roll(spec, arm)
    for body in spec.bodies:
        if body.name != "world":
            body.gravcomp = 1.0
    if arm.kp is not None:
        for act in spec.actuators:
            if act.target in arm.joints:
                act.gainprm[0] = arm.kp
                act.biasprm[1] = -arm.kp
                act.biasprm[2] = 0.0
    return spec


@lru_cache(maxsize=32)
def _native_geometry(arm_id: str) -> dict:
    """Roll axis, approach and closing directions and the fingertip of an arm's native jaw, in flange frame."""
    arm = ARMS[arm_id]
    spec = _load_arm(arm)
    m = spec.compile()
    d = mujoco.MjData(m)
    mujoco.mj_kinematics(m, d)
    fb = m.body(arm.flange).id
    pf, rf = d.xpos[fb], d.xmat[fb].reshape(3, 3)
    a_ids = [m.body(b).id for b in arm.jaw_a]
    b_ids = [m.body(b).id for b in arm.jaw_b]
    pa = np.vstack(
        [_geom_points(m, d, g) for g in _collision_geoms(m, a_ids, arm.pad_hint)]
    )
    pb = np.vstack(
        [_geom_points(m, d, g) for g in _collision_geoms(m, b_ids, arm.pad_hint)]
    )
    roll = m.joint(arm.joints[-1])
    axis = rf @ m.jnt_axis[roll.id]
    anchor = pf + rf @ m.jnt_pos[roll.id]
    tips = np.vstack([pa, pb])
    if (tips.mean(0) - anchor) @ axis < 0:
        axis = -axis
    close = pb.mean(0) - pa.mean(0)
    close = close - axis * (close @ axis)
    close /= np.linalg.norm(close)
    # where the parts that stay end along the roll axis (points near the axis only)
    keep = []
    parent = m.body_parentid[fb]
    strip_mesh_ids = {m.mesh(n).id for n in arm.strip_meshes}
    strip_body_ids = {m.body(b).id for b in arm.strip_bodies}
    for g in range(m.ngeom):
        if m.geom_group[g] != 2 or m.geom_bodyid[g] not in (fb, parent):
            continue
        if m.geom_bodyid[g] in strip_body_ids:
            continue
        if (
            m.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH
            and m.geom_dataid[g] in strip_mesh_ids
        ):
            continue
        p = _geom_points(m, d, g)
        rel = p - anchor
        radial = np.linalg.norm(rel - np.outer(rel @ axis, axis), axis=1)
        keep.append(p[radial < 0.03])
    kept = np.vstack(keep) if keep else anchor[None]
    end = float(((kept - anchor) @ axis).max())
    to_local = lambda v: rf.T @ v
    return {
        "axis": to_local(axis),
        "close": to_local(close),
        "anchor": rf.T @ (anchor - pf),
        "mount_point": rf.T @ (anchor + axis * end - pf),
    }


@lru_cache(maxsize=16)
def _effector_geometry(ee_id: str) -> dict:
    """Closing axis of a sourced effector in its root body frame (approach is the root's +z)."""
    ee = EFFECTORS[ee_id]
    assert ee is not None
    m = mujoco.MjModel.from_xml_path(str(REPO / ee.source.path))
    d = mujoco.MjData(m)
    mujoco.mj_kinematics(m, d)
    rb = m.body(ee.root).id
    rr = d.xmat[rb].reshape(3, 3)
    pa = np.vstack(
        [
            _geom_points(m, d, g)
            for g in _collision_geoms(m, [m.body(b).id for b in ee.jaw_a], "")
        ]
    )
    pb = np.vstack(
        [
            _geom_points(m, d, g)
            for g in _collision_geoms(m, [m.body(b).id for b in ee.jaw_b], "")
        ]
    )
    close = rr.T @ (pb.mean(0) - pa.mean(0))
    close[2] = 0.0
    close /= np.linalg.norm(close)
    return {"close": close, "mass": float(m.body_subtreemass[rb])}


@lru_cache(maxsize=64)
def _mount(arm_id: str, ee_id: str) -> dict:
    """Frame on the flange where a swapped effector's root goes: on the roll axis, approach along it."""
    nat = _native_geometry(arm_id)
    eg = _effector_geometry(ee_id)
    # effector frame: z = approach, x = closing; flange frame: z -> roll axis, x -> native closing axis
    r_flange = _frame(nat["close"], nat["axis"])
    r_ee = _frame(eg["close"], np.array([0.0, 0.0, 1.0]))
    rot = r_flange @ r_ee.T
    return {"pos": nat["mount_point"].tolist(), "quat": _quat(rot)}


# ------------------------------------------------------------------ the robot record


@dataclass
class Robot:
    id: str
    arm: str
    wrist: str
    effector: str
    arm_joints: list[str]
    grip_actuator: str
    tcp_site: str
    tcp_body: str
    home: np.ndarray
    g_closed: float
    g_open: float
    g_squeeze: float
    centre_open: float
    centre_grasp: float
    centre_closed: float
    grip_scale: float = (
        1.0  # ctrl = action * grip_scale (tendon grippers take a force-like ctrl)
    )
    finger_bodies: list[str] = field(default_factory=list)
    widths: list[list[float]] = field(
        default_factory=list
    )  # [gripper action, opening m]
    wrist_cam: dict = field(default_factory=dict)

    @property
    def n_arm(self) -> int:
        return len(self.arm_joints)

    @property
    def act_dim(self) -> int:
        return self.n_arm + 1

    @property
    def default(self) -> bool:
        return self.id == DEFAULT

    def stroke(self) -> tuple[float, float]:
        w = [x[1] for x in self.widths]
        return (
            (max(min(w), 0.0), max(w)) if w else (0.0, 0.0)
        )  # model fingers may pass through each other


def _so101_default() -> Robot:
    from . import scene as S

    return Robot(
        id=DEFAULT,
        arm="so101",
        wrist="native",
        effector="native",
        arm_joints=list(S.JOINTS[:5]),
        grip_actuator="gripper",
        tcp_site="gripperframe",
        tcp_body="gripper",
        home=S.HOME.copy(),
        g_closed=S.GRIPPER_CLOSED,
        g_open=0.7,
        g_squeeze=0.08,
        centre_open=0.024,
        centre_grasp=0.0135,
        centre_closed=0.006,
        finger_bodies=["gripper", "moving_jaw_so101_v1"],
    )


def _finger_sets(robot_id: str, m: mujoco.MjModel) -> tuple[list[int], list[int], str]:
    arm_id, _w, ee_id = parse_id(robot_id)
    arm = ARMS[arm_id]
    ee = EFFECTORS[ee_id]
    if ee is None:
        return (
            [m.body(b).id for b in arm.jaw_a],
            [m.body(b).id for b in arm.jaw_b],
            arm.pad_hint,
        )
    return (
        [m.body("ee_" + b).id for b in ee.jaw_a],
        [m.body("ee_" + b).id for b in ee.jaw_b],
        "",
    )


def _seed_table(
    m: mujoco.MjModel, site: int, joints: list[str], n: int = 6000
) -> np.ndarray:
    """Random joint configurations whose tool frame points down: rows of [x, y, z, q...], IK starting points."""
    d = mujoco.MjData(m)
    qadr = [m.joint(j).qposadr[0] for j in joints]
    jid = [m.joint(j).id for j in joints]
    lo, hi = m.jnt_range[jid, 0], m.jnt_range[jid, 1]
    rng = np.random.default_rng(0)
    rows = []
    for _ in range(n):
        q = rng.uniform(lo, hi)
        d.qpos[qadr] = q
        mujoco.mj_kinematics(m, d)
        if d.site_xmat[site].reshape(3, 3)[2, 0] < -0.8:
            rows.append(np.r_[d.site_xpos[site], q])
    return np.array(rows) if rows else np.zeros((0, 3 + len(joints)))


_SEEDS: dict[str, np.ndarray] = {}


def seeds_for(
    robot_id: str, m: mujoco.MjModel, site: int, joints: list[str]
) -> np.ndarray:
    if robot_id not in _SEEDS:
        _SEEDS[robot_id] = _seed_table(m, site, joints)
    return _SEEDS[robot_id]


def solve_topdown(
    m, d, site, joints, home, target, heading=None, seeds=None, iters=160, w_ori=0.1
):
    """Damped least squares to a top-down tool pose, started from home and from the nearest down-pointing
    seeds. Returns (q, position error m, orientation error) of the best start."""
    qadr = np.array([m.joint(j).qposadr[0] for j in joints])
    dof = np.array([m.joint(j).dofadr[0] for j in joints])
    jid = [m.joint(j).id for j in joints]
    lo, hi = m.jnt_range[jid, 0], m.jnt_range[jid, 1]
    jp, jr = np.zeros((3, m.nv)), np.zeros((3, m.nv))
    n = len(joints)
    starts = [np.clip(home, lo, hi)]
    if seeds is not None and len(seeds):
        near = np.argsort(np.linalg.norm(seeds[:, :3] - target, axis=1))[:3]
        starts += [seeds[i, 3:] for i in near]
    best = (starts[0], 9.0, 9.0)
    for q0 in starts:
        q = q0.copy()
        for _ in range(iters):
            d.qpos[qadr] = q
            mujoco.mj_kinematics(m, d)
            mujoco.mj_comPos(m, d)
            mat = d.site_xmat[site].reshape(3, 3)
            ep = target - d.site_xpos[site]
            eo = np.cross(mat[:, 0], [0, 0, -1.0])
            if heading is not None and n >= 5:
                eo = eo + np.cross(mat[:, 2], heading)
            mujoco.mj_jacSite(m, d, jp, jr, site)
            J = np.vstack([jp[:, dof], w_ori * jr[:, dof]])
            e = np.concatenate([ep, w_ori * eo])
            JJt = J @ J.T + 1e-3 * np.eye(6)
            dq = J.T @ np.linalg.solve(JJt, e)
            N = np.eye(n) - J.T @ np.linalg.solve(JJt, J)
            dq += N @ (0.02 * (home - q))
            q = np.clip(q + np.clip(dq, -0.2, 0.2), lo, hi)
        d.qpos[qadr] = q
        mujoco.mj_kinematics(m, d)
        mat = d.site_xmat[site].reshape(3, 3)
        ep_n = float(np.linalg.norm(target - d.site_xpos[site]))
        eo_n = float(np.linalg.norm(np.cross(mat[:, 0], [0, 0, -1.0])))
        if ep_n + 0.05 * eo_n < best[1] + 0.05 * best[2]:
            best = (q.copy(), ep_n, eo_n)
    d.qpos[qadr] = best[0]
    mujoco.mj_kinematics(m, d)
    return best


@lru_cache(maxsize=32)
def calibrate(robot_id: str) -> Robot:
    """Everything the expert, the scene and the policy need to know about a robot id."""
    arm_id, wrist, ee_id = parse_id(robot_id)
    if so101_jaw(robot_id):
        # the stock jaw keeps its grasp pads and constants; only the joint set can change (roll locked)
        r = _so101_default()
        r.id, r.wrist = robot_id, wrist
        if wrist == "locked":
            r.arm_joints, r.home = r.arm_joints[:4], r.home[:4].copy()
        r.widths = _measure_widths_default()
        return r
    arm = ARMS[arm_id]
    ee = EFFECTORS[ee_id]
    spec = _base_spec(robot_id)
    spec.option.gravity = [0.0, 0.0, 0.0]
    m = spec.compile()
    fix_qpos0(m)
    d = mujoco.MjData(m)
    joints = [j for j in arm.joints if not (wrist == "locked" and j == arm.joints[-1])]
    grip_act = arm.grip_actuator if ee is None else "ee_" + ee.actuator
    gi = m.actuator(grip_act).id
    # ctrl = gain * x + bias1 * length: a position-like actuator whose target length is ctrl * gain / -bias1
    gain, b1 = m.actuator_gainprm[gi, 0], m.actuator_biasprm[gi, 1]
    scale = float(-b1 / gain) if b1 < 0 and gain > 0 else 1.0
    lo, hi = m.actuator_ctrlrange[gi]
    a_ids, b_ids, hint = _finger_sets(robot_id, m)
    tcp_body = arm.flange if ee is None else "ee_" + ee.root
    tb = m.body(tcp_body).id

    def faces_toward(gs: list[int], axis: np.ndarray | None) -> list[int]:
        # box plates whose thin side is not along the closing axis are side walls, not grasp faces
        if axis is None:
            return gs
        keep = []
        for g in gs:
            if m.geom_type[g] == mujoco.mjtGeom.mjGEOM_BOX:
                thin = d.geom_xmat[g].reshape(3, 3)[:, int(np.argmin(m.geom_size[g]))]
                if abs(float(thin @ axis)) < 0.9:
                    continue
            keep.append(g)
        return keep or gs

    def opening(u: float, axis: np.ndarray | None = None):
        mujoco.mj_resetData(m, d)
        d.ctrl[:] = 0
        for j in joints:
            adr = m.joint(j).qposadr[0]
            d.ctrl[m.actuator(_act_for_joint(m, j)).id] = d.qpos[adr]
        d.ctrl[gi] = u
        for _ in range(900):
            mujoco.mj_step(m, d)
        pa = np.vstack(
            [
                _geom_points(m, d, g)
                for g in faces_toward(_collision_geoms(m, a_ids, hint), axis)
            ]
        )
        pb = np.vstack(
            [
                _geom_points(m, d, g)
                for g in faces_toward(_collision_geoms(m, b_ids, hint), axis)
            ]
        )
        return pa, pb, float(d.actuator_length[gi])

    pa, pb, _ = opening(hi)
    pf, rf = d.xpos[tb], d.xmat[tb].reshape(3, 3)
    if ee is None:
        nat = _native_geometry(arm_id)
        approach = rf @ nat["axis"] if arm.flange == tcp_body else None
        approach = approach if approach is not None else np.array([0, 0, -1.0])
    else:
        approach = rf[:, 2]
    close = pb.mean(0) - pa.mean(0)
    close -= approach * (close @ approach)
    close /= np.linalg.norm(close)

    widths = []
    faces = []
    for u in np.linspace(lo, hi, 9):
        pa, pb, length = opening(float(u), close)
        allp = np.vstack([pa, pb])
        tip = float((allp @ approach).max())
        za = pa[(pa @ approach) > tip - TIP_ZONE]
        zb = pb[(pb @ approach) > tip - TIP_ZONE]
        if not len(za) or not len(zb):
            za, zb = pa, pb
        fa, fbb = float((za @ close).max()), float((zb @ close).min())
        widths.append([round(length / 1.0, 5), round(fbb - fa, 5)])
        faces.append((u, length, fa, fbb, tip, allp))
    # gripper actions are in actuator-length units (the state the policy reads); ctrl = action * scale
    lens = np.array([w[0] for w in widths])
    ws = np.array([w[1] for w in widths])
    order = np.argsort(lens)
    lens, ws = lens[order], ws[order]
    faces = [faces[i] for i in order]
    open_i = int(np.argmax(ws))
    closed_i = int(np.argmin(ws))
    g_open_full, g_closed = float(lens[open_i]), float(lens[closed_i])

    def length_for(width: float) -> float:
        # widths are monotone between the closed and open ends; interpolate along that stretch
        i0, i1 = sorted((closed_i, open_i))
        seg_l, seg_w = lens[i0 : i1 + 1], ws[i0 : i1 + 1]
        o = np.argsort(seg_w)
        return float(np.interp(width, seg_w[o], seg_l[o]))

    w_max, w_min = float(ws.max()), float(ws.min())
    if (
        w_min < 0
    ):  # the source fingers pass through each other past touching; closed = where they meet
        g_closed = length_for(0.0)
        w_min = 0.0
    g_open = length_for(min(w_max, BLOCK_W + 0.035)) if w_max > BLOCK_W else g_open_full
    g_squeeze = length_for(max(w_min, BLOCK_W - SQUEEZE_BELOW))
    # tool frame: at the fingertip, centred between the faces when the fingers are at block width
    u_block = length_for(min(max(BLOCK_W, w_min), w_max))
    k = int(np.argmin(np.abs(lens - u_block)))
    _u, _l, fa, fbb, tip, allp = faces[k]
    third = np.cross(close, approach)
    zone = allp[(allp @ approach) > tip - TIP_ZONE]
    mid_close = (fa + fbb) / 2
    centre_w = approach * tip + close * mid_close + third * float((zone @ third).mean())
    # asymmetric jaws (one finger fixed to the flange) shift the gap centre as they open
    fixed_a = ee is None and arm.flange in arm.jaw_a

    def centre_at(g: float) -> float:
        if not fixed_a:
            return 0.0
        return (float(np.interp(g, lens, ws)) - (fbb - fa)) / 2

    rot_w = _frame(approach, close)  # x approach, z closing
    pos_local = rf.T @ (centre_w - pf)
    rot_local = rf.T @ rot_w
    # wrist camera: behind the fingertips on the side, looking along the jaws
    cam_pos = rf.T @ (
        centre_w - approach * 0.11 + np.cross(close, approach) * 0.05 - pf
    )
    cam_z = -(approach * 0.9 - np.cross(close, approach) * 0.35)
    cam_z /= np.linalg.norm(cam_z)
    cam_x = close - cam_z * (close @ cam_z)
    cam_x /= np.linalg.norm(cam_x)
    cam_y = np.cross(cam_z, cam_x)
    r = Robot(
        id=robot_id,
        arm=arm_id,
        wrist=wrist,
        effector=ee_id,
        arm_joints=joints,
        grip_actuator=grip_act,
        tcp_site="tcp",
        tcp_body=tcp_body,
        home=np.zeros(len(joints)),
        g_closed=g_closed,
        g_open=g_open,
        g_squeeze=g_squeeze,
        centre_open=centre_at(g_open),
        centre_grasp=centre_at(u_block),
        centre_closed=centre_at(g_closed),
        grip_scale=scale,
        finger_bodies=[m.body(i).name for i in a_ids + b_ids],
        widths=[[float(a), float(b)] for a, b in zip(lens, ws, strict=True)],
        wrist_cam={
            "pos": cam_pos.tolist(),
            "xyaxes": (rf.T @ cam_x).tolist() + (rf.T @ cam_y).tolist(),
        },
    )
    r._tcp = {"pos": pos_local.tolist(), "quat": _quat(rot_local)}  # type: ignore[attr-defined]
    # home pose: tool above the table centre, pointing down
    m2 = robot_spec(robot_id, _r=r).compile()
    fix_qpos0(m2)
    d2 = mujoco.MjData(m2)
    site = m2.site("tcp").id
    seeds = seeds_for(robot_id, m2, site, joints)
    jid = [m2.joint(j).id for j in joints]
    mid = m2.jnt_range[jid].mean(1)
    for z in (
        HOME_TCP[2],
        0.12,
        0.09,
        0.06,
    ):  # the highest top-down rest pose the arm can hold
        q, ep, eo = solve_topdown(
            m2, d2, site, joints, mid, np.array([*HOME_TCP[:2], z]), None, seeds, 300
        )
        if ep < 0.005 and eo < 0.1:
            break
    r.home = q
    return r


def _act_for_joint(m: mujoco.MjModel, joint: str) -> int:
    jid = m.joint(joint).id
    for a in range(m.nu):
        if (
            m.actuator_trntype[a] == mujoco.mjtTrn.mjTRN_JOINT
            and m.actuator_trnid[a, 0] == jid
        ):
            return a
    raise KeyError(joint)


def _measure_widths_default() -> list[list[float]]:
    """SO-101 jaw opening per gripper command, on the scene's own grasp pads."""
    from .scene import GRIPPER_CLOSED, GRIPPER_OPEN, _arm_spec, _pad_frames

    spec = _arm_spec()
    m = spec.compile()
    d = mujoco.MjData(m)
    frames = _pad_frames()
    out = []
    for g in np.linspace(GRIPPER_CLOSED, GRIPPER_OPEN, 9):
        d.qpos[m.joint("gripper").qposadr[0]] = g
        mujoco.mj_kinematics(m, d)
        pts = []
        for body, (pos, _q, size) in frames.items():
            b = m.body(body).id
            pts.append(d.xpos[b] + d.xmat[b].reshape(3, 3) @ pos)
        s = m.site("gripperframe").id
        z = d.site_xmat[s].reshape(3, 3)[:, 2]
        gap = abs(float((pts[1] - pts[0]) @ z)) - 2 * frames["gripper"][2][0] * 0
        out.append(
            [float(g), round(gap - 0.007, 5)]
        )  # pad faces: centres minus the two 3.5 mm pad halves
    return out


def so101_jaw(robot_id: str) -> bool:
    """The SO-101's own jaw (stock grasp pads, `gripperframe` tool site), with either wrist setting."""
    arm, _w, ee = parse_id(robot_id)
    return arm == "so101" and ee == "native"


def robot_spec(robot_id: str, _r: Robot | None = None) -> mujoco.MjSpec:
    """A fresh MjSpec of the robot alone (no scene), with the tool site, ready to add a scene to."""
    if is_default(robot_id):
        from .scene import _arm_spec

        return _arm_spec()
    if so101_jaw(robot_id):
        return _base_spec(robot_id)
    r = _r or calibrate(robot_id)
    spec = _base_spec(robot_id)
    body = _find_body(spec, r.tcp_body)
    body.add_site(
        name="tcp", pos=r._tcp["pos"], quat=r._tcp["quat"], size=[0.004] * 3, group=4
    )  # type: ignore[attr-defined]
    return spec


# ------------------------------------------------------------------ checks and feasibility

# the table workspace the task sampler uses (tasks.BLOCK_R / GOAL_R, BLOCK_ANG / GOAL_ANG)
WS_R = (0.125, 0.285)
WS_ANG = math.radians(38)
GRASP_Z = 0.015 - 0.012  # tool frame height when gripping a block (expert GRIP_DEPTH)
HOVER_Z = 0.07
PUSH_Z = 0.012
REACH_TOL = 0.006
# how far from straight down the tool may tilt (|x cross down|): tight where fingers meet the block,
# loose in transit, where the expert's IK weights orientation softly too
ORI_TOL = {"grasp": 0.12, "push": 0.2, "hover": 0.5}


@lru_cache(maxsize=32)
def reachability(robot_id: str, n_r: int = 7, n_a: int = 9) -> dict:
    """Top-down reach over the table workspace at grasp, push and hover height."""
    r = calibrate(robot_id)
    m = robot_spec(robot_id).compile()
    fix_qpos0(m)
    d = mujoco.MjData(m)
    site = m.site(r.tcp_site).id
    seeds = seeds_for(robot_id, m, site, r.arm_joints)
    pts = []
    for rr in np.linspace(*WS_R, n_r):
        for aa in np.linspace(-WS_ANG, WS_ANG, n_a):
            xy = np.array([rr * math.cos(aa), rr * math.sin(aa)])
            radial = np.array([*xy / np.linalg.norm(xy), 0.0])
            row = {"x": round(float(xy[0]), 4), "y": round(float(xy[1]), 4)}
            for name, z in (("grasp", GRASP_Z), ("push", PUSH_Z), ("hover", HOVER_Z)):
                _q, ep, eo = solve_topdown(
                    m,
                    d,
                    site,
                    r.arm_joints,
                    r.home,
                    np.array([*xy, z]),
                    radial,
                    seeds,
                    w_ori=0.02 if name == "hover" else 0.1,
                )
                row[name] = bool(ep < REACH_TOL and eo < ORI_TOL[name])
            pts.append(row)
    frac = {
        k: round(sum(p[k] for p in pts) / len(pts), 3)
        for k in ("grasp", "push", "hover")
    }
    return {"points": pts, "fraction": frac, "tol_m": REACH_TOL}


@lru_cache(maxsize=32)
def report(robot_id: str) -> dict:
    """The checks the Build tab shows: loads, action space, joint limits, finger opening, reach, holding torque."""
    parse_id(robot_id)
    out: dict = {"id": robot_id, "checks": []}

    def check(name, ok, detail, level="block"):
        out["checks"].append(
            {"name": name, "ok": bool(ok), "detail": detail, "level": level}
        )

    try:
        r = calibrate(robot_id)
        from .scene import Layout, build_spec

        m = build_spec(Layout(), robot=robot_id).compile()
    except (
        ValueError,
        KeyError,
    ) as e:  # a part combination that does not compile is reported, not raised
        check("Model compiles", False, f"{type(e).__name__}: {e}")
        out["ok"] = False
        out["failure"] = f"Model compiles: {e}"
        return out
    check(
        "Model compiles",
        True,
        f"{m.nbody - 1} bodies, {m.njnt} joints, {m.nu} actuators, {m.nmesh} meshes",
    )
    out["act_dim"] = r.act_dim
    out["joints"] = []
    sane = True
    for j, h in zip(r.arm_joints, r.home, strict=True):
        jj = m.joint(j)
        lo, hi = (float(v) for v in m.jnt_range[jj.id])
        lim = bool(m.jnt_limited[jj.id])
        ok = (
            lim
            and hi - lo > 0.2
            and lo - 1e-6 <= h <= hi + 1e-6
            and hi - lo < 2 * math.pi + 0.2
        )
        sane &= ok
        out["joints"].append(
            {
                "name": j,
                "lo": round(lo, 3),
                "hi": round(hi, 3),
                "home": round(float(h), 3),
                "ok": ok,
            }
        )
    check(
        "Joint limits sane",
        sane,
        "every arm joint limited, range > 0.2 rad, home inside the range"
        if sane
        else "a joint is unlimited or home is outside its range",
    )
    check(
        "Action space",
        True,
        f"{r.n_arm} arm joints + gripper = {r.act_dim}-D actions"
        + (
            ""
            if r.act_dim == 6
            else f" (the stock SO-101 is 6-D; the policy head is resized for {r.act_dim})"
        ),
        "info",
    )
    w_lo, w_hi = r.stroke()
    out["stroke_mm"] = [round(w_lo * 1000, 1), round(w_hi * 1000, 1)]
    check(
        "Fingers fit a 3 cm block",
        w_hi > BLOCK_W + 0.008 and w_lo < BLOCK_W - 0.002,
        f"opening {w_lo * 1000:.0f} to {w_hi * 1000:.0f} mm (measured in sim at the fingertips)",
    )
    reach = reachability(robot_id)
    out["reach"] = reach
    f = reach["fraction"]
    check(
        "Reaches the table workspace",
        f["grasp"] >= 0.9 and f["hover"] >= 0.9,
        f"top-down: {f['grasp']:.0%} of grasp points, {f['push']:.0%} push, {f['hover']:.0%} hover "
        f"(r {WS_R[0]:.3f}-{WS_R[1]:.3f} m, +/-{math.degrees(WS_ANG):.0f} deg)",
    )
    tq = holding_torque(robot_id)
    out["torque"] = tq
    check(
        "Holds a block at full reach",
        tq["ok"],
        tq["detail"],
        "block" if not tq["ok"] else "info",
    )
    # self-contact at home
    from .scene import Scene

    sc = Scene(Layout(), robot=robot_id)
    sc.reset(Layout())
    for _ in range(60):
        mujoco.mj_step(sc.model, sc.data)
    pen = 0.0
    for c in sc.data.contact[: sc.data.ncon]:
        b1, b2 = sc.model.geom_bodyid[c.geom[0]], sc.model.geom_bodyid[c.geom[1]]
        if b1 and b2:
            pen = max(pen, -float(c.dist))
    sc.close()
    check(
        "No self-collision at home",
        pen < 0.004,
        f"deepest robot contact {pen * 1000:.1f} mm",
    )
    out["mass_kg"] = round(float(m.body_subtreemass[1]), 3) if m.nbody > 1 else 0.0
    out["ok"] = all(c["ok"] for c in out["checks"] if c["level"] == "block")
    bad = next(
        (c for c in out["checks"] if c["level"] == "block" and not c["ok"]), None
    )
    out["failure"] = f"{bad['name']}: {bad['detail']}" if bad else None
    return out


@lru_cache(maxsize=32)
def holding_torque(robot_id: str) -> dict:
    """Gravity torque per joint (the sim compensates gravity, a real arm would not) with a 30 g block held at the
    far edge of the workspace, against the actuator force limits in the source model."""
    r = calibrate(robot_id)
    spec = robot_spec(robot_id)
    for b in spec.bodies:
        b.gravcomp = 0.0
    m = spec.compile()
    d = mujoco.MjData(m)
    site = m.site(r.tcp_site).id
    fix_qpos0(m)
    solve_topdown(
        m,
        d,
        site,
        r.arm_joints,
        r.home,
        np.array([WS_R[1], 0.0, HOVER_Z]),
        np.array([1.0, 0, 0]),
        seeds_for(robot_id, m, site, r.arm_joints),
    )
    q = d.qpos.copy()
    mujoco.mj_resetData(m, d)
    d.qpos[:] = q
    mujoco.mj_forward(m, d)
    tau = d.qfrc_bias.copy()
    # the block: 30 g at the tool frame
    jp = np.zeros((3, m.nv))
    mujoco.mj_jacSite(m, d, jp, None, site)
    tau += jp.T @ np.array([0.0, 0.0, 0.03 * 9.81])
    rows, ok, worst = [], True, 0.0
    for j in r.arm_joints:
        a = _act_for_joint(m, j)
        lim = float(m.actuator_forcerange[a, 1]) if m.actuator_forcelimited[a] else None
        t = abs(float(tau[m.joint(j).dofadr[0]]))
        if lim:
            worst = max(worst, t / lim)
            ok &= t <= lim
        rows.append({"joint": j, "torque_nm": round(t, 3), "limit_nm": lim})
    lims = [x for x in rows if x["limit_nm"]]
    if not lims:
        detail = (
            "the source model sets no actuator force limits, so torque is not checked"
        )
    else:
        hi = max(lims, key=lambda x: x["torque_nm"] / x["limit_nm"])
        detail = (
            f"worst joint {hi['joint']}: {hi['torque_nm']:.2f} of {hi['limit_nm']:.2f} N m "
            f"({worst:.0%} of the model's actuator limit) with a 30 g block at {WS_R[1]:.2f} m"
        )
    return {
        "ok": bool(ok),
        "rows": rows,
        "worst_frac": round(worst, 3),
        "detail": detail,
    }


GRASP_SKILLS = ("lift", "pick_place", "stack")


def robot_feasibility(robot_id: str, task) -> tuple[bool, str]:
    """Can this robot do this task? `task` is a TaskSpec, a dict with a "skill" key, or a skill name.

    Returns (ok, reason). Used by the router (skill library) and the Build tab. Reasons are plain sentences.
    Checks, in order: the model composes; the fingers open wider than a 3 cm block and close below it (grasping
    skills); top-down reach over the table workspace at the height the skill works at (>= 90% of points);
    gravity torque with a held block within the source model's actuator limits.
    """
    skill = (
        task
        if isinstance(task, str)
        else (task["skill"] if isinstance(task, dict) else task.skill)
    )
    try:
        rep = report(robot_id)
    except ValueError as e:
        return False, str(e)
    if not rep.get("ok", False) and any(
        c["name"] == "Model compiles" and not c["ok"] for c in rep["checks"]
    ):
        return False, "the parts do not compose into a valid MuJoCo model"
    w_lo, w_hi = rep["stroke_mm"]
    if skill in GRASP_SKILLS:
        if w_hi < BLOCK_W * 1000 + 8:
            return (
                False,
                f"the fingers open only {w_hi:.0f} mm; a 3 cm block needs at least 38 mm",
            )
        if w_lo > BLOCK_W * 1000 - 2:
            return (
                False,
                f"the fingers close only to {w_lo:.0f} mm, so they cannot squeeze a 3 cm block",
            )
    f = rep["reach"]["fraction"]
    need = {"reach": "hover", "push": "push"}.get(skill, "grasp")
    for h in (need, "hover"):
        if f[h] < 0.9:
            return False, (
                f"it reaches only {f[h]:.0%} of the table workspace at {h} height pointing down "
                f"(needs 90%); blocks sit {WS_R[0]:.2f}-{WS_R[1]:.2f} m from the base"
            )
    if not rep["torque"]["ok"]:
        return (
            False,
            "holding a block at full reach exceeds an actuator's force limit: "
            + rep["torque"]["detail"],
        )
    return (
        True,
        f"reaches {f[need]:.0%} of the workspace at {need} height; fingers open {w_lo:.0f}-{w_hi:.0f} mm",
    )


# ------------------------------------------------------------------ catalog for the page


def catalog() -> dict:
    def src(s: Source) -> dict:
        return {
            "name": s.name,
            "maker": s.maker,
            "licence": s.licence,
            "url": s.url,
            "path": s.path,
            "notes": s.notes,
        }

    arms = []
    for a in ARMS.values():
        arms.append(
            {
                "id": a.id,
                "slot": "arm",
                "dof": len(a.joints),
                "reach_m": a.reach_m,
                "joints": list(a.joints),
                **src(a.source),
            }
        )
    effs = [
        {
            "id": "native",
            "slot": "effector",
            "name": "Native jaw",
            "maker": "the selected arm's own gripper",
            "licence": "same as the arm",
            "url": "",
            "path": "",
            "notes": "the jaw that ships with the arm model",
        }
    ]
    for e in EFFECTORS.values():
        if e is None:
            continue
        effs.append({"id": e.id, "slot": "effector", "kind": e.kind, **src(e.source)})
    wrists = [
        {
            "id": k,
            "slot": "wrist",
            **v,
            "licence": "same as the arm",
            "maker": "the selected arm",
        }
        for k, v in WRISTS.items()
    ]
    return {
        "arms": arms,
        "wrists": wrists,
        "effectors": effs,
        "not_offered": NOT_OFFERED,
        "default": DEFAULT,
    }


def part_robot_id(slot: str, part: str) -> str:
    """The robot id used to preview a single part: an arm with its native jaw, or an effector on the SO-101."""
    if slot == "arm":
        return make_id(part)
    if slot == "wrist":
        return make_id("so101", part)
    return make_id("so101", "native", part)
