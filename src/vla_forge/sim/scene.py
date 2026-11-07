"""MuJoCo tabletop scene for the SO-101 arm.

The arm is TheRobotStudio's SO-101 MJCF (vendor/so101, from the SO-ARM100 repo), loaded unmodified with MjSpec.
Everything else is added in code, per layout:

* a table whose top surface is z = 0; the arm base sits at the origin facing +x
* blocks: 3 cm painted wooden cubes, 30 g, free joints
* bowls: shallow round bowls (base disc + ring of wall segments), mocap bodies so a layout can move them
* pads: a coloured square of tape (visual only)
* cameras: "front" (clamp-mounted webcam view of the workspace) and "wrist" (on the gripper, along the jaws)

Control follows the Stand-In sim this scene grew out of: gravity compensation on every arm body and the
STS3215 gains from vendor/so101/joints_properties.xml (kp 17.8). Physics runs at 300 Hz, control at 50 Hz and
the policy at 10 Hz, so all three land on whole physics steps.

Grasping uses two thin pads fitted to the inner finger faces: the vendor jaw collision meshes are convex hulls
that bulge into the grasp gap, so they stop colliding with blocks and the pads do the grasping.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

import mujoco
import numpy as np

REPO = Path(__file__).resolve().parents[3]
MJCF = REPO / "vendor" / "so101" / "so101_new_calib.xml"

JOINTS = [
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
    "gripper",
]
HOME = np.array([0.0, -0.6, 0.9, 0.9, 0.0])
GRIPPER_CLOSED = -0.1745
GRIPPER_OPEN = 1.7453

TIMESTEP = 1.0 / 300.0
STEPS_PER_CONTROL = 6  # 50 Hz control
CONTROLS_PER_POLICY = 5  # 10 Hz policy
STEPS_PER_POLICY = STEPS_PER_CONTROL * CONTROLS_PER_POLICY
POLICY_DT = STEPS_PER_POLICY * TIMESTEP
KP = 17.8

CUBE_HALF = 0.015
CUBE_MASS = 0.03
BOWL_R = 0.05
BOWL_WALL_H = 0.025
BOWL_WALL_T = 0.006
BOWL_SEGMENTS = 20
PAD_HALF = 0.04

OBS_SIZE = 168  # policy camera resolution (12 x 12 DINOv2 patches)

# colour names a prompt can use, as linear-ish sRGB
BLOCK_COLORS = {
    "red": (0.86, 0.14, 0.11),
    "green": (0.16, 0.62, 0.27),
    "blue": (0.13, 0.32, 0.86),
    "yellow": (0.95, 0.78, 0.10),
    "orange": (0.96, 0.46, 0.08),
    "purple": (0.48, 0.24, 0.72),
}
BOWL_COLORS = {
    "white": (0.93, 0.92, 0.88),
    "blue": (0.20, 0.40, 0.80),
    "green": (0.25, 0.58, 0.38),
    "pink": (0.90, 0.55, 0.62),
    "black": (0.12, 0.12, 0.13),
}
PAD_COLORS = {
    "blue": (0.10, 0.55, 0.95),
    "green": (0.15, 0.70, 0.35),
    "orange": (0.98, 0.55, 0.10),
    "white": (0.95, 0.95, 0.95),
}

# grasp pads, measured on the vendor meshes (see module docstring); coordinates in the gripperframe site frame
GRIP_REF = 0.0
PAD_X = (-0.042, -0.002)
PAD_HALF_Y = 0.009
FIXED_PAD_Z = (-0.010, -0.003)
MOVING_PAD_Z = (0.017, 0.024)
PAD_FRICTION = 1.4
GRASP_SOLREF = [0.007, 1.0]
GRASP_SOLIMP = [0.95, 0.99, 0.001, 0.5, 2.0]
MESH_BIT, CUBE_BIT = 1, 2


@dataclass
class Layout:
    """What is on the table, and where. Blocks: name -> (color, x, y, yaw); bowls and pads: name -> (color, x, y)."""

    blocks: dict[str, tuple[str, float, float, float]] = field(default_factory=dict)
    bowls: dict[str, tuple[str, float, float]] = field(default_factory=dict)
    pads: dict[str, tuple[str, float, float]] = field(default_factory=dict)
    # factory-line cells (sim/line.py): totes name -> (color, x, y), and whether a conveyor runs past the arm
    bins: dict[str, tuple[str, float, float]] = field(default_factory=dict)
    belt: bool = False
    # fixtures from sim/props.py (kitting tray, sockets, ...): name -> (kind, color, x, y, yaw)
    props: dict[str, tuple[str, str, float, float, float]] = field(default_factory=dict)
    # a factory dressing from scenes/<name>.json (assets_scene.py): scenery, floor, lamps, the hero product
    dressing: str = ""
    # a library model (assets/library/<id>) shown beside the arm, visual only: the model a prompt was given as its reference
    scenery: str = ""
    # real library tools lying on the table (sim/toolbox.py): name -> (tool id, x, y, yaw)
    tools: dict[str, tuple[str, float, float, float]] = field(default_factory=dict)

    def key(self) -> tuple:
        """Structure only (names and colours): layouts with the same key share a compiled model."""
        return (
            tuple((n, v[0]) for n, v in sorted(self.blocks.items())),
            tuple((n, v[0]) for n, v in sorted(self.bowls.items())),
            tuple((n, v[0]) for n, v in sorted(self.pads.items())),
            tuple((n, v[0]) for n, v in sorted(self.bins.items())),
            self.belt,
            *(
                [tuple((n, v[0], v[1]) for n, v in sorted(self.props.items()))]
                if self.props
                else []
            ),
            *(["dressing:" + self.dressing] if self.dressing else []),
            *(["scenery:" + self.scenery] if self.scenery else []),
            *([tuple((n, v[0]) for n, v in sorted(self.tools.items()))] if self.tools else []),
        )

    def fixtures(self) -> dict[str, tuple[str, float, float]]:
        """Every mocap fixture (bowls, squares, totes, props) as name -> (color, x, y)."""
        return {
            **self.bowls,
            **self.pads,
            **self.bins,
            **{n: (v[1], v[2], v[3]) for n, v in self.props.items()},
        }

    def to_json(self) -> dict:
        d = {"blocks": self.blocks, "bowls": self.bowls, "pads": self.pads}
        if self.bins or self.belt:
            d.update(bins=self.bins, belt=self.belt)
        if self.props:
            d["props"] = self.props
        if self.dressing:
            d["dressing"] = self.dressing
        if self.scenery:
            d["scenery"] = self.scenery
        if self.tools:
            d["tools"] = self.tools
        return d

    @staticmethod
    def from_json(d: dict) -> Layout:
        return Layout(
            {k: tuple(v) for k, v in d.get("blocks", {}).items()},
            {k: tuple(v) for k, v in d.get("bowls", {}).items()},
            {k: tuple(v) for k, v in d.get("pads", {}).items()},
            {k: tuple(v) for k, v in d.get("bins", {}).items()},
            bool(d.get("belt", False)),
            {k: tuple(v) for k, v in d.get("props", {}).items()},
            str(d.get("dressing", "")),
            str(d.get("scenery", "")),
            {k: tuple(v) for k, v in d.get("tools", {}).items()},
        )


@dataclass
class Look:
    """Domain randomization for the policy cameras."""

    table_rgb: tuple = (0.62, 0.52, 0.40)
    light_dir: tuple = (-0.3, 0.3, -1.0)
    light_gain: float = 0.8
    cam_jitter: tuple = (0.0, 0.0, 0.0)
    floor_rgb: tuple = (0.22, 0.23, 0.25)

    @staticmethod
    def sample(rng: np.random.Generator, strength: float = 1.0) -> Look:
        s = strength
        base = np.array([0.62, 0.52, 0.40])
        return Look(
            table_rgb=tuple(
                np.clip(base + rng.uniform(-0.12, 0.12, 3) * s, 0.05, 0.95)
            ),
            light_dir=(
                float(-0.3 + rng.uniform(-0.4, 0.4) * s),
                float(0.3 + rng.uniform(-0.4, 0.4) * s),
                -1.0,
            ),
            light_gain=float(0.8 + rng.uniform(-0.25, 0.2) * s),
            cam_jitter=tuple(rng.uniform(-0.012, 0.012, 3) * s),
            floor_rgb=tuple(
                np.clip(
                    np.array([0.22, 0.23, 0.25]) + rng.uniform(-0.08, 0.08, 3) * s, 0, 1
                )
            ),
        )


def look_at(pos, target, up=(0.0, 0.0, 1.0)):
    pos, target, up = map(np.asarray, (pos, target, up))
    fwd = target - pos
    fwd = fwd / np.linalg.norm(fwd)
    right = np.cross(fwd, up)
    right /= np.linalg.norm(right)
    return np.concatenate([right, np.cross(right, fwd)])


FRONT_POS = np.array([0.55, 0.0, 0.42])
FRONT_TARGET = np.array([0.18, 0.0, 0.0])


def _arm_spec() -> mujoco.MjSpec:
    spec = mujoco.MjSpec.from_file(str(MJCF))
    spec.option.timestep = TIMESTEP
    spec.option.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
    for body in spec.bodies:
        if body.name != "world":
            body.gravcomp = 1.0
    for act in spec.actuators:
        act.gainprm[0] = KP
        act.biasprm[1] = -KP
        act.biasprm[2] = 0.0
    return spec


@lru_cache(maxsize=1)
def _pad_frames():
    """Pad centre/orientation in the gripper and moving-jaw body frames, measured on the bare arm."""
    model = _arm_spec().compile()
    data = mujoco.MjData(model)
    data.qpos[model.joint("gripper").qposadr[0]] = GRIP_REF
    mujoco.mj_kinematics(model, data)
    s = model.site("gripperframe").id
    ps, rs = data.site_xpos[s], data.site_xmat[s].reshape(3, 3)
    out = {}
    for body, zr in (("gripper", FIXED_PAD_Z), ("moving_jaw_so101_v1", MOVING_PAD_Z)):
        b = model.body(body).id
        pb, rb = data.xpos[b], data.xmat[b].reshape(3, 3)
        centre = np.array([np.mean(PAD_X), 0.0, np.mean(zr)])
        pos = rb.T @ (ps + rs @ centre - pb)
        quat = np.zeros(4)
        mujoco.mju_mat2Quat(quat, (rb.T @ rs).flatten())
        size = [(PAD_X[1] - PAD_X[0]) / 2, PAD_HALF_Y, (zr[1] - zr[0]) / 2]
        out[body] = (pos, quat, size)
    return out


def build_spec(
    layout: Layout, offscreen: int = OBS_SIZE, robot: str = "so101"
) -> mujoco.MjSpec:
    """The table scene around a robot. `robot` is a robot id (sim/robots.py); "so101" is the stock arm."""
    from .robots import calibrate, is_default, robot_spec, so101_jaw

    default = is_default(robot)
    stock_jaw = so101_jaw(robot)  # the SO-101 jaw grasps with the scene's own pads
    spec = _arm_spec() if default else robot_spec(robot)
    spec.visual.global_.offwidth = max(offscreen, 640)
    spec.visual.global_.offheight = max(offscreen, 640)
    spec.visual.quality.shadowsize = 2048
    world = spec.worldbody

    spec.add_texture(
        name="floor_tex",
        type=mujoco.mjtTexture.mjTEXTURE_2D,
        builtin=mujoco.mjtBuiltin.mjBUILTIN_CHECKER,
        rgb1=[0.22, 0.23, 0.25],
        rgb2=[0.18, 0.19, 0.21],
        width=256,
        height=256,
    )
    spec.add_material(name="floor_mat", textures=["", "floor_tex"], texrepeat=[6, 6])
    spec.add_material(name="table_mat", rgba=[0.62, 0.52, 0.40, 1.0])
    world.add_light(
        name="key",
        pos=[0.4, -0.4, 1.4],
        dir=[-0.3, 0.3, -1.0],
        diffuse=[0.8] * 3,
        specular=[0.2] * 3,
        castshadow=True,
    )
    world.add_light(
        name="fill",
        pos=[-0.3, 0.5, 1.2],
        dir=[0.3, -0.4, -1.0],
        diffuse=[0.35] * 3,
        castshadow=False,
    )
    world.add_geom(
        name="floor",
        type=mujoco.mjtGeom.mjGEOM_PLANE,
        size=[3, 3, 0.1],
        pos=[0, 0, -0.75],
        material="floor_mat",
    )
    world.add_geom(
        name="table",
        type=mujoco.mjtGeom.mjGEOM_BOX,
        size=[0.45, 0.6, 0.02],
        pos=[0.2, 0.0, -0.02],
        material="table_mat",
        friction=[0.6, 0.005, 0.0001],
    )

    for name, (color, *_xy) in layout.pads.items():
        spec.add_material(name=f"{name}_mat", rgba=[*PAD_COLORS[color], 1.0])
        pad = world.add_body(name=name, mocap=True, pos=[0.25, 0, 0])
        t, w, h = PAD_HALF, 0.006, 0.0005
        for i, (px, py, sx, sy) in enumerate(
            [(t, 0, w, t + w), (-t, 0, w, t + w), (0, t, t + w, w), (0, -t, t + w, w)]
        ):
            pad.add_geom(
                name=f"{name}_tape{i}",
                type=mujoco.mjtGeom.mjGEOM_BOX,
                size=[sx, sy, h],
                pos=[px, py, h],
                material=f"{name}_mat",
                contype=0,
                conaffinity=0,
                group=1,
            )

    for name, (color, *_xy) in layout.bowls.items():
        spec.add_material(name=f"{name}_mat", rgba=[*BOWL_COLORS[color], 1.0])
        bowl = world.add_body(name=name, mocap=True, pos=[0.25, 0, 0])
        bowl.add_geom(
            name=f"{name}_base",
            type=mujoco.mjtGeom.mjGEOM_CYLINDER,
            size=[BOWL_R + BOWL_WALL_T, 0.002, 0],
            pos=[0, 0, 0.002],
            material=f"{name}_mat",
        )
        seg_half = (BOWL_R + BOWL_WALL_T / 2) * np.tan(np.pi / BOWL_SEGMENTS) * 1.05
        for i in range(BOWL_SEGMENTS):
            a = 2 * np.pi * i / BOWL_SEGMENTS
            r = BOWL_R + BOWL_WALL_T / 2
            bowl.add_geom(
                name=f"{name}_wall{i}",
                type=mujoco.mjtGeom.mjGEOM_BOX,
                size=[BOWL_WALL_T / 2, seg_half, BOWL_WALL_H / 2],
                pos=[r * np.cos(a), r * np.sin(a), BOWL_WALL_H / 2],
                quat=[np.cos(a / 2), 0, 0, np.sin(a / 2)],
                material=f"{name}_mat",
            )

    if layout.props:
        from .props import add_prop

        for name, (kind, color, *_pose) in layout.props.items():
            add_prop(spec, name, kind, color)

    if layout.tools:
        from .toolbox import add_tools

        add_tools(spec, layout.tools)

    if layout.belt or layout.bins:
        from . import line

        if layout.belt:
            line.add_belt(spec)
        for name, (color, *_xy) in layout.bins.items():
            line.add_bin(spec, name, color)

    if not stock_jaw:
        _robot_contacts(spec, calibrate(robot).finger_bodies)
    # jaw meshes stop touching blocks; the pads do the grasping; everything else touches blocks
    for g in spec.geoms if stock_jaw else []:
        if not g.contype:
            continue
        if g.parent.name in ("gripper", "moving_jaw_so101_v1"):
            g.contype, g.conaffinity = MESH_BIT, MESH_BIT
        else:
            g.conaffinity = MESH_BIT | CUBE_BIT
    for body, (pos, quat, size) in _pad_frames().items() if stock_jaw else []:
        b = next(bb for bb in spec.bodies if bb.name == body)
        b.add_geom(
            name=f"pad_{'fixed' if body == 'gripper' else 'moving'}",
            type=mujoco.mjtGeom.mjGEOM_BOX,
            size=size,
            pos=pos,
            quat=quat,
            friction=[PAD_FRICTION, 0.02, 0.002],
            condim=6,
            contype=CUBE_BIT,
            conaffinity=CUBE_BIT,
            solref=GRASP_SOLREF,
            solimp=GRASP_SOLIMP,
            group=3,
        )

    for name, (color, *_pose) in layout.blocks.items():
        spec.add_material(name=f"{name}_mat", rgba=[*BLOCK_COLORS[color], 1.0])
        blk = world.add_body(name=name, pos=[0.2, 0.0, CUBE_HALF])
        blk.add_freejoint(name=f"{name}_free")
        blk.add_geom(
            name=f"{name}_geom",
            type=mujoco.mjtGeom.mjGEOM_BOX,
            size=[CUBE_HALF] * 3,
            mass=CUBE_MASS,
            material=f"{name}_mat",
            friction=[0.6, 0.005, 0.0001],
            condim=4,
            contype=CUBE_BIT,
            conaffinity=CUBE_BIT,
            solref=GRASP_SOLREF,
            solimp=GRASP_SOLIMP,
        )

    world.add_camera(
        name="front",
        pos=FRONT_POS.tolist(),
        xyaxes=look_at(FRONT_POS, FRONT_TARGET).tolist(),
        fovy=52,
    )
    if stock_jaw:
        grip = next(b for b in spec.bodies if b.name == "gripper")
        grip.add_camera(
            name="wrist", pos=[0.0, 0.045, -0.02], xyaxes=[1, 0, 0, 0, 1, 0], fovy=75
        )
    else:
        r = calibrate(robot)
        grip = next(b for b in spec.bodies if b.name == r.tcp_body)
        grip.add_camera(name="wrist", fovy=75, **r.wrist_cam)
    if layout.dressing:
        from ..assets_scene import dress

        dress(spec, layout.dressing)
    if layout.scenery:
        from ..assets_scene import add_scenery

        add_scenery(spec, layout.scenery)
    return spec


def _robot_contacts(spec: mujoco.MjSpec, fingers: list[str]) -> None:
    """A composed robot: every collision geom touches blocks; the fingertips get the grasp contact settings."""
    for g in spec.geoms:
        if not g.contype:
            continue
        g.conaffinity = MESH_BIT | CUBE_BIT
        g.contype = g.contype or MESH_BIT
        if g.parent.name in fingers:
            g.contype |= CUBE_BIT
            g.friction = [PAD_FRICTION, 0.02, 0.002]
            g.condim = 6
            g.solref = GRASP_SOLREF
            g.solimp = GRASP_SOLIMP


def _belt_top(x: float) -> float:
    from .line import BELT_TOP, on_belt

    return BELT_TOP if on_belt(x) else 0.0


class Scene:
    """A compiled layout plus helpers. Reuse one Scene for every episode with the same layout structure."""

    def __init__(self, layout: Layout, robot: str = "so101"):
        from .robots import calibrate, fix_qpos0

        self.layout = layout
        self.robot = calibrate(robot)
        self.model = build_spec(layout, robot=robot).compile()
        fix_qpos0(self.model)
        self.model.light_castshadow[:] = (
            0  # policy cameras: no shadow pass (render cost), same in train and eval
        )
        self.data = mujoco.MjData(self.model)
        m = self.model
        if self.robot.default:
            self.qadr = np.array([m.joint(j).qposadr[0] for j in JOINTS])
            self.dofadr = np.array([m.joint(j).dofadr[0] for j in JOINTS])
        else:  # arm joints; the gripper state is its actuator length (joint or tendon)
            arm = self.robot.arm_joints
            self.qadr = np.array([m.joint(j).qposadr[0] for j in arm])
            self.dofadr = np.array([m.joint(j).dofadr[0] for j in arm])
        self._grip = m.actuator(self.robot.grip_actuator).id
        self._site = m.site(self.robot.tcp_site).id
        self._renderer: mujoco.Renderer | None = None
        self._front_cam = m.camera("front").id
        self._front_pos0 = m.cam_pos[self._front_cam].copy()
        self._key = m.light("key").id
        self._table_mat = m.material("table_mat").id

    # state
    def joint_pos(self) -> np.ndarray:
        if self.robot.default:
            return self.data.qpos[self.qadr].copy()
        return np.r_[self.data.qpos[self.qadr], self.data.actuator_length[self._grip]]

    def to_ctrl(self, action: np.ndarray) -> np.ndarray:
        """Action (arm joint targets + gripper in actuator-length units) -> the model's ctrl vector."""
        if self.robot.default:
            return action
        c = np.array(action, dtype=float)
        c[-1] = c[-1] * self.robot.grip_scale
        return c

    def body_pos(self, name: str) -> np.ndarray:
        return self.data.xpos[self.model.body(name).id].copy()

    def body_yaw(self, name: str) -> float:
        mat = self.data.xmat[self.model.body(name).id].reshape(3, 3)
        return float(np.arctan2(mat[1, 0], mat[0, 0]))

    def ee_pos(self) -> np.ndarray:
        return self.data.site_xpos[self._site].copy()

    def reset(
        self, layout: Layout, look: Look | None = None, arm_q: np.ndarray | None = None
    ) -> None:
        m, d = self.model, self.data
        mujoco.mj_resetData(m, d)
        r = self.robot
        arm_q = r.home if arm_q is None else arm_q
        if r.default:
            q = np.append(arm_q, GRIPPER_CLOSED)
            d.qpos[self.qadr] = q
            d.ctrl[:] = q
        else:
            d.qpos[self.qadr] = arm_q
            d.ctrl[:] = self.to_ctrl(np.append(arm_q, r.g_closed))
        for name, (_c, x, y, yaw) in layout.blocks.items():
            adr = m.joint(f"{name}_free").qposadr[0]
            z = CUBE_HALF + (_belt_top(x) if layout.belt else 0.0)
            d.qpos[adr : adr + 3] = [x, y, z]
            d.qpos[adr + 3 : adr + 7] = [np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)]
        if layout.tools:
            from .toolbox import reset_tools

            reset_tools(self, layout)
        for name, (_c, x, y) in layout.fixtures().items():
            d.mocap_pos[m.body(name).mocapid[0]] = [x, y, 0.0]
        for name, (*_k, yaw) in layout.props.items():
            d.mocap_quat[m.body(name).mocapid[0]] = [
                np.cos(yaw / 2),
                0,
                0,
                np.sin(yaw / 2),
            ]
        self.layout = layout
        self.apply_look(look or Look())
        mujoco.mj_forward(m, d)

    def apply_look(self, look: Look) -> None:
        m = self.model
        m.mat_rgba[self._table_mat, :3] = look.table_rgb
        dvec = np.asarray(look.light_dir, float)
        m.light_dir[self._key] = dvec / np.linalg.norm(dvec)
        m.light_diffuse[self._key] = [look.light_gain] * 3
        m.cam_pos[self._front_cam] = self._front_pos0 + np.asarray(look.cam_jitter)

    # cameras
    def render_obs(self) -> tuple[np.ndarray, np.ndarray]:
        if self._renderer is None:
            self._renderer = mujoco.Renderer(self.model, OBS_SIZE, OBS_SIZE)
        r = self._renderer
        r.update_scene(self.data, camera="front")
        front = r.render().copy()
        r.update_scene(self.data, camera="wrist")
        wrist = r.render().copy()
        return front, wrist

    def close(self) -> None:
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None

    # replay poses for the web renderer
    def pose_frame(self) -> list[float]:
        """World position + quaternion of every body (bodies 1..nbody-1), flattened, rounded."""
        d = self.data
        out = np.concatenate([d.xpos[1:], d.xquat[1:]], axis=1)
        return np.round(out, 5).flatten().tolist()
