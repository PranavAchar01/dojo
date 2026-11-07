"""MuJoCo table for the bridge: two SO-101 arms, the loose bricks, and the snap-on-contact clutch.

Clutch model (honest version, also in docs/hero-demo.md): a real stud connection holds a brick with a few newtons of
clamp. We do not simulate stud interference. Instead each brick has a weld equality to the world at its target pose
that stays off until the brick is released near its slot. The weld switches on only if, at that moment, the brick is
within the success tolerance (2 mm, 5 deg) of the target pose and its underside is at seating height; it never pulls
a brick that is further off. Once on, the weld holds the brick for the rest of the episode (the next brick sits on
it, as it would on real studs).
"""

from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np

from . import hero_bricks as db
from . import hero_render as dr
from .sim import scene as S

TOL_POS = 0.002  # m, success tolerance
TOL_YAW = np.radians(5.0)
SEAT_Z = 0.0015  # underside may be this far above the seated height and still engages the studs
ARMS = {"a_": ([0.20, -0.26, 0.0], 90.0), "b_": ([0.20, 0.26, 0.0], -90.0)}
JAW_BODIES = ("gripper", "moving_jaw_so101_v1")

# loose-brick slots on the table, (x, y, yaw deg), near the arm that serves them; bricks take them in build order
SLOTS = (
    (0.145, -0.115, 20.0),
    (0.255, -0.105, -30.0),
    (0.115, -0.060, 70.0),
    (0.230, 0.115, -15.0),
    (0.150, 0.105, 40.0),
    (0.290, 0.110, 70.0),
    (0.300, -0.060, 10.0),
    (0.100, 0.050, -20.0),
    (0.200, -0.140, 0.0),
    (0.200, 0.140, 0.0),
)


def loose() -> dict[str, tuple[float, float, float]]:
    return {b.name: SLOTS[i % len(SLOTS)] for i, b in enumerate(db.DUCK)}


PAD_X = (-0.006, 0.004)  # pad span along the jaw about the gripper site: low on the jaw tip, sized for 9.6 mm bricks


def pad_frames() -> dict:
    """Pads for the jaw tips. The stock pads (sim/scene.py) sit 2 to 42 mm up the jaw for 30 mm cubes; a plain brick is
    only 9.6 mm tall, so the same measurement is repeated with a pad span at the tip."""
    keep = S.PAD_X
    S._pad_frames.cache_clear()
    S.PAD_X = PAD_X
    try:
        return S._pad_frames()
    finally:
        S.PAD_X = keep
        S._pad_frames.cache_clear()


def mass(b: db.Brick) -> float:
    """About 0.3 g per stud of a 3-plate brick (a 2x2 brick weighs 1.2 g)."""
    nx, ny = db.PART_FOOTPRINT[b.part]
    return 0.0003 * nx * ny * db.PART_PLATES[b.part] / 3


def build_xy() -> np.ndarray:
    """Table xy of the assembly's low-x, low-y stud corner, set so the whole bridge is centred on (0.20, 0)."""
    c = db.centre()
    return np.array([0.20 - c[0], -c[1]])


def target_pose(b: db.Brick) -> tuple[np.ndarray, np.ndarray]:
    """World pose of the seated brick: the bridge sits at build_xy() on the table top."""
    return b.target_pos() + np.array([*build_xy(), 0.0]), b.target_quat()


def build_model(size: int = 960) -> mujoco.MjModel:
    spec = mujoco.MjSpec()
    spec.option.timestep = S.TIMESTEP
    spec.option.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
    spec.visual.global_.offwidth = size * 2
    spec.visual.global_.offheight = size * 2
    spec.visual.quality.shadowsize = 4096
    for pre, (pos, yaw) in ARMS.items():
        spec.attach(
            S._arm_spec(),
            prefix=pre,
            frame=spec.worldbody.add_frame(pos=pos, euler=[0, 0, yaw]),
        )
    w = spec.worldbody
    spec.add_texture(
        name="sky",
        type=mujoco.mjtTexture.mjTEXTURE_SKYBOX,
        builtin=mujoco.mjtBuiltin.mjBUILTIN_GRADIENT,
        rgb1=[0.86, 0.88, 0.92],
        rgb2=[0.62, 0.66, 0.74],
        width=64,
        height=64,
    )
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
    spec.add_material(name="table_mat", rgba=[0.84, 0.80, 0.72, 1.0])
    w.add_light(
        name="key",
        pos=[0.4, -0.5, 1.4],
        dir=[-0.2, 0.4, -1.0],
        diffuse=[0.8] * 3,
        castshadow=True,
    )
    w.add_light(
        name="fill",
        pos=[-0.3, 0.5, 1.2],
        dir=[0.3, -0.4, -1.0],
        diffuse=[0.35] * 3,
        castshadow=False,
    )
    w.add_geom(
        name="floor",
        type=mujoco.mjtGeom.mjGEOM_PLANE,
        size=[3, 3, 0.1],
        pos=[0, 0, -0.75],
        material="floor_mat",
    )
    w.add_geom(
        name="table",
        type=mujoco.mjtGeom.mjGEOM_BOX,
        size=[0.45, 0.6, 0.02],
        pos=[0.2, 0.0, -0.02],
        material="table_mat",
        friction=[0.8, 0.005, 0.0001],
    )
    for g in spec.geoms:  # the vendor arm prints yellow like the bricks: tint it slate so the bricks read
        if g.type == mujoco.mjtGeom.mjGEOM_MESH and g.parent.name.split("_", 1)[0] in ("a", "b"):
            g.material = ""
            g.rgba = [0.30, 0.34, 0.42, 1.0]
    # collision bits as in sim/scene.py: jaw meshes do not touch bricks, the pads do, everything else does
    for g in spec.geoms:
        if not g.contype:
            continue
        if g.parent.name.split("_", 1)[-1] in JAW_BODIES or g.parent.name in JAW_BODIES:
            g.contype, g.conaffinity = S.MESH_BIT, S.MESH_BIT
        else:
            g.conaffinity = S.MESH_BIT | S.CUBE_BIT
    for pre in ARMS:
        for body, (pos, quat, psize) in pad_frames().items():
            b = next(bb for bb in spec.bodies if bb.name == pre + body)
            b.add_geom(
                name=f"{pre}pad_{'fixed' if body == 'gripper' else 'moving'}",
                type=mujoco.mjtGeom.mjGEOM_BOX,
                size=psize,
                pos=pos,
                quat=quat,
                friction=[S.PAD_FRICTION, 0.02, 0.002],
                condim=6,
                contype=S.CUBE_BIT,
                conaffinity=S.CUBE_BIT,
                solref=S.GRASP_SOLREF,
                solimp=S.GRASP_SOLIMP,
                group=3,
            )
    for b in db.DUCK:
        body = w.add_body(name=b.name, pos=[0.2, 0, 0.02])
        body.add_freejoint(name=f"{b.name}_free")
        nx, ny = db.PART_FOOTPRINT[b.part]
        h = db.PART_PLATES[b.part] * db.PLATE
        body.add_geom(
            name=f"{b.name}_box",
            type=mujoco.mjtGeom.mjGEOM_BOX,
            size=[nx * db.STUD / 2 - 1e-4, ny * db.STUD / 2 - 1e-4, h / 2],
            pos=[0, 0, h / 2],
            mass=mass(b),
            friction=[0.7, 0.005, 0.0001],
            condim=4,
            contype=S.CUBE_BIT,
            conaffinity=S.CUBE_BIT,
            solref=S.GRASP_SOLREF,
            solimp=S.GRASP_SOLIMP,
            rgba=[0, 0, 0, 0],
            group=3,
        )
        dr.add_brick_visual(spec, body, b, b.name)
        pos, quat = target_pose(b)
        eq = spec.add_equality()
        eq.type = mujoco.mjtEq.mjEQ_WELD
        eq.objtype = mujoco.mjtObj.mjOBJ_BODY
        eq.name1, eq.name2 = "world", b.name
        eq.name = f"snap_{b.name}"
        eq.active = False
        eq.data[:11] = [0, 0, 0, *pos, *quat, 1.0]
    for name, pos, target, fov in (
        ("overview", [0.85, 0.0, 0.40], [0.20, 0.0, 0.03], 42),
        ("closeup", [0.55, -0.02, 0.17], [0.20, -0.01, 0.01], 40),
    ):
        w.add_camera(
            name=name, pos=pos, xyaxes=S.look_at(pos, target).tolist(), fovy=fov
        )
    return spec.compile()


@dataclass
class Table:
    m: mujoco.MjModel
    d: mujoco.MjData

    @staticmethod
    def new(size: int = 960) -> Table:
        m = build_model(size)
        return Table(m, mujoco.MjData(m))

    def arm_q(self, pre: str) -> np.ndarray:
        return np.array(
            [self.d.qpos[self.m.joint(pre + j).qposadr[0]] for j in S.JOINTS]
        )

    def reset(
        self, rng: np.random.Generator | None = None, jitter: float = 0.0
    ) -> None:
        m, d = self.m, self.d
        mujoco.mj_resetData(m, d)
        for pre in ARMS:
            q = np.append(S.HOME, S.GRIPPER_CLOSED)
            for j, v in zip(S.JOINTS, q):
                d.qpos[m.joint(pre + j).qposadr[0]] = v
                d.ctrl[m.actuator(pre + j).id] = v
        for name, (x, y, yaw) in loose().items():
            if rng is not None:
                x, y = (
                    x + rng.uniform(-jitter, jitter),
                    y + rng.uniform(-jitter, jitter),
                )
                yaw = yaw + rng.uniform(-30, 30) * (jitter > 0)
            adr = m.joint(f"{name}_free").qposadr[0]
            h = np.radians(yaw) / 2
            d.qpos[adr : adr + 7] = [x, y, 0.0003, np.cos(h), 0, 0, np.sin(h)]
        mujoco.mj_forward(m, d)

    # --- pose errors and the clutch -------------------------------------------------------------------------
    def pose(self, name: str) -> tuple[np.ndarray, float]:
        bid = self.m.body(name).id
        mat = self.d.xmat[bid].reshape(3, 3)
        return self.d.xpos[bid].copy(), float(np.arctan2(mat[1, 0], mat[0, 0]))

    def error(self, name: str) -> tuple[float, float, float]:
        """(position error m, yaw error rad modulo the brick's 180 degree symmetry, underside height above seat m)."""
        b = db.brick_map()[name]
        tp, tq = target_pose(b)
        p, yaw = self.pose(name)
        ty = 2 * np.arctan2(tq[3], tq[0])
        dyaw = (
            yaw - ty + np.pi / 2
        ) % np.pi - np.pi / 2  # 2x4, 2x3, 2x2 and 1x2 bricks are 180 degree symmetric
        if db.PART_FOOTPRINT[b.part][0] == db.PART_FOOTPRINT[b.part][1]:
            dyaw = (yaw - ty + np.pi / 4) % (np.pi / 2) - np.pi / 4
        return float(np.linalg.norm(p - tp)), float(abs(dyaw)), float(p[2] - tp[2])

    def seated(self, name: str) -> bool:
        pos_err, yaw_err, dz = self.error(name)
        return pos_err <= TOL_POS and yaw_err <= TOL_YAW and dz <= SEAT_Z

    def try_snap(self, name: str) -> bool:
        """Switch the weld on if the brick is within tolerance and seated. Returns whether it is (now) snapped."""
        eq = self.m.equality(f"snap_{name}").id
        if self.d.eq_active[eq]:
            return True
        if self.seated(name):
            self.d.eq_active[eq] = 1
            return True
        return False

    def snapped(self, name: str) -> bool:
        return bool(self.d.eq_active[self.m.equality(f"snap_{name}").id])

    def set_ctrl(self, pre: str, q: np.ndarray) -> None:
        for j, v in zip(S.JOINTS, q):
            self.d.ctrl[self.m.actuator(pre + j).id] = v

    def step(self, n: int = 1) -> None:
        for _ in range(n):
            mujoco.mj_step(self.m, self.d)

    def render(self, cam: str, w: int, h: int) -> np.ndarray:
        r = mujoco.Renderer(self.m, height=h, width=w)
        try:
            r.update_scene(self.d, camera=cam)
            return r.render().copy()
        finally:
            r.close()
