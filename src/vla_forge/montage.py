"""Renders the optimizer's montage: one small real MuJoCo frame per scene per moment, packed into sprite sheets.

A recorded rollout is a list of body poses (rollout.py). Here each pose is put back into the compiled scene,
the arm's joint angles are recovered from the bodies' relative rotations, and the scene is rendered from a fixed
three-quarter camera. One sheet holds all scenes at one moment (10 x 10 tiles), so the page can play every scene
at once by flipping through the sheets.
"""

from __future__ import annotations

from pathlib import Path

import mujoco
import numpy as np
from PIL import Image

from .sim.display import restyle
from .sim.robots import calibrate, fix_qpos0
from .sim.scene import JOINTS, Layout, build_spec

TILE_W, TILE_H = 96, 72
COLS = 10
MOMENTS = 8  # frames per scene in the grid
HERO_W, HERO_H = 320, 240
HERO_FRAMES = 16


def _qmul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    w1, x1, y1, z1 = a
    w2, x2, y2, z2 = b
    return np.array(
        [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ]
    )


def _conj(q: np.ndarray) -> np.ndarray:
    return np.array([q[0], -q[1], -q[2], -q[3]])


class Stage:
    """Compiled scenes by structure; a pose frame goes in, an RGB image comes out."""

    def __init__(self, robot: str = "so101"):
        self.robot = robot
        self.models: dict[tuple, tuple] = {}

    def _model(self, lay: Layout):
        key = lay.key()
        if key not in self.models:
            m = build_spec(lay, robot=self.robot).compile()
            fix_qpos0(m)
            restyle(m)
            self.models[key] = (m, mujoco.MjData(m), {})
        return self.models[key]

    def render(
        self,
        layout_json: dict,
        frames: np.ndarray,
        idx: list[int],
        w: int,
        h: int,
        distance: float = 0.46,
    ) -> list[np.ndarray]:
        lay = Layout.from_json(layout_json)
        m, d, rs = self._model(lay)
        if (w, h) not in rs:
            rs[(w, h)] = mujoco.Renderer(m, h, w)
        rend = rs[(w, h)]
        rb = calibrate(self.robot)
        joints = JOINTS if rb.default else rb.arm_joints
        for name, (_c, x, y) in lay.fixtures().items():
            d.mocap_pos[m.body(name).mocapid[0]] = [x, y, 0.0]
        cam = mujoco.MjvCamera()
        cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        cam.lookat[:] = [0.15, 0.0, 0.07]
        cam.distance = distance
        cam.azimuth = 148.0
        cam.elevation = -34.0
        out = []
        for i in idx:
            f = np.asarray(frames[i], float).reshape(-1, 7)
            for j in joints:
                jid = m.joint(j).id
                b = int(m.jnt_bodyid[jid])
                p = int(m.body_parentid[b])
                qp = f[p - 1, 3:] if p > 0 else np.array([1.0, 0, 0, 0])
                qc = f[b - 1, 3:]
                rel = _qmul(_conj(m.body_quat[b]), _qmul(_conj(qp), qc))
                axis = m.jnt_axis[jid]
                ang = 2.0 * np.arctan2(float(rel[1:] @ axis), float(rel[0]))
                ang = (ang + np.pi) % (2 * np.pi) - np.pi
                d.qpos[m.jnt_qposadr[jid]] = ang
            for name in lay.blocks:
                bid = m.body(name).id
                adr = m.joint(f"{name}_free").qposadr[0]
                d.qpos[adr : adr + 7] = f[bid - 1]
            mujoco.mj_forward(m, d)
            rend.update_scene(d, camera=cam)
            out.append(rend.render().copy())
        return out

    def close(self) -> None:
        for _m, _d, rs in self.models.values():
            for r in rs.values():
                r.close()
        self.models.clear()


def moment_index(n_frames: int, k: int) -> list[int]:
    return [round(v) for v in np.linspace(0, n_frames - 1, k)]


def write_sheets(
    stage: Stage, scenes: list[dict], out_dir: Path, moments: int = MOMENTS
) -> list[str]:
    """scenes: [{layout, frames (np [T, nb*7]), success}]. One JPEG per moment, tiles in scene order."""
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = (len(scenes) + COLS - 1) // COLS
    sheets = [
        Image.new("RGB", (COLS * TILE_W, rows * TILE_H), (240, 242, 245))
        for _ in range(moments)
    ]
    for si, sc in enumerate(scenes):
        imgs = stage.render(
            sc["layout"],
            sc["frames"],
            moment_index(len(sc["frames"]), moments),
            TILE_W,
            TILE_H,
        )
        for k, im in enumerate(imgs):
            sheets[k].paste(
                Image.fromarray(im), ((si % COLS) * TILE_W, (si // COLS) * TILE_H)
            )
    names = []
    for k, sh in enumerate(sheets):
        name = f"m{k}.jpg"
        sh.save(out_dir / name, quality=72)
        names.append(name)
    return names


def write_hero(stage: Stage, scene: dict, out: Path) -> None:
    """One scene at higher resolution as a horizontal filmstrip (HERO_FRAMES frames)."""
    imgs = stage.render(
        scene["layout"],
        scene["frames"],
        moment_index(len(scene["frames"]), HERO_FRAMES),
        HERO_W,
        HERO_H,
        distance=0.62,
    )
    strip = Image.new("RGB", (HERO_W * len(imgs), HERO_H))
    for i, im in enumerate(imgs):
        strip.paste(Image.fromarray(im), (i * HERO_W, 0))
    out.parent.mkdir(parents=True, exist_ok=True)
    strip.save(out, quality=78)
