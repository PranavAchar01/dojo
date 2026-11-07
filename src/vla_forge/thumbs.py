"""Still images of the robot builder's cards: one per arm, per gripper, and per wrist or jaw option on each arm.

Rendered once with MuJoCo's offscreen renderer from the same source models the live preview uses, framed like the
live preview, on a transparent background, and kept as webp files under assets/thumbs/. The cards show the image;
nothing assembles in the browser until an arm is picked. `python -m vla_forge.thumbs` fills in whatever is missing
(`--force` redraws everything), then warms the per-configuration cache in robotcache.
"""

from __future__ import annotations

import math
import re
import sys
from pathlib import Path

import mujoco
import numpy as np
from PIL import Image

from . import robotcache
from .sim import robots as R
from .viz import VISUAL_GROUP, part_posed, robot_posed, visual_bbox

DIR = R.REPO / "assets" / "thumbs"
W, H = 480, 320
FOVY = 30.0


def thumb_name(slot: str, part: str, arm: str | None = None) -> str:
    on_arm = slot == "wrist" or (slot == "effector" and part == "native")
    return f"{slot}-{part}-on-{arm}.webp" if on_arm else f"{slot}-{part}.webp"


def wanted() -> list[tuple[str, str, str | None]]:
    out: list[tuple[str, str, str | None]] = [("arm", a, None) for a in R.ARMS]
    out += [("effector", e, None) for e, v in R.EFFECTORS.items() if v is not None]
    for a in R.ARMS:
        out += [("wrist", w, a) for w in R.WRISTS]
        out.append(("effector", "native", a))
    return out


# the robots are drawn black and white: the same roles and colours as web/studio.js robotMaterial, so a card
# picture matches the library and the live preview
SHELL, SERVO, STEEL = (
    (0.945, 0.949, 0.953, 1),
    (0.078, 0.082, 0.094, 1),
    (0.44, 0.455, 0.48, 1),
)
_SERVO_NAME = re.compile(
    r"motor|sts3215|horn|prop|gripper_bar|finger|_7_gripper|_6_wrist|_2_shoulder|hand_|robotiq.*(base|driver|coupler|follower|spring)",
    re.IGNORECASE,
)
_NOT_SERVO = re.compile(r"moving_jaw|Moving_Jaw|gripper_moving_finger|static_finger$")


def _recolour(m: mujoco.MjModel) -> None:
    for g in range(m.ngeom):
        if (
            m.geom_group[g] != VISUAL_GROUP
            or m.geom_type[g] != mujoco.mjtGeom.mjGEOM_MESH
        ):
            continue
        name = (
            mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_MESH, int(m.geom_dataid[g])) or ""
        )
        mat = m.geom_matid[g]
        rgba = m.mat_rgba[mat] if mat >= 0 else m.geom_rgba[g]
        lum = 0.299 * rgba[0] + 0.587 * rgba[1] + 0.114 * rgba[2]
        if _SERVO_NAME.search(name) and not _NOT_SERVO.search(name):
            colour = SERVO
        else:
            colour = SHELL if lum > 0.62 else SERVO if lum < 0.3 else STEEL
        m.geom_rgba[g] = colour
        m.geom_matid[g] = -1


def _on_arm(slot: str, part: str, arm: str) -> str:
    return R.make_id(arm, part if slot == "wrist" else "native")


def _render(
    m: mujoco.MjModel, d: mujoco.MjData, box, direction, pad: float
) -> Image.Image:
    _recolour(m)
    lo, hi = np.asarray(box[0]), np.asarray(box[1])
    centre = (lo + hi) / 2
    size = max(float(np.linalg.norm(hi - lo)), 0.05)
    dirv = np.asarray(direction, float)
    dirv /= np.linalg.norm(dirv)
    for g in range(
        m.ngeom
    ):  # the live preview draws visual meshes only: no floors, no helper shapes
        if (
            m.geom_group[g] != VISUAL_GROUP
            or m.geom_type[g] != mujoco.mjtGeom.mjGEOM_MESH
        ):
            m.geom_rgba[g, 3] = 0
            m.geom_matid[g] = -1
    m.vis.global_.offwidth, m.vis.global_.offheight = 2 * W, 2 * H
    m.vis.global_.fovy = FOVY
    m.vis.quality.offsamples = 4
    m.vis.headlight.ambient[:] = 0.45
    m.vis.headlight.diffuse[:] = 0.65
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.lookat[:] = centre
    cam.distance = pad * size / (2 * math.tan(math.radians(FOVY) / 2))
    cam.azimuth = math.degrees(math.atan2(-dirv[1], -dirv[0]))
    cam.elevation = -math.degrees(math.asin(dirv[2]))
    opt = mujoco.MjvOption()
    opt.geomgroup[:] = 0
    opt.geomgroup[VISUAL_GROUP] = 1
    opt.sitegroup[:] = 0
    with mujoco.Renderer(m, 2 * H, 2 * W) as r:
        r.update_scene(d, camera=cam, scene_option=opt)
        rgb = r.render().copy()
        r.enable_segmentation_rendering()
        seg = r.render()
    alpha = ((seg[..., 0] >= 0) * 255).astype(np.uint8)
    img = Image.fromarray(np.dstack([rgb, alpha]), "RGBA")
    return img.resize((W, H), Image.LANCZOS)


def render(slot: str, part: str, arm: str | None = None) -> Image.Image:
    if arm is None:
        m, d = part_posed(slot, part)
        lo, hi = visual_bbox(m, d)
        return _render(m, d, (lo, hi), (1.1, -0.9, 0.7), 0.8)
    try:
        m, d, r = robot_posed(_on_arm(slot, part, arm))
    except Exception:  # noqa: BLE001 - an arm that cannot take this option shows it on the stock SO-101, like the live preview
        m, d, r = robot_posed(_on_arm(slot, part, "so101"))
    lo, hi = visual_bbox(m, d)
    h = 0.19 * float(
        np.linalg.norm(hi - lo)
    )  # a box around the tool point big enough for the wrist and the jaw
    c = d.site_xpos[m.site(r.tcp_site).id].copy()
    c[2] += 0.5 * h
    return _render(m, d, (c - h, c + h), (0.7, -1.0, 0.5), 1.45)


def save(slot: str, part: str, arm: str | None = None) -> Path:
    DIR.mkdir(parents=True, exist_ok=True)
    f = DIR / thumb_name(slot, part, arm)
    render(slot, part, arm).save(f, "WEBP", quality=88, method=6)
    return f


def missing() -> list[tuple[str, str, str | None]]:
    return [w for w in wanted() if not (DIR / thumb_name(*w)).exists()]


def warm_needed() -> bool:
    return bool(missing()) or any(
        not robotcache.warmed(c) for c in robotcache.configs()
    )


def main(argv: list[str] | None = None) -> None:
    force = "--force" in (argv if argv is not None else sys.argv[1:])
    for item in wanted() if force else missing():
        save(*item)
    for rid in robotcache.configs():
        robotcache.warm(rid)


if __name__ == "__main__":
    main()
