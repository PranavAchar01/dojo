"""Renders of the duck with the real LDraw meshes: the instruction photo and the step-by-step exploded sheet."""

from __future__ import annotations

import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from . import duck_bricks as db

BG = (0.93, 0.93, 0.91)


def _font(size: int):
    for p in ("/System/Library/Fonts/Menlo.ttc", "/System/Library/Fonts/Helvetica.ttc"):
        try:
            return ImageFont.truetype(p, size)
        except OSError:
            continue
    return ImageFont.load_default()


def add_brick_visual(spec, body, brick: db.Brick, tag: str, rgba=None) -> None:
    """The real mesh (studs and underside tubes included) as a non-colliding visual geom on `body`."""
    v, f = db.part_mesh_cached(brick.part)
    mname = f"{brick.part}_m"
    if not any(m.name == mname for m in spec.meshes):
        spec.add_mesh(
            name=mname,
            uservert=v.astype(np.float32).ravel(),
            userface=f.astype(np.int32).ravel(),
        )
    body.add_geom(
        name=f"{tag}_vis",
        type=mujoco.mjtGeom.mjGEOM_MESH,
        meshname=mname,
        rgba=list(rgba or brick.rgba),
        contype=0,
        conaffinity=0,
        group=1,
        mass=0,
    )


def add_eyes(body) -> None:
    """The printed eye of the kit's 2x2 head, as two thin discs on each side face (a decal, no physics)."""
    half = db.STUD  # the 2x2 head is 16 mm wide in y, so its side faces sit at y = +-8 mm
    for side in (-1, 1):
        for r, rgba, dy in ((0.0030, (1, 1, 1, 1), 0.00005), (0.0016, (0.02, 0.02, 0.02, 1), 0.0001)):
            body.add_geom(
                type=mujoco.mjtGeom.mjGEOM_CYLINDER,
                size=[r, 0.00005, 0],
                pos=[0.0025, side * (half + dy), 0.0050],
                euler=[90, 0, 0],
                rgba=list(rgba),
                contype=0,
                conaffinity=0,
                mass=0,
            )


def scene(
    poses: dict[str, tuple[np.ndarray, np.ndarray]],
    tint: dict | None = None,
    size: int = 1024,
):
    """A bare studio scene: matte floor, soft light, every brick in `poses` (name -> (pos, quat))."""
    spec = mujoco.MjSpec()
    spec.visual.global_.offwidth = size * 2
    spec.visual.global_.offheight = size * 2
    spec.visual.quality.shadowsize = 4096
    spec.visual.headlight.ambient = [0.45, 0.45, 0.45]
    spec.visual.headlight.diffuse = [0.35, 0.35, 0.35]
    spec.visual.rgba.haze = [*BG, 1]
    w = spec.worldbody
    w.add_light(
        pos=[0.12, -0.25, 0.35],
        dir=[-0.15, 0.45, -1],
        diffuse=[0.75] * 3,
        castshadow=True,
    )
    w.add_geom(
        type=mujoco.mjtGeom.mjGEOM_PLANE,
        size=[3, 3, 0.01],
        rgba=[*BG, 1],
        pos=[0, 0, 0],
    )
    bm = db.brick_map()
    for name, (pos, quat) in poses.items():
        b = w.add_body(name=name, pos=list(pos), quat=list(quat))
        add_brick_visual(spec, b, bm[name], name, (tint or {}).get(name))
        if name == "head":
            add_eyes(b)
    return spec.compile()


def camera(
    az: float = -35, el: float = 28, dist: float = 0.16, lookat=None
) -> mujoco.MjvCamera:
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.azimuth, cam.elevation, cam.distance = az, -el, dist
    cam.lookat[:] = db.CENTRE if lookat is None else lookat
    return cam


def render(model, cam, w: int, h: int) -> np.ndarray:
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    r = mujoco.Renderer(model, height=h, width=w)
    try:
        r.update_scene(data, camera=cam)
        r.scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = 1
        return r.render().copy()
    finally:
        r.close()


def target_poses(upto: int | None = None, lift: dict | None = None) -> dict:
    out = {}
    for b in db.DUCK[:upto]:
        p = b.target_pos().copy()
        if lift and b.name in lift:
            p[2] += lift[b.name]
        out[b.name] = (p, b.target_quat())
    return out


def photo(path, size: int = 1024) -> None:
    """The finished duck as a clean instruction photo (the image Muse Spark sees)."""
    img = render(
        scene(target_poses(), size=size), camera(az=65, el=24, dist=0.11), size, size
    )
    Image.fromarray(img).save(path)


def exploded(path, panel: int = 640) -> None:
    """Six panels, one per build step: the placed bricks in colour, the new brick lifted above its slot and tinted."""
    labels = {
        "foot": "Red 2x3 plate flat on the table",
        "leg": "2x2 on the front studs of the plate",
        "body": "2x4 across the leg, rear hangs out",
        "beak": "Red plate, one stud past the body",
        "head": "2x2 head (eye) on the beak plate",
        "tail": "1x2 flat on the rear studs",
    }
    sheet = Image.new(
        "RGB", (panel * 3, int(panel * 2 + 40)), tuple(int(c * 255) for c in BG)
    )
    d = ImageDraw.Draw(sheet)
    f, f2 = _font(26), _font(20)
    for i, b in enumerate(db.DUCK):
        lift = {b.name: 0.022}
        poses = target_poses(i + 1, lift)
        img = render(
            scene(poses, size=panel), camera(az=65, el=26, dist=0.145), panel, panel
        )
        x, y = (i % 3) * panel, (i // 3) * (panel + 20)
        sheet.paste(Image.fromarray(img), (x, y))
        d.text(
            (x + 18, y + 14),
            f"{i + 1}  {db.PART_NAME[b.part]}",
            fill=(30, 30, 30),
            font=f,
        )
        d.text(
            (x + 18, y + 48),
            f"LDraw {b.part}   {labels[b.name]}",
            fill=(70, 70, 70),
            font=f2,
        )
    sheet.save(path)
