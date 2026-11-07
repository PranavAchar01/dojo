"""Table fixtures for the harder skills, built procedurally from boxes (no downloaded meshes).

* tray: a 2 x 2 kitting tray. Thin base plate, a cross of dividers and a rim; four square slots, 5.5 cm inside,
  centres 3.05 cm from the tray centre. Dividers stand 1.6 cm above the table.
* socket: a square recess in a raised plate. The plate is 2 cm tall, the hole is 3.7 cm square (3.5 mm clearance
  per side around a 3 cm cube) and the table is the hole's floor. The last 8 mm of the hole's wall are a 45 degree
  lead-in funnel made of tilted thin boxes. The body frame sits at the hole centre and its yaw is the hole's yaw
  (the "diamond" socket is the same fixture turned 45 degrees).
* marker: no geometry, a mocap point used as the slot goal of a tray (kept out of the browser export).

Every prop is a mocap body, so a layout can move and turn it. `prop_parts` is the single source for the
physics geoms and for the browser export (viz.py), so what is drawn is what collides.
"""

from __future__ import annotations

import math

import mujoco
import numpy as np

from .scene import CUBE_HALF

PROP_COLORS = {
    "gray": (0.50, 0.52, 0.56),
    "white": (0.90, 0.89, 0.85),
    "blue": (0.22, 0.38, 0.74),
    "green": (0.26, 0.56, 0.38),
    "tan": (0.78, 0.66, 0.46),
    "black": (0.16, 0.16, 0.18),
}
PROP_KINDS = ("tray", "socket", "marker")

# tray
SLOT_INNER = 0.055
SLOT_PITCH = 0.061  # centre to centre
TRAY_HALF = 0.064
TRAY_BASE = 0.004  # top of the base plate
DIVIDER_TOP = 0.016
DIVIDER_T = 0.003  # half thickness

# socket
SOCKET_CLEAR = 0.0035
SOCKET_HOLE = CUBE_HALF + SOCKET_CLEAR  # half width of the hole
SOCKET_HALF = 0.04
SOCKET_H = 0.02
SOCKET_CHAMFER = 0.008
SOCKET_STEP = SOCKET_H - SOCKET_CHAMFER


def _qz(a: float) -> np.ndarray:
    return np.array([math.cos(a / 2), 0.0, 0.0, math.sin(a / 2)])


def _qy(a: float) -> np.ndarray:
    return np.array([math.cos(a / 2), 0.0, math.sin(a / 2), 0.0])


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


def _box(size, pos, quat=(1, 0, 0, 0), shade=1.0) -> dict:
    return {
        "type": "box",
        "size": [round(float(v), 6) for v in size],
        "pos": [round(float(v), 6) for v in pos],
        "quat": [round(float(v), 6) for v in quat],
        "shade": shade,
    }


def _tray_parts() -> list[dict]:
    t, h = TRAY_HALF, TRAY_HALF
    mid = (TRAY_BASE + DIVIDER_TOP) / 2
    hz = (DIVIDER_TOP - TRAY_BASE) / 2
    parts = [_box([t, h, TRAY_BASE / 2], [0, 0, TRAY_BASE / 2], shade=0.82)]
    parts.append(_box([t, DIVIDER_T, hz], [0, 0, mid]))
    parts.append(_box([DIVIDER_T, h, hz], [0, 0, mid]))
    rim = t - DIVIDER_T
    for s in (1, -1):
        parts.append(_box([t, DIVIDER_T, hz], [0, s * rim, mid]))
        parts.append(_box([DIVIDER_T, h, hz], [s * rim, 0, mid]))
    return parts


def _socket_parts() -> list[dict]:
    S, H, Z, z0, c = SOCKET_HALF, SOCKET_HOLE, SOCKET_H, SOCKET_STEP, SOCKET_CHAMFER
    parts = []
    # full-height plate around a hole that is c wider on every side, then a lower lip down to the true hole
    for inner, top in ((H + c, Z), (H, z0)):
        for s in (1, -1):
            w = (S - inner) / 2
            parts.append(_box([S, w, top / 2], [0, s * (inner + w), top / 2]))
            parts.append(_box([w, inner, top / 2], [s * (inner + w), 0, top / 2]))
    # lead-in: a thin box tilted 45 degrees on each side, from the lip's inner edge up to the plate top
    half_len = c * math.sqrt(2) / 2
    for k in range(4):
        phi = k * math.pi / 2
        centre = np.array([H + c / 2, 0.0, (z0 + Z) / 2])
        pos = [
            centre[0] * math.cos(phi) - centre[1] * math.sin(phi),
            centre[0] * math.sin(phi) + centre[1] * math.cos(phi),
            centre[2],
        ]
        quat = _qmul(_qz(phi), _qy(-math.pi / 4))
        parts.append(_box([half_len, H + c, 0.0012], pos, quat, shade=0.9))
    return parts


def prop_parts(kind: str) -> list[dict]:
    """The geoms of a prop in its body frame (MuJoCo half sizes; all of them collide)."""
    if kind == "tray":
        return _tray_parts()
    if kind == "socket":
        return _socket_parts()
    if kind == "marker":
        return []
    raise ValueError(f"unknown prop kind {kind}")


def prop_rgb(color: str) -> tuple[float, float, float]:
    return PROP_COLORS[color]


def add_prop(spec: mujoco.MjSpec, name: str, kind: str, color: str) -> None:
    body = spec.worldbody.add_body(name=name, mocap=True, pos=[0.25, 0, 0])
    rgb = np.array(prop_rgb(color))
    for i, p in enumerate(prop_parts(kind)):
        body.add_geom(
            name=f"{name}_p{i}",
            type=mujoco.mjtGeom.mjGEOM_BOX,
            size=p["size"],
            pos=p["pos"],
            quat=p["quat"],
            rgba=[*np.clip(rgb * p["shade"], 0, 1), 1.0],
            friction=[0.6, 0.005, 0.0001],
        )


def slot_offset(slot: str) -> tuple[float, float]:
    """Tray-frame offset of a named slot. Front is toward the camera (+x), left is the viewer's left (-y)."""
    fb, lr = slot.split("-")
    return (
        (SLOT_PITCH / 2) * (1 if fb == "front" else -1),
        (SLOT_PITCH / 2) * (-1 if lr == "left" else 1),
    )


KIT_SLOTS = ("front-left", "front-right", "back-left", "back-right")
