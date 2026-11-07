"""The scripted build: two SO-101s on the table pick every loose brick in build order and seat it with `hero_expert`.

Nothing here is learned. Each brick goes to the arm that stands on its side of the table; after the last brick the
build is judged by `hero_rules.check` on the bricks that really ended up snapped, and the whole run is recorded as
poses for the library player (`hero_replay`).
"""

from __future__ import annotations

import time

import mujoco
import numpy as np

from . import hero_bricks as hb
from . import hero_expert as he
from . import hero_rules as hr
from . import hero_sim as ds

KEEP_EVERY = 3  # control ticks per recorded frame (50 Hz control -> ~16.7 Hz)
FPS = 50 / KEEP_EVERY
ATTEMPTS = 3  # pick-and-place tries per brick; a try that leaves the brick unseated picks it up again


def arm_for(name: str) -> str:
    return "a_" if ds.loose()[name][1] < 0 else "b_"


def thumb_view(t: ds.Table, w: int = 960, h: int = 720) -> np.ndarray:
    """The card's picture: both arms and the bridge, seen from the front and above."""
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.lookat[:] = [0.2, 0.0, 0.03]
    cam.distance, cam.azimuth, cam.elevation = 0.62, 180.0, -32.0
    r = mujoco.Renderer(t.m, height=h, width=w)
    try:
        r.update_scene(t.d, camera=cam)
        return r.render().copy()
    finally:
        r.close()


def pose_frame(t: ds.Table) -> list[float]:
    d = t.d
    a = np.concatenate([d.xpos[1:], d.xquat[1:]], axis=1)
    return np.round(a, 4).ravel().tolist()


def _finite(x):
    """JSON has no NaN or inf: a failed pick leaves them in the error fields, so they become null."""
    if isinstance(x, float):
        return x if np.isfinite(x) else None
    if isinstance(x, dict):
        return {k: _finite(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_finite(v) for v in x]
    return x


def run(
    seed: int = 0,
    jitter: float = 0.02,
    record: bool = True,
    bricks=None,
    rules: hb.Rules | None = None,
    trace=None,
) -> dict:
    """Build the bridge once. Returns per-brick results, the rule results, and (if asked) the recorded frames."""
    if bricks is not None:
        hb.set_assembly(bricks)
    order = [b.name for b in hb.DUCK]
    t = ds.Table.new(960)
    t.reset(np.random.default_rng(seed), jitter=jitter)
    t.step(30)
    frames: list[list[float]] = []
    marks: list[dict] = []
    tick = [0]
    cur = [""]
    bm = hb.brick_map()

    def on_tick(_t):
        tick[0] += 1
        if record and tick[0] % KEEP_EVERY == 0:
            frames.append(pose_frame(t))

    arms = {p: he.Expert(t, p, on_tick) for p in ds.ARMS}
    for a in arms.values():
        a.trace = trace
    if record:
        frames.append(pose_frame(t))
    t0 = time.time()
    per: list[dict] = []
    for name in order:
        e = arms[arm_for(name)]
        ok, err = False, ""
        e.pre_snap = (float("nan"),) * 3
        cur[0] = name
        marks.append({"brick": name, "frame": len(frames)})
        lifted, tries = False, 0
        for tries in range(1, ATTEMPTS + 1):
            try:
                e.pick(name)
                lifted = bool(t.pose(name)[0][2] > 0.02)
                if not lifted:
                    e.home()
                    continue
                ok = bool(e.place(name))
                e.home()
            except (RuntimeError, ValueError, mujoco.FatalError) as ex:
                err = repr(ex)[:120]
                try:
                    e.home()
                except (RuntimeError, ValueError, mujoco.FatalError):
                    pass
            if ok:
                break
        pe, ye, dz = e.pre_snap
        per.append(
            {
                "brick": name,
                "arm": e.pre,
                "lifted": lifted,
                "seated": ok,
                "attempts": tries,
                "pos_err_mm": round(pe * 1e3, 2),
                "yaw_err_deg": round(float(np.degrees(ye)), 2),
                "dz_mm": round(dz * 1e3, 2),
                "error": err,
            }
        )
    end_view = thumb_view(t) if record else None  # the card's picture: the finished bridge, arms at home
    seated = [r["brick"] for r in per if r["seated"]]
    built = tuple(bm[n] for n in seated)
    rules = rules or hb.Rules()
    results = hr.check(built, rules) if built else []
    return {
        "seed": seed,
        "bricks": _finite(per),
        "seated": len(seated),
        "total": len(order),
        "rules": _finite(hr.to_json(results)),
        "rules_passed": sum(r.ok for r in results),
        "rules_total": len(results),
        "ik_err_mm": _finite(round(max(e.max_ik_err for e in arms.values()) * 1e3, 2)),
        "sim_seconds": round(tick[0] * he.DT, 1),
        "wall_seconds": round(time.time() - t0, 1),
        "frames": frames,
        "marks": marks,
        "thumb": end_view,
        "table": t,
    }
