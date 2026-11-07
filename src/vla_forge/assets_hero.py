"""The factory hero: two SO-101 arms assemble a real consumer product (the cassette player) in the dressed cell.

    uv run --group assets python -m vla_forge.assets_hero [--parts 6] [--seed 7]

Per part, on the conveyor and through two station cells, each with its own arm:
  1. the part (a workpiece carrier with the player's body and an unseated cassette and lid) rides the belt to the
     photo-eye; arm 1 picks it and seats it in the pallet nest (task "kit"); the cassette seats;
  2. it rides on to station 2; arm 2 picks it and inserts it into the socket (task "insert"); the lid closes.
Success is the cell's own physics predicates (sim/tasks.py: in_slot, seated, aligned, still), read from the
MuJoCo state, per station and for the whole product. The controller is the scripted expert with privileged state,
labelled as such; the trained-VLA path goes through the normal pipeline (the dressed Layout is an ordinary Layout).

Writes runs/hero/factory_hero.json (scene export, both cells' recorded episodes, success counts) for web/factory.html.
"""

from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path

import mujoco
import numpy as np

from . import factory as F
from .assets_scene import PITCH, bind_product
from .sim import line as L
from .sim.expert import Expert
from .sim.scene import Layout, Look, Scene
from .sim.tasks import check
from .viz import export_scene

CELL = "factory_cell"
OUT = F.ROOT / "runs" / "hero"
KIT = F.Station(
    1,
    "Seat the cassette carrier in the pallet",
    "red",
    "gray",
    task="kit",
    slot="front-left",
)
INSERT = F.Station(
    2, "Insert into the socket, close the lid", "red", "gray", task="insert"
)


def dressed(lay: Layout) -> Layout:
    lay.dressing = CELL
    return lay


def _arrive(st: F.Station, color: str, yaw: float):
    """The part rides the belt into the cell and stops at the photo-eye; returns frames and where it stopped."""
    lay = dressed(F._cell_layout(st, color, L.BELT_X, L.ENTRY_Y, yaw))
    sc = Scene(lay, st.robot)
    sc.reset(lay)
    frames = [sc.pose_frame(), *F._belt_phase(sc, L.WINDOW_Y + 0.011)]
    p = sc.body_pos("target")
    stop = (float(p[0]), float(p[1]), F._yaw_of(frames[-1], sc.model.body("target").id))
    sc.close()
    return frames, stop


def _station(st: F.Station, color: str, stop, rng: np.random.Generator, i: int):
    """Run the expert in a station cell. Returns (ok, frames, failure) with the product following the physics."""
    lay = dressed(F._cell_layout(st, color, *stop, rng=rng))
    spec = st.spec(color)
    sc = Scene(lay, st.robot)
    sc.reset(lay, Look.sample(np.random.default_rng(i)))
    drv = bind_product(sc, CELL)
    drv.reset(cassette_in=st.task == "insert")
    frames: list[list[float]] = []

    def tick(s: Scene) -> None:
        done = check(spec, s, lay)
        drv.step(
            cassette_in=st.task == "insert" or done,
            lid_closed=st.task == "insert" and done,
        )
        frames.append(s.pose_frame())

    Expert(sc, spec, on_tick=tick, rng=np.random.default_rng(i)).run()
    ok = bool(check(spec, sc, lay))
    for _ in range(6):  # let the product pose finish easing after the last tick
        drv.step(
            cassette_in=st.task == "insert" or ok, lid_closed=st.task == "insert" and ok
        )
        frames.append(sc.pose_frame())
    result = (ok, F._frames_30hz(frames), "" if ok else "part not seated")
    sc.close()
    return result


def run(n_parts: int = 6, seed: int = 7) -> dict:
    t0 = time.time()
    rng = np.random.default_rng(seed)
    parts = []
    yaws = [float(rng.uniform(-0.6, 0.6)) for _ in range(n_parts)]
    for i in range(n_parts):
        part = {"part": i, "stations": []}
        yaw = math.atan2(L.WINDOW_Y, L.BELT_X) + yaws[i]
        ok_all = True
        for st in (KIT, INSERT):
            arrive, stop = _arrive(st, "red", yaw)
            ok, frames, why = _station(
                st,
                "red",
                stop,
                np.random.default_rng(seed * 7919 + st.index * 131 + i),
                i,
            )
            part["stations"].append(
                {
                    "station": st.index,
                    "task": st.task,
                    "ok": ok,
                    "failure": why,
                    "arrive": [[round(v, 4) for v in f] for f in arrive[::2]],
                    "frames": [[round(v, 4) for v in f] for f in frames[::2]],
                }
            )
            print(f"part {i} station {st.index} ({st.task}) ok={ok}", flush=True)
            if not ok:
                ok_all = False
                break
            yaw = stop[2]
        part["product_ok"] = ok_all
        parts.append(part)
    per = {
        st.task: {
            "tried": sum(
                len([s for s in p["stations"] if s["task"] == st.task]) for p in parts
            ),
            "ok": sum(
                1
                for p in parts
                for s in p["stations"]
                if s["task"] == st.task and s["ok"]
            ),
        }
        for st in (KIT, INSERT)
    }
    lay = dressed(F._cell_layout(KIT, "red", L.BELT_X, L.WINDOW_Y, 0.0))
    result = {
        "cell": CELL,
        "controller": "scripted expert with privileged state (not a trained VLA)",
        "n_parts": n_parts,
        "seed": seed,
        "assembled": sum(p["product_ok"] for p in parts),
        "success_rate": round(sum(p["product_ok"] for p in parts) / n_parts, 3),
        "per_station": per,
        "wall_seconds": round(time.time() - t0, 1),
        "scene": export_scene(lay),
        "parts": parts,
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "factory_hero.json").write_text(json.dumps(result))
    return result


_SCENERY = ("scn", "product", "belt_item", "guard")


def palette(m: mujoco.MjModel) -> None:
    """The current Dojo look for a dressed cell: white shells, black servos, graphite steel on the arms (the roles
    and colours of sim/display.py, which the library tiles use), the tile floor and table greys. Library models keep
    their own materials. Display only."""
    from .sim.display import _COLORS, CONCRETE, robot_role

    for g in range(m.ngeom):
        if m.geom(g).name == "table":
            m.geom_rgba[g] = CONCRETE
            m.geom_matid[g] = -1
            continue
        body = m.body(m.geom_bodyid[g]).name
        if (
            m.geom_type[g] != mujoco.mjtGeom.mjGEOM_MESH
            or m.geom_group[g] != 2
            or body.startswith(_SCENERY)
        ):
            continue
        mesh = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_MESH, m.geom_dataid[g]) or ""
        rgba = (
            m.mat_rgba[m.geom_matid[g]] if m.geom_matid[g] >= 0 else m.geom_rgba[g]
        ).copy()
        m.geom_rgba[g] = np.asarray(_COLORS[robot_role(rgba, mesh)])
        m.geom_matid[g] = -1


class Cell:
    """One station's compiled scene, posed, with the product following the physics (for stills)."""

    def __init__(
        self, st: F.Station, stop, hide_static: bool = False, idle: bool = False
    ):
        self.st = st
        self.lay = dressed(F._cell_layout(st, "red", *stop))
        self.spec = st.spec("red")
        self.sc = Scene(self.lay, st.robot)
        self.sc.reset(self.lay)
        self.drv = bind_product(self.sc, CELL)
        self.drv.reset(cassette_in=st.task == "insert")
        palette(self.sc.model)
        if (
            hide_static
        ):  # a second arm drawn in the same picture: only what moves with it
            m = self.sc.model
            keep = {
                "base", "shoulder", "upper_arm", "lower_arm", "wrist", "gripper",
                "moving_jaw_so101_v1", "tray", "goal", "target",
            }  # fmt: skip
            for g in range(m.ngeom):
                b = m.body(m.geom_bodyid[g]).name
                if b not in keep and not b.startswith("product"):
                    m.geom_group[g] = 5
        if idle:  # a cell with nothing in it: the arm at home, the part and its product not drawn
            m = self.sc.model
            for g in range(m.ngeom):
                b = m.body(m.geom_bodyid[g]).name
                if b == "target" or b.startswith("product"):
                    m.geom_group[g] = 5
        mujoco.mj_forward(self.sc.model, self.sc.data)

    def run(self, on_tick=None) -> bool:
        def tick(_s):
            self.tick()
            if on_tick:
                on_tick(self)

        Expert(self.sc, self.spec, on_tick=tick, rng=np.random.default_rng(0)).run()
        return self.done()

    def done(self) -> bool:
        return bool(check(self.spec, self.sc, self.lay))

    def tick(self) -> None:
        d = self.done()
        self.drv.step(
            cassette_in=self.st.task == "insert" or d,
            lid_closed=self.st.task == "insert" and d,
        )

    def close(self) -> None:
        self.sc.close()


def _remap(scn, n0: int, src: mujoco.MjModel, dst: mujoco.MjModel) -> None:
    """Geoms added from another model's data refer to its ids; point them at the renderer's model by name."""
    for i in range(n0, scn.ngeom):
        g = scn.geoms[i]
        for attr, kind in (
            ("dataid", mujoco.mjtObj.mjOBJ_MESH),
            ("matid", mujoco.mjtObj.mjOBJ_MATERIAL),
            ("texid", mujoco.mjtObj.mjOBJ_TEXTURE),
        ):
            sid = getattr(g, attr)
            if sid < 0:
                continue
            name = mujoco.mj_id2name(src, kind, sid)
            setattr(g, attr, mujoco.mj_name2id(dst, kind, name) if name else -1)


CAMERA = {
    "lookat": [0.2, 0.62, 0.02],
    "distance": 1.7,
    "azimuth": 150.0,
    "elevation": -26.0,
}


def render_line(
    path: Path,
    a: Cell,
    b: Cell,
    size=(1440, 900),
    camera=None,
    shift: float | None = None,
    hide_front: bool = False,
) -> None:
    """One MuJoCo frame of the whole line: station 1's cell, and station 2's arm one pitch along the belt."""
    from PIL import Image

    ma, mb = a.sc.model, b.sc.model
    if hide_front:  # the guard panels, posts and gate nearest the camera
        for g in range(ma.ngeom):
            if ma.geom(g).name.startswith("gf_"):
                ma.geom_group[g] = 5
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    c = camera or CAMERA
    cam.lookat[:] = c["lookat"]
    cam.distance, cam.azimuth, cam.elevation = (
        c["distance"], c["azimuth"], c["elevation"],
    )  # fmt: skip
    ma.vis.global_.offwidth, ma.vis.global_.offheight = size
    for i in range(ma.nlight):
        ma.light_castshadow[i] = 1 if ma.light(i).name.startswith("key") else 0
    r = mujoco.Renderer(ma, size[1], size[0])
    r.update_scene(a.sc.data, camera=cam)
    scn = r.scene
    n0 = scn.ngeom
    opt = mujoco.MjvOption()
    mujoco.mjv_addGeoms(
        mb, b.sc.data, opt, mujoco.MjvPerturb(), mujoco.mjtCatBit.mjCAT_ALL, scn
    )
    _remap(scn, n0, mb, ma)
    for i in range(n0, scn.ngeom):
        scn.geoms[i].pos[1] += PITCH if shift is None else shift
    r.render()  # draws the scene as composed
    Image.fromarray(r.render()).save(path)
    r.close()


def shots(out_dir: Path, prefix: str = "2026-11-07-dojo-assets") -> None:
    """The stills of the cassette-player task and the cell: the expert runs, a frame is taken at each moment."""
    stop = (0.135, 0.11, 0.0)
    close1 = {
        "lookat": [0.2, 0.15, 0.04],
        "distance": 0.8,
        "azimuth": 150.0,
        "elevation": -30.0,
    }
    cell = {"lookat": [0.25, 0.25, 0.08], "distance": 0.95, "azimuth": 180, "elevation": -35}
    wide = {
        "lookat": [0.25, 0.25, 0.05],
        "distance": 1.15,
        "azimuth": 160.0,
        "elevation": -26.0,
    }
    size = (1200, 800)

    def pair(live_first: bool, st_live, idle_cell: bool = True):
        live = Cell(st_live, stop, hide_static=not live_first)
        rest = Cell(
            INSERT if live_first else KIT, stop, hide_static=live_first, idle=idle_cell
        )
        return (live, rest) if live_first else (rest, live)

    a, b = pair(True, KIT)
    render_line(out_dir / f"{prefix}-factory-mujoco.png", a, b, size, cell, hide_front=True)
    render_line(out_dir / f"{prefix}-factory-mujoco-wide.png", a, b, size, wide)
    a.close(), b.close()

    a, b = pair(True, KIT)
    seen = {"n": 0}

    def step1(c: Cell) -> None:
        if c.drv.cas_t >= 1.0 and not seen["n"]:
            seen["n"] = 1
            render_line(out_dir / f"{prefix}-cassette-step1.png", c, b, size, close1)

    ok1 = a.run(step1)
    a.close(), b.close()

    live = Cell(INSERT, stop)
    rest = Cell(KIT, stop, hide_static=True, idle=True)
    ok2 = live.run()
    for _ in range(8):
        live.tick()
    render_line(
        out_dir / f"{prefix}-cassette-done.png", live, rest, size, close1, shift=-PITCH
    )
    live.close(), rest.close()
    print(json.dumps({"kit_ok": ok1, "insert_ok": ok2}))


if __name__ == "__main__":
    a = sys.argv[1:]
    if "--shots" in a:
        shots(Path(a[a.index("--shots") + 1]))
        raise SystemExit
    n = int(a[a.index("--parts") + 1]) if "--parts" in a else 6
    s = int(a[a.index("--seed") + 1]) if "--seed" in a else 7
    res = run(n, s)
    print(
        json.dumps(
            {k: v for k, v in res.items() if k not in ("scene", "parts")}, indent=1
        )
    )
