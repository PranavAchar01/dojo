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
from .assets_scene import bind_product
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


def render(path: Path, size=(1440, 900), pose: str = "seated") -> None:
    """One MuJoCo frame of the dressed cell (offscreen renderer, lamp shadows on)."""
    lay = dressed(F._cell_layout(KIT, "red", F.TRAY_AT[0], F.TRAY_AT[1], 0.0))
    sc = Scene(lay)
    m = sc.model
    for i in range(m.nlight):
        m.light_castshadow[i] = 1 if m.light(i).name in ("key", "lamp0") else 0
    sc.reset(lay)
    drv = bind_product(sc, CELL)
    drv.reset(cassette_in=pose == "seated")
    mujoco.mj_forward(m, sc.data)
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.lookat[:] = [0.22, 0.0, 0.0]
    cam.distance, cam.azimuth, cam.elevation = 1.35, 150.0, -24.0
    m.vis.global_.offwidth, m.vis.global_.offheight = size
    r = mujoco.Renderer(m, size[1], size[0])
    r.update_scene(sc.data, camera=cam)
    from PIL import Image

    Image.fromarray(r.render()).save(path)
    r.close()
    sc.close()


if __name__ == "__main__":
    a = sys.argv[1:]
    n = int(a[a.index("--parts") + 1]) if "--parts" in a else 6
    s = int(a[a.index("--seed") + 1]) if "--seed" in a else 7
    res = run(n, s)
    print(
        json.dumps(
            {k: v for k, v in res.items() if k not in ("scene", "parts")}, indent=1
        )
    )
