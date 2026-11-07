"""The zoomed view behind a montage tile: the recorded MuJoCo rollout of one scene in one round, replayed frame by frame.

The server owns the replay. A rollout is stored as the pose of every body at 30 fps (rollout.py); montage.Stage puts each
pose back into the compiled scene and MuJoCo renders it from a camera the page orbits. Runs made by the population loop
keep the whole trajectory of every round (runs/opt-<id>/traj/r<round>.npz). Older runs kept only the picture sheets, so for
those the scene is rolled out again on request with the policy of that round (R0 is the skill as trained, the best round is
final.pt) and cached; other rounds of an older run have no stored trajectory and say so.
"""

from __future__ import annotations

import io
import json
from concurrent.futures import ThreadPoolExecutor
from threading import Lock

import numpy as np
from PIL import Image

from . import montage, optimizer
from .encoders import device
from .policy import VLA
from .registry import ROOT, Registry
from .sim.tasks import sample_layout

FPS = 30
# MuJoCo's offscreen renderer is bound to the thread that made it: one thread draws every zoomed frame
POOL = ThreadPoolExecutor(1, thread_name_prefix="zoom")
_STAGES: dict[str, montage.Stage] = {}
_CACHE: dict[tuple, dict] = {}
_LOCK = Lock()


def _run_dir(oid: str):
    if not oid.replace("-", "").isalnum():
        raise KeyError(oid)
    d = optimizer.RUNS / f"opt-{oid}"
    if not (d / "run.json").exists():
        raise KeyError(oid)
    return d


def _layout(spec, scene: int) -> dict:
    return sample_layout(spec, np.random.default_rng(optimizer.SEED0 + scene)).to_json()


def _load(oid: str, rnd: int, scene: int) -> dict | None:
    """{frames, success, failure, steps, stats, layout, robot, source} or None when this round has no stored rollout."""
    key = (oid, rnd, scene)
    with _LOCK:
        if key in _CACHE:
            return _CACHE[key]
    d = _run_dir(oid)
    run = json.loads((d / "run.json").read_text())
    sk = Registry().skills[run["skill"]["id"]]
    spec = sk.task()
    full, meta = d / "traj" / f"r{rnd}.npz", d / "traj" / f"r{rnd}.json"
    rec = None
    if full.exists() and meta.exists():
        z = np.load(full)
        if f"s{scene}" in z:
            rec = {
                **json.loads(meta.read_text())[scene],
                "frames": z[f"s{scene}"],
                "source": "stored trajectory",
            }
    if rec is None:
        cached = d / "traj" / f"ondemand_r{rnd}_s{scene}.npz"
        if cached.exists():
            z = np.load(cached, allow_pickle=False)
            rec = {
                **json.loads(str(z["meta"])),
                "frames": z["frames"],
                "source": "rolled out again from the saved policy",
            }
        else:
            if rnd == 0:
                path = ROOT / sk.model
            elif rnd == run["best_round"] and (d / "final.pt").exists():
                path = d / "final.pt"
            elif (d / "weights" / f"p{rnd}.pt").exists():
                path = d / "weights" / f"p{rnd}.pt"
            else:
                return None
            m, _ = VLA.load(path, device())
            traj: dict[int, np.ndarray] = {}
            sc = optimizer.score(
                m,
                spec,
                [_layout(spec, scene)],
                sk.robot,
                lambda n: None,
                first=scene,
                traj=traj,
            )[0]
            from .optimizer_evo import light

            meta_d = light({**sc, "index": scene})
            np.savez_compressed(cached, frames=traj[0], meta=json.dumps(meta_d))
            rec = {
                **meta_d,
                "frames": traj[0],
                "source": "rolled out again from the saved policy",
            }
    rec.update(layout=_layout(spec, scene), robot=sk.robot)
    with _LOCK:
        _CACHE[key] = rec
        if len(_CACHE) > 40:
            _CACHE.pop(next(iter(_CACHE)))
    return rec


def _note(run: dict, rnd: int, scene: int) -> dict | None:
    """What the model said about this scene, from the run's own record. A note written about this scene wins; otherwise
    the diagnosis written when it looked at this round's failures, labelled as a pattern and not as a note on the scene."""
    rows = run["rounds"]
    looked = [r for r in rows[1:] if r.get("parent", None) == rnd] or (
        [rows[rnd + 1]] if rnd + 1 < len(rows) else []
    )
    for r in [*looked, *rows[1:]]:
        for n in (r.get("change") or {}).get("scene_notes", []):
            if n["scene"] == scene:
                return {
                    "text": n["note"],
                    "kind": "scene",
                    "round": r["round"],
                    "backend": r.get("backend"),
                }
    for r in looked:
        dg = (r.get("change") or {}).get("diagnosis")
        if dg:
            return {
                "text": dg,
                "kind": "pattern",
                "round": r["round"],
                "backend": r.get("backend"),
            }
    return None


def info(oid: str, rnd: int, scene: int) -> dict:
    d = _run_dir(oid)
    run = json.loads((d / "run.json").read_text())
    rows = run["rounds"]
    if (
        not (0 <= rnd < len(rows))
        or rows[rnd].get("success") is None
        or not (0 <= scene < len(rows[rnd]["outcomes"]))
    ):
        raise KeyError("no such round or scene")
    row = rows[rnd]
    best = run["best_round"]

    def at(r: int) -> dict:
        o = rows[r]["outcomes"][scene]
        return {"round": r, "success": bool(o)}

    rec = POOL.submit(_load, oid, rnd, scene).result()
    ch = row.get("change") or {}
    return {
        "scene": scene,
        "round": rnd,
        "n_scenes": run["n_scenes"],
        "success": bool(row["outcomes"][scene]),
        "failure": (rec or {}).get("failure", ""),
        "steps": (rec or {}).get("steps"),
        "available": rec is not None,
        "frames": len(rec["frames"]) if rec else 0,
        "fps": FPS,
        "source": rec["source"]
        if rec
        else "no trajectory was stored for this round of an older run",
        "change": {
            "text": ch.get("text", "the skill as trained"),
            "reason": ch.get("reason", ""),
            "status": row.get("status"),
            "round_success": row.get("success"),
            "vs": row.get("vs"),
        },
        "note": _note(run, rnd, scene),
        "before": at(0),
        "after": at(best),
        "best_round": best,
        "model": "Llama 4 Maverick",
    }


def _render(
    oid: str,
    rnd: int,
    scene: int,
    i: int,
    w: int,
    h: int,
    az: float,
    el: float,
    dist: float,
) -> bytes:
    rec = _load(oid, rnd, scene)
    if rec is None:
        raise KeyError("no trajectory")
    st = _STAGES.setdefault(rec["robot"], montage.Stage(rec["robot"]))
    i = min(max(i, 0), len(rec["frames"]) - 1)
    img = st.render(
        rec["layout"], rec["frames"], [i], w, h, distance=dist, azimuth=az, elevation=el
    )[0]
    buf = io.BytesIO()
    Image.fromarray(img).save(buf, "JPEG", quality=82)
    return buf.getvalue()


def frame(
    oid: str,
    rnd: int,
    scene: int,
    i: int,
    w: int = 480,
    h: int = 360,
    az: float = 148.0,
    el: float = -34.0,
    dist: float = 0.46,
) -> bytes:
    w, h = min(max(w, 160), 800), min(max(h, 120), 600)
    az, el, dist = az % 360.0, min(max(el, -85.0), -5.0), min(max(dist, 0.2), 1.2)
    return POOL.submit(_render, oid, rnd, scene, i, w, h, az, el, dist).result()
