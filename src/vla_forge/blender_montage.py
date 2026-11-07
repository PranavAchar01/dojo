"""The optimizer montage rendered in Blender.

Physics stays MuJoCo's: a rollout is a list of body poses (rollout.py), and Blender only draws them. For each layout
structure the compiled, restyled MuJoCo scene is exported once (geoms, their body, their local pose, mesh vertices and
faces from the SO-101 assets the model loaded) as an .npz. Rollouts are then split over N headless Blender processes
(`blender -b -P scripts/blender_worker.py`), each of which builds that geometry with a concrete floor and the
black-and-white arm, sets every geom from the recorded poses and renders the requested moments with EEVEE.
The result is packed into the same sprite sheets the Optimize panel already plays (montage.write_sheets' contract).
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import tempfile
import time
from pathlib import Path

import mujoco
import numpy as np
from PIL import Image

from . import montage
from .registry import ROOT
from .sim.scene import Layout

WORKER = ROOT / "scripts" / "blender_worker.py"
APP = Path("/Applications/Blender.app/Contents/MacOS/Blender")
RENDER_SCALE = 2  # tiles are rendered at 2x and averaged down
SAMPLES = 8
HERO_SCALE = 2
MIN_FREE_PCT = 10
GB_PER_WORKER = 1.5


def blender_bin() -> str | None:
    if (
        os.environ.get("VLA_FORGE_NO_BLENDER") == "1"
    ):  # the GPU is busy or absent: draw the grid with MuJoCo
        return None
    return shutil.which("blender") or (str(APP) if APP.exists() else None)


def free_memory_pct() -> int:
    try:
        out = subprocess.run(
            ["memory_pressure"], check=False, capture_output=True, text=True, timeout=20
        ).stdout
        m = re.search(r"free percentage:\s*(\d+)%", out)
        return int(m.group(1)) if m else 100
    except (OSError, subprocess.SubprocessError):
        return 100


def worker_count(jobs: int, cap: int | None = None) -> int:
    """As many Blender processes as memory allows: cores, ~1.5 GB each, and none while free memory is under 10%."""
    if free_memory_pct() < MIN_FREE_PCT:
        return 1
    cores = os.cpu_count() or 4
    try:
        total_gb = int(
            subprocess.run(
                ["sysctl", "-n", "hw.memsize"],
                check=False,
                capture_output=True,
                text=True,
            ).stdout
        ) / (1 << 30)
    except (OSError, ValueError):
        total_gb = 16.0
    free_gb = total_gb * free_memory_pct() / 100
    n = int(min(cores - 1, (free_gb - 0.1 * total_gb) // GB_PER_WORKER, jobs))
    if cap:
        n = min(n, cap)
    if os.environ.get("VLA_FORGE_BLENDER_WORKERS"):
        n = min(n, int(os.environ["VLA_FORGE_BLENDER_WORKERS"]))
    return max(1, n)


def export_geometry(stage: montage.Stage, layout_json: dict, path: Path) -> None:
    """Every visible geom of the compiled scene, with the body it rides on and its pose in that body's frame."""
    lay = Layout.from_json(layout_json)
    m, _d, _ = stage._model(lay)
    rows = []
    for g in range(m.ngeom):
        name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g) or ""
        if name == "floor" or m.geom_group[g] > 2 or m.geom_type[g] == 0:
            continue
        rgba = m.geom_rgba[g].copy()
        if m.geom_matid[g] >= 0:
            rgba = m.mat_rgba[m.geom_matid[g]].copy()
        role = "table" if name == "table" else "arm" if m.geom_type[g] == 7 else "prop"
        if m.geom_type[g] == 7 and name.startswith(("block", "cube")):
            role = "prop"
        rows.append((g, rgba, role))
    verts, faces = [], []
    vadr, vnum, fadr, fnum = [], [], [], []
    for g, _r, _role in rows:
        if m.geom_type[g] == 7:
            mid = int(m.geom_dataid[g])
            va, vn = int(m.mesh_vertadr[mid]), int(m.mesh_vertnum[mid])
            fa, fn = int(m.mesh_faceadr[mid]), int(m.mesh_facenum[mid])
            vadr.append(len(verts))
            vnum.append(vn)
            verts.extend(m.mesh_vert[va : va + vn].tolist())
            fadr.append(len(faces))
            fnum.append(fn)
            faces.extend((m.mesh_face[fa : fa + fn]).tolist())
        else:
            vadr.append(0)
            vnum.append(0)
            fadr.append(0)
            fnum.append(0)
    np.savez_compressed(
        path,
        gbody=np.array([m.geom_bodyid[g] for g, _, _ in rows], np.int32),
        gpos=np.array([m.geom_pos[g] for g, _, _ in rows]),
        gquat=np.array([m.geom_quat[g] for g, _, _ in rows]),
        gtype=np.array([m.geom_type[g] for g, _, _ in rows], np.int32),
        gsize=np.array([m.geom_size[g] for g, _, _ in rows]),
        grgba=np.array([r for _, r, _ in rows]),
        grole=np.array([role for _, _, role in rows]),
        vadr=np.array(vadr, np.int32),
        vnum=np.array(vnum, np.int32),
        fadr=np.array(fadr, np.int32),
        fnum=np.array(fnum, np.int32),
        verts=np.array(verts, np.float32).reshape(-1, 3),
        faces=np.array(faces, np.int32).reshape(-1, 3),
    )


WORKER_TIMEOUT_S = int(os.environ.get("VLA_FORGE_BLENDER_TIMEOUT", "600"))


def _run_workers(specs: list[Path], log_dir: Path) -> list[float]:
    """Launch one Blender per spec, wait for all, return each one's wall time. Only these PIDs are ever signalled."""
    binary = blender_bin()
    if binary is None:
        raise RuntimeError("Blender is not installed")
    procs = []
    for i, spec in enumerate(specs):
        log = open(log_dir / f"worker{i}.log", "w")  # noqa: SIM115
        p = subprocess.Popen(
            [
                "nice",
                "-n",
                "19",
                "taskpolicy",
                "-b",
                binary,
                "-b",
                "--factory-startup",
                "-P",
                str(WORKER),
                "--",
                str(spec),
            ],
            stdout=log,
            stderr=subprocess.STDOUT,
        )
        procs.append((p, log, time.time()))
    times = []
    deadline = time.time() + WORKER_TIMEOUT_S
    for p, log, t0 in procs:
        try:
            rc = p.wait(timeout=max(1.0, deadline - time.time()))
        except subprocess.TimeoutExpired:
            for q, _, _ in procs:
                if q.poll() is None:
                    q.kill()
            log.close()
            raise RuntimeError(
                f"Blender workers timed out after {WORKER_TIMEOUT_S}s; see {log_dir}"
            ) from None
        times.append(round(time.time() - t0, 1))
        log.close()
        if rc != 0:
            raise RuntimeError(f"a Blender worker exited with {rc}; see {log_dir}")
    return times


def render_sheets(
    stage: montage.Stage,
    scenes: list[dict],
    out_dir: Path,
    moments: int = montage.MOMENTS,
    workers: int | None = None,
) -> dict:
    """Same output as montage.write_sheets (m0..m7.jpg, 10 tiles wide) but drawn in Blender, N processes at once."""
    t0 = time.time()
    out_dir.mkdir(parents=True, exist_ok=True)
    work = Path(tempfile.mkdtemp(prefix="blender-montage-", dir=out_dir.parent))
    try:
        geo: dict[tuple, Path] = {}
        jobs = []
        tw, th = montage.TILE_W * RENDER_SCALE, montage.TILE_H * RENDER_SCALE
        for si, sc in enumerate(scenes):
            key = Layout.from_json(sc["layout"]).key()
            if key not in geo:
                geo[key] = work / f"geo{len(geo)}.npz"
                export_geometry(stage, sc["layout"], geo[key])
            fr = work / f"frames{si}.npy"
            np.save(fr, np.asarray(sc["frames"], np.float32))
            idx = montage.moment_index(len(sc["frames"]), moments)
            jobs.append(
                {
                    "geo": str(geo[key]),
                    "frames": str(fr),
                    "idx": idx,
                    "out": [str(work / f"t{si}_{k}.png") for k in range(len(idx))],
                    "scene": si,
                }
            )
        n = workers or worker_count(len(jobs))
        jobs.sort(
            key=lambda j: j["geo"]
        )  # a worker rebuilds the scene only when the geometry changes
        specs = []
        for w in range(n):
            part = jobs[w::n]
            sp = work / f"spec{w}.json"
            sp.write_text(
                json.dumps({"res": [tw, th], "samples": SAMPLES, "jobs": part})
            )
            specs.append(sp)
        times = _run_workers(specs, work)
        cols = montage.COLS
        rows = (len(scenes) + cols - 1) // cols
        sheets = [
            Image.new(
                "RGB",
                (cols * montage.TILE_W, rows * montage.TILE_H),
                (240, 242, 245),
            )
            for _ in range(moments)
        ]
        renders = 0
        for j in jobs:
            si = j["scene"]
            for k, path in enumerate(j["out"]):
                im = Image.open(path).convert("RGB")
                im = im.resize((montage.TILE_W, montage.TILE_H), Image.LANCZOS)
                sheets[k].paste(
                    im, ((si % cols) * montage.TILE_W, (si // cols) * montage.TILE_H)
                )
                renders += 1
        for k, sh in enumerate(sheets):
            sh.save(out_dir / f"m{k}.jpg", quality=80)
    finally:
        shutil.rmtree(work, ignore_errors=True)
    return {
        "engine": "blender-eevee",
        "scenes": len(scenes),
        "renders": renders,
        "workers": n,
        "worker_seconds": times,
        "seconds": round(time.time() - t0, 1),
    }


def render_hero(
    stage: montage.Stage, scene: dict, out: Path, workers: int | None = None
) -> dict:
    """The before/after filmstrip (HERO_FRAMES frames, split over the workers) at twice the size, averaged down."""
    t0 = time.time()
    work = Path(tempfile.mkdtemp(prefix="blender-hero-", dir=out.parent))
    try:
        geo = work / "geo.npz"
        export_geometry(stage, scene["layout"], geo)
        fr = work / "frames.npy"
        np.save(fr, np.asarray(scene["frames"], np.float32))
        idx = montage.moment_index(len(scene["frames"]), montage.HERO_FRAMES)
        n = max(1, min(workers or worker_count(len(idx)), len(idx), 4))
        res = [montage.HERO_W * HERO_SCALE, montage.HERO_H * HERO_SCALE]
        specs = []
        for w in range(n):
            sel = list(range(w, len(idx), n))
            sp = work / f"spec{w}.json"
            sp.write_text(
                json.dumps(
                    {
                        "res": res,
                        "samples": 24,
                        "jobs": [
                            {
                                "geo": str(geo),
                                "frames": str(fr),
                                "idx": [idx[k] for k in sel],
                                "out": [str(work / f"h{k}.png") for k in sel],
                                "distance": 0.62,
                            }
                        ],
                    }
                )
            )
            specs.append(sp)
        _run_workers(specs, work)
        strip = Image.new("RGB", (montage.HERO_W * len(idx), montage.HERO_H))
        for k in range(len(idx)):
            im = Image.open(work / f"h{k}.png").convert("RGB")
            strip.paste(
                im.resize((montage.HERO_W, montage.HERO_H), Image.LANCZOS),
                (k * montage.HERO_W, 0),
            )
        out.parent.mkdir(parents=True, exist_ok=True)
        strip.save(out, quality=84)
    finally:
        shutil.rmtree(work, ignore_errors=True)
    return {"renders": len(idx), "workers": n, "seconds": round(time.time() - t0, 1)}
