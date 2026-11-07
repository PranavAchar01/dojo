"""Demonstration generation: the scripted expert in randomized scenes, recorded at the policy's 10 Hz.

Runs in a pool of worker processes (mujoco + numpy only, no torch). Each successful episode is written to one
.npz (images uint8 [T, 2, 168, 168, 3], states and joint commands [T, 6]); failed expert episodes are dropped.
"""

from __future__ import annotations

import multiprocessing as mp
import os
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from .sim.expert import Expert
from .sim.scene import Layout, Look, Scene
from .sim.tasks import TaskSpec, check, sample_layout


def demo_job(args: dict) -> dict:
    """One expert episode. args: spec, seed, out, look_strength, near, layout, robot (a robot id, default so101)."""
    t0 = time.time()
    spec = TaskSpec.from_json(args["spec"])
    rng = np.random.default_rng(args["seed"])
    layout = (
        Layout.from_json(args["layout"])
        if args.get("layout")
        else sample_layout(
            spec, rng, near=args.get("near"), robot=args.get("robot", "so101")
        )
    )
    look = Look.sample(rng, args.get("look_strength", 1.0))
    sc = Scene(layout, args.get("robot", "so101"))
    sc.reset(layout, look)
    imgs, states, ctrls, aux = [], [], [], []
    has_goal = "goal" in {**layout.fixtures(), **layout.blocks}

    def on_tick(s: Scene):
        f, w = s.render_obs()
        imgs.append(np.stack([f, w]))
        states.append(s.joint_pos())
        ctrls.append(
            s.cmd.copy()
        )  # the expert's clean command (the executed one may carry DART noise)
        g = s.body_pos("goal")[:2] if has_goal else np.zeros(2)
        aux.append(np.concatenate([s.body_pos("target"), g]))

    noise = args.get("noise", 0.0) * float(rng.uniform(0.3, 1.0))
    ex = Expert(sc, spec, on_tick=on_tick, noise=noise, rng=rng)
    ex.run()
    ok = check(spec, sc, layout)
    sc.close()
    out = {
        "ok": bool(ok),
        "T": len(states),
        "layout": layout.to_json(),
        "seconds": time.time() - t0,
        "seed": args["seed"],
    }
    if ok:
        path = Path(args["out"])
        path.parent.mkdir(parents=True, exist_ok=True)
        ctrls_a = np.array(ctrls, np.float32)
        actions = np.concatenate(
            [ctrls_a[1:], ctrls_a[-1:]], 0
        )  # command in force at the end of the next 0.1 s
        np.savez(
            path,
            images=np.array(imgs, np.uint8),
            states=np.array(states, np.float32),
            actions=actions,
            aux=np.array(aux, np.float32),
        )
        out["path"] = str(path)
    return out


_POOL: ProcessPoolExecutor | None = None


def _warm(_):
    from .sim.scene import _pad_frames

    _pad_frames()
    return os.getpid()


def pool(workers: int | None = None) -> ProcessPoolExecutor:
    """A persistent spawn pool; warmed once so a run never pays process start-up."""
    global _POOL
    if _POOL is None:
        n = workers or int(os.environ.get("VLA_FORGE_DATA_WORKERS", "0")) or 4
        _POOL = ProcessPoolExecutor(n, mp_context=mp.get_context("spawn"))
        list(_POOL.map(_warm, range(n)))
    return _POOL
