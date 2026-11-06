"""Closed-loop evaluation of a trained VLA in the simulator.

Environments live in persistent worker processes (physics + camera rendering in parallel); the main process
batches every environment's images through DINOv2 and the action head on MPS. Every 0.1 s each environment
renders its two cameras, the head predicts a 1 s chunk of joint targets, and overlapping chunks are blended
with ACT's temporal ensembling. Between policy steps the joint command is interpolated at 50 Hz. Body poses
are recorded at 30 fps so the web viewer replays exactly what happened. Success is the task's programmatic
check, held for HOLD_STEPS consecutive policy steps.
"""

from __future__ import annotations

import multiprocessing as mp
import os
import time
from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np

EVAL_SEED0 = 1_000_000  # eval layouts never overlap training seeds
HOLD_STEPS = {"lift": 10, "pick_place": 5, "push": 5, "stack": 5}
MAX_STEPS = {"pick_place": 150, "push": 150, "lift": 100, "stack": 150}
ENSEMBLE_M = 0.1


@dataclass
class Rollout:
    index: int
    seed: int
    layout: dict
    success: bool = False
    steps: int = 0
    failure: str = ""
    frames: list = field(default_factory=list)
    stats: dict = field(default_factory=dict)


# ------------------------------------------------------------------ worker side (no torch)


def _classify(skill: str, st: dict) -> str:
    from .sim.scene import CUBE_HALF

    if st["distractor_lift"] > 0.03:
        return "grabbed the wrong block"
    if skill == "push":
        return (
            "never reached the block"
            if st["target_moved"] < 0.02
            else "pushed off course"
        )
    if st["target_max_z"] < CUBE_HALF + 0.015:
        return (
            "missed the grasp"
            if st["min_grip_dist"] < 0.03
            else "never reached the block"
        )
    if skill == "lift":
        return "dropped the block"
    if st["final_goal_dist"] > 0.06:
        return "dropped it on the way"
    return "missed the target"


def _worker(conn):
    import mujoco

    from .sim.scene import (
        CONTROLS_PER_POLICY,
        CUBE_HALF,
        STEPS_PER_CONTROL,
        Layout,
        Look,
        Scene,
    )
    from .sim.tasks import TaskSpec, check

    envs: dict[int, dict] = {}
    spec = None
    while True:
        cmd, arg = conn.recv()
        if cmd == "init":
            for e in envs.values():
                e["sc"].close()
            envs.clear()
            spec = TaskSpec.from_json(arg["spec"])
            out = {}
            for i, lay_j, look_seed, strength in arg["envs"]:
                lay = Layout.from_json(lay_j)
                sc = Scene(lay)
                sc.reset(lay, Look.sample(np.random.default_rng(look_seed), strength))
                envs[i] = {
                    "sc": sc,
                    "frames": [sc.pose_frame()],
                    "start": sc.body_pos("target")[:2].copy(),
                    "stats": {
                        "target_max_z": 0.0,
                        "distractor_lift": 0.0,
                        "min_grip_dist": 9.0,
                        "target_moved": 0.0,
                        "final_goal_dist": 0.0,
                    },
                }
                out[i] = (np.stack(sc.render_obs()), sc.joint_pos(), False)
            conn.send(out)
        elif cmd == "step":
            out = {}
            for i, target in arg["targets"].items():
                e = envs[i]
                sc = e["sc"]
                m, d = sc.model, sc.data
                prev = d.ctrl.copy()
                for c in range(CONTROLS_PER_POLICY):
                    d.ctrl[:] = prev + (target - prev) * (c + 1) / CONTROLS_PER_POLICY
                    for s in range(STEPS_PER_CONTROL):
                        mujoco.mj_step(m, d)
                        if arg["record"] and (c * STEPS_PER_CONTROL + s) % 10 == 9:
                            e["frames"].append(sc.pose_frame())
                st = e["stats"]
                cpos = sc.body_pos("target")
                st["target_max_z"] = max(st["target_max_z"], float(cpos[2]))
                st["min_grip_dist"] = min(
                    st["min_grip_dist"], float(np.linalg.norm(sc.ee_pos() - cpos))
                )
                st["target_moved"] = max(
                    st["target_moved"], float(np.linalg.norm(cpos[:2] - e["start"]))
                )
                for name in sc.layout.blocks:
                    if name.startswith("distractor"):
                        st["distractor_lift"] = max(
                            st["distractor_lift"],
                            float(sc.body_pos(name)[2] - CUBE_HALF),
                        )
                if "goal" in {**sc.layout.bowls, **sc.layout.pads, **sc.layout.blocks}:
                    st["final_goal_dist"] = float(
                        np.linalg.norm(cpos[:2] - sc.body_pos("goal")[:2])
                    )
                ok = check(spec, sc)
                imgs = np.stack(sc.render_obs()) if arg["render"].get(i, True) else None
                out[i] = (imgs, sc.joint_pos(), ok)
            conn.send(out)
        elif cmd == "finish":
            out = {}
            for i, e in envs.items():
                out[i] = {
                    "frames": e["frames"],
                    "stats": e["stats"],
                    "failure": _classify(spec.skill, e["stats"]),
                }
                e["sc"].close()
            envs.clear()
            conn.send(out)
        elif cmd == "ping":
            conn.send(os.getpid())
        elif cmd == "stop":
            break


class EnvPool:
    """Persistent simulator workers for evaluation."""

    def __init__(self, workers: int = 5):
        ctx = mp.get_context("spawn")
        self.conns, self.procs = [], []
        for _ in range(workers):
            a, b = ctx.Pipe()
            p = ctx.Process(target=_worker, args=(b,), daemon=True)
            p.start()
            self.conns.append(a)
            self.procs.append(p)
        for c in self.conns:
            c.send(("ping", None))
        self.pids = [c.recv() for c in self.conns]

    def _scatter(self, cmd: str, per_worker: list) -> dict:
        for c, arg in zip(self.conns, per_worker, strict=True):
            c.send((cmd, arg))
        out = {}
        for c in self.conns:
            out.update(c.recv())
        return out

    def close(self):
        for c in self.conns:
            c.send(("stop", None))
        for p in self.procs:
            p.join(timeout=5)


_POOL: EnvPool | None = None


def env_pool() -> EnvPool:
    global _POOL
    if _POOL is None:
        _POOL = EnvPool(int(os.environ.get("VLA_FORGE_EVAL_WORKERS", "5")))
    return _POOL


# ------------------------------------------------------------------ main side


def evaluate(
    model,
    spec,
    n: int = 20,
    seed0: int = EVAL_SEED0,
    layouts: list | None = None,
    record: bool = True,
    on_progress: Callable[[dict], None] | None = None,
    look_strength: float = 1.0,
) -> list[Rollout]:
    import torch

    from .encoders import VIS_DIM, VIS_TOKENS, device, encode_images, encode_text
    from .sim.tasks import sample_layout

    dev = device()
    model.to(dev).eval()
    lang = encode_text([spec.instruction]).to(dev)
    pool = env_pool()
    w = len(pool.conns)
    rolls = []
    init = [{"spec": spec.to_json(), "envs": []} for _ in range(w)]
    for i in range(n):
        rng = np.random.default_rng(seed0 + i)
        lay = layouts[i] if layouts else sample_layout(spec, rng).to_json()
        rolls.append(Rollout(i, seed0 + i, lay))
        init[i % w]["envs"].append((i, lay, seed0 + i + 7, look_strength))
    obs = pool._scatter("init", init)
    k = model.cfg.chunk
    max_steps = MAX_STEPS[spec.skill]
    hold = HOLD_STEPS[spec.skill]
    preds = np.zeros((n, max_steps + k, k, 6), np.float32)
    streak = np.zeros(n, int)
    active = np.ones(n, bool)
    t_start = time.time()
    for t in range(max_steps):
        idx = [i for i in range(n) if active[i]]
        if not idx:
            break
        imgs = np.stack([obs[i][0] for i in idx])
        vis = (
            encode_images(imgs.reshape(-1, *imgs.shape[2:]), dev)
            .view(len(idx), 2, VIS_TOKENS, VIS_DIM)
            .to(dev)
        )
        state = torch.from_numpy(np.stack([obs[i][1] for i in idx])).float().to(dev)
        chunk = model.act(vis, lang.expand(len(idx), -1), state).float().cpu().numpy()
        per = [{"targets": {}, "record": record, "render": {}} for _ in range(w)]
        for bi, i in enumerate(idx):
            preds[i, t] = chunk[bi]
            lo = max(0, t - k + 1)
            cand = np.stack([preds[i, tp, t - tp] for tp in range(lo, t + 1)])
            wt = np.exp(-ENSEMBLE_M * np.arange(len(cand)))
            per[i % w]["targets"][i] = (wt[:, None] * cand).sum(0) / wt.sum()
        obs.update(pool._scatter("step", per))
        for i in idx:
            rolls[i].steps = t + 1
            streak[i] = streak[i] + 1 if obs[i][2] else 0
            if streak[i] >= hold:
                rolls[i].success = True
                active[i] = False
        if on_progress and (t % 5 == 4 or not active.any()):
            on_progress(
                {
                    "step": t + 1,
                    "max_steps": max_steps,
                    "done": int((~active).sum()),
                    "succeeded": int(sum(r.success for r in rolls)),
                    "n": n,
                    "elapsed": round(time.time() - t_start, 1),
                }
            )
    fin = pool._scatter("finish", [None] * w)
    for r in rolls:
        f = fin[r.index]
        r.frames, r.stats = f["frames"], f["stats"]
        if not r.success:
            r.failure = f["failure"]
    return rolls


def summarize(rolls: list[Rollout]) -> dict:
    fails: dict[str, int] = {}
    for r in rolls:
        if not r.success:
            fails[r.failure] = fails.get(r.failure, 0) + 1
    n = len(rolls)
    s = sum(r.success for r in rolls)
    return {
        "n": n,
        "succeeded": s,
        "success_rate": round(s / max(n, 1), 3),
        "failures": fails,
        "failed_starts": [
            {
                "index": r.index,
                "failure": r.failure,
                "target_xy": [round(v, 3) for v in r.layout["blocks"]["target"][1:3]],
                "goal_xy": _goal_xy(r.layout),
                "stats": {k: round(v, 3) for k, v in r.stats.items()},
            }
            for r in rolls
            if not r.success
        ],
    }


def _goal_xy(lay: dict):
    for kind in ("bowls", "pads", "blocks"):
        if "goal" in lay.get(kind, {}):
            v = lay[kind]["goal"]
            return [round(v[1], 3), round(v[2], 3)]
    return None
