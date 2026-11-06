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
HOLD_STEPS = {"reach": 5, "lift": 10, "pick_place": 5, "push": 5, "stack": 5}
MAX_STEPS = {"reach": 80, "pick_place": 150, "push": 150, "lift": 100, "stack": 150}
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


def _named(spec) -> tuple[str, str | None]:
    """(moved block, goal body) of a spec's core predicate."""
    p = spec.success[0]
    return p.get("block", "target"), p.get("bowl") or p.get("pad") or p.get("base")


def _new_stats(sc, spec) -> dict:
    blk, _ = _named(spec)
    return {
        "blk": blk,
        "start": sc.body_pos(blk)[:2].copy(),
        "s": {
            "target_max_z": 0.0,
            "distractor_lift": 0.0,
            "min_grip_dist": 9.0,
            "target_moved": 0.0,
            "final_goal_dist": 0.0,
        },
    }


def _track(sc, st: dict, spec) -> None:
    """Failure statistics for one step of a composed task (same fields as a single-skill rollout)."""
    from .sim.scene import CUBE_HALF

    blk, goal = _named(spec)
    s = st["s"]
    c = sc.body_pos(blk)
    s["target_max_z"] = max(s["target_max_z"], float(c[2]))
    s["min_grip_dist"] = min(s["min_grip_dist"], float(np.linalg.norm(sc.ee_pos() - c)))
    s["target_moved"] = max(
        s["target_moved"], float(np.linalg.norm(c[:2] - st["start"]))
    )
    for name in sc.layout.blocks:
        if name not in (blk, goal):
            s["distractor_lift"] = max(
                s["distractor_lift"], float(sc.body_pos(name)[2] - CUBE_HALF) - 0.012
            )
    if goal:
        s["final_goal_dist"] = float(np.linalg.norm(c[:2] - sc.body_pos(goal)[:2]))


def _classify(skill: str, st: dict) -> str:
    from .sim.scene import CUBE_HALF

    if st["distractor_lift"] > 0.03:
        return "grabbed the wrong block"
    if skill == "reach":
        return (
            "knocked the block"
            if st["target_moved"] > 0.02
            else "never reached the block"
            if st["min_grip_dist"] > 0.05
            else "missed the hover"
        )
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
    cspecs: list = []
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
        elif cmd == "cinit":  # a composed task: several steps, one scene (compose.py)
            for e in envs.values():
                e["sc"].close()
            envs.clear()
            cspecs = [TaskSpec.from_json(j) for j in arg["specs"]]
            out = {}
            for i, lay_j, look_seed, strength in arg["envs"]:
                lay = Layout.from_json(lay_j)
                sc = Scene(lay)
                sc.reset(lay, Look.sample(np.random.default_rng(look_seed), strength))
                envs[i] = {
                    "sc": sc,
                    "frames": [sc.pose_frame()],
                    "stats": [_new_stats(sc, sp) for sp in cspecs],
                }
                out[i] = (np.stack(sc.render_obs()), sc.joint_pos(), False, 0.0)
            conn.send(out)
        elif cmd == "cstep":
            from .sim.scene import HOME

            out = {}
            for i, (phase, target) in arg["targets"].items():
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
                _track(sc, e["stats"][phase], cspecs[phase])
                ok = check(cspecs[phase], sc)
                home = float(np.linalg.norm(sc.joint_pos()[:5] - HOME))
                out[i] = (np.stack(sc.render_obs()), sc.joint_pos(), ok, home)
            conn.send(out)
        elif cmd == "cfinish":
            out = {}
            for i, e in envs.items():
                sc = e["sc"]
                final = [bool(check(sp, sc)) for sp in cspecs]
                out[i] = {
                    "frames": e["frames"],
                    "final": final,
                    "failures": [
                        "" if ok else _classify(sp.skill, st["s"])
                        for ok, sp, st in zip(final, cspecs, e["stats"], strict=True)
                    ],
                }
                sc.close()
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


# ------------------------------------------------------------------ composed tasks (tier 3)

RETURN_STEPS = 30  # after a step succeeds, its policy keeps running (up to 3 s) until the arm is back home
HOME_NEAR = 0.25  # rad, joint-space distance that counts as home


@dataclass
class ComposedRollout:
    index: int
    seed: int
    layout: dict
    success: bool = False
    steps_ok: list = field(default_factory=list)  # each step's predicate at the end
    held: list = field(
        default_factory=list
    )  # each step's predicate held while that step ran
    failures: list = field(default_factory=list)
    phase_frames: list = field(
        default_factory=list
    )  # first recorded frame of each step
    ticks: int = 0
    frames: list = field(default_factory=list)

    @property
    def failure(self) -> str:
        bad = [f"step {k + 1}: {f}" for k, f in enumerate(self.failures) if f]
        return "; ".join(bad)


def evaluate_composed(
    models: list,
    specs: list,
    layouts: list[dict],
    seed0: int = EVAL_SEED0,
    record: bool = True,
    on_progress: Callable[[dict], None] | None = None,
) -> list[ComposedRollout]:
    """Run the steps' policies one after another in each scene, all scenes in parallel.

    Every motion comes from the step's own trained policy: a step runs until its success check holds (or its
    time runs out), then the same policy keeps running until the arm is back home, as its demonstrations end,
    and the next step's policy takes over from there."""
    import torch

    from .encoders import VIS_DIM, VIS_TOKENS, device, encode_images, encode_text

    dev = device()
    for m in models:
        m.to(dev).eval()
    langs = [encode_text([s.instruction]).to(dev) for s in specs]
    pool = env_pool()
    w = len(pool.conns)
    n = len(layouts)
    rolls = [ComposedRollout(i, seed0 + i, layouts[i]) for i in range(n)]
    init = [{"specs": [s.to_json() for s in specs], "envs": []} for _ in range(w)]
    for i in range(n):
        init[i % w]["envs"].append((i, layouts[i], seed0 + i + 7, 1.0))
    obs = pool._scatter("cinit", init)
    k = models[0].cfg.chunk
    nph = len(specs)
    limit = [MAX_STEPS[s.skill] for s in specs]
    hold = [HOLD_STEPS[s.skill] for s in specs]
    longest = max(limit) + RETURN_STEPS
    preds = np.zeros((n, longest + k, k, 6), np.float32)
    phase = np.zeros(n, int)
    t_in = np.zeros(n, int)
    streak = np.zeros(n, int)
    post = np.full(
        n, -1
    )  # -1 while the step is still working; then ticks spent returning home
    held = np.zeros((n, nph), bool)
    for r in rolls:
        r.phase_frames = [0]
    tick = 0
    t_start = time.time()
    while True:
        idx = [i for i in range(n) if phase[i] < nph]
        if not idx:
            break
        imgs = np.stack([obs[i][0] for i in idx])
        vis = (
            encode_images(imgs.reshape(-1, *imgs.shape[2:]), dev)
            .view(len(idx), 2, VIS_TOKENS, VIS_DIM)
            .to(dev)
        )
        state = torch.from_numpy(np.stack([obs[i][1] for i in idx])).float().to(dev)
        chunk = np.zeros((len(idx), k, 6), np.float32)
        for p in sorted({int(phase[i]) for i in idx}):
            sel = [bi for bi, i in enumerate(idx) if phase[i] == p]
            chunk[sel] = (
                models[p]
                .act(vis[sel], langs[p].expand(len(sel), -1), state[sel])
                .float()
                .cpu()
                .numpy()
            )
        per = [{"targets": {}, "record": record} for _ in range(w)]
        for bi, i in enumerate(idx):
            t = t_in[i]
            preds[i, t] = chunk[bi]
            lo = max(0, t - k + 1)
            cand = np.stack([preds[i, tp, t - tp] for tp in range(lo, t + 1)])
            wt = np.exp(-ENSEMBLE_M * np.arange(len(cand)))
            per[i % w]["targets"][i] = (
                int(phase[i]),
                (wt[:, None] * cand).sum(0) / wt.sum(),
            )
        obs.update(pool._scatter("cstep", per))
        tick += 1
        for i in idx:
            p = phase[i]
            t_in[i] += 1
            if post[i] < 0:
                streak[i] = streak[i] + 1 if obs[i][2] else 0
                if streak[i] >= hold[p]:
                    held[i, p] = True
                    post[i] = 0
                elif t_in[i] >= limit[p]:
                    post[i] = RETURN_STEPS  # out of time: hand over at once
            else:
                post[i] += 1
            if post[i] >= 0 and (obs[i][3] < HOME_NEAR or post[i] >= RETURN_STEPS):
                phase[i] += 1
                t_in[i] = 0
                streak[i] = 0
                post[i] = -1
                preds[i] = 0
                rolls[i].ticks = tick
                if phase[i] < nph:
                    rolls[i].phase_frames.append(1 + 3 * tick)
        if on_progress and (tick % 5 == 0 or not idx):
            on_progress(
                {
                    "tick": tick,
                    "phases": [int((phase == p).sum()) for p in range(nph + 1)],
                    "n": n,
                    "elapsed": round(time.time() - t_start, 1),
                }
            )
    fin = pool._scatter("cfinish", [None] * w)
    for r in rolls:
        f = fin[r.index]
        r.frames = f["frames"]
        r.steps_ok = f["final"]
        r.held = [bool(v) for v in held[r.index]]
        r.failures = f["failures"]
        r.success = all(r.steps_ok)
    return rolls


def summarize_composed(rolls: list[ComposedRollout], nsteps: int) -> dict:
    n = len(rolls)
    s = sum(r.success for r in rolls)
    fails: dict[str, int] = {}
    for r in rolls:
        for k, f in enumerate(r.failures):
            if f:
                key = f"step {k + 1}: {f}"
                fails[key] = fails.get(key, 0) + 1
    return {
        "n": n,
        "succeeded": s,
        "success_rate": round(s / max(n, 1), 3),
        "failures": fails,
        "per_step": [sum(r.steps_ok[k] for r in rolls) for k in range(nsteps)],
    }
