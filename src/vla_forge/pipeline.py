"""The work behind each agent: one prompt becomes a trained, measured, published skill.

Agents (agents.py) decide what happens next and pass uAgents messages; the heavy stages here run in worker
threads so the agents' event loop stays responsive. Every stage reports to the event bus that the web UI
streams over SSE.
"""

from __future__ import annotations

import asyncio
import hashlib
import re
import threading
import time
from concurrent.futures import as_completed
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch

from . import llm
from .datagen import demo_job, pool
from .encoders import device, encode_images, encode_text
from .learn import Demos, train
from .policy import VLA
from .registry import BASE_PATH, ROOT, Registry, Skill
from .rollout import env_pool, evaluate, summarize
from .sim.scene import Layout, Scene
from .sim.tasks import TaskSpec, sample_layout
from .viz import export_scene

DEADLINE = 172.0  # seconds from prompt to published skill (the brief's budget is 180)
TARGET = 0.8
EVAL_N = 20
FIRST_DEMOS = 60
FIRST_TRAIN = 55.0
DART_NOISE = 0.05
REPLAY_EPISODES = 400
EVAL_SECONDS = 15.0  # measured: 20 rollouts in 12-14 s


@dataclass
class Round:
    index: int
    demos: int = 0
    train: dict = field(default_factory=dict)
    eval: dict = field(default_factory=dict)
    plan: dict = field(default_factory=dict)
    successes: list[bool] = field(default_factory=list)
    loss: list[list[float]] = field(
        default_factory=list
    )  # [seconds, loss ema] as logged


@dataclass
class Run:
    id: str
    prompt: str
    t0: float = field(default_factory=time.time)
    status: str = "queued"
    spec: TaskSpec | None = None
    parse_info: dict = field(default_factory=dict)
    layouts: list[dict] = field(default_factory=list)
    data: Demos = field(default_factory=Demos)
    model: VLA | None = None
    init_from: dict = field(default_factory=dict)
    rounds: list[Round] = field(default_factory=list)
    rollouts: dict[int, list] = field(default_factory=dict)  # round -> list[Rollout]
    best_round: int = -1
    best_state: dict | None = None
    skill_id: str = ""
    seconds: float = 0.0
    reply_to: str = ""  # chat sender to answer when the skill is published
    error: str = ""

    def elapsed(self) -> float:
        return time.time() - self.t0

    def left(self) -> float:
        return DEADLINE - self.elapsed()

    def summary(self) -> dict:
        return {
            "id": self.id,
            "prompt": self.prompt,
            "status": self.status,
            "elapsed": round(self.elapsed() if not self.seconds else self.seconds, 1),
            "spec": self.spec.to_json() if self.spec else None,
            "parse": self.parse_info,
            "init_from": self.init_from,
            "rounds": [
                {
                    "index": r.index,
                    "demos": r.demos,
                    "train": r.train,
                    "eval": r.eval,
                    "plan": r.plan,
                    "successes": r.successes,
                    "loss": r.loss,
                }
                for r in self.rounds
            ],
            "best_round": self.best_round,
            "skill_id": self.skill_id,
            "error": self.error,
        }


class Forge:
    """Shared state: the registry, warmed pools and models, the runs, and the event bus."""

    def __init__(self):
        self.registry = Registry()
        self.runs: dict[str, Run] = {}
        self.order: list[str] = []
        self.loop: asyncio.AbstractEventLoop | None = None
        self.subscribers: set[asyncio.Queue] = set()
        self.replay: Demos | None = None
        self.compute = threading.Lock()  # one run's heavy stages at a time
        self.ready = False

    # events
    def emit(self, run_id: str, kind: str, agent: str = "", **data) -> None:
        ev = {
            "t": round(time.time(), 3),
            "run": run_id,
            "kind": kind,
            "agent": agent,
            **data,
        }
        if self.loop is None:
            return
        self.loop.call_soon_threadsafe(self._fanout, ev)

    def _fanout(self, ev: dict) -> None:
        for q in list(self.subscribers):
            if q.qsize() < 2000:
                q.put_nowait(ev)

    # start-up
    def warm(self) -> None:
        t0 = time.time()
        dev = device()
        encode_images(np.zeros((2, 168, 168, 3), np.uint8), dev)
        encode_text(["warm up"])
        pool()
        env_pool()
        self.replay = load_replay(REPLAY_EPISODES)
        if self.replay.episodes:
            self.replay.pack(dev)
        export_scene(Layout())
        self.ready = True
        self.emit(
            "", "ready", seconds=round(time.time() - t0, 1), replay=self.replay.episodes
        )

    def new_run(self, prompt: str) -> Run:
        rid = (
            time.strftime("%H%M%S")
            + "-"
            + hashlib.sha1(f"{prompt}{time.time()}".encode()).hexdigest()[:4]
        )
        run = Run(rid, prompt)
        self.runs[rid] = run
        self.order.append(rid)
        return run

    # stages (blocking; called from agent handlers via asyncio.to_thread)
    def parse(self, run: Run) -> TaskSpec | None:
        run.t0 = time.time()
        run.status = "running"
        spec, info = llm.parse_task(run.prompt)
        run.parse_info = info
        run.spec = spec
        return spec

    def scene(self, run: Run) -> dict:
        assert run.spec is not None
        seed = int(hashlib.sha1(run.id.encode()).hexdigest()[:8], 16)
        rng = np.random.default_rng(seed)
        run.layouts = [sample_layout(run.spec, rng).to_json() for _ in range(EVAL_N)]
        lay = Layout.from_json(run.layouts[0])
        sc = Scene(lay)
        sc.reset(lay)
        frame = sc.pose_frame()
        sc.close()
        return {"layout": run.layouts[0], "frame": frame}

    def collect(self, run: Run, n: int, near: list[dict] | None, agent: str) -> dict:
        """Expert demos in fresh randomized scenes; each episode is encoded as soon as it lands."""
        assert run.spec is not None
        t0 = time.time()
        out = ROOT / "runs" / run.id / f"round{len(run.rounds)}"
        base_seed = 10_000 * (len(run.rounds) + 1) + len(run.data.vis)
        jobs = []
        for i in range(n):
            j = {
                "spec": run.spec.to_json(),
                "seed": base_seed + i,
                "out": str(out / f"ep{i:03d}.npz"),
                "noise": DART_NOISE,
            }
            if near:
                j["near"] = near[i % len(near)]
            jobs.append(j)
        p = pool()
        with self.compute:
            return self._collect(run, n, jobs, p, t0, agent)

    def _collect(self, run: Run, n: int, jobs: list, p, t0: float, agent: str) -> dict:
        assert run.spec is not None
        futs = [p.submit(demo_job, j) for j in jobs]
        ok = frames = 0
        for done, f in enumerate(as_completed(futs), 1):
            r = f.result()
            if r["ok"]:
                ok += 1
                frames += run.data.add_episode(r["path"], run.spec.instruction)
                Path(r["path"]).unlink()
            if done % 4 == 0 or done == n:
                self.emit(
                    run.id,
                    "demos",
                    agent,
                    done=done,
                    ok=ok,
                    n=n,
                    frames=frames,
                    seconds=round(time.time() - t0, 1),
                )
        return {
            "ok": ok,
            "n": n,
            "frames": frames,
            "seconds": round(time.time() - t0, 1),
        }

    def train(self, run: Run, seconds: float, lr: float, agent: str) -> dict:
        assert run.spec is not None
        if run.model is None:
            sid, path, sim, reason = self.registry.nearest(run.spec)
            if not path.exists():
                path = BASE_PATH
                sid, reason = None, "starting from the multi-task base policy"
            if path.exists():
                run.model, _meta = VLA.load(path, device())
            else:  # no base trained yet on this machine: learn from scratch
                sid, reason = None, "no base policy yet, training from scratch"
                run.model = VLA(norm=run.data.norm()).to(device())
            run.init_from = {
                "skill": sid,
                "similarity": round(sim, 3),
                "reason": reason,
                "path": str(path.name),
            }
            self.emit(run.id, "init", agent, **run.init_from)

        def log(e):
            run.rounds[-1].loss.append([e["t"], round(e["ema"], 5)])
            self.emit(run.id, "loss", agent, round=len(run.rounds) - 1, **e)

        with self.compute:
            st = train(
                run.model,
                run.data,
                seconds,
                lr=lr,
                on_log=log,
                replay=self.replay,
                log_every=0.4,
            )
        return st

    def evaluate(self, run: Run, agent: str) -> dict:
        assert run.spec is not None and run.model is not None
        rnd = len(run.rounds) - 1

        def prog(p):
            self.emit(run.id, "eval_progress", agent, round=rnd, **p)

        t0 = time.time()
        with self.compute:
            rolls = evaluate(
                run.model,
                run.spec,
                EVAL_N,
                layouts=run.layouts,
                record=True,
                on_progress=prog,
            )
        s = summarize(rolls)
        s["seconds"] = round(time.time() - t0, 1)
        run.rollouts[rnd] = rolls
        if s["success_rate"] > (
            run.rounds[run.best_round].eval["success_rate"]
            if run.best_round >= 0
            else -1
        ):
            run.best_round = rnd
            run.best_state = {
                k: v.detach().cpu().clone() for k, v in run.model.state_dict().items()
            }
        return s

    def critic(self, run: Run, report: dict) -> tuple[dict, str]:
        hist = [
            {
                "round": r.index,
                "demos": r.demos,
                "success_rate": r.eval.get("success_rate"),
                "train_seconds": r.train.get("seconds"),
                "plan": r.plan,
            }
            for r in run.rounds
        ]
        settings = {
            "demos_so_far": run.data.episodes,
            "eval_rollouts": EVAL_N,
            "seconds_per_demo_batch_of_30": 9,
            "seconds_per_eval": EVAL_SECONDS,
        }
        plan, model = llm.critic_plan(report, hist, settings, run.left(), TARGET)
        # the plan must fit the deadline: demos (~0.3 s each over the pool) + training + one more evaluation
        budget = run.left() - EVAL_SECONDS - 0.3 * plan["extra_demos"] - 6
        if plan["action"] == "retrain" and budget < 8:
            plan["action"] = "publish"
            plan["rationale"] += (
                " (no time left in the 3-minute budget for another round)"
            )
        plan["train_seconds"] = float(min(plan["train_seconds"], max(budget, 8)))
        return plan, model

    def publish(self, run: Run) -> Skill:
        assert (
            run.spec is not None
            and run.model is not None
            and run.best_state is not None
        )
        run.model.load_state_dict(run.best_state)
        slug = re.sub(r"[^a-z0-9]+", "-", run.spec.instruction.lower()).strip("-")
        sid = f"{slug}-{run.id[-4:]}"
        path = Path("skills") / f"{sid}.pt"
        best = run.rounds[run.best_round]
        run.model.save(
            ROOT / path, {"instruction": run.spec.instruction, "run": run.id}
        )
        ok_roll = next(
            (r for r in run.rollouts[run.best_round] if r.success),
            run.rollouts[run.best_round][0],
        )
        skill = Skill(
            id=sid,
            instruction=run.spec.instruction,
            spec=run.spec.to_json(),
            model=str(path),
            parent=run.init_from.get("skill"),
            success_rate=best.eval["success_rate"],
            n_eval=best.eval["n"],
            seconds=round(run.elapsed(), 1),
            thumb=ok_roll.frames[-1] if ok_roll.frames else [],
        )
        self.registry.add(skill)
        run.skill_id = sid
        run.seconds = round(run.elapsed(), 1)
        run.status = "published"
        return skill

    def rollout_payload(self, run: Run, rnd: int, index: int) -> dict:
        r = run.rollouts[rnd][index]
        return {
            "round": rnd,
            "index": index,
            "success": r.success,
            "failure": r.failure,
            "steps": r.steps,
            "fps": 30,
            "scene": export_scene(Layout.from_json(r.layout)),
            "frames": r.frames,
        }


def load_replay(n: int) -> Demos:
    """A slice of the base policy's own training data, mixed into every fine-tune so old skills stay."""
    d = Demos()
    shards = sorted((ROOT / "models" / "base_data").glob("shard_*.pt"))
    rng = np.random.default_rng(0)
    rng.shuffle(shards)
    for p in shards:
        if d.episodes >= n:
            break
        try:
            z = torch.load(p, weights_only=False)
        except RuntimeError:  # a shard still being written by a running pretrain
            continue
        for v, s, a, x, ins in zip(
            z["vis"], z["states"], z["actions"], z["aux"], z["instr"], strict=True
        ):
            d.add_encoded(v, s, a, x, ins)
            if d.episodes >= n:
                break
    return d


FORGE = Forge()
