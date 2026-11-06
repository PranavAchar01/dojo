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

from . import llm, tiles
from . import router as router_mod
from .compose import Step, step_layout
from .compose import sample_layout as compose_layout
from .datagen import demo_job, pool
from .encoders import device, encode_images, encode_text
from .learn import Demos, train
from .policy import VLA
from .registry import BASE_PATH, ROBOT, ROOT, Registry, Skill
from .rollout import (
    env_pool,
    evaluate,
    evaluate_composed,
    summarize,
    summarize_composed,
)
from .sim.scene import Layout, Scene
from .sim.tasks import TIER, TaskSpec, sample_layout
from .viz import export_scene, preview

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
    # tier 3: a composed task holds its steps; each missing step is a child run of the normal pipeline
    steps: list[Step] = field(default_factory=list)
    plan: list[dict] = field(
        default_factory=list
    )  # per step: instruction, skill_id, source, child, ...
    parent: str = ""
    step_index: int = -1
    # library: how this prompt was asked for, and what the router decided
    mode: str = "auto"  # auto | train (never reuse) | scratch (from the base) | remix (from from_skill)
    from_skill: str = ""
    route: dict = field(default_factory=dict)

    @property
    def composite(self) -> bool:
        return bool(self.steps)

    @property
    def tier(self) -> int:
        if self.steps:
            return 3
        return TIER.get(self.spec.skill, 1) if self.spec else 0

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
            "tier": self.tier,
            "steps": [st.to_json() for st in self.steps],
            "plan": self.plan,
            "parent": self.parent,
            "step_index": self.step_index,
            "mode": self.mode,
            "from_skill": self.from_skill,
            "route": self.route,
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
        self.tests: dict[str, dict] = {}  # test id -> {skill, rolls, instructions}

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

    def new_run(self, prompt: str, mode: str = "auto", from_skill: str = "") -> Run:
        rid = (
            time.strftime("%H%M%S")
            + "-"
            + hashlib.sha1(f"{prompt}{time.time()}".encode()).hexdigest()[:4]
        )
        run = Run(rid, prompt, mode=mode, from_skill=from_skill)
        self.runs[rid] = run
        self.order.append(rid)
        return run

    # stages (blocking; called from agent handlers via asyncio.to_thread)
    def parse(self, run: Run) -> TaskSpec | list[Step] | None:
        run.t0 = time.time()
        run.status = "running"
        out, info = llm.parse_task(run.prompt)
        run.parse_info = info
        if isinstance(out, list):
            run.steps = out
        else:
            run.spec = out
        return out

    def route(self, run: Run) -> router_mod.Route:
        """What the prompt needs (reuse / train / chain) or why it is refused; see router.py."""
        parsed = run.steps if run.composite else run.spec
        r = router_mod.route(
            run.prompt, parsed, run.parse_info, self.registry, run.mode, run.from_skill
        )
        run.route = r.to_json()
        return r

    # tier 3
    def plan(self, run: Run) -> list[dict]:
        """Which published skill agent does each step, and which steps must be trained first."""
        seed = int(hashlib.sha1(run.id.encode()).hexdigest()[:8], 16)
        run.layouts = [compose_layout(run.steps, np.random.default_rng(seed)).to_json()]
        run.plan = []
        for k, st in enumerate(run.steps):
            sk = self.registry.exact(st.instruction)
            run.plan.append(
                {
                    "step": k,
                    "instruction": st.instruction,
                    "skill": st.skill,
                    "tier": TIER[st.skill],
                    "skill_id": sk.id if sk else "",
                    "address": sk.agent_address if sk else "",
                    "success_rate": sk.success_rate if sk else None,
                    "source": "reuse" if sk else "train",
                    "child": "",
                    "seconds": 0.0,
                }
            )
        return run.plan

    def child_run(self, parent: Run, k: int) -> Run:
        """A missing step becomes its own one-skill run (with its own 3-minute budget)."""
        st = parent.steps[k]
        child = self.new_run(st.instruction)
        child.spec = st.single()
        child.parent, child.step_index = parent.id, k
        child.status, child.t0 = "running", time.time()
        child.parse_info = {"model": "planner", "kind": "single"}
        parent.plan[k]["child"] = child.id
        return child

    def compose_eval(self, run: Run, agent: str) -> dict:
        """Run the composed task with the step skills' own policies in EVAL_N held-out composed scenes."""
        seed = int(hashlib.sha1(run.id.encode()).hexdigest()[:8], 16)
        rng = np.random.default_rng(seed)
        run.layouts = [compose_layout(run.steps, rng).to_json() for _ in range(EVAL_N)]
        models = []
        for p in run.plan:
            sk = self.registry.skills[p["skill_id"]]
            m, _ = VLA.load(ROOT / sk.model, device())
            models.append(m)
        specs = [st.in_scene() for st in run.steps]

        def prog(p):
            self.emit(run.id, "compose_progress", agent, **p)

        t0 = time.time()
        with self.compute:
            rolls = evaluate_composed(
                models, specs, run.layouts, record=True, on_progress=prog
            )
        s = summarize_composed(rolls, len(specs))
        s["seconds"] = round(time.time() - t0, 1)
        run.rollouts[0] = rolls
        run.rounds = [Round(0, eval=s, successes=[r.success for r in rolls])]
        run.best_round = 0
        return s

    def publish_composite(self, run: Run) -> Skill:
        slug = re.sub(r"[^a-z0-9]+", "-", run.prompt.lower()).strip("-")[:48]
        sid = f"{slug}-{run.id[-4:]}"
        ev = run.rounds[0].eval
        ok_roll = next((r for r in run.rollouts[0] if r.success), run.rollouts[0][0])
        skill = Skill(
            id=sid,
            instruction=run.prompt,
            spec={},
            model="",
            parent=None,
            success_rate=ev["success_rate"],
            n_eval=ev["n"],
            seconds=round(run.elapsed(), 1),
            thumb=ok_roll.frames[-1] if ok_roll.frames else [],
            tier=3,
            steps=[
                {"instruction": p["instruction"], "skill_id": p["skill_id"]}
                for p in run.plan
            ],
            prompt=run.prompt,
            robot=ROBOT,
            origin="chain",
            train_seconds=round(
                sum(
                    self.registry.skills[p["skill_id"]].train_seconds
                    for p in run.plan
                    if p["source"] == "train"
                ),
                1,
            ),
        )
        self.registry.add(skill)
        self._save_tile(skill, run.rollouts[0], [st.instruction for st in run.steps])
        run.skill_id = sid
        run.seconds = round(run.elapsed(), 1)
        run.status = "published"
        return skill

    def _save_tile(
        self, skill: Skill, rolls: list, instructions=None, source="held-out eval"
    ) -> None:
        """The tile's thumbnail and replay, written next to the registry, then the 'saved' moment."""
        self.emit("", "saving", "registry", skill=skill.id)
        fields = tiles.save_assets(skill, rolls, source, instructions)
        sk = self.registry.update(skill.id, **fields)
        self.emit("", "saved", "registry", tile=tiles.tile(sk))

    def _layout(self, run: Run, rng: np.random.Generator) -> Layout:
        """A training or held-out scene. A step trained for a composed task sees that task's tables."""
        assert run.spec is not None
        if run.parent:
            return step_layout(self.runs[run.parent].steps, run.step_index, rng)
        return sample_layout(run.spec, rng)

    def scene(self, run: Run) -> dict:
        assert run.spec is not None
        seed = int(hashlib.sha1(run.id.encode()).hexdigest()[:8], 16)
        rng = np.random.default_rng(seed)
        run.layouts = [self._layout(run, rng).to_json() for _ in range(EVAL_N)]
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
            if run.parent:
                j["layout"] = self._layout(
                    run, np.random.default_rng(base_seed + i)
                ).to_json()
            elif near:
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
            src = self.registry.skills.get(run.from_skill)
            if run.mode == "remix" and src is not None and src.model:
                sid, path, sim = src.id, ROOT / src.model, 1.0
                reason = f"remix: fine-tuning from '{src.name or src.instruction}'"
            elif run.mode == "scratch":
                sid, path, sim = None, BASE_PATH, 0.0
                reason = "from scratch: starting from the multi-task base policy, not the nearest tile"
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

        self.emit(
            run.id,
            "train_start",
            agent,
            round=len(run.rounds) - 1,
            seconds=round(seconds, 1),
        )

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
            tier=TIER[run.spec.skill],
            prompt=run.prompt,
            robot=ROBOT,
            origin="step"
            if run.parent
            else run.mode
            if run.mode in ("scratch", "remix")
            else "prompt",
            train_seconds=round(
                sum(r.train.get("seconds", 0.0) for r in run.rounds), 1
            ),
        )
        self.registry.add(skill)
        self._save_tile(skill, run.rollouts[run.best_round])
        run.skill_id = sid
        run.seconds = round(run.elapsed(), 1)
        run.status = "published"
        return skill

    # library: Test a tile on scenes nobody has seen
    def test_skill(self, sid: str, tid: str, n: int) -> dict:
        sk = self.registry.skills[sid]
        seed0 = int(np.random.default_rng().integers(2_000_000, 3_000_000))
        rng = np.random.default_rng(seed0)

        def prog(p):
            self.emit(f"test-{tid}", "test_progress", "eval", skill=sid, **p)

        t0 = time.time()
        if sk.composite:
            subs = [self.registry.skills[st["skill_id"]] for st in sk.steps]
            steps = [Step(s.task().skill, s.task().obj, s.task().target) for s in subs]
            lays = [compose_layout(steps, rng).to_json() for _ in range(n)]
            self.emit(
                f"test-{tid}",
                "test_start",
                "eval",
                skill=sid,
                n=n,
                scene=preview(Layout.from_json(lays[0])),
            )
            models = [VLA.load(ROOT / s.model, device())[0] for s in subs]
            with self.compute:
                rolls = evaluate_composed(
                    models,
                    [st.in_scene() for st in steps],
                    lays,
                    seed0=seed0,
                    on_progress=prog,
                )
            instructions = [st.instruction for st in steps]
        else:
            spec = sk.task()
            lays = [sample_layout(spec, rng).to_json() for _ in range(n)]
            self.emit(
                f"test-{tid}",
                "test_start",
                "eval",
                skill=sid,
                n=n,
                scene=preview(Layout.from_json(lays[0])),
            )
            model, _ = VLA.load(ROOT / sk.model, device())
            with self.compute:
                rolls = evaluate(
                    model, spec, n, seed0=seed0, layouts=lays, on_progress=prog
                )
            instructions = None
        res = {
            "t": round(time.time(), 1),
            "n": n,
            "succeeded": int(sum(r.success for r in rolls)),
            "seconds": round(time.time() - t0, 1),
            "seed0": seed0,
        }
        self.tests[tid] = {"skill": sid, "rolls": rolls, "instructions": instructions}
        sk = self.registry.add_test(sid, res)
        if not sk.replay:  # tiles from before the library: their first test gives them a picture and a replay
            self._save_tile(sk, rolls, instructions, source="test")
            sk = self.registry.skills[sid]
        self.emit(
            f"test-{tid}",
            "test_done",
            "eval",
            skill=sid,
            **res,
            successes=[r.success for r in rolls],
            tile=tiles.tile(sk),
        )
        return res

    def test_payload(self, tid: str, index: int) -> dict:
        t = self.tests[tid]
        return {"round": 0, **tiles.payload(t["rolls"][index], t["instructions"])}

    def rollout_payload(self, run: Run, rnd: int, index: int) -> dict:
        r = run.rollouts[rnd][index]
        if run.composite:
            return {
                "round": rnd,
                "index": index,
                "success": r.success,
                "failure": r.failure,
                "steps_ok": r.steps_ok,
                "held": r.held,
                "phase_frames": r.phase_frames,
                "instructions": [st.instruction for st in run.steps],
                "fps": 30,
                "scene": export_scene(Layout.from_json(r.layout)),
                "frames": r.frames,
            }
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
