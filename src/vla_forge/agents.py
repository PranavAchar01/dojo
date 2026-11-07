"""The pipeline that turns a sentence into a trained skill, in process, with Muse Spark and SubQ thinking side by side.

    task -> router -> scene -> data -> trainer -> eval -> critic -> (data again | registry)
    composed tasks: task -> router -> planner -> (a child run per missing step) -> executor -> registry

Each stage is a plain function call (heavy ones on a thread); the stage names are kept because the page shows them.
Two moments ask both models at once (lanes.both), each on what it is built for (docs/meta-subq.md):
  - task: Muse Spark parses the sentence; SubQ reads the whole skill library and picks the saved skill that already
    does it, if any. The router still decides reuse; SubQ's pick is shown next to it.
  - critic: Muse Spark writes the next round's plan; SubQ reads every rollout of every round of this run and gives the
    probability that another round raises success. The plan decides; SubQ's number is shown with whether they agree.
The optimizer's rounds (optimizer.py) are where both proposals are trained and scored head to head.

Events are the ones the page has always read (log, thought, task, route, scene, eval, critic, published, reused,
rejected, plan, child, offer, error), plus llm_call from lanes.py.
"""

from __future__ import annotations

import asyncio
import json

from . import lanes, llama, subq, tiles
from .pipeline import EVAL_N, FIRST_DEMOS, FIRST_TRAIN, FORGE, Round, Run
from .registry import ROOT, Skill

DECISION_WORDS = {
    "reuse": "reuse a tile",
    "train": "train a new skill",
    "chain": "chain skills",
    "too_complex": "too complex",
    "not_feasible": "not feasible",
}
# stage -> (what it does, which model lane it asks: "meta", "subq", "both" or "")
STAGES = {
    "task": ("Turns the sentence into a task spec and a success check.", "both"),
    "router": (
        "Reuses a library tile, trains, chains, or refuses with the reason.",
        "",
    ),
    "scene": ("Builds the MuJoCo scene and the held-out evaluation layouts.", ""),
    "data": ("Generates randomized expert demonstrations with DART noise.", ""),
    "trainer": ("Fine-tunes the VLA from the nearest skill in the library.", ""),
    "eval": ("Runs the policy in 20 held-out scenes and measures success.", ""),
    "critic": ("Reads the failures and plans the next round, or publishes.", "both"),
    "registry": ("Saves each trained skill as a library tile.", ""),
    "planner": (
        "Splits a multi-step task into skills and trains the missing ones.",
        "",
    ),
    "executor": (
        "Runs a composed task with each step's policy in 20 held-out scenes.",
        "",
    ),
    "optimizer": (
        "Both models propose a change each round; both are scored in simulation.",
        "both",
    ),
}
EDGES = [
    ["task", "router"],
    ["router", "scene"],
    ["router", "planner"],
    ["scene", "data"],
    ["data", "trainer"],
    ["trainer", "eval"],
    ["eval", "critic"],
    ["critic", "data"],
    ["critic", "registry"],
    ["planner", "executor"],
    ["executor", "registry"],
]


def thought(run_id: str, agent: str, text: str, lane: str = "", **extra) -> None:
    """One line for the Thinking panel; `lane` names the model that produced it (meta, subq) or "" for plain code."""
    FORGE.emit(run_id, "thought", agent, text=text, llm=bool(lane), lane=lane, **extra)


class StageError(Exception):
    def __init__(self, stage: str, exc: Exception):
        super().__init__(f"{stage}: {type(exc).__name__}: {exc}")
        self.stage = stage


def library_state(run: Run) -> tuple[dict, dict]:
    """(state, options) for SubQ's library question: every saved skill, newest first, plus `none`."""
    skills = FORGE.registry.ordered()[:254]
    state = {
        "request": run.prompt,
        "robot": run.robot,
        "library": [
            {
                "id": s.id,
                "instruction": s.instruction,
                "name": s.name,
                "task": s.spec,  # raw: a composite skill has steps, not one task
                "robot": s.robot,
                "success_rate": s.success_rate,
                "n_eval": s.n_eval,
                "composite": bool(s.composite),
            }
            for s in skills
        ],
    }
    options = {f"skill_{s.id}": s.instruction[:200] for s in skills}
    options["none"] = "no saved skill does this request on this robot"
    return state, options


def subq_library(run: Run) -> dict:
    state, options = library_state(run)
    ans = subq.decide(
        state,
        {
            "match": {
                "type": "choice",
                "instructions": "Which saved skill already does exactly this request (same motion, same block, same "
                "target, same robot)? Answer none when no skill matches exactly.",
                "criteria": options,
            }
        },
        "library match",
    )
    pick, p = subq.choice(ans["match"])
    return {"pick": pick.removeprefix("skill_") if pick != "none" else "", "p": p}


def subq_critic(run: Run, report: dict) -> float:
    """P(another round raises held-out success), from every rollout of every round of this run."""
    state = {
        "task": run.spec.to_json() if run.spec else {},
        "latest_evaluation": report,
        "rounds": [
            {
                "round": r.index,
                "demos": r.demos,
                "train": r.train,
                "eval": r.eval,
                "plan": r.plan,
                "scenes": [
                    {
                        "scene": i,
                        "success": bool(ro.success),
                        "failure": ro.failure,
                    }
                    for i, ro in enumerate(run.rollouts.get(r.index, []))
                ],
            }
            for r in run.rounds
        ],
        "seconds_left": round(run.left()),
    }
    ans = subq.decide(
        state,
        {
            "more": {
                "type": "noul",
                "instructions": "Will one more round of extra demonstrations and training raise success on the "
                "same held-out scenes, given the time left?",
            }
        },
        "critic: another round?",
    )
    return float(ans["more"]["noul"])


class Forge:
    """Runs prompts through the stages; one training run at a time, later prompts queue."""

    def __init__(self, loop: asyncio.AbstractEventLoop):
        self.loop = loop
        self.pending: dict[str, asyncio.Future] = {}
        self.run_lock = asyncio.Lock()
        self.active = ""

    async def start(self) -> None:
        """Nothing to start: the stages are function calls."""

    # ------------------------------------------------------------------ helpers
    def log(self, run_id: str, agent: str, text: str, lane: str = "", **extra) -> None:
        FORGE.emit(run_id, "log", agent, text=text, **extra)
        thought(run_id, agent, text, lane)

    def _done(self, run: Run, text: str) -> None:
        fut = self.pending.pop(run.id, None)
        if fut and not fut.done():
            fut.set_result(text)

    async def _stage(self, name: str, fn, *a):
        try:
            return await asyncio.to_thread(fn, *a)
        except Exception as e:
            raise StageError(name, e) from e

    # ------------------------------------------------------------------ entry
    async def submit(
        self,
        prompt: str,
        mode: str = "auto",
        from_skill: str = "",
        station: dict | None = None,
        scenery: str = "",
    ) -> Run:
        run = FORGE.new_run(prompt, mode, from_skill, station, scenery)
        self.pending[run.id] = self.loop.create_future()
        self.loop.create_task(self._go(run))
        return run

    async def _go(self, run: Run) -> None:
        # app.js starts following a run on this first log line
        self.log(run.id, "task", f'chat message from the page: "{run.prompt}"')
        async with self.run_lock:
            self.active = run.id
            try:
                await self._task(run)
            except StageError as e:
                self._fail(run, str(e))
            finally:
                self.active = ""

    def _fail(self, run: Run, error: str) -> None:
        run.status, run.error = "error", error
        FORGE.emit(run.id, "error", error.split(":", 1)[0], error=error)
        if run.parent:
            parent = FORGE.runs[run.parent]
            parent.status, parent.error = "error", f"step {run.step_index + 1}: {error}"
            FORGE.emit(parent.id, "error", "planner", error=parent.error)
            self._done(parent, f"error|{parent.error}")
        self._done(run, f"error|{error}")

    # ------------------------------------------------------------------ task + router
    async def _task(self, run: Run) -> None:
        with lanes.about(run.id, "parse the task"):
            if run.station:  # the line planner already wrote the task; no model call
                spec, match = FORGE.parse(run), None
            else:
                spec, match = await asyncio.to_thread(
                    lanes.both, lambda: FORGE.parse(run), lambda: subq_library(run)
                )
                if isinstance(spec, Exception):
                    raise StageError("task", spec)
        info = run.parse_info
        note = f". {info['note']}" if info.get("note") else ""
        lane = "meta" if str(info.get("model", "")).startswith(llama.MODEL_NAME) else ""
        if run.composite:
            FORGE.emit(
                run.id,
                "task",
                "task",
                steps=[st.to_json() for st in run.steps],
                parse=info,
                tier=3,
            )
            self.log(
                run.id,
                "task",
                f"parsed by {info.get('model')} in {info.get('seconds', 0):.1f}s: a tier 3 task in "
                f"{len(run.steps)} steps{note}",
                lane,
            )
        elif spec is None:
            self.log(
                run.id,
                "task",
                f"parsed by {info.get('model')}: no task this arm can learn{note}",
                lane,
            )
        else:
            FORGE.emit(
                run.id, "task", "task", spec=spec.to_json(), parse=info, tier=run.tier
            )
            self.log(
                run.id,
                "task",
                f"parsed by {info.get('model')} in {info.get('seconds', 0):.1f}s: "
                f"{spec.skill} {spec.obj}{' -> ' + spec.target if spec.target else ''}; "
                f"success = {' & '.join(p['op'] for p in spec.success)}{note}",
                lane,
            )
        if isinstance(match, Exception):
            self.log(run.id, "task", f"SubQ did not answer: {match}")
            match = None
        elif match is not None:
            sk = FORGE.registry.skills.get(match["pick"])
            self.log(
                run.id,
                "task",
                f"SubQ read all {len(FORGE.registry.skills)} saved skills: "
                + (
                    f"'{sk.instruction}' already does this (p {match['p']:.2f})"
                    if sk
                    else f"none does exactly this (p {match['p']:.2f})"
                ),
                "subq",
            )
        await self._route(run, match)

    async def _route(self, run: Run, match: dict | None) -> None:
        r = await self._stage("router", FORGE.route, run)
        FORGE.emit(run.id, "route", "router", **r.to_json())
        self.log(
            run.id,
            "router",
            f"{DECISION_WORDS[r.decision]}: {r.reason}",
            checks=[c.__dict__ for c in r.checks],
        )
        if match is not None:
            same = (
                (match["pick"] == r.skill_id)
                if r.decision == "reuse"
                else not match["pick"]
            )
            self.log(
                run.id,
                "router",
                f"SubQ {'agrees' if same else 'disagrees'}: it picked "
                + (f"skill {match['pick']}" if match["pick"] else "no saved skill"),
                "subq",
                subq_agrees=same,
            )
        if r.refused:
            run.status, run.error = "rejected", r.reason
            run.seconds = round(run.elapsed(), 1)
            FORGE.emit(
                run.id,
                "rejected",
                "router",
                reason=r.reason,
                decision=r.decision,
                suggestion=r.suggestion,
            )
            self._done(run, f"rejected|{r.reason}")
            return
        if r.decision == "reuse":
            run.status, run.skill_id = "reused", r.skill_id
            run.seconds = round(run.elapsed(), 1)
            sk = FORGE.registry.skills[r.skill_id]
            FORGE.emit(run.id, "reused", "router", skill=sk.id, tile=tiles.tile(sk))
            self._done(run, f"reused|{sk.id}")
            return
        if run.composite:
            await self._compose(run)
            return
        await self._train(run)

    # ------------------------------------------------------------------ one skill
    async def _train(self, run: Run) -> Skill | None:
        info = await self._stage("scene", FORGE.scene, run)
        FORGE.emit(run.id, "scene", "scene", **info)
        self.log(
            run.id,
            "scene",
            f"SO-101 + {len(info['layout']['blocks'])} blocks; {EVAL_N} held-out layouts sampled",
        )
        n, near, secs, lr = FIRST_DEMOS, [], FIRST_TRAIN, 3e-4
        while True:
            self.log(
                run.id,
                "data",
                f"generating {n} expert demos"
                + (f" focused near {len(near)} failure spots" if near else ""),
            )
            st = await self._stage("data", FORGE.collect, run, n, near or None, "data")
            run.rounds.append(Round(index=len(run.rounds), demos=st["ok"]))
            self.log(
                run.id,
                "data",
                f"{st['ok']}/{st['n']} demos ok, {st['frames']} frames encoded in {st['seconds']}s",
            )
            self.log(
                run.id,
                "trainer",
                f"training on {run.data.episodes} episodes (+ replay) for {secs:.0f}s",
            )
            tr = await self._stage("trainer", FORGE.train, run, secs, lr, "trainer")
            run.rounds[-1].train = tr
            self.log(
                run.id, "trainer", f"{tr['steps']} steps, loss {tr['loss'] or 0:.3f}"
            )
            self.log(
                run.id, "eval", f"rolling out the policy in {EVAL_N} held-out scenes"
            )
            rep = await self._stage("eval", FORGE.evaluate, run, "eval")
            rnd = run.rounds[-1]
            rnd.eval = {
                k: rep[k]
                for k in ("n", "succeeded", "success_rate", "failures", "seconds")
            }
            rnd.successes = [r.success for r in run.rollouts[rnd.index]]
            FORGE.emit(
                run.id,
                "eval",
                "eval",
                round=rnd.index,
                **rnd.eval,
                successes=rnd.successes,
            )
            self.log(
                run.id,
                "eval",
                f"{rep['succeeded']}/{rep['n']} succeeded in {rep['seconds']}s"
                + (
                    f"; failures: {', '.join(f'{v} {k}' for k, v in rep['failures'].items())}"
                    if rep["failures"]
                    else ""
                ),
            )
            plan = await self._critic(run, rep)
            if plan["action"] != "retrain":
                break
            near = (
                [
                    {
                        "target": f["target_xy"],
                        **({"goal": f["goal_xy"]} if f["goal_xy"] else {}),
                    }
                    for f in rep["failed_starts"]
                ]
                if plan["focus_failures"]
                else []
            )
            n, secs, lr = plan["extra_demos"], plan["train_seconds"], plan["lr"]
        return await self._publish(run)

    async def _critic(self, run: Run, rep: dict) -> dict:
        with lanes.about(run.id, f"critic, round {len(run.rounds)}"):
            res, p_more = await asyncio.to_thread(
                lanes.both,
                lambda: FORGE.critic(run, rep),
                lambda: subq_critic(run, rep),
            )
        if isinstance(res, Exception):
            raise StageError("critic", res)
        plan, model = res
        sq = None if isinstance(p_more, Exception) else round(p_more, 3)
        run.rounds[-1].plan = {**plan, "model": model, "subq_p_more": sq}
        FORGE.emit(
            run.id,
            "critic",
            "critic",
            round=len(run.rounds) - 1,
            plan=plan,
            model=model,
            subq_p_more=sq,
        )
        lane = "meta" if model == llama.MODEL_NAME else ""
        self.log(
            run.id, "critic", f"[{model}] {plan['action']}: {plan['rationale']}", lane
        )
        if sq is None:
            self.log(run.id, "critic", f"SubQ did not answer: {p_more}")
        else:
            agree = (sq >= 0.5) == (plan["action"] == "retrain")
            self.log(
                run.id,
                "critic",
                f"SubQ read all {sum(len(v) for v in run.rollouts.values())} rollouts of this run: "
                f"P(another round helps) = {sq:.2f}, {'agrees' if agree else 'disagrees'} with the plan",
                "subq",
            )
        return plan

    async def _publish(self, run: Run) -> Skill:
        skill = await self._stage(
            "registry", FORGE.publish_composite if run.composite else FORGE.publish, run
        )
        FORGE.emit(
            run.id,
            "published",
            "registry",
            skill=skill.id,
            address="",
            success_rate=skill.success_rate,
            seconds=run.seconds,
            best_round=run.best_round,
        )
        self.log(
            run.id,
            "registry",
            f"saved skill-{skill.id} "
            + (
                f"({skill.success_rate:.0%} end to end, {len(skill.steps)} skills in a row, {run.seconds:.0f}s total)"
                if run.composite
                else f"({skill.success_rate:.0%}, round {run.best_round + 1}, {run.seconds:.0f}s total)"
            ),
        )
        if not run.parent:
            self._done(run, f"published|{skill.id}")
        return skill

    # ------------------------------------------------------------------ composed tasks
    def _offer(self, run: Run, step: int, sk: Skill) -> None:
        run.plan[step].update(
            offered=True, skill_id=sk.id, address="", success_rate=sk.success_rate
        )
        FORGE.emit(run.id, "offer", "planner", step=step, plan=run.plan)
        self.log(
            run.id,
            "planner",
            f"step {step + 1}: skill-{sk.id} ({sk.success_rate:.0%} measured on {sk.n_eval} scenes)",
        )

    async def _compose(self, run: Run) -> None:
        plan = FORGE.plan(run)
        FORGE.emit(run.id, "plan", "planner", plan=plan)
        FORGE.emit(run.id, "scene", "planner", blocks=len(run.steps))
        reuse = sum(p["source"] == "reuse" for p in plan)
        self.log(
            run.id,
            "planner",
            f"{len(plan)} steps: {reuse} saved skill{'s' if reuse != 1 else ''} to use, {len(plan) - reuse} to train",
        )
        for p in plan:
            if p["skill_id"]:
                self._offer(run, p["step"], FORGE.registry.skills[p["skill_id"]])
        for p in plan:
            if p["skill_id"]:
                continue
            child = FORGE.child_run(run, p["step"])
            FORGE.emit(
                run.id,
                "child",
                "planner",
                step=p["step"],
                child=child.id,
                plan=run.plan,
            )
            self.log(
                run.id,
                "planner",
                f"step {p['step'] + 1}: no saved skill for '{p['instruction']}' yet; training it now",
            )
            try:
                sk = await self._train(child)
            except StageError as e:
                self._fail(child, str(e))
                return
            p.update(seconds=child.seconds)
            self._offer(run, p["step"], sk)
        self.log(
            run.id,
            "executor",
            f"running the {len(run.steps)} skill policies in turn in {EVAL_N} held-out scenes",
        )
        rep = await self._stage("executor", FORGE.compose_eval, run, "executor")
        FORGE.emit(
            run.id,
            "eval",
            "executor",
            round=0,
            **rep,
            successes=run.rounds[0].successes,
        )
        per = ", ".join(
            f"step {k + 1} {v}/{rep['n']}" for k, v in enumerate(rep["per_step"])
        )
        self.log(
            run.id,
            "executor",
            f"{rep['succeeded']}/{rep['n']} done end to end in {rep['seconds']}s ({per})",
        )
        await self._publish(run)

    # ------------------------------------------------------------------ for the page
    def models(self) -> dict:
        """Who answers on each lane right now, in plain words (live_backends may probe Meta once per process)."""
        live = llama.live_backends()
        backend = (
            f"meta:{llama.MODEL}"
            if "meta" in live
            else f"openai:{llama.OPENAI_MODEL}"
            if "openai" in live
            else "stub"
        )
        return {
            "meta": {**llama.meta_status(), "backend": backend},
            "subq": subq.status(),
        }

    def graph(self) -> dict:
        nodes = [
            {"id": n, "description": d, "llm": bool(lane), "lane": lane, "skill": False}
            for n, (d, lane) in STAGES.items()
        ]
        return {
            "nodes": nodes,
            "edges": EDGES,
            "models": self.models(),
            "backends": llama.live_backends(),
        }


def run_once(skill: Skill) -> dict:
    """One rollout of a saved skill in a fresh random scene."""
    import numpy as np

    from .encoders import device
    from .policy import VLA
    from .rollout import evaluate

    seed = int(np.random.default_rng().integers(3_000_000, 4_000_000))
    if skill.composite:
        from .compose import Step
        from .compose import sample_layout as compose_layout
        from .rollout import evaluate_composed

        subs = [FORGE.registry.skills[st["skill_id"]] for st in skill.steps]
        steps = [
            Step(
                **{
                    k: v
                    for k, v in s.task().to_json().items()
                    if k in ("skill", "obj", "target")
                }
            )
            for s in subs
        ]
        lay = compose_layout(steps, np.random.default_rng(seed)).to_json()
        models = [VLA.load(ROOT / s.model, device())[0] for s in subs]
        with FORGE.compute:
            r = evaluate_composed(
                models,
                [st.in_scene() for st in steps],
                [lay],
                seed0=seed,
                record=False,
                robot=skill.robot,
            )[0]
        return {"success": r.success, "failure": r.failure, "steps": r.ticks}
    model, _ = VLA.load(ROOT / skill.model, device())
    with FORGE.compute:
        r = evaluate(
            model, skill.task(), 1, seed0=seed, record=False, robot=skill.robot
        )[0]
    return {"success": r.success, "failure": r.failure, "steps": r.steps}


def describe_run_reply(skill: Skill, res: dict) -> str:
    return (
        f"{skill.instruction}: {'success' if res['success'] else 'failed (' + res['failure'] + ')'}"
        f" in {res['steps'] / 10:.1f}s of robot time"
    )


__all__ = ["Forge", "describe_run_reply", "json", "run_once"]
