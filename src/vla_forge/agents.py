"""Fetch.ai uAgents: one agent per stage, talking over real uAgents messages inside one Bureau.

    gateway --ChatMessage--> task --TaskReady--> scene --SceneReady--> data --DemosReady--> trainer
    trainer --PolicyTrained--> eval --EvalReport--> critic --(Retrain)--> data   (the self-improvement loop)
                                                   critic --(Publish)--> registry --SkillPublished--> task
    task --ChatMessage--> whoever asked (the web UI's gateway, or ASI:One once registered on Agentverse)

Every published skill becomes its own agent (skill-<name>) that speaks the chat protocol: send it "run" and it
executes its policy in a fresh random scene and answers with the measured outcome.

Agentverse: everything runs locally. When AGENTVERSE_API_KEY and VLA_FORGE_PUBLIC_URL (a public https URL that
forwards to the bureau port) are set, `register_on_agentverse` registers the task agent and each skill agent
with the chat protocol so ASI:One can reach them. No key is created or stored by this code.
"""

from __future__ import annotations

import asyncio
import functools
import json
import os
import secrets
from datetime import UTC, datetime
from uuid import uuid4

from uagents import Agent, Bureau, Context, Model, Protocol
from uagents_core.contrib.protocols.chat import (
    ChatAcknowledgement,
    ChatMessage,
    TextContent,
    chat_protocol_spec,
)

from .pipeline import EVAL_N, FIRST_DEMOS, FIRST_TRAIN, FORGE, Round, Run
from .registry import ROOT, Skill

BUREAU_PORT = int(os.environ.get("VLA_FORGE_AGENT_PORT", "8102"))
SEED_FILE = (
    ROOT / ".agent-seed"
)  # local secret, gitignored: gives every agent a stable address


class TaskReady(Model):
    run_id: str
    spec: str


class SceneReady(Model):
    run_id: str
    eval_layouts: int


class CollectDemos(Model):
    run_id: str
    n: int
    near: str = "[]"  # json list of {"target": [x, y], "goal": [x, y]} the critic wants more practice near
    train_seconds: float = FIRST_TRAIN
    lr: float = 3e-4


class DemosReady(Model):
    run_id: str
    episodes: int
    train_seconds: float
    lr: float


class PolicyTrained(Model):
    run_id: str
    steps: int
    loss: float


class EvalReport(Model):
    run_id: str
    report: str


class Publish(Model):
    run_id: str


class SkillPublished(Model):
    run_id: str
    skill_id: str
    success_rate: float


def _seed(name: str) -> str:
    if not SEED_FILE.exists():
        SEED_FILE.write_text(secrets.token_hex(32))
        SEED_FILE.chmod(0o600)
    return f"{SEED_FILE.read_text().strip()}-{name}"


def _text(msg: ChatMessage) -> str:
    return " ".join(c.text for c in msg.content if isinstance(c, TextContent)).strip()


def _chat(text: str) -> ChatMessage:
    return ChatMessage(
        timestamp=datetime.now(UTC),
        msg_id=uuid4(),
        content=[TextContent(type="text", text=text)],
    )


class Forge:
    """Builds the agents and the bureau on the running event loop."""

    def __init__(self, loop: asyncio.AbstractEventLoop):
        self.loop = loop
        public = os.environ.get("VLA_FORGE_PUBLIC_URL")
        self.endpoint = [f"{public.rstrip('/')}/submit"] if public else None
        self.bureau = Bureau(
            port=BUREAU_PORT, loop=loop, endpoint=self.endpoint, log_level="WARNING"
        )
        self.agents: dict[str, Agent] = {}
        self.pending: dict[str, asyncio.Future] = {}
        self.pending_skill: dict[str, tuple[str, asyncio.Future]] = {}
        self.run_lock = asyncio.Lock()
        self.active = ""
        self._build()

    def finish(self, run: Run) -> None:
        if self.active == run.id and self.run_lock.locked():
            self.active = ""
            self.run_lock.release()

    def guarded(self, agent: str):
        """Stage handlers: a crash marks the run failed and frees the queue instead of hanging it."""

        def wrap(fn):
            @functools.wraps(fn)
            async def inner(ctx: Context, sender: str, msg):
                try:
                    await fn(ctx, sender, msg)
                except Exception as e:
                    run = FORGE.runs.get(getattr(msg, "run_id", "") or self.active)
                    if run is None:
                        raise
                    run.status, run.error = "error", f"{agent}: {type(e).__name__}: {e}"
                    FORGE.emit(run.id, "error", agent, error=run.error)
                    self.finish(run)
                    if run.reply_to:
                        await ctx.send(run.reply_to, _chat(f"{run.id}|error|{run.error}"))

            return inner

        return wrap

    def agent(self, name: str, description: str) -> Agent:
        a = Agent(
            name=name,
            seed=_seed(name),
            loop=self.loop,
            log_level="WARNING",
            enable_agent_inspector=False,
            publish_agent_details=bool(self.endpoint),
            report_events=bool(self.endpoint),
            description=description,
        )
        self.agents[name] = a
        self.bureau.add(a)
        return a

    def log(self, run_id: str, agent: str, text: str, **extra) -> None:
        FORGE.emit(run_id, "log", agent, text=text, **extra)

    def _build(self) -> None:
        gateway = self.agent(
            "gateway", "The web UI's identity: forwards prompts to the task agent."
        )
        task = self.agent(
            "task",
            "Turns 'have a robot do X' into a task spec and a success check (chat protocol).",
        )
        scene = self.agent(
            "scene", "Builds the MuJoCo scene and the held-out evaluation layouts."
        )
        data = self.agent(
            "data", "Generates randomized expert demonstrations with DART noise."
        )
        trainer = self.agent(
            "trainer", "Fine-tunes the VLA from the nearest skill in the library."
        )
        evaler = self.agent(
            "eval",
            "Runs the trained policy in 20 held-out scenes and measures success.",
        )
        critic = self.agent(
            "critic", "Reads the failures and decides the next round, or publishes."
        )
        registry = self.agent(
            "registry", "Publishes each trained skill as its own callable agent."
        )
        a = self.agents

        # gateway: the UI talks to the task agent exactly as ASI:One would, over the chat protocol
        gw_chat = Protocol(spec=chat_protocol_spec)

        @gw_chat.on_message(ChatMessage)
        async def gw_reply(ctx: Context, sender: str, msg: ChatMessage):
            await ctx.send(
                sender,
                ChatAcknowledgement(
                    timestamp=datetime.now(UTC), acknowledged_msg_id=msg.msg_id
                ),
            )
            meta = _text(msg)
            if sender in self.pending_skill:
                _key, fut = self.pending_skill.pop(sender)
            else:
                fut = self.pending.pop(meta.split("|", 1)[0], None)
            if fut and not fut.done():
                fut.set_result(meta)

        @gw_chat.on_message(ChatAcknowledgement)
        async def gw_ack(ctx: Context, sender: str, msg: ChatAcknowledgement):
            pass

        gateway.include(gw_chat)

        # task agent: chat protocol in, TaskReady out; answers the asker when the skill is published
        t_chat = Protocol(spec=chat_protocol_spec)

        @t_chat.on_message(ChatMessage)
        @self.guarded("task")
        async def on_prompt(ctx: Context, sender: str, msg: ChatMessage):
            await ctx.send(
                sender,
                ChatAcknowledgement(
                    timestamp=datetime.now(UTC), acknowledged_msg_id=msg.msg_id
                ),
            )
            text = _text(msg)
            run_id, _, prompt = (
                text.partition("|")
                if text.count("|") == 1 and sender == a["gateway"].address
                else ("", "", text)
            )
            run = FORGE.runs.get(run_id) or FORGE.new_run(prompt)
            run.reply_to = sender
            self.log(
                run.id, "task", f'chat message from {sender[:14]}..: "{run.prompt}"'
            )
            await self.run_lock.acquire()  # one prompt trains at a time; later ones queue here
            self.active = run.id
            spec = await asyncio.to_thread(FORGE.parse, run)
            info = run.parse_info
            if spec is None:
                self.finish(run)
                run.status = "rejected"
                run.error = info.get("reason", "not a tabletop task this robot can do")
                FORGE.emit(run.id, "rejected", "task", reason=run.error)
                await ctx.send(sender, _chat(f"{run.id}|rejected|{run.error}"))
                return
            FORGE.emit(run.id, "task", "task", spec=spec.to_json(), parse=info)
            self.log(
                run.id,
                "task",
                f"parsed with {info.get('model')} in {info.get('seconds', 0):.1f}s: "
                f"{spec.skill} {spec.obj}{' -> ' + spec.target if spec.target else ''}; "
                f"success = {' & '.join(p['op'] for p in spec.success)}",
            )
            await ctx.send(
                a["scene"].address,
                TaskReady(run_id=run.id, spec=json.dumps(spec.to_json())),
            )

        @t_chat.on_message(ChatAcknowledgement)
        async def t_ack(ctx: Context, sender: str, msg: ChatAcknowledgement):
            pass

        task.include(t_chat, publish_manifest=bool(self.endpoint))

        @task.on_message(SkillPublished)
        @self.guarded("task")
        async def on_published(ctx: Context, sender: str, msg: SkillPublished):
            run = FORGE.runs[msg.run_id]
            self.finish(run)
            text = (
                f"{run.id}|published|Trained '{run.spec.instruction}' in {run.seconds:.0f}s: "
                f"{msg.success_rate:.0%} success over {EVAL_N} held-out scenes. Skill agent: skill-{msg.skill_id}"
            )
            self.log(run.id, "task", "replying to the asker over the chat protocol")
            await ctx.send(run.reply_to, _chat(text))

        @scene.on_message(TaskReady)
        @self.guarded("scene")
        async def on_task(ctx: Context, sender: str, msg: TaskReady):
            run = FORGE.runs[msg.run_id]
            info = await asyncio.to_thread(FORGE.scene, run)
            FORGE.emit(run.id, "scene", "scene", **info)
            self.log(
                run.id,
                "scene",
                f"SO-101 + {len(info['layout']['blocks'])} blocks; {EVAL_N} held-out layouts sampled",
            )
            await ctx.send(
                a["data"].address, CollectDemos(run_id=run.id, n=FIRST_DEMOS)
            )

        @data.on_message(CollectDemos)
        @self.guarded("data")
        async def on_collect(ctx: Context, sender: str, msg: CollectDemos):
            run = FORGE.runs[msg.run_id]
            near = json.loads(msg.near)
            self.log(
                run.id,
                "data",
                f"generating {msg.n} expert demos"
                + (f" focused near {len(near)} failure spots" if near else ""),
            )
            st = await asyncio.to_thread(
                FORGE.collect, run, msg.n, near or None, "data"
            )
            run.rounds.append(Round(index=len(run.rounds), demos=st["ok"]))
            self.log(
                run.id,
                "data",
                f"{st['ok']}/{st['n']} demos ok, {st['frames']} frames encoded in {st['seconds']}s",
            )
            await ctx.send(
                a["trainer"].address,
                DemosReady(
                    run_id=run.id,
                    episodes=run.data.episodes,
                    train_seconds=msg.train_seconds,
                    lr=msg.lr,
                ),
            )

        @trainer.on_message(DemosReady)
        @self.guarded("trainer")
        async def on_demos(ctx: Context, sender: str, msg: DemosReady):
            run = FORGE.runs[msg.run_id]
            self.log(
                run.id,
                "trainer",
                f"training on {msg.episodes} episodes (+ replay) for {msg.train_seconds:.0f}s",
            )
            st = await asyncio.to_thread(
                FORGE.train, run, msg.train_seconds, msg.lr, "trainer"
            )
            run.rounds[-1].train = st
            self.log(run.id, "trainer", f"{st['steps']} steps, loss {st['loss']:.3f}")
            await ctx.send(
                a["eval"].address,
                PolicyTrained(run_id=run.id, steps=st["steps"], loss=st["loss"] or 0.0),
            )

        @evaler.on_message(PolicyTrained)
        @self.guarded("eval")
        async def on_trained(ctx: Context, sender: str, msg: PolicyTrained):
            run = FORGE.runs[msg.run_id]
            self.log(
                run.id, "eval", f"rolling out the policy in {EVAL_N} held-out scenes"
            )
            rep = await asyncio.to_thread(FORGE.evaluate, run, "eval")
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
            await ctx.send(
                a["critic"].address, EvalReport(run_id=run.id, report=json.dumps(rep))
            )

        @critic.on_message(EvalReport)
        @self.guarded("critic")
        async def on_report(ctx: Context, sender: str, msg: EvalReport):
            run = FORGE.runs[msg.run_id]
            rep = json.loads(msg.report)
            plan, model = await asyncio.to_thread(FORGE.critic, run, rep)
            run.rounds[-1].plan = {**plan, "model": model}
            FORGE.emit(
                run.id,
                "critic",
                "critic",
                round=len(run.rounds) - 1,
                plan=plan,
                model=model,
            )
            self.log(
                run.id, "critic", f"[{model}] {plan['action']}: {plan['rationale']}"
            )
            if plan["action"] == "retrain":
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
                await ctx.send(
                    a["data"].address,
                    CollectDemos(
                        run_id=run.id,
                        n=plan["extra_demos"],
                        near=json.dumps(near),
                        train_seconds=plan["train_seconds"],
                        lr=plan["lr"],
                    ),
                )
            else:
                await ctx.send(a["registry"].address, Publish(run_id=run.id))

        @registry.on_message(Publish)
        @self.guarded("registry")
        async def on_publish(ctx: Context, sender: str, msg: Publish):
            run = FORGE.runs[msg.run_id]
            skill = await asyncio.to_thread(FORGE.publish, run)
            agent = self.skill_agent(skill)
            FORGE.registry.set_address(skill.id, agent.address)
            FORGE.emit(
                run.id,
                "published",
                "registry",
                skill=skill.id,
                address=agent.address,
                success_rate=skill.success_rate,
                seconds=run.seconds,
                best_round=run.best_round,
            )
            self.log(
                run.id,
                "registry",
                f"published skill-{skill.id} at {agent.address[:18]}.. "
                f"({skill.success_rate:.0%}, round {run.best_round + 1}, {run.seconds:.0f}s total)",
            )
            await ctx.send(
                a["task"].address,
                SkillPublished(
                    run_id=run.id, skill_id=skill.id, success_rate=skill.success_rate
                ),
            )

    def skill_agent(self, skill: Skill) -> Agent:
        name = f"skill-{skill.id}"
        if name in self.agents:
            return self.agents[name]
        ag = self.agent(
            name,
            f"Callable skill: {skill.instruction} ({skill.success_rate:.0%} measured).",
        )
        proto = Protocol(spec=chat_protocol_spec)

        @proto.on_message(ChatMessage)
        async def run_skill(ctx: Context, sender: str, msg: ChatMessage):
            await ctx.send(
                sender,
                ChatAcknowledgement(
                    timestamp=datetime.now(UTC), acknowledged_msg_id=msg.msg_id
                ),
            )
            res = await asyncio.to_thread(run_once, skill)
            await ctx.send(
                sender,
                _chat(
                    f"{skill.instruction}: {'success' if res['success'] else 'failed (' + res['failure'] + ')'}"
                    f" in {res['steps'] / 10:.1f}s of robot time"
                ),
            )

        @proto.on_message(ChatAcknowledgement)
        async def s_ack(ctx: Context, sender: str, msg: ChatAcknowledgement):
            pass

        ag.include(proto, publish_manifest=bool(self.endpoint))
        if self.running:
            ag.setup()
        return ag

    running = False

    async def start(self) -> None:
        for s in FORGE.registry.ordered():
            self.skill_agent(s)
        self.running = True
        await self.bureau.run_async()

    async def submit(self, prompt: str) -> Run:
        run = FORGE.new_run(prompt)
        self.pending[run.id] = self.loop.create_future()
        gw = self.agents["gateway"]
        await gw._build_context().send(self.agents["task"].address, _chat(f"{run.id}|{prompt}"))
        return run


def run_once(skill: Skill) -> dict:
    """One rollout of a published skill in a fresh random scene (what a skill agent does when called)."""
    import numpy as np

    from .encoders import device
    from .policy import VLA
    from .rollout import evaluate

    model, _ = VLA.load(ROOT / skill.model, device())
    seed = int(np.random.default_rng().integers(3_000_000, 4_000_000))
    with FORGE.compute:
        r = evaluate(model, skill.task(), 1, seed0=seed, record=False)[0]
    return {"success": r.success, "failure": r.failure, "steps": r.steps}


def register_on_agentverse(forge: Forge) -> list[str]:
    """Register the task agent and every skill agent on Agentverse (needs his key and a public URL)."""
    key = os.environ.get("AGENTVERSE_API_KEY")
    if not key or not forge.endpoint:
        return []
    from uagents_core.utils.registration import (
        RegistrationRequestCredentials,
        register_chat_agent,
    )

    done = []
    for name, ag in forge.agents.items():
        if name != "task" and not name.startswith("skill-"):
            continue
        ok = register_chat_agent(
            name=f"vla-forge-{name}",
            endpoint=forge.endpoint[0],
            active=True,
            credentials=RegistrationRequestCredentials(
                agentverse_api_key=key, agent_seed_phrase=_seed(name)
            ),
            description=ag._description if hasattr(ag, "_description") else None,
        )
        if ok:
            done.append(name)
    return done
