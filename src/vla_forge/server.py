"""FastAPI server: the one-page UI, a JSON API and the server-sent event stream. Agents run in the same loop.

VLA_FORGE_TIMED=1 HF_HUB_OFFLINE=1 scripts/heavy.sh uv run vla-forge     # http://127.0.0.1:8100
"""

from __future__ import annotations

import asyncio
import json
import os
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

import numpy as np
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from uagents_core.contrib.protocols.chat import ChatMessage, TextContent

from . import tiles
from .agents import Forge as Agents
from .agents import register_on_agentverse
from .pipeline import FORGE
from .sim import robots as R
from .sim.scene import Layout
from .sim.tasks import TaskSpec, sample_layout
from .viz import part_view, preview, robot_meshes, robot_view

WEB = Path(__file__).resolve().parents[2] / "web"
AGENTS: Agents | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global AGENTS
    loop = asyncio.get_running_loop()
    FORGE.loop = loop
    AGENTS = Agents(loop)
    task = loop.create_task(AGENTS.start())
    await asyncio.to_thread(FORGE.warm)
    registered = await asyncio.to_thread(register_on_agentverse, AGENTS)
    FORGE.emit(
        "",
        "agents",
        "",
        agents={n: a.address for n, a in AGENTS.agents.items()},
        agentverse=registered,
    )
    yield
    task.cancel()


app = FastAPI(lifespan=lifespan)


class Prompt(BaseModel):
    prompt: str
    mode: str = "auto"  # auto | train (never reuse) | scratch (from the base) | remix (from from_skill)
    from_skill: str = ""


class TestReq(BaseModel):
    n: int = 5


class Rename(BaseModel):
    name: str


@app.get("/")
def index():
    return FileResponse(WEB / "index.html")


@app.get("/api/state")
def state():
    assert AGENTS is not None
    return {
        "ready": FORGE.ready,
        "robot": FORGE.robot,
        "active": AGENTS.active,
        "runs": [FORGE.runs[r].summary() for r in FORGE.order[-60:]],
        "skills": [s.__dict__ for s in FORGE.registry.ordered()],
        "agents": {n: a.address for n, a in AGENTS.agents.items()},
    }


@app.post("/api/prompt")
async def prompt(p: Prompt):
    assert AGENTS is not None
    text = " ".join(p.prompt.split())[:300]
    if not text:
        raise HTTPException(400, "empty prompt")
    if p.mode not in ("auto", "train", "scratch", "remix"):
        raise HTTPException(400, "mode is auto, train, scratch or remix")
    if p.mode == "remix" and p.from_skill not in FORGE.registry.skills:
        raise HTTPException(404, "no such tile to remix")
    run = await AGENTS.submit(text, p.mode, p.from_skill if p.mode == "remix" else "")
    return {"run": run.id}


# ---------------------------------------------------------------- library


@app.get("/api/library")
def library():
    return {
        "tiles": [tiles.tile(s) for s in FORGE.registry.ordered()],
        "active": FORGE.runs[AGENTS.active].summary()
        if AGENTS and AGENTS.active in FORGE.runs
        else None,
    }


def _skill(sid: str):
    return FORGE.registry.skills.get(sid) or _404()


@app.get("/api/skills/{sid}/thumb.png")
def thumb(sid: str):
    sk = _skill(sid)
    if not sk.thumb_png:
        _404()
    return FileResponse(
        tiles.ROOT / sk.thumb_png,
        media_type="image/png",
        headers={"Cache-Control": "no-cache"},
    )


@app.get("/api/skills/{sid}/replay/{index}")
def replay(sid: str, index: int):
    sk = _skill(sid)
    if not sk.replay:
        _404()
    rolls = tiles.load_replay(sk)
    if not 0 <= index < len(rolls):
        _404()
    summary = [{"success": r["success"], "failure": r["failure"]} for r in rolls]
    return Response(
        json.dumps({**rolls[index], "all": summary, "source": sk.replay_source}),
        media_type="application/json",
    )


@app.patch("/api/skills/{sid}")
def rename(sid: str, r: Rename):
    _skill(sid)
    name = " ".join(r.name.split())[:80]
    return tiles.tile(FORGE.registry.update(sid, name=name))


@app.post("/api/skills/{sid}/test")
async def test_skill(sid: str, r: TestReq):
    """Run the tile's policy on fresh random scenes (never its training or held-out seeds); progress over SSE."""
    _skill(sid)
    n = min(max(r.n, 1), 20)
    tid = uuid4().hex[:8]
    task = asyncio.get_running_loop().run_in_executor(
        None, FORGE.test_skill, sid, tid, n
    )
    task.add_done_callback(
        lambda f: (
            f.exception()
            and FORGE.emit(
                f"test-{tid}", "test_error", "eval", skill=sid, error=str(f.exception())
            )
        )
    )
    return {"test": tid, "n": n}


@app.get("/api/tests/{tid}/rollout/{index}")
def test_rollout(tid: str, index: int):
    t = FORGE.tests.get(tid) or _404()
    if not 0 <= index < len(t["rolls"]):
        _404()
    return FORGE.test_payload(tid, index)


@app.get("/api/runs/{rid}")
def run_summary(rid: str):
    run = FORGE.runs.get(rid) or _404()
    return run.summary()


@app.get("/api/runs/{rid}/scene")
def run_scene(rid: str):
    run = FORGE.runs.get(rid) or _404()
    if not run.layouts:
        raise HTTPException(409, "scene not built yet")
    return preview(Layout.from_json(run.layouts[0]), run.robot)


def _robot_id(robot: str) -> str:
    try:
        R.parse_id(robot)
    except (KeyError, ValueError):
        raise HTTPException(404, f"unknown robot {robot!r}") from None
    return robot


@app.get("/api/meshes")
def meshes(robot: str = "so101"):
    return robot_meshes(_robot_id(robot))


@app.get("/api/preview")
def idle_preview(robot: str | None = None):
    """A resting scene for the empty page: the active arm at home behind a few blocks."""
    rng = np.random.default_rng(3)
    return preview(
        sample_layout(TaskSpec("pick_place", "red", "white", 2), rng),
        _robot_id(robot or FORGE.robot),
    )


# ---- Build tab: the robot customizer ----


class UseRobot(BaseModel):
    robot: str


@app.get("/api/robots")
def robots():
    """The part catalog (arms, wrists, end effectors) with sources and licences, plus the active robot."""
    return {**R.catalog(), "active": FORGE.robot}


@app.get("/api/robots/{robot}/report")
async def robot_report(robot: str):
    """Assemble the robot and run every check: MJCF compiles, joint limits, stroke, reach of the table."""
    rid = _robot_id(robot)

    def run() -> dict:
        ok, why, feas = R.usable(rid)
        return {**R.report(rid), "skills": feas, "usable": ok, "usable_reason": why}

    return await asyncio.to_thread(run)


@app.get("/api/robots/{robot}/view")
async def robot_preview(robot: str):
    return await asyncio.to_thread(robot_view, _robot_id(robot))


@app.get("/api/parts/{slot}/{part}/view")
async def part_preview(slot: str, part: str):
    table = {"arm": R.ARMS, "wrist": R.WRISTS, "effector": R.EFFECTORS}.get(slot)
    if table is None or part not in table:
        _404()
    return await asyncio.to_thread(part_view, slot, part)


@app.post("/api/robot/use")
async def use_robot(u: UseRobot):
    """Make this robot the embodiment the next skill trains on. It must compose and pass every check that is not
    task-specific, and fit at least one skill; the router refuses the skills it cannot do, with the reason."""
    ok, why, _ = await asyncio.to_thread(R.usable, _robot_id(u.robot))
    if not ok:
        raise HTTPException(409, why)
    FORGE.robot = u.robot
    return {"robot": FORGE.robot}


@app.get("/api/runs/{rid}/rollout/{rnd}/{index}")
def rollout(rid: str, rnd: int, index: int):
    run = FORGE.runs.get(rid) or _404()
    if rnd not in run.rollouts or not 0 <= index < len(run.rollouts[rnd]):
        _404()
    return FORGE.rollout_payload(run, rnd, index)


@app.post("/api/skills/{sid}/run")
async def run_skill(sid: str):
    """Call a published skill's own agent over the chat protocol, as any other agent would."""
    assert AGENTS is not None
    name = f"skill-{sid}"
    if name not in AGENTS.agents:
        _404()
    fut = AGENTS.loop.create_future()
    key = f"skill:{uuid4().hex[:8]}"
    AGENTS.pending_skill[AGENTS.agents[name].address] = (key, fut)
    gw = AGENTS.agents["gateway"]
    await gw._build_context().send(
        AGENTS.agents[name].address,
        ChatMessage(
            timestamp=datetime.now(UTC),
            msg_id=uuid4(),
            content=[TextContent(type="text", text="run")],
        ),
    )
    try:
        reply = await asyncio.wait_for(fut, 120)
    except TimeoutError:
        raise HTTPException(504, "skill agent did not answer") from None
    return {"reply": reply}


@app.post("/api/capture")
async def capture(req: Request):
    """The page's own MediaRecorder upload (demo recording, ?capture=1); saved under demo/."""
    out = WEB.parent / "demo" / f"vla-forge-{datetime.now():%Y%m%d-%H%M%S}.webm"
    out.parent.mkdir(exist_ok=True)
    out.write_bytes(await req.body())
    return {"path": str(out), "bytes": out.stat().st_size}


@app.get("/api/events")
async def events():
    q: asyncio.Queue = asyncio.Queue()
    FORGE.subscribers.add(q)

    async def gen():
        try:
            yield "retry: 1500\n\n"
            while True:
                try:
                    ev = await asyncio.wait_for(q.get(), 15)
                    yield f"data: {json.dumps(ev)}\n\n"
                except TimeoutError:
                    yield ": ping\n\n"
        finally:
            FORGE.subscribers.discard(q)

    return StreamingResponse(
        gen(), media_type="text/event-stream", headers={"Cache-Control": "no-cache"}
    )


def _404():
    raise HTTPException(404, "not found")


app.mount("/static", StaticFiles(directory=WEB), name="static")


def main():
    import uvicorn

    uvicorn.run(
        app,
        host="127.0.0.1",
        port=int(os.environ.get("VLA_FORGE_PORT", "8100")),
        log_level="warning",
    )
