"""FastAPI server: the one-page UI, a JSON API and the server-sent event stream. Agents run in the same loop.

VLA_FORGE_TIMED=1 HF_HUB_OFFLINE=1 scripts/heavy.sh uv run vla-forge     # http://127.0.0.1:8100
"""

from __future__ import annotations

import asyncio
import json
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

import numpy as np
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from uagents_core.contrib.protocols.chat import ChatMessage, TextContent

from .agents import Forge as Agents
from .agents import register_on_agentverse
from .pipeline import FORGE
from .sim.scene import Layout
from .sim.tasks import TaskSpec, sample_layout
from .viz import preview, robot_meshes

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


@app.get("/")
def index():
    return FileResponse(WEB / "index.html")


@app.get("/api/state")
def state():
    assert AGENTS is not None
    return {
        "ready": FORGE.ready,
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
    run = await AGENTS.submit(text)
    return {"run": run.id}


@app.get("/api/runs/{rid}")
def run_summary(rid: str):
    run = FORGE.runs.get(rid) or _404()
    return run.summary()


@app.get("/api/runs/{rid}/scene")
def run_scene(rid: str):
    run = FORGE.runs.get(rid) or _404()
    if not run.layouts:
        raise HTTPException(409, "scene not built yet")
    return preview(Layout.from_json(run.layouts[0]))


@app.get("/api/meshes")
def meshes():
    return robot_meshes()


@app.get("/api/preview")
def idle_preview():
    """A resting scene for the empty page: the arm at home behind a few blocks."""
    rng = np.random.default_rng(3)
    return preview(sample_layout(TaskSpec("pick_place", "red", "white", 2), rng))


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

    uvicorn.run(app, host="127.0.0.1", port=8100, log_level="warning")
