"""FastAPI server: the one-page UI, a JSON API and the server-sent event stream. The pipeline runs in the same loop.

VLA_FORGE_TIMED=1 HF_HUB_OFFLINE=1 scripts/heavy.sh uv run vla-forge     # http://127.0.0.1:8100
"""

from __future__ import annotations

import asyncio
import gzip
import json
import os
import sys
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from uuid import uuid4

import numpy as np
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from . import (
    agents_api,
    assets_api,
    factory,
    optimizer_api,
    robotcache,
    thumbs,
    tiles,
)
from .agents import Forge as Agents
from .agents import describe_run_reply, run_once
from .pipeline import FORGE
from .sim import robots as R
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
    agents_api.HOLD["agents"] = AGENTS
    if await asyncio.to_thread(
        thumbs.warm_needed
    ):  # fill the Robot tab's thumbnails and caches in a quiet process
        await asyncio.create_subprocess_exec(
            "nice", "-n", "19", sys.executable, "-m", "vla_forge.thumbs"
        )
    await asyncio.to_thread(FORGE.warm)
    FORGE.emit("", "agents", "", models=await asyncio.to_thread(AGENTS.models))
    yield


app = FastAPI(lifespan=lifespan)
app.include_router(agents_api.router)
app.include_router(optimizer_api.router)
app.include_router(assets_api.router)


class Prompt(BaseModel):
    prompt: str
    mode: str = "auto"  # auto | train (never reuse) | scratch (from the base) | remix (from from_skill)
    from_skill: str = ""


class TestReq(BaseModel):
    n: int = 5


class Rename(BaseModel):
    name: str


@app.get("/")
def landing():
    return FileResponse(WEB / "landing.html")


@app.get("/app")
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


def _blob(data: bytes, key: str, request: Request) -> Response:
    """A stored gzip JSON payload: revalidated by its parts hash, so a reload costs a 304."""
    headers = {
        "ETag": f'"{key}"',
        "Cache-Control": "no-cache",
        "Vary": "Accept-Encoding",
    }
    if request.headers.get("if-none-match") == headers["ETag"]:
        return Response(status_code=304, headers=headers)
    if "gzip" in request.headers.get("accept-encoding", ""):
        headers["Content-Encoding"] = "gzip"
    else:
        data = gzip.decompress(data)
    return Response(data, media_type="application/json", headers=headers)


@app.get("/api/robots/{robot}/report")
async def robot_report(robot: str, request: Request):
    """Assemble the robot and run every check: MJCF compiles, joint limits, stroke, reach of the table."""
    data, key = await asyncio.to_thread(robotcache.report_blob, _robot_id(robot))
    return _blob(data, key, request)


@app.get("/api/robots/{robot}/view")
async def robot_preview(robot: str, request: Request):
    data, key = await asyncio.to_thread(robotcache.view_blob, _robot_id(robot))
    return _blob(data, key, request)


@app.get("/api/parts/{slot}/{part}/view")
async def part_preview(slot: str, part: str, request: Request):
    table = {"arm": R.ARMS, "wrist": R.WRISTS, "effector": R.EFFECTORS}.get(slot)
    if table is None or part not in table:
        _404()
    if slot == "wrist" or (slot == "effector" and part == "native"):
        data, key = await asyncio.to_thread(
            robotcache.view_blob, R.part_robot_id(slot, part)
        )
    else:
        data, key = await asyncio.to_thread(robotcache.part_blob, slot, part)
    return _blob(data, key, request)


@app.get("/api/thumbs/{slot}/{part}")
def part_thumb(slot: str, part: str, arm: str | None = None):
    """The card image of a part (a wrist or the native jaw as it sits on `arm`), rendered once and kept on disk."""
    table = {"arm": R.ARMS, "wrist": R.WRISTS, "effector": R.EFFECTORS}.get(slot)
    if table is None or part not in table or (arm is not None and arm not in R.ARMS):
        _404()
    on_arm = slot == "wrist" or (slot == "effector" and part == "native")
    f = thumbs.DIR / thumbs.thumb_name(slot, part, (arm or "so101") if on_arm else None)
    if not f.exists():
        _404()
    return FileResponse(
        f, media_type="image/webp", headers={"Cache-Control": "public, max-age=86400"}
    )


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
    """Run a saved skill once in a fresh random scene and say what happened."""
    sk = _skill(sid)
    res = await asyncio.to_thread(run_once, sk)
    return {"reply": describe_run_reply(sk, res)}


@app.post("/api/capture")
async def capture(req: Request):
    """The page's own MediaRecorder upload (demo recording, ?capture=1); saved under demo/."""
    out = WEB.parent / "demo" / f"vla-forge-{datetime.now():%Y%m%d-%H%M%S}.webm"
    out.parent.mkdir(exist_ok=True)
    out.write_bytes(await req.body())
    return {"path": str(out), "bytes": out.stat().st_size}


# ---------------------------------------------------------------- factory lines

LINES: dict[str, factory.Line] = factory.load_all()
LINE_SCENES: dict[str, list] = {}


class LinePrompt(BaseModel):
    prompt: str


class LineRun(BaseModel):
    n_parts: int = 12
    seed: int = 7


class StationEdit(BaseModel):
    controller: str | None = None  # vla | scripted
    robot: str | None = None


def _line(lid: str) -> factory.Line:
    return LINES.get(lid) or _404()


def _line_json(ln: factory.Line, full: bool = False) -> dict:
    """The line plus each station's skill record (held-out success and how it was trained)."""
    d = ln.to_json(full)
    for st in d["stations"]:
        sk = FORGE.registry.skills.get(st["skill_id"]) if st["skill_id"] else None
        st["skill"] = sk and {
            "success_rate": sk.success_rate,
            "n_eval": sk.n_eval,
            "seconds": sk.seconds,
        }
    return d


def _line_emit(ln: factory.Line, kind: str, **data) -> None:
    FORGE.emit(f"line-{ln.id}", f"line_{kind}", "line", line=ln.id, **data)


@app.get("/api/lines")
def lines():
    return [_line_json(ln) for ln in sorted(LINES.values(), key=lambda x: -x.created)]


@app.post("/api/line/plan")
async def line_plan(p: LinePrompt):
    """Words to stations: the planner splits the line, the router checks every station."""
    text = " ".join(p.prompt.split())[:600]
    if not text:
        raise HTTPException(400, "empty description")
    ln = await asyncio.to_thread(factory.plan_line, text)
    await asyncio.to_thread(factory.route_line, ln, FORGE.registry)
    if not ln.stations:
        ln.status = "refused"
    LINES[ln.id] = ln
    factory.save(ln)
    return _line_json(ln)


@app.patch("/api/line/{lid}/stations/{index}")
async def line_station(lid: str, index: int, e: StationEdit):
    ln = _line(lid)
    if ln.status in ("building", "running") or not 0 <= index < len(ln.stations):
        raise HTTPException(409, "line is busy or no such station")
    st = ln.stations[index]
    if e.controller in ("vla", "scripted"):
        st.controller = e.controller
    if e.robot in factory.ROBOTS:
        st.robot = e.robot
    st.skill_id = ""
    r = await asyncio.to_thread(factory.route_station, st, FORGE.registry)
    st.route = r.to_json()
    st.status = {"reuse": "ready", "train": "train", "scripted": "scripted"}.get(
        r.decision, "refused"
    )
    st.skill_id = r.skill_id if r.decision == "reuse" else ""
    LINE_SCENES.pop(lid, None)
    factory.save(ln)
    return _line_json(ln)


async def _run_line(ln: factory.Line, n_parts: int, seed: int) -> None:
    ln.status = "running"
    _line_emit(ln, "status", status=ln.status)

    def progress(p: dict) -> None:
        _line_emit(ln, "station_done", **p)

    try:
        ln.result = await asyncio.to_thread(
            factory.run_line, ln, FORGE.registry, FORGE.compute, n_parts, seed, progress
        )
        ln.status = "done"
    except Exception as exc:  # noqa: BLE001  (surfaced to the page; the line stays inspectable)
        ln.status, ln.result = "failed", {"error": str(exc)}
    factory.save(ln)
    _line_emit(
        ln,
        "status",
        status=ln.status,
        result={k: v for k, v in ln.result.items() if k != "playback"},
    )


def _with_skill(kind: str, d: dict) -> dict:
    if kind == "station" and d["station"].get("skill_id"):
        sk = FORGE.registry.skills.get(d["station"]["skill_id"])
        d["station"]["skill"] = sk and {
            "success_rate": sk.success_rate,
            "n_eval": sk.n_eval,
            "seconds": sk.seconds,
        }
    return d


@app.post("/api/line/{lid}/build")
async def line_build(lid: str, r: LineRun):
    """Train every station that needs a skill (one agent run each), then run the whole line."""
    assert AGENTS is not None
    ln = _line(lid)
    if ln.status in ("building", "running"):
        raise HTTPException(409, "line is busy")
    if any(s.status == "refused" for s in ln.stations) or not ln.stations:
        raise HTTPException(409, "a station is not feasible; change it first")
    n = min(max(r.n_parts, 2), 30)

    async def go() -> None:
        _line_emit(ln, "status", status="building")
        await factory.build_line(
            ln,
            AGENTS,
            FORGE,
            lambda kind, **d: _line_emit(ln, kind, **_with_skill(kind, d)),
        )
        factory.save(ln)
        if any(s.status != "ready" and s.controller == "vla" for s in ln.stations):
            ln.status = "failed"
            factory.save(ln)
            _line_emit(ln, "status", status=ln.status)
            return
        await _run_line(ln, n, r.seed)

    asyncio.get_running_loop().create_task(go())
    return {"line": ln.id}


@app.post("/api/line/{lid}/run")
async def line_run(lid: str, r: LineRun):
    ln = _line(lid)
    if ln.status in ("building", "running"):
        raise HTTPException(409, "line is busy")
    if any(s.controller == "vla" and not s.skill_id for s in ln.stations):
        raise HTTPException(409, "train the stations first")
    asyncio.get_running_loop().create_task(
        _run_line(ln, min(max(r.n_parts, 2), 30), r.seed)
    )
    return {"line": ln.id}


@app.get("/api/line/{lid}")
def line_get(lid: str):
    return _line_json(_line(lid), full=True)


@app.get("/api/line/{lid}/scene")
async def line_scene(lid: str):
    ln = _line(lid)
    if lid not in LINE_SCENES:
        LINE_SCENES[lid] = await asyncio.to_thread(factory.cell_scenes, ln)
    return {
        "cells": LINE_SCENES[lid],
        "pitch": factory.L.EXIT_Y - factory.L.ENTRY_Y,
        "block_colors": factory.block_colors(),
    }


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
        host=os.environ.get(
            "VLA_FORGE_HOST", "127.0.0.1"
        ),  # the mini's Tailscale IP to reach it from the Air
        port=int(os.environ.get("VLA_FORGE_PORT", "8100")),
        log_level="warning",
    )
