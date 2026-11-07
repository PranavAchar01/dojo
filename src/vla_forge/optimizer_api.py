"""HTTP surface of the optimizer (web/optimizer.js is its only client). Mounted by server.py with one line."""

from __future__ import annotations

import asyncio

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse, Response
from pydantic import BaseModel, Field

from . import optimizer, optimizer_zoom
from .pipeline import FORGE

router = APIRouter(prefix="/api/optimizer")
LIVE: dict[str, optimizer.Opt] = {}


class Start(BaseModel):
    skill: str
    rounds: int = Field(5, ge=0, le=60)  # 0: perpetual, until stopped
    train_seconds: float = Field(optimizer.DEFAULTS["train_seconds"], ge=15, le=90)


@router.get("/skills")
def skills():
    """Skills the optimizer can help: single-arm, below 90% on their published check."""
    return [
        {
            "id": s.id,
            "name": s.name or s.instruction,
            "instruction": s.instruction,
            "success_rate": s.success_rate,
            "n_eval": s.n_eval,
        }
        for s in optimizer.eligible(FORGE.registry.skills)
    ]


@router.post("/start")
async def start(r: Start):
    sk = FORGE.registry.skills.get(r.skill)
    if sk is None or sk not in optimizer.eligible(FORGE.registry.skills):
        raise HTTPException(404, "no such skill with headroom")
    if any(o.status in ("starting", "running") for o in LIVE.values()):
        raise HTTPException(409, "an optimizer run is already going")
    opt = optimizer.new_opt(sk, r.rounds)
    _launch(sk, opt, r.train_seconds, False)
    return {"id": opt.id}


def _launch(sk, opt, train_seconds: float, resume: bool) -> None:
    LIVE[opt.id] = opt

    def go() -> None:
        optimizer.run(
            sk,
            opt.n_rounds,
            opt,
            FORGE.replay,
            FORGE.compute,
            FORGE.emit,
            train_seconds,
            "random",
            resume=resume,
        )

    task = asyncio.get_running_loop().run_in_executor(None, go)
    task.add_done_callback(lambda f: f.exception())


@router.post("/runs/{oid}/stop")
def stop(oid: str):
    """End a running (perpetual) run after the round it is in; it can be resumed."""
    if oid not in LIVE or LIVE[oid].status not in ("starting", "running"):
        raise HTTPException(404, "that run is not going")
    (optimizer.RUNS / f"opt-{oid}" / "STOP").write_text("stop")
    return {"id": oid, "stopping": True}


@router.post("/runs/{oid}/test")
def test_now(oid: str):
    """Ask a running run to score TEST on the current best at the end of this round (counted as a peek)."""
    if oid not in LIVE or LIVE[oid].status not in ("starting", "running"):
        raise HTTPException(404, "that run is not going")
    (optimizer.RUNS / f"opt-{oid}" / "TEST_REQUEST").write_text("test")
    return {"id": oid, "requested": True}


@router.post("/runs/{oid}/resume")
async def resume(oid: str):
    if any(o.status in ("starting", "running") for o in LIVE.values()):
        raise HTTPException(409, "an optimizer run is already going")
    try:
        opt = optimizer.load_opt(oid)
    except ValueError:
        raise HTTPException(404, "no such run") from None
    if opt.mode != "perpetual" or not (optimizer.RUNS / f"opt-{oid}" / "evo").exists():
        raise HTTPException(409, "only runs made by the population loop can be resumed")
    sk = FORGE.registry.skills.get(opt.skill["id"])
    _launch(sk, opt, optimizer.DEFAULTS["train_seconds"], True)
    return {"id": oid}


@router.get("/runs/{oid}/scene")
def scene(oid: str, round: int, scene: int):
    try:
        return optimizer_zoom.info(oid, round, scene)
    except KeyError:
        raise HTTPException(404, "no such scene") from None


@router.get("/runs/{oid}/frame")
def frame(
    oid: str,
    round: int,
    scene: int,
    i: int = 0,
    w: int = 480,
    h: int = 360,
    az: float = 148.0,
    el: float = -34.0,
    d: float = 0.46,
):
    try:
        jpg = optimizer_zoom.frame(oid, round, scene, i, w, h, az, el, d)
    except KeyError:
        raise HTTPException(404, "no stored rollout for that round") from None
    return Response(
        jpg, media_type="image/jpeg", headers={"Cache-Control": "max-age=300"}
    )


@router.get("/runs")
def runs():
    return optimizer.list_runs()


@router.get("/runs/{oid}")
def run(oid: str):
    if oid in LIVE:
        return LIVE[oid].to_json()
    d = optimizer.load_run(oid)
    if d is None:
        raise HTTPException(404, "no such run")
    return d


@router.get("/runs/{oid}/img/{rel:path}")
def img(oid: str, rel: str):
    p = optimizer.asset(oid, rel)
    if p is None:
        raise HTTPException(404, "no such image")
    return FileResponse(
        p, media_type="image/jpeg", headers={"Cache-Control": "no-cache"}
    )
