"""HTTP surface of the optimizer (web/optimizer.js is its only client). Mounted by server.py with one line."""

from __future__ import annotations

import asyncio

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from . import optimizer
from .pipeline import FORGE

router = APIRouter(prefix="/api/optimizer")
LIVE: dict[str, optimizer.Opt] = {}


class Start(BaseModel):
    skill: str
    rounds: int = Field(5, ge=1, le=8)
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
    LIVE[opt.id] = opt

    def go() -> None:
        optimizer.run(
            sk, r.rounds, opt, FORGE.replay, FORGE.compute, FORGE.emit, r.train_seconds
        )

    task = asyncio.get_running_loop().run_in_executor(None, go)
    task.add_done_callback(
        lambda f: f.exception()
    )  # the failure is already on the run record
    return {"id": opt.id}


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
