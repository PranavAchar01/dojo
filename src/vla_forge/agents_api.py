"""HTTP surface of the two model lanes: the stage graph with who answers on each lane, and the live call record."""

from __future__ import annotations

import asyncio

from fastapi import APIRouter

from . import lanes, optimizer

router = APIRouter()
HOLD: dict = {"agents": None}  # set by server.py's lifespan


@router.get("/api/agents")
async def graph():
    return await asyncio.to_thread(HOLD["agents"].graph)


@router.get("/api/llm")
async def llm():
    """Totals per lane, calls in flight, recent calls, this session's optimizer duels and who answers on each lane.
    When this session has no duels yet, the duels of the latest saved optimizer run that has them."""
    snap = lanes.snapshot()
    snap["models"] = await asyncio.to_thread(HOLD["agents"].models)
    if not snap["duels"]:
        snap["saved_duels"] = await asyncio.to_thread(optimizer.saved_duels)
    return snap
