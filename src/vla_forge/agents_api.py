"""HTTP surface of the agent team: the graph, the live message log, protocol manifests, and the optimizer start
route that goes through the gateway -> optimizer agent protocol (server.py registers it ahead of optimizer_api)."""

from __future__ import annotations

import asyncio

from fastapi import APIRouter

from . import optimizer_api
from .agentbus import BUS

router = APIRouter()
HOLD: dict = {"agents": None}  # set by server.py's lifespan


@router.get("/api/agents")
async def graph():
    return await asyncio.to_thread(HOLD["agents"].graph)


@router.get("/api/agents/messages")
def messages(since: int = 0, limit: int = 300):
    return BUS.since(since, limit)


@router.get("/api/agents/manifests")
def manifests():
    return HOLD["agents"].manifests()


@router.post("/api/optimizer/start")
async def start(r: optimizer_api.Start):
    """Same body and answer as optimizer_api.start, but the request travels gateway -> optimizer as a message."""
    return {"id": await HOLD["agents"].optimize(r.skill, r.rounds, r.train_seconds)}
