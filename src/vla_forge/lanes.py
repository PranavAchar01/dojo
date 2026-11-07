"""The two model lanes, Meta (Muse Spark) and SubQ: every call is timed, counted and streamed to the page as `llm_call` events.

A call opens with phase "start" (lane, what it is for, the run it belongs to) and closes with phase "end" (the backend
that really answered, tokens in and out as the API reported them, latency, or the error). `both` runs one Meta call and
one SubQ call at the same moment on threads. `duel` records one optimizer round's two proposals and the winner.
Nothing here makes up a number: totals are sums of what the APIs returned.
"""

from __future__ import annotations

import contextvars
import itertools
import threading
import time
from collections import deque
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from typing import Any

LANES = ("meta", "subq")
_seq = itertools.count(1)
_lock = threading.Lock()
_pool = ThreadPoolExecutor(max_workers=4, thread_name_prefix="lane")
# (run id, what the call is for) of the code asking; set with `about`, read when a call starts
CTX: contextvars.ContextVar[tuple[str, str]] = contextvars.ContextVar(
    "lane_ctx", default=("", "")
)


def _empty() -> dict:
    return {
        "calls": 0,
        "errors": 0,
        "tokens_in": 0,
        "tokens_out": 0,
        "ms": 0,
        "backends": {},
    }


TOTALS: dict[str, dict] = {lane: _empty() for lane in LANES}
RECENT: deque[dict] = deque(maxlen=120)  # finished calls, newest last
OPEN: dict[int, dict] = {}  # calls in flight
DUELS: list[dict] = []  # one per optimizer round this session


def _emit(run: str, kind: str, agent: str, **data) -> None:
    from .pipeline import FORGE

    FORGE.emit(run, kind, agent, **data)


@contextmanager
def about(run: str, purpose: str):
    """Tag the model calls made inside this block with a run id and a short purpose for the page."""
    tok = CTX.set((run, purpose))
    try:
        yield
    finally:
        CTX.reset(tok)


def start(lane: str, kind: str, model: str) -> dict:
    run, purpose = CTX.get()
    call = {
        "id": next(_seq),
        "lane": lane,
        "call_kind": kind,  # not "kind": that is the event's own kind
        "purpose": purpose or kind,
        "model": model,
        "run": run,
        "t0": time.time(),
    }
    with _lock:
        OPEN[call["id"]] = call
    _emit(run, "llm_call", lane, phase="start", **_public(call))
    return call


def end(
    call: dict,
    backend: str,
    tokens_in: int = 0,
    tokens_out: int = 0,
    error: str = "",
    note: str = "",
) -> dict:
    ms = round((time.time() - call["t0"]) * 1000)
    call.update(
        backend=backend,
        tokens_in=int(tokens_in or 0),
        tokens_out=int(tokens_out or 0),
        ms=ms,
        error=error,
        note=note,
    )
    with _lock:
        OPEN.pop(call["id"], None)
        t = TOTALS[call["lane"]]
        t["calls"] += 1
        t["errors"] += bool(error)
        t["tokens_in"] += call["tokens_in"]
        t["tokens_out"] += call["tokens_out"]
        t["ms"] += ms
        t["backends"][backend] = t["backends"].get(backend, 0) + 1
        RECENT.append(_public(call))
    _emit(call["run"], "llm_call", call["lane"], phase="end", **_public(call))
    return call


def _public(call: dict) -> dict:
    return {k: v for k, v in call.items() if k not in ("run",)}


def both(meta_fn: Callable[[], Any], subq_fn: Callable[[], Any]) -> tuple[Any, Any]:
    """Run the Meta call and the SubQ call at the same time. An exception in one lane is returned in its slot, not
    raised, so the other lane's answer is never lost."""
    fa = _pool.submit(contextvars.copy_context().run, meta_fn)
    fb = _pool.submit(contextvars.copy_context().run, subq_fn)
    out = []
    for f in (fa, fb):
        try:
            out.append(f.result())
        except Exception as e:  # noqa: BLE001 - handed back to the caller, which records it
            out.append(e)
    return out[0], out[1]


def duel(run: str, row: dict) -> None:
    """One optimizer round's head to head: both changes, both SELECT scores, the winner."""
    with _lock:
        DUELS.append(row)
    _emit(run, "duel", "optimizer", **row)


def tally(duels: list[dict]) -> dict:
    t = {"meta": 0, "subq": 0, "tie": 0, "agreed": 0, "solo": 0}
    for d in duels:
        t[d.get("winner", "solo")] = t.get(d.get("winner", "solo"), 0) + 1
    return t


def snapshot() -> dict:
    """What /api/llm returns: totals per lane, the calls in flight, the recent calls, and this session's duels."""
    with _lock:
        now = time.time()
        return {
            "totals": {
                k: {**v, "backends": dict(v["backends"])} for k, v in TOTALS.items()
            },
            "open": [
                {**_public(c), "elapsed_ms": round((now - c["t0"]) * 1000)}
                for c in OPEN.values()
            ],
            "recent": list(RECENT),
            "duels": list(DUELS),
            "tally": tally(DUELS),
        }
