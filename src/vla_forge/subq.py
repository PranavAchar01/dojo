"""Subquadratic's SubQ Snap: a decision model that reads a very long context and picks.

SubQ Snap (subq-snap-1.1) is called through the Decisions API, `POST {SUBQUADRATIC_BASE_URL}/systemone`: a `state` (any
JSON or text, up to 393,216 tokens) and named questions of type `choice` (named options, returns the pick and a
probability for every option), `noul` (yes/no probability) or `score`. It does not take chat completions (the API says
so for every Snap model on our key). Input costs $0.04 per 1M tokens and output is free; a 189K-token state answered in
2.4 s when measured, so it is the lane that reads everything: every trajectory, every past round, the whole library.

Key and URL: SUBQUADRATIC_API_KEY / SUBQUADRATIC_BASE_URL from the environment, else ~/.config/jarvis/keys.env (read
here, never printed). The free grant is 100M tokens; `runs/subq-spend.json` counts the input tokens the API reports and
calls stop at VLA_FORGE_SUBQ_CAP_TOKENS (default 80M). Each call is streamed on the SubQ lane (lanes.py).
"""

from __future__ import annotations

import json
import os
import threading
from functools import lru_cache
from pathlib import Path

import httpx

from . import lanes

RUNS = Path(__file__).resolve().parents[2] / "runs"
KEYS_ENV = Path.home() / ".config" / "jarvis" / "keys.env"
MODEL = os.environ.get("VLA_FORGE_SUBQ_MODEL", "subq-snap-1.1")
NAME = "SubQ Snap 1.1" if MODEL == "subq-snap-1.1" else MODEL
CAP_TOKENS = int(os.environ.get("VLA_FORGE_SUBQ_CAP_TOKENS", str(80_000_000)))
CONTEXT = 393_216
MAX_STATE_TOKENS = 300_000  # leave room for the questions under the 384K window
BYTES_PER_TOKEN = 1.9  # measured: 368,031 bytes of JSON state = 189,392 tokens
SPEND = RUNS / "subq-spend.json"
TIMEOUT = 120.0
_lock = threading.Lock()
WHY: dict[str, str] = {}  # why SubQ stopped answering in this process


class Unavailable(RuntimeError):
    """SubQ cannot answer: no key, the token cap, or the API refused."""


def _from_env_file(name: str) -> str | None:
    try:
        for line in KEYS_ENV.read_text().splitlines():
            k, _, v = line.partition("=")
            if k.strip().removeprefix("export ").strip() == name:
                return v.strip().strip("'\"") or None
    except OSError:
        pass
    return None


@lru_cache(maxsize=1)
def api_key() -> str | None:
    return os.environ.get("SUBQUADRATIC_API_KEY") or _from_env_file(
        "SUBQUADRATIC_API_KEY"
    )


@lru_cache(maxsize=1)
def base_url() -> str:
    return (
        os.environ.get("SUBQUADRATIC_BASE_URL")
        or _from_env_file("SUBQUADRATIC_BASE_URL")
        or "https://api.subq.ai/v1"
    ).rstrip("/")


def tokens_used() -> int:
    try:
        return int(json.loads(SPEND.read_text())["tokens"])
    except (OSError, ValueError, KeyError):
        return 0


def _charge(tokens: int) -> None:
    with _lock:
        used = tokens_used() + int(tokens)
        SPEND.parent.mkdir(parents=True, exist_ok=True)
        SPEND.write_text(json.dumps({"tokens": used, "cap": CAP_TOKENS}))


def status() -> dict:
    if os.environ.get("VLA_FORGE_SUBQ_OFF") == "1":
        why = "SubQ is switched off (VLA_FORGE_SUBQ_OFF=1)"
    elif not api_key():
        why = "no SUBQUADRATIC_API_KEY is set"
    elif tokens_used() >= CAP_TOKENS:
        why = f"the {CAP_TOKENS // 1_000_000}M-token cap on the free grant is reached"
    else:
        why = WHY.get("api", "")
    return {
        "available": not why,
        "why": why,
        "name": NAME,
        "model": MODEL,
        "backend": f"subq:{MODEL}" if not why else "unavailable",
        "tokens_used": tokens_used(),
        "cap_tokens": CAP_TOKENS,
        "context": CONTEXT,
    }


def fit(state, max_tokens: int = MAX_STATE_TOKENS):
    """The state as given if it fits, else its JSON text cut to the budget (the caller puts what matters first)."""
    text = state if isinstance(state, str) else json.dumps(state, separators=(",", ":"))
    limit = int(max_tokens * BYTES_PER_TOKEN)
    if len(text) <= limit:
        return state
    return text[:limit] + " [cut here to fit the context window]"


def decide(state, questions: dict, kind: str, http: httpx.Client | None = None) -> dict:
    """Ask SubQ the named questions about `state`. Returns the `answers` object. Raises Unavailable."""
    st = status()
    call = lanes.start("subq", kind, MODEL)
    if not st["available"]:
        lanes.end(call, "unavailable", error=st["why"])
        raise Unavailable(st["why"])
    body = {"model": MODEL, "state": fit(state), "questions": questions}
    try:
        r = (http or httpx.Client(timeout=TIMEOUT)).post(
            f"{base_url()}/systemone",
            headers={
                "Authorization": f"Bearer {api_key()}",
                "Content-Type": "application/json",
            },
            json=body,
        )
        r.raise_for_status()
        out = r.json()
    except httpx.HTTPStatusError as e:
        code = e.response.status_code
        msg = {
            401: "SubQ rejected the API key",
            402: "SubQ has no credit left",
            403: "this key cannot use the model",
        }.get(code, f"SubQ answered HTTP {code}")
        if code in (401, 402, 403):
            WHY["api"] = f"{msg} ({code})"
        lanes.end(call, f"subq:{MODEL}", error=msg)
        raise Unavailable(msg) from None
    except (httpx.HTTPError, ValueError) as e:
        lanes.end(call, f"subq:{MODEL}", error=type(e).__name__)
        raise Unavailable(f"SubQ call failed: {type(e).__name__}") from None
    used = int((out.get("usage") or {}).get("input_tokens") or 0)
    _charge(used)
    lanes.end(call, f"subq:{MODEL}", used, 0)
    return out["answers"]


def choice(answer: dict) -> tuple[str, float]:
    """(picked option, its probability) from a `choice` answer."""
    pick = answer["choice"]
    return pick, float((answer.get("probabilities") or {}).get(pick, 0.0))
