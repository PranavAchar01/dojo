"""The agents' message log and thought stream.

Every uAgents message the Bureau sends is recorded here (sender, receiver, protocol, message type, a short summary
built from the message's own fields, timestamp) and fanned out on the app's event stream as `agent_msg`. What each
agent says it is doing goes out as `thought` events, tagged with the agent and whether a Llama call produced the
text. Nothing here invents content: the log wraps `Context.send`, and thoughts are the agents' own log lines.
"""

from __future__ import annotations

import itertools
import time
from collections import deque

from uagents.context import ExternalContext, InternalContext

# agents whose step asks the model (llm.parse_task, llm.critic_plan, optimizer round proposals)
LLM_AGENTS = {"task", "critic", "optimizer"}

CHAT = "AgentChatProtocol 0.3.0"
OPTIMIZE = "forge-optimize 1.0"
PIPELINE = "forge-pipeline (typed uAgents models)"
OPTIMIZE_TYPES = {
    "OptimizeRequest",
    "OptimizeAccepted",
    "OptimizeProgress",
    "OptimizeResult",
}
CHAT_TYPES = {"ChatMessage", "ChatAcknowledgement"}


def protocol_of(mtype: str) -> str:
    if mtype in CHAT_TYPES:
        return CHAT
    if mtype in OPTIMIZE_TYPES:
        return OPTIMIZE
    return PIPELINE


def _short(v, n: int = 64) -> str:
    s = str(v).replace("\n", " ")
    return s if len(s) <= n else s[: n - 1] + "…"


def summarize(msg) -> str:
    """A short line from the message's own fields (the run id and big JSON blobs are left out)."""
    name = type(msg).__name__
    if name == "ChatMessage":
        text = "".join(getattr(c, "text", "") for c in msg.content)
        head, sep, rest = text.partition("|")
        return _short(rest if sep and len(head) < 24 else text, 90)
    if name == "ChatAcknowledgement":
        return "acknowledged"
    parts = []
    for k, v in msg.dict().items():
        if k in ("run_id", "request_id"):
            continue
        if isinstance(v, str) and v[:1] in "[{":
            parts.append(f"{k}: {len(v)} chars")
        elif isinstance(v, float):
            parts.append(f"{k}: {v:.2f}")
        elif v not in ("", None):
            parts.append(f"{k}: {_short(v, 40)}")
    return _short(", ".join(parts), 110)


class Bus:
    def __init__(self) -> None:
        self.log: deque[dict] = deque(maxlen=600)
        self.names: dict[str, str] = {}  # agent address -> name
        self.seq = itertools.count(1)

    def record(self, sender: str, dest: str, msg) -> dict:
        from .pipeline import FORGE

        mtype = type(msg).__name__
        run_id = getattr(msg, "run_id", "") or ""
        if not run_id and mtype == "ChatMessage":
            text = "".join(getattr(c, "text", "") for c in msg.content)
            head, sep, _ = text.partition("|")
            run_id = head if sep and len(head) < 24 else ""
        rec = {
            "seq": next(self.seq),
            "ts": round(time.time(), 3),
            "sender": sender,
            "receiver": self.names.get(dest, dest[:12] + "…"),
            "protocol": protocol_of(mtype),
            "type": mtype,
            "summary": summarize(msg),
            "ack": mtype == "ChatAcknowledgement",
            "run": run_id,
        }
        self.log.append(rec)
        FORGE.emit(
            run_id, "agent_msg", sender, **{k: v for k, v in rec.items() if k != "run"}
        )
        return rec

    def since(self, seq: int = 0, limit: int = 300) -> list[dict]:
        return [r for r in self.log if r["seq"] > seq][-limit:]


BUS = Bus()
_patched = False


def patch_send() -> None:
    """Record every message sent through a Bureau agent's context. Applied once, before the Bureau starts."""
    global _patched
    if _patched:
        return
    _patched = True
    for cls in (InternalContext, ExternalContext):
        orig = cls.send

        def wrap(orig):
            async def send(self, destination, message, timeout=30):
                status = await orig(self, destination, message, timeout)
                BUS.record(self.agent.name, destination, message)
                return status

            return send

        cls.send = wrap(orig)


def thought(run_id: str, agent: str, text: str, **extra) -> None:
    """One line of an agent's reasoning or narration for the Thinking panel."""
    from .pipeline import FORGE

    FORGE.emit(
        run_id,
        "thought",
        agent,
        text=text,
        llm=agent in LLM_AGENTS,
        **extra,
    )
