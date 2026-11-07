"""Meta's Model API client (Muse Spark), with OpenAI and a stub behind it when Meta cannot answer.

Targets Meta's OpenAI-compatible chat completions endpoint, https://api.meta.ai/v1/chat/completions
(dev.meta.ai/docs), with `Authorization: Bearer <key>` and a schema-constrained reply
(`response_format: {"type": "json_schema", "json_schema": {"name", "schema"}}`; decoding is constrained). Images go as
`image_url` data URLs in the user message. The key is META_API_KEY (or LLAMA_API_KEY), else the macOS Keychain item of
the same name, read once per process (a new key is picked up on restart). Model: VLA_FORGE_META_LLM, default
muse-spark-1.3, a reasoning model: reasoning tokens count against max_tokens, so the budget is generous and the effort
is VLA_FORGE_META_EFFORT (default "low").

Backends: Meta when its key works (probed once per process), else OpenAI behind the same interface. The backend that
really answered is returned, recorded per optimizer round and shown on the page; when Meta is off or refuses the key,
`meta_status()` says why and the page says Muse Spark is unavailable instead of putting another model under its name.
Every call is streamed as an `llm_call` event on the Meta lane (lanes.py). The stub answers the same four calls (task,
plan, critic, optimizer) with schema-valid replies and no network: when VLA_FORGE_LLAMA_STUB=1, when no key is set, or
when every live call fails. Planner, task and critic replies come from the keyword rules; optimizer replies replay the
changes recorded in a saved optimizer run.
"""

from __future__ import annotations

import base64
import json
import os
import subprocess
from functools import lru_cache
from pathlib import Path

import httpx

BASE_URL = os.environ.get("VLA_FORGE_META_URL", "https://api.meta.ai/v1")
MODEL = os.environ.get(
    "VLA_FORGE_META_LLM", "muse-spark-1.3"
)  # task, line plan, critic
OPT_MODEL = os.environ.get(
    "VLA_FORGE_META_OPT_LLM", "muse-wonder-1.0"
)  # the optimizer's proposer, head to head with SubQ
NAMES = {
    "muse-spark-1.3": "Muse Spark",
    "muse-spark-1.2": "Muse Spark 1.2",
    "muse-spark-1.1": "Muse Spark 1.1",
    "muse-wonder-1.0": "Muse Wonder",
}
MODEL_NAME = NAMES.get(MODEL, MODEL)


def model_for(kind: str) -> str:
    """The optimizer asks Muse Wonder; every other call asks Muse Spark. Never a *-contributor model (their inputs
    train Meta's models)."""
    m = OPT_MODEL if kind == "optimizer" else MODEL
    return m.removesuffix("-contributor")


EFFORT = os.environ.get("VLA_FORGE_META_EFFORT", "low")
REPLAY_RUN = os.environ.get("VLA_FORGE_STUB_RUN", "opt-234442-a649")
RUNS = Path(__file__).resolve().parents[2] / "runs"
OPENAI_URL = os.environ.get("VLA_FORGE_OPENAI_URL", "https://api.openai.com/v1")
OPENAI_MODEL = os.environ.get("VLA_FORGE_OPENAI_MODEL", "gpt-5.5")
KEYS_ENV = Path.home() / ".config" / "jarvis" / "keys.env"
TIMEOUT = 60.0


def _keychain(name: str) -> str | None:
    try:
        out = subprocess.run(
            ["security", "find-generic-password", "-s", name, "-w"],
            check=False,
            capture_output=True,
            text=True,
            timeout=5,
        )
        return out.stdout.strip() or None
    except (OSError, subprocess.SubprocessError):
        return None


@lru_cache(maxsize=1)
def api_key() -> str | None:
    for name in ("META_API_KEY", "LLAMA_API_KEY"):
        key = os.environ.get(name) or _keychain(name)
        if key:
            return key
    return None


@lru_cache(maxsize=1)
def openai_key() -> str | None:
    """OPENAI_API_KEY from the environment, else from ~/.config/jarvis/keys.env (read here, never printed)."""
    if os.environ.get("OPENAI_API_KEY"):
        return os.environ["OPENAI_API_KEY"]
    try:
        for line in KEYS_ENV.read_text().splitlines():
            name, _, val = line.partition("=")
            if name.strip().removeprefix("export ").strip() == "OPENAI_API_KEY":
                return val.strip().strip("'\"") or None
    except OSError:
        pass
    return None


def _content(user: str, images: list[bytes] | None) -> list[dict] | str:
    if not images:
        return user
    return [{"type": "text", "text": user}] + [
        {
            "type": "image_url",
            "image_url": {
                "url": "data:image/png;base64," + base64.b64encode(b).decode()
            },
        }
        for b in images
    ]


class MetaClient:
    """One schema-constrained chat call against Meta's Model API (images included)."""

    def __init__(
        self, key: str, model: str = MODEL, base_url: str = BASE_URL, http=None
    ):
        self.key, self.model, self.base_url = key, model, base_url.rstrip("/")
        self.http = http or httpx.Client(timeout=TIMEOUT * 2)
        self.usage: dict = {}

    def request(
        self, system: str, user: str, schema: dict, images: list[bytes] | None = None
    ) -> dict:
        return {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": _content(user, images)},
            ],
            "response_format": {
                "type": "json_schema",
                "json_schema": {"name": "out", "schema": schema},
            },
            "max_tokens": 6000,  # reasoning tokens count here too
            "reasoning_effort": EFFORT,
        }

    def chat_json(
        self, system: str, user: str, schema: dict, images: list[bytes] | None = None
    ) -> dict:
        r = self.http.post(
            f"{self.base_url}/chat/completions",
            headers={
                "Authorization": f"Bearer {self.key}",
                "Content-Type": "application/json",
            },
            json=self.request(system, user, schema, images),
        )
        r.raise_for_status()
        body = r.json()
        self.usage = body.get("usage") or {}
        _charge(self.usage, system + user, self.model)
        text = body["choices"][0]["message"]["content"]
        if not text:
            raise ValueError(
                "Meta returned no content (the reasoning used the whole budget)"
            )
        return json.loads(text)


class Stub:
    """Schema-valid replies with no network."""

    model = MODEL

    def chat_json(self, kind: str, system: str, user: str, schema: dict) -> dict:
        if kind == "task":
            return self._task(user)
        if kind == "plan":
            from .factory import _rule_plan

            plan = _rule_plan(user)
            if plan.get("assembly") and "crate" not in user.lower():
                plan["stations"] = plan["stations"][
                    :2
                ]  # no crating arm unless the request asks for one
            return plan
        if kind == "critic":
            from .llm import _rule_critic

            d = json.loads(user)
            out = _rule_critic(
                d["evaluation"],
                d["settings"],
                d["seconds_left"],
                d["target_success_rate"],
            )
            out["rationale"] = (
                "Add demonstrations at the failed starts and keep training."
                if out["action"] == "retrain"
                else "The success rate is at the bar, so publish."
            )
            return out
        if kind == "optimizer":
            return self._optimizer(json.loads(user))
        raise ValueError(f"unknown call {kind!r}")

    @staticmethod
    def _task(prompt: str) -> dict:
        from .llm import rule_parse

        s = rule_parse(prompt)
        return {
            "feasible": True,
            "note": "",
            "kind": "single",
            "steps": [
                {"skill": s.skill, "object_color": s.obj, "target_color": s.target}
            ],
            "distractors": 2,
            "checks": [],
            "mentions": [],
        }

    @staticmethod
    def _optimizer(d: dict) -> dict:
        rounds = json.loads((RUNS / REPLAY_RUN / "run.json").read_text())["rounds"]
        i = min(max(int(d.get("round", 1)), 1), len(rounds) - 1)
        c = rounds[i]["change"]
        return {"knob": c["knob"], "value": str(c["value"]), "reason": c["reason"]}


def use_stub() -> bool:
    return os.environ.get("VLA_FORGE_LLAMA_STUB") == "1" or not live_backends()


class OpenAIClient:
    """The same schema-constrained call against OpenAI, behind the MetaClient interface (including images).

    Used when the Meta key does not work or the credit is off. The page then says Muse Spark is unavailable and names this
    backend; every optimizer round records the backend that really answered in its `backend` field."""

    def __init__(
        self, key: str, model: str = OPENAI_MODEL, http=None, base_url: str = OPENAI_URL
    ):
        self.key, self.model, self.base_url = key, model, base_url.rstrip("/")
        self.http = http or httpx.Client(timeout=TIMEOUT * 2)
        self.usage: dict = {}

    @property
    def name(self) -> str:
        return f"openai:{self.model}"

    def request(
        self, system: str, user: str, schema: dict, images: list[bytes] | None = None
    ) -> dict:
        content = _content(user, images)
        return {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": content},
            ],
            "response_format": {
                "type": "json_schema",
                "json_schema": {"name": "out", "schema": schema, "strict": False},
            },
            "max_completion_tokens": 4000,
        }

    def chat_json(
        self, system: str, user: str, schema: dict, images: list[bytes] | None = None
    ) -> dict:
        r = self.http.post(
            f"{self.base_url}/chat/completions",
            headers={
                "Authorization": f"Bearer {self.key}",
                "Content-Type": "application/json",
            },
            json=self.request(system, user, schema, images),
        )
        r.raise_for_status()
        body = r.json()
        self.usage = body.get("usage") or {}
        return json.loads(body["choices"][0]["message"]["content"])


_DEAD: set[str] = (
    set()
)  # backends that refused their key this process (tested once, not every call)
_ALIVE: set[str] = set()  # backends whose key answered a probe this process
META_WHY: dict[str, str] = {}  # why Meta is not answering, for the page


# Meta runs on the hackathon's free $100 credit only. Calls stay off until VLA_FORGE_META_CREDITS=1 (set once the
# credit shows under Billing > free credits), and stop for good once the estimated spend reaches the cap. Prices are
# Meta's standard (not contributor) list prices; reasoning tokens are billed as output and are already in
# usage.completion_tokens. An unknown model is charged at Wonder's prices so the estimate runs high.
CREDIT_CAP_USD = float(os.environ.get("VLA_FORGE_META_CAP_USD", "80"))
PRICES = {  # USD per 1M tokens: (input, cached input, output)
    "muse-wonder-1.0": (4.00, 0.20, 20.00),
    "muse-spark-1.3": (1.25, 0.15, 4.25),
    "muse-spark-1.2": (1.25, 0.15, 4.25),
}
SPEND = RUNS / "meta-spend.json"


def meta_spent() -> float:
    try:
        return float(json.loads(SPEND.read_text())["usd"])
    except (OSError, ValueError, KeyError):
        return 0.0


def _charge(usage: dict, prompt: str, model: str = MODEL) -> None:
    pin, pcache, pout = PRICES.get(model, PRICES["muse-wonder-1.0"])
    tin = usage.get("prompt_tokens")
    tin = len(prompt) // 3 if tin is None else tin  # no usage reported: estimate high
    cached = (usage.get("prompt_tokens_details") or {}).get("cached_tokens") or 0
    tout = usage.get("completion_tokens")
    tout = 1000 if tout is None else tout
    usd = meta_spent() + ((tin - cached) * pin + cached * pcache + tout * pout) / 1e6
    SPEND.parent.mkdir(parents=True, exist_ok=True)
    SPEND.write_text(json.dumps({"usd": round(usd, 4), "cap": CREDIT_CAP_USD}))


def meta_allowed() -> bool:
    return (
        os.environ.get("VLA_FORGE_META_CREDITS") == "1"
        and meta_spent() < CREDIT_CAP_USD
    )


def _meta_alive(key: str) -> bool:
    """One cheap call to find out whether the Meta key works; a 401, 402 or 403 retires the backend for this process."""
    if "meta" in _DEAD or not meta_allowed():
        return False
    if "meta" in _ALIVE:
        return True
    try:
        MetaClient(key).chat_json("Reply with JSON.", "ping", {"type": "object"})
        _ALIVE.add("meta")
        return True
    except httpx.HTTPStatusError as e:
        if e.response.status_code in (401, 402, 403):
            _DEAD.add("meta")
            META_WHY["meta"] = _refusal(e)
            return False
        return True  # a transient server error is not a dead key
    except (httpx.HTTPError, ValueError, KeyError, IndexError):
        return True


def _refusal(e: httpx.HTTPStatusError) -> str:
    code = e.response.status_code
    words = {401: "rejected the API key", 402: "has no credit", 403: "refused this key"}
    return f"Meta's Model API {words.get(code, 'refused the call')} ({code})"


def meta_status() -> dict:
    """Whether Muse Spark itself can answer in this process, and if not, why (shown on the page in plain words)."""
    if os.environ.get("VLA_FORGE_LLAMA_STUB") == "1":
        why = "stub mode is on (VLA_FORGE_LLAMA_STUB=1)"
    elif not api_key():
        why = "no Meta API key is set"
    elif os.environ.get("VLA_FORGE_META_CREDITS") != "1":
        why = "Meta calls are off until the free credit is confirmed (VLA_FORGE_META_CREDITS=1)"
    elif meta_spent() >= CREDIT_CAP_USD:
        why = f"the ${CREDIT_CAP_USD:.0f} free-credit cap is reached"
    elif "meta" in _DEAD:
        why = META_WHY.get("meta", "Meta's Model API refused the key")
    else:
        why = ""
    return {
        "available": not why,
        "why": why,
        "model": MODEL,
        "name": MODEL_NAME,
        "optimizer_model": OPT_MODEL,
        "optimizer_name": NAMES.get(OPT_MODEL, OPT_MODEL),
        "spent_usd": round(meta_spent(), 4),
        "cap_usd": CREDIT_CAP_USD,
        "fallback": f"openai:{OPENAI_MODEL}" if openai_key() else "stub",
    }


def label(backend: str) -> str:
    """Display name for whatever answered: Muse Spark only when Meta did."""
    if backend.startswith("meta:"):
        m = backend.split(":", 1)[1]
        return NAMES.get(m, m)
    if backend.startswith("openai:"):
        return f"OpenAI {backend.split(':', 1)[1]} (Meta unavailable)"
    if backend.startswith("subq:"):
        from .subq import NAME

        return NAME
    if backend == "random":
        return "random (control arm)"
    return "rules (no model answered)"


def live_backends() -> list[str]:
    """Names in the order they are tried: Meta's Model API if its key works, then OpenAI."""
    out = []
    if os.environ.get("VLA_FORGE_LLAMA_STUB") == "1":
        return out
    k = api_key()
    if k and _meta_alive(k):
        out.append("meta")
    if openai_key():
        out.append("openai")
    return out


def _client(name: str, kind: str = "task"):
    if name == "meta":
        m = model_for(kind)
        return MetaClient(api_key() or "", m), f"meta:{m}"
    c = OpenAIClient(openai_key() or "")
    return c, c.name


def chat_json_ex(
    kind: str,
    system: str,
    user: str,
    schema: dict,
    images: list[bytes] | None = None,
    backends: list[str] | None = None,
) -> tuple[dict, str, str]:
    """(parsed JSON, display name of what answered, backend that really answered). Live backends first, then the stub.

    The display name is Muse Spark only when Meta answered (see `label`). The call is streamed on the Meta lane."""
    from . import lanes

    call = lanes.start("meta", kind, model_for(kind))
    errors = []
    for name in backends or live_backends():
        if name == "meta" and not meta_allowed():
            continue
        try:
            c, backend = _client(name, kind)
            raw = c.chat_json(system, user, schema, images)
            u = c.usage or {}
            lanes.end(
                call,
                backend,
                u.get("prompt_tokens") or 0,
                u.get("completion_tokens") or 0,
                note="; ".join(errors)
                or ("" if name == "meta" else meta_status()["why"]),
            )
            return raw, label(backend), backend
        except httpx.HTTPStatusError as e:
            if name == "meta" and e.response.status_code in (401, 402, 403):
                _DEAD.add("meta")
                META_WHY["meta"] = _refusal(e)
            errors.append(f"{name}: HTTP {e.response.status_code}")
        except (httpx.HTTPError, ValueError, KeyError, IndexError) as e:
            errors.append(f"{name}: {type(e).__name__}")
    raw = Stub().chat_json(kind, system, user, schema)
    lanes.end(
        call,
        "stub",
        note="; ".join(errors) or meta_status()["why"] or "no live backend",
    )
    return raw, label("stub"), "stub"


def chat_json(kind: str, system: str, user: str, schema: dict) -> tuple[dict, str]:
    """(parsed JSON, display name of what answered). A failed live call falls through to the next backend, then the
    stub."""
    raw, model, _backend = chat_json_ex(kind, system, user, schema)
    return raw, model
