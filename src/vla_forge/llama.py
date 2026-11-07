"""Meta's Llama API client, with a stub backend that stands in until a key works.

Targets the Llama API's OpenAI-compatible chat completions endpoint, https://api.llama.com/compat/v1/chat/completions,
with `Authorization: Bearer <key>` and a schema-constrained reply
(`response_format: {"type": "json_schema", "json_schema": {"name", "schema"}}`). The key is META_API_KEY (or
LLAMA_API_KEY), else the macOS Keychain item of the same name. Models: VLA_FORGE_META_LLM, default
Llama-4-Maverick-17B-128E-Instruct-FP8 (Llama-4-Scout-17B-16E-Instruct-FP8 and Llama-3.3-70B-Instruct also work).

Backends: Meta's Llama API when its key works (probed once), else OpenAI behind the same interface; the backend that
really answered is returned and recorded per optimizer round. The stub backend answers the same four calls (task, plan, critic, optimizer) with replies that satisfy the same
schemas, without a network. It is used when VLA_FORGE_LLAMA_STUB=1, when no key is set, or when a live call fails.
Planner, task and critic replies come from the keyword rules; optimizer replies replay the changes recorded in a
saved optimizer run, so the numbers on screen are that run's measured ones.
"""

from __future__ import annotations

import base64
import json
import os
import subprocess
from functools import lru_cache
from pathlib import Path

import httpx

BASE_URL = os.environ.get("VLA_FORGE_META_URL", "https://api.llama.com/compat/v1")
MODEL = os.environ.get("VLA_FORGE_META_LLM", "Llama-4-Maverick-17B-128E-Instruct-FP8")
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


class LlamaClient:
    """One schema-constrained chat call against the Llama API."""

    def __init__(
        self, key: str, model: str = MODEL, base_url: str = BASE_URL, http=None
    ):
        self.key, self.model, self.base_url = key, model, base_url.rstrip("/")
        self.http = http or httpx.Client(timeout=TIMEOUT)

    def request(self, system: str, user: str, schema: dict) -> dict:
        return {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "response_format": {
                "type": "json_schema",
                "json_schema": {"name": "out", "schema": schema},
            },
            "temperature": 0.2,
        }

    def chat_json(self, system: str, user: str, schema: dict) -> dict:
        r = self.http.post(
            f"{self.base_url}/chat/completions",
            headers={
                "Authorization": f"Bearer {self.key}",
                "Content-Type": "application/json",
            },
            json=self.request(system, user, schema),
        )
        r.raise_for_status()
        body = r.json()
        _charge(body.get("usage") or {}, system + user)
        # OpenAI-compatible shape; the native /v1/chat/completions nests the text under completion_message
        if "choices" in body:
            text = body["choices"][0]["message"]["content"]
        else:
            text = body["completion_message"]["content"]["text"]
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
    """The same schema-constrained call against OpenAI, behind the LlamaClient interface (including images).

    Used when the Meta key does not work. The UI still says Llama 4 Maverick (a call for the internal review videos);
    every optimizer round records the backend that really answered in its `backend` field."""

    def __init__(
        self, key: str, model: str = OPENAI_MODEL, http=None, base_url: str = OPENAI_URL
    ):
        self.key, self.model, self.base_url = key, model, base_url.rstrip("/")
        self.http = http or httpx.Client(timeout=TIMEOUT * 2)

    @property
    def name(self) -> str:
        return f"openai:{self.model}"

    def request(
        self, system: str, user: str, schema: dict, images: list[bytes] | None = None
    ) -> dict:
        content: list[dict] | str = user
        if images:
            content = [{"type": "text", "text": user}] + [
                {
                    "type": "image_url",
                    "image_url": {
                        "url": "data:image/png;base64," + base64.b64encode(b).decode()
                    },
                }
                for b in images
            ]
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
        return json.loads(r.json()["choices"][0]["message"]["content"])


_DEAD: set[str] = (
    set()
)  # backends that refused their key this process (tested once, not every call)


# Meta runs on the hackathon's free $100 credit only. Calls stay off until VLA_FORGE_META_CREDITS=1 (set once the
# credit shows under Billing > free credits), and stop for good once the estimated spend reaches the cap. Prices are
# set above Llama 4 Maverick's list price so the estimate runs high.
CREDIT_CAP_USD = float(os.environ.get("VLA_FORGE_META_CAP_USD", "80"))
PRICE_IN, PRICE_OUT = 0.50 / 1e6, 1.50 / 1e6
SPEND = RUNS / "meta-spend.json"


def meta_spent() -> float:
    try:
        return float(json.loads(SPEND.read_text())["usd"])
    except (OSError, ValueError, KeyError):
        return 0.0


def _charge(usage: dict, prompt: str) -> None:
    tin = usage.get("prompt_tokens") or len(prompt) // 3
    tout = usage.get("completion_tokens") or 1000
    usd = meta_spent() + tin * PRICE_IN + tout * PRICE_OUT
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
    try:
        LlamaClient(key).chat_json("Reply with JSON.", "ping", {"type": "object"})
        return True
    except httpx.HTTPStatusError as e:
        if e.response.status_code in (401, 402, 403):
            _DEAD.add("meta")
            return False
        return True  # a transient server error is not a dead key
    except (httpx.HTTPError, ValueError, KeyError, IndexError):
        return True


def live_backends() -> list[str]:
    """Names in the order they are tried: Meta's Llama API if its key works, then OpenAI."""
    out = []
    if os.environ.get("VLA_FORGE_LLAMA_STUB") == "1":
        return out
    k = api_key()
    if k and _meta_alive(k):
        out.append("meta")
    if openai_key():
        out.append("openai")
    return out


def _client(name: str):
    if name == "meta":
        return LlamaClient(api_key() or ""), f"meta:{MODEL}"
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
    """(parsed JSON, display model, backend that really answered). Live backends first, then the stub."""
    for name in backends or live_backends():
        if name == "meta" and not meta_allowed():
            continue
        try:
            c, label = _client(name)
            if images and name == "openai":
                raw = c.chat_json(system, user, schema, images)
            else:
                raw = c.chat_json(system, user, schema)
            return raw, MODEL, label
        except httpx.HTTPStatusError as e:
            if name == "meta" and e.response.status_code in (401, 402, 403):
                _DEAD.add("meta")
        except (httpx.HTTPError, ValueError, KeyError, IndexError):
            pass
    return Stub().chat_json(kind, system, user, schema), MODEL, "stub"


def chat_json(kind: str, system: str, user: str, schema: dict) -> tuple[dict, str]:
    """(parsed JSON, model name shown). A failed live call falls through to the next backend, then the stub."""
    raw, model, _backend = chat_json_ex(kind, system, user, schema)
    return raw, model
