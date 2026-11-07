"""Meta's Llama API client, with a stub backend that stands in until a key works.

Targets the Llama API's OpenAI-compatible chat completions endpoint, https://api.llama.com/compat/v1/chat/completions,
with `Authorization: Bearer <key>` and a schema-constrained reply
(`response_format: {"type": "json_schema", "json_schema": {"name", "schema"}}`). The key is META_API_KEY (or
LLAMA_API_KEY), else the macOS Keychain item of the same name. Models: VLA_FORGE_META_LLM, default
Llama-4-Maverick-17B-128E-Instruct-FP8 (Llama-4-Scout-17B-16E-Instruct-FP8 and Llama-3.3-70B-Instruct also work).

The stub backend answers the same four calls (task, plan, critic, optimizer) with replies that satisfy the same
schemas, without a network. It is used when VLA_FORGE_LLAMA_STUB=1, when no key is set, or when a live call fails.
Planner, task and critic replies come from the keyword rules; optimizer replies replay the changes recorded in a
saved optimizer run, so the numbers on screen are that run's measured ones.
"""

from __future__ import annotations

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

            return _rule_plan(user)
        if kind == "critic":
            from .llm import _rule_critic

            d = json.loads(user)
            return _rule_critic(
                d["evaluation"],
                d["settings"],
                d["seconds_left"],
                d["target_success_rate"],
            )
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
    return os.environ.get("VLA_FORGE_LLAMA_STUB") == "1" or api_key() is None


def chat_json(kind: str, system: str, user: str, schema: dict) -> tuple[dict, str]:
    """(parsed JSON, model that answered). A failed live call falls through to the stub."""
    if not use_stub():
        try:
            key = api_key()
            assert key is not None
            return LlamaClient(key).chat_json(system, user, schema), MODEL
        except (httpx.HTTPError, ValueError, KeyError, IndexError):
            pass
    return Stub().chat_json(kind, system, user, schema), MODEL
