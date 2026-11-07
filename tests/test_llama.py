import json

import httpx
import pytest

from vla_forge import llama, llm

SCHEMA = {"type": "object", "properties": {"a": {"type": "string"}}}


def test_client_sends_the_documented_request():
    seen = {}

    def handler(req: httpx.Request) -> httpx.Response:
        seen["url"], seen["auth"] = str(req.url), req.headers["authorization"]
        seen["body"] = json.loads(req.content)
        return httpx.Response(
            200, json={"choices": [{"message": {"content": '{"a": "x"}'}}]}
        )

    c = llama.LlamaClient(
        "k", http=httpx.Client(transport=httpx.MockTransport(handler))
    )
    assert c.chat_json("sys", "usr", SCHEMA) == {"a": "x"}
    assert seen["url"] == "https://api.llama.com/compat/v1/chat/completions"
    assert seen["auth"] == "Bearer k"
    rf = seen["body"]["response_format"]
    assert rf["type"] == "json_schema" and rf["json_schema"]["schema"] == SCHEMA
    assert seen["body"]["model"] == llama.MODEL


def test_client_reads_the_native_reply_shape():
    def handler(req):
        return httpx.Response(
            200,
            json={
                "completion_message": {
                    "content": {"type": "text", "text": '{"a": "y"}'}
                }
            },
        )

    c = llama.LlamaClient(
        "k", http=httpx.Client(transport=httpx.MockTransport(handler))
    )
    assert c.chat_json("s", "u", SCHEMA) == {"a": "y"}


def test_stub_is_used_without_a_key(monkeypatch):
    monkeypatch.setattr(llama, "api_key", lambda: None)
    assert llama.use_stub()
    raw, model = llm._chat_json(
        "s", "put the green block in the back-right slot of the tray", {}, "task"
    )
    assert model == llama.MODEL
    assert (
        raw["steps"][0]["skill"] == "kit"
        and raw["steps"][0]["target_color"] == "back-right"
    )


def test_failed_live_call_falls_back_to_the_stub(monkeypatch):
    monkeypatch.setattr(llama, "api_key", lambda: "k")
    monkeypatch.delenv("VLA_FORGE_LLAMA_STUB", raising=False)

    def boom(self, *a):
        raise httpx.ConnectError("down")

    monkeypatch.setattr(llama.LlamaClient, "chat_json", boom)
    raw, _ = llama.chat_json("task", "s", "touch the red block", {})
    assert raw["steps"][0]["skill"] == "reach"


def test_stub_optimizer_replays_the_saved_run():
    raw = llama.Stub().chat_json("optimizer", "s", json.dumps({"round": 1}), {})
    assert raw["knob"] == "extra_demos_near_failures" and raw["value"] == "50"


def test_stub_plan_and_critic_are_schema_shaped():
    plan = llama.Stub().chat_json(
        "plan",
        "s",
        "two arms: seat the housing in the pallet then insert it into the socket",
        {},
    )
    assert plan["stations"]
    crit = llama.Stub().chat_json(
        "critic",
        "s",
        json.dumps(
            {
                "evaluation": {"success_rate": 0.5},
                "history": [],
                "settings": {},
                "seconds_left": 120,
                "target_success_rate": 0.8,
            }
        ),
        {},
    )
    assert crit["action"] in ("publish", "retrain")


def test_unknown_call_kind_is_rejected():
    with pytest.raises(ValueError):
        llama.Stub().chat_json("nope", "s", "u", {})
