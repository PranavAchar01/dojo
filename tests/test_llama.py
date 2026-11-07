import json

import httpx
import pytest

from vla_forge import llama, llm

SCHEMA = {"type": "object", "properties": {"a": {"type": "string"}}}


@pytest.fixture(autouse=True)
def free_credit(monkeypatch, tmp_path):
    monkeypatch.setenv("VLA_FORGE_META_CREDITS", "1")
    monkeypatch.setattr(llama, "SPEND", tmp_path / "meta-spend.json")


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
    monkeypatch.setattr(llama, "openai_key", lambda: None)
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
    monkeypatch.setattr(llama, "openai_key", lambda: None)
    monkeypatch.setattr(llama, "_DEAD", set())
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
    assert len(plan["stations"]) == 2
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


def _mock(handler):
    return httpx.Client(transport=httpx.MockTransport(handler))


def test_a_dead_meta_key_falls_back_to_openai(monkeypatch):
    calls = []

    def meta(req: httpx.Request) -> httpx.Response:
        calls.append("meta")
        return httpx.Response(401, json={"error": "invalid_api_key"})

    seen = {}

    def openai(req: httpx.Request) -> httpx.Response:
        seen["url"], seen["body"] = str(req.url), json.loads(req.content)
        return httpx.Response(
            200, json={"choices": [{"message": {"content": '{"a": "z"}'}}]}
        )

    monkeypatch.setattr(llama, "_DEAD", set())
    monkeypatch.setattr(llama, "api_key", lambda: "dead")
    monkeypatch.setattr(llama, "openai_key", lambda: "ok")
    real_llama, real_openai = llama.LlamaClient, llama.OpenAIClient
    monkeypatch.setattr(
        llama, "LlamaClient", lambda key: real_llama(key, http=_mock(meta))
    )
    monkeypatch.setattr(
        llama, "OpenAIClient", lambda key: real_openai(key, http=_mock(openai))
    )
    assert llama.live_backends() == ["openai"]  # probed once, then retired
    assert llama.live_backends() == ["openai"] and calls == ["meta"]
    raw, model, backend = llama.chat_json_ex("optimizer", "s", "u", SCHEMA, [b"png"])
    assert raw == {"a": "z"} and backend == f"openai:{llama.OPENAI_MODEL}"
    assert model == llama.MODEL  # the UI keeps the Llama name
    assert seen["url"].endswith("/chat/completions")
    parts = seen["body"]["messages"][1]["content"]
    assert parts[1]["image_url"]["url"].startswith("data:image/png;base64,")


def test_a_working_meta_key_is_used_first(monkeypatch):
    def meta(req):
        return httpx.Response(
            200, json={"choices": [{"message": {"content": '{"a": "m"}'}}]}
        )

    monkeypatch.setattr(llama, "_DEAD", set())
    monkeypatch.setattr(llama, "api_key", lambda: "good")
    monkeypatch.setattr(llama, "openai_key", lambda: "ok")
    real = llama.LlamaClient
    monkeypatch.setattr(llama, "LlamaClient", lambda key: real(key, http=_mock(meta)))
    assert llama.live_backends() == ["meta", "openai"]
    raw, _model, backend = llama.chat_json_ex("optimizer", "s", "u", SCHEMA)
    assert raw == {"a": "m"} and backend == f"meta:{llama.MODEL}"


def test_both_backends_failing_ends_on_the_stub(monkeypatch):
    def boom(req):
        return httpx.Response(500)

    monkeypatch.setattr(llama, "_DEAD", set())
    monkeypatch.setattr(llama, "api_key", lambda: None)
    monkeypatch.setattr(llama, "openai_key", lambda: "ok")
    real = llama.OpenAIClient
    monkeypatch.setattr(llama, "OpenAIClient", lambda key: real(key, http=_mock(boom)))
    _raw, _m, backend = llama.chat_json_ex(
        "task", "s", "put the green block in the back-right slot of the tray", {}
    )
    assert backend == "stub"


def test_meta_stays_off_without_the_free_credit_and_past_the_cap(monkeypatch):
    calls = []

    def meta(req: httpx.Request) -> httpx.Response:
        calls.append("meta")
        return httpx.Response(
            200,
            json={
                "choices": [{"message": {"content": '{"a": "m"}'}}],
                "usage": {"prompt_tokens": 1000, "completion_tokens": 1000},
            },
        )

    monkeypatch.setattr(llama, "_DEAD", set())
    monkeypatch.setattr(llama, "api_key", lambda: "ok")
    monkeypatch.setattr(llama, "openai_key", lambda: None)
    real_llama = llama.LlamaClient
    monkeypatch.setattr(
        llama, "LlamaClient", lambda key: real_llama(key, http=_mock(meta))
    )
    monkeypatch.delenv("VLA_FORGE_META_CREDITS")
    assert llama.live_backends() == [] and calls == []
    monkeypatch.setenv("VLA_FORGE_META_CREDITS", "1")
    _raw, _model, backend = llama.chat_json_ex(
        "plan", "s", "u", SCHEMA, backends=["meta"]
    )
    assert backend.startswith("meta:") and llama.meta_spent() == pytest.approx(0.002)
    monkeypatch.setattr(llama, "CREDIT_CAP_USD", 0.001)
    _raw, _model, backend = llama.chat_json_ex(
        "plan", "s", "u", SCHEMA, backends=["meta"]
    )
    assert backend == "stub" and len(calls) == 1
