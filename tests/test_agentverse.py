import asyncio

import pytest

from vla_forge import agents


@pytest.fixture
def forge(monkeypatch):
    monkeypatch.setenv("AGENTVERSE_API_KEY", "av-secret-key")
    monkeypatch.delenv("VLA_FORGE_PUBLIC_URL", raising=False)
    loop = asyncio.new_event_loop()
    yield agents.Forge(loop)
    loop.close()


def test_registers_task_and_optimizer_only(forge, monkeypatch):
    calls = []
    monkeypatch.setattr(
        "uagents_core.utils.registration.register_agent",
        lambda req, cfg, creds: calls.append((req, creds)) or True,
    )
    done = agents.register_on_agentverse(forge)
    assert set(done) == {"task", "optimizer"}
    assert [c[0].name for c in calls] == ["VLA Forge", "VLA Forge Optimizer"]
    for req, creds in calls:
        assert req.type == "mailbox" and "mailbox" in req.endpoint
        low = req.readme.lower()
        assert "mujoco" in low and "uagents" in low and "llama" in low
        assert not any(w in low for w in ("vividly", "deepseek", "gemini"))
    assert calls[0][1].agent_seed_phrase == agents._seed("task")
    forge.registered = list(done)
    note = forge.graph()["agentverse"]["note"]
    assert note.startswith("registered on Agentverse")
    assert forge.agents["task"].address in note and "VLA Forge Optimizer" in note


def test_failure_keeps_old_footer_and_shows_error(forge, monkeypatch):
    def boom(req, cfg, creds):
        raise RuntimeError(f"401 for {creds.agentverse_api_key}")

    monkeypatch.setattr("uagents_core.utils.registration.register_agent", boom)
    assert agents.register_on_agentverse(forge) == {}
    note = forge.graph()["agentverse"]["note"]
    assert note.startswith("local Bureau, not registered on Agentverse")
    assert "401" in note and "av-secret-key" not in note
