from uagents import Model

from vla_forge.agentbus import BUS, protocol_of, summarize


class Ping(Model):
    run_id: str = ""
    n: int = 3
    rate: float = 0.5
    blob: str = '{"a": 1}'


def test_summary_uses_message_fields_only():
    s = summarize(Ping())
    assert "n: 3" in s and "rate: 0.50" in s and "blob: 8 chars" in s
    assert "run_id" not in s


def test_protocol_names():
    assert protocol_of("OptimizeRequest") == "forge-optimize 1.0"
    assert protocol_of("ChatMessage").startswith("AgentChatProtocol")
    assert protocol_of("RouteTask").startswith("forge-pipeline")


def test_record_names_receiver_and_orders():
    BUS.names["agent1xyz"] = "router"
    a = BUS.record("task", "agent1xyz", Ping())
    b = BUS.record("router", "agent1unknown00000", Ping())
    assert a["receiver"] == "router" and b["receiver"].endswith("…")
    assert b["seq"] == a["seq"] + 1
    assert [r["seq"] for r in BUS.since(a["seq"])] == [b["seq"]]
