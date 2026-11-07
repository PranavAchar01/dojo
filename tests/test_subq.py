import json

import httpx
import pytest

from vla_forge import lanes, subq


@pytest.fixture(autouse=True)
def fresh(monkeypatch, tmp_path):
    monkeypatch.setattr(subq, "SPEND", tmp_path / "subq-spend.json")
    monkeypatch.setattr(subq, "api_key", lambda: "k")
    monkeypatch.setattr(subq, "base_url", lambda: "https://api.subq.test/v1")
    monkeypatch.setattr(subq, "WHY", {})
    monkeypatch.delenv("VLA_FORGE_SUBQ_OFF", raising=False)


def _mock(handler):
    return httpx.Client(transport=httpx.MockTransport(handler))


def test_decide_posts_to_the_decisions_api_and_counts_tokens():
    seen = {}

    def handler(req):
        seen["url"], seen["auth"] = str(req.url), req.headers["authorization"]
        seen["body"] = json.loads(req.content)
        return httpx.Response(
            200,
            json={
                "answers": {
                    "pick": {
                        "choice": "b",
                        "probabilities": {"a": 0.2, "b": 0.8},
                        "confidence": 0.8,
                    }
                },
                "usage": {"input_tokens": 1234, "output_tokens": 0},
            },
        )

    q = {"pick": {"type": "choice", "options": ["a", "b"]}}
    out = subq.decide({"x": 1}, q, "optimizer", http=_mock(handler))
    assert seen["url"] == "https://api.subq.test/v1/systemone"
    assert seen["auth"] == "Bearer k"
    assert seen["body"]["model"] == subq.MODEL and seen["body"]["questions"] == q
    assert subq.choice(out["pick"]) == ("b", 0.8)
    assert subq.tokens_used() == 1234
    last = lanes.RECENT[-1]
    assert last["lane"] == "subq" and last["tokens_in"] == 1234 and not last["error"]


def test_token_cap_stops_calls_before_they_are_sent(monkeypatch):
    monkeypatch.setattr(subq, "CAP_TOKENS", 10)
    subq._charge(10)
    sent = []
    with pytest.raises(subq.Unavailable):
        subq.decide(
            "s", {}, "task", http=_mock(lambda r: sent.append(r) or httpx.Response(200))
        )
    assert sent == [] and not subq.status()["available"]
    assert "cap" in subq.status()["why"]


def test_a_refused_key_marks_subq_unavailable_with_the_reason():
    with pytest.raises(subq.Unavailable):
        subq.decide("s", {}, "task", http=_mock(lambda r: httpx.Response(401)))
    st = subq.status()
    assert not st["available"] and "401" in st["why"] and st["backend"] == "unavailable"


def test_long_state_is_cut_to_the_context_budget():
    big = {"rows": ["x" * 100] * 10_000}
    out = subq.fit(big, max_tokens=1000)
    assert isinstance(out, str) and len(out) < 1000 * subq.BYTES_PER_TOKEN + 60
    assert subq.fit({"a": 1}) == {"a": 1}


def test_both_runs_the_two_lanes_at_once_and_keeps_errors_in_their_slot():
    import threading

    gate = threading.Barrier(2, timeout=5)

    def a():
        gate.wait()  # only passes if the other lane is running at the same time
        return "meta"

    def b():
        gate.wait()
        raise RuntimeError("subq down")

    ra, rb = lanes.both(a, b)
    assert ra == "meta" and isinstance(rb, RuntimeError)


def test_tally_counts_each_winner():
    t = lanes.tally(
        [
            {"winner": "meta"},
            {"winner": "subq"},
            {"winner": "subq"},
            {"winner": "agreed"},
        ]
    )
    assert t["meta"] == 1 and t["subq"] == 2 and t["agreed"] == 1 and t["tie"] == 0


def test_menu_skips_reverted_changes_and_no_ops():
    from vla_forge import optimizer as O

    cfg = dict(O.DEFAULTS)
    hist = [{"change": {"knob": "lr", "value": 2e-4}, "kept": False}]
    opts = list(O.menu(cfg, hist).values())
    assert not any(c["knob"] == "lr" and c["value"] == 2e-4 for c in opts)
    assert not any(c["knob"] == "batch" and c["value"] == cfg["batch"] for c in opts)
    assert any(c["knob"] == "extra_demos_near_failures" for c in opts)
    assert 2 <= len(opts) <= 255


def test_subq_change_maps_the_pick_back_to_a_valid_change(monkeypatch):
    from vla_forge import optimizer as O

    def fake(state, questions, kind, http=None):
        crit = questions["change"]["criteria"]
        names = list(crit)
        return {
            "change": {
                "choice": names[3],
                "probabilities": {names[3]: 0.6, names[0]: 0.3, names[1]: 0.1},
            }
        }

    monkeypatch.setattr(subq, "decide", fake)
    ch, p = O.subq_change({}, dict(O.DEFAULTS), [])
    assert p == 0.6 and ch == {**ch, **O.clean_change({**ch, "reason": ""})} | {
        "reason": ch["reason"],
        "runners_up": ch["runners_up"],
    }
    assert len(ch["runners_up"]) == 2


def test_propose_pair_returns_both_and_keeps_meta_when_subq_fails(monkeypatch):
    from vla_forge import optimizer as O

    monkeypatch.setattr(
        O, "propose", lambda *a: ({"knob": "lr", "value": 1e-4}, "meta:muse-wonder-1.0")
    )

    def down(*a):
        raise subq.Unavailable("SubQ rejected the API key")

    monkeypatch.setattr(O, "subq_change", down)
    (_m, be), (q, _p, err) = O.propose_pair({}, [], 1, None, None, {}, {})
    assert be == "meta:muse-wonder-1.0" and q is None and "rejected" in err
