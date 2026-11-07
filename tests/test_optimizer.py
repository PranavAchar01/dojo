import numpy as np
import pytest

from vla_forge import llm, montage, optimizer


def test_wilson_interval_brackets_the_rate():
    lo, hi = optimizer.wilson(60, 100)
    assert lo < 0.6 < hi
    assert 0.50 < lo < 0.52 and 0.68 < hi < 0.70
    assert optimizer.wilson(0, 0) == [0.0, 0.0]


def test_clean_change_clamps_and_rounds():
    c = optimizer.clean_change({"knob": "lr", "value": "5", "reason": "x"})
    assert c["value"] == 6e-4
    c = optimizer.clean_change({"knob": "batch", "value": "200", "reason": "x"})
    assert c["value"] == 256
    c = optimizer.clean_change(
        {"knob": "extra_demos_random", "value": "3.7", "reason": "x"}
    )
    assert c["value"] == 10
    c = optimizer.clean_change(
        {"knob": "scope", "value": "Action_Layers", "reason": "a — b"}
    )
    assert c["value"] == "action_layers" and "—" not in c["reason"]


def test_clean_change_rejects_unknown_knobs_and_scopes():
    with pytest.raises(ValueError):
        optimizer.clean_change({"knob": "rm -rf", "value": "1", "reason": ""})
    with pytest.raises(ValueError):
        optimizer.clean_change({"knob": "scope", "value": "everything", "reason": ""})
    with pytest.raises(ValueError):
        optimizer.clean_change({"knob": "lr", "value": "fast", "reason": ""})


def _scored(n=30):
    rng = np.random.default_rng(0)
    out = []
    for i in range(n):
        xy = [float(rng.uniform(0.15, 0.28)), float(rng.uniform(-0.1, 0.1))]
        out.append(
            {
                "success": i % 3 != 0,
                "failure": "jammed on the socket rim",
                "target_xy": xy,
                "goal_xy": None,
                "stats": {"min_grip_dist": 0.02 + 0.001 * i},
            }
        )
    return out


def test_failure_report_counts_tags():
    rep = optimizer.failure_report(_scored(), {"lr": 1e-4})
    assert rep["scenes"] == 30 and rep["succeeded"] == 20
    assert rep["failure_tags"] == {"jammed on the socket rim": 10}
    assert sum(v["scenes"] for v in rep["by_distance_from_base"].values()) == 30


def test_propose_validates_model_output(monkeypatch):
    monkeypatch.setattr(
        llm,
        "_chat_json",
        lambda *a: ({"knob": "lr", "value": "0.0001", "reason": "ok"}, "Llama-test"),
    )
    ch, model = optimizer.propose({}, [], 1)
    assert ch["knob"] == "lr" and model == "Llama-test"


def test_propose_falls_back_to_the_fixed_menu(monkeypatch):
    def boom(*a):
        raise RuntimeError("no language model available")

    monkeypatch.setattr(llm, "_chat_json", boom)
    ch, model = optimizer.propose({}, [], 2)
    assert model == "rules" and ch["knob"] in optimizer.KNOBS
    monkeypatch.setattr(
        llm,
        "_chat_json",
        lambda *a: ({"knob": "bogus", "value": "1", "reason": ""}, "m"),
    )
    assert optimizer.propose({}, [], 1)[1] == "rules"


def test_asset_refuses_paths_outside_the_run(tmp_path, monkeypatch):
    monkeypatch.setattr(optimizer, "RUNS", tmp_path)
    (tmp_path / "opt-abc").mkdir()
    (tmp_path / "opt-abc" / "hero_r0.jpg").write_bytes(b"x")
    (tmp_path / "secret.jpg").write_bytes(b"x")
    assert optimizer.asset("abc", "hero_r0.jpg") is not None
    assert optimizer.asset("abc", "../secret.jpg") is None
    assert optimizer.asset("../abc", "hero_r0.jpg") is None
    assert optimizer.asset("abc", "run.json") is None


def test_quaternion_helpers_compose():
    q = np.array([np.cos(0.3), 0, 0, np.sin(0.3)])
    ident = montage._qmul(montage._conj(q), q)
    assert np.allclose(ident, [1, 0, 0, 0])


def test_describe_covers_every_knob():
    for knob, (lo, hi, _) in optimizer.KNOBS.items():
        v = "action_layers" if knob == "scope" else str(lo)
        ch = optimizer.clean_change({"knob": knob, "value": v, "reason": ""})
        assert optimizer.describe(ch)
