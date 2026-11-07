import json

import numpy as np
import pytest

from vla_forge import llama, montage, optimizer
from vla_forge.registry import Registry


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
                "index": i,
                "steps": 100,
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


def _ask(monkeypatch, replies):
    it = iter(replies)
    monkeypatch.setattr(llama, "chat_json_ex", lambda *a, **k: next(it))


def test_propose_validates_model_output(monkeypatch):
    _ask(
        monkeypatch,
        [
            (
                {"knob": "lr", "value": "0.0001", "diagnosis": "d", "reason": "ok"},
                "m",
                "openai:test",
            )
        ],
    )
    ch, backend = optimizer.propose({}, [], 1)
    assert ch["knob"] == "lr" and backend == "openai:test" and ch["diagnosis"] == "d"


def test_propose_falls_back_to_the_fixed_menu(monkeypatch):
    def boom(*a, **k):
        raise RuntimeError("no language model available")

    monkeypatch.setattr(llama, "chat_json_ex", boom)
    ch, backend = optimizer.propose({}, [], 2)
    assert backend == "rules" and ch["knob"] in optimizer.KNOBS
    _ask(monkeypatch, [({"knob": "bogus", "value": "1", "reason": ""}, "m", "b")])
    assert optimizer.propose({}, [], 1)[1] == "rules"


def _hist(knob, value, kept):
    return {
        "round": 1,
        "change": {"knob": knob, "value": value, "reason": ""},
        "success_before": 60,
        "success_after": 58,
        "kept": kept,
    }


def test_a_reverted_change_may_not_come_back(monkeypatch):
    hist = [_hist("lr", 0.0001, False)]
    again = ({"knob": "lr", "value": "0.0001", "reason": "x"}, "m", "openai:t")
    other = ({"knob": "lr", "value": "0.0003", "reason": "x"}, "m", "openai:t")
    seen = []
    it = iter([again, other])

    def ask(kind, system, user, schema, images=None, backends=None):
        seen.append(json.loads(user))
        return next(it)

    monkeypatch.setattr(llama, "chat_json_ex", ask)
    ch, backend = optimizer.propose({}, hist, 2)
    assert ch["value"] == 0.0003 and backend == "openai:t"
    assert seen[0]["rejected"] == [{"knob": "lr", "value": 0.0001}]
    assert "rejected_this_round" in seen[1]
    # twice the same: the fixed menu takes over, and it also skips the barred change
    _ask(monkeypatch, [again, again])
    ch, backend = optimizer.propose({}, hist, 2)
    assert backend == "rules" and not (ch["knob"] == "lr" and ch["value"] == 0.0001)


def test_kept_changes_are_not_barred():
    assert optimizer.barred(
        [_hist("lr", 0.0001, True), _hist("batch", 128, False)]
    ) == [{"knob": "batch", "value": 128}]


def test_split_sets_are_disjoint_and_stable():
    sets = optimizer.split(100, 100)
    assert len(sets["select"]) == 100 and len(sets["test"]) == 100
    all_seeds = sets["train"] + sets["select"] + sets["test"]
    assert len(all_seeds) == len(set(all_seeds))
    assert sets == optimizer.split(100, 100)  # fixed seeds
    # the real draw order of the demonstrations stays inside the train set
    used = set(
        range(optimizer.START_SEED0, optimizer.START_SEED0 + optimizer.START_DEMOS)
    )
    for r in range(1, optimizer.MAX_ROUNDS + 1):
        used |= {optimizer.ROUND_SEED0 + 1000 * r + i for i in range(60)}
    assert used <= set(sets["train"])


def test_overlapping_sets_are_refused():
    with pytest.raises(ValueError):
        optimizer.check_disjoint({"train": [1, 2], "select": [2, 3], "test": [4]})
    with pytest.raises(ValueError):
        optimizer.check_disjoint({"a": [1, 1], "b": [2]})


def test_select_and_test_scenes_differ():
    from vla_forge.sim.tasks import sample_layout

    spec = Registry().skills["insert-the-green-block-into-the-socket-f611"].task()
    a = sample_layout(spec, np.random.default_rng(optimizer.SEED0)).to_json()
    b = sample_layout(spec, np.random.default_rng(optimizer.TEST_SEED0)).to_json()
    assert a != b


def test_mcnemar_and_paired_bootstrap():
    before = [0] * 20 + [1] * 10
    after = [1] * 14 + [0] * 6 + [1] * 10  # 14 fixed, 0 broken
    m = optimizer.mcnemar_exact(before, after)
    assert m["fixed"] == 14 and m["broken"] == 0 and m["p"] < 0.001
    assert optimizer.mcnemar_exact(before, before) == {
        "fixed": 0,
        "broken": 0,
        "p": 1.0,
    }
    c = optimizer.compare(before, after)
    assert c["delta"] == round(14 / 30, 3)
    lo, hi = c["delta_ci"]
    assert lo < c["delta"] < hi and lo > 0
    # a 5/5 split of discordant pairs is not evidence of anything
    even = [1, 0] * 5 + [0, 1] * 5
    swap = [0, 1] * 5 + [1, 0] * 5
    assert optimizer.mcnemar_exact(even, swap)["p"] == 1.0


def test_control_changes_are_random_valid_and_skip_reverted():
    rng = np.random.default_rng(3)
    hist = [_hist("extra_demos_random", 30, False)]
    knobs = set()
    for _ in range(40):
        ch = optimizer.random_change(rng, hist)
        again = optimizer.clean_change({**ch, "reason": ""})
        assert again["knob"] == ch["knob"] and again["value"] == ch["value"]
        assert not (ch["knob"] == "extra_demos_random" and ch["value"] == 30)
        assert ch["reason"].startswith("control")
        knobs.add(ch["knob"])
    assert len(knobs) >= 5  # not one knob in disguise


def test_failure_report_carries_per_failure_evidence():
    sc = _scored()
    for i, s in enumerate(sc):
        s["index"], s["steps"] = i, 100
        s["stats"] = {"min_grip_dist": 0.02, "target_max_z": 0.03, "target_moved": 0.01}
    rep = optimizer.failure_report(sc, {})
    f = rep["failures"][0]
    assert {"scene", "tag", "target_xy", "min_grip_dist", "target_max_z"} <= set(f)
    assert rep["per_tag_mean_stats"]["jammed on the socket rim"]["target_max_z"] == 0.03


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


def test_a_frozen_scope_trains_without_dropout_in_the_frozen_layers():
    import torch

    from vla_forge.learn import quiet_frozen_layers, scoped_params
    from vla_forge.policy import VLA

    m = VLA().train()
    scoped_params(m, "action_layers")
    quiet_frozen_layers(m)
    assert not m.head.xattn.training and m.head.enc.training
    assert isinstance(m, torch.nn.Module)


def test_evidence_scenes_spread_and_rotate():
    sc = _scored(60)
    a = optimizer.pick_evidence(sc, 1)
    b = optimizer.pick_evidence(sc, 2)
    assert len(a) == 4 and all(not sc[i]["success"] for i in a)
    assert a != b
    assert optimizer.pick_evidence(_scored(5), 1) == [0, 3]


def test_scene_notes_are_cleaned():
    c = optimizer.clean_change(
        {
            "knob": "lr",
            "value": "1e-4",
            "reason": "r",
            "scene_notes": [
                {"scene": 3, "note": "block — on rim"},
                {"scene": "x"},
                {"note": "n"},
            ],
        }
    )
    assert c["scene_notes"] == [{"scene": 3, "note": "block , on rim"}]


def test_split_stays_disjoint_for_a_long_perpetual_run():
    sets = optimizer.split(100, 100, max_rounds=400)
    assert len(sets["train"]) + 200 == len(
        set(sets["train"] + sets["select"] + sets["test"])
    )


def test_population_database_roundtrip(tmp_path):
    from openevolve.database import Program

    from vla_forge import optimizer_evo as E

    rec = {"cfg": dict(optimizer.DEFAULTS), "plan": []}
    db = E.new_db(tmp_path)
    for i, k in enumerate((50, 55)):
        db.add(
            Program(
                id=f"p{i}",
                code=__import__("json").dumps(rec),
                language="json",
                metrics=E.metrics_of(k, 100, rec),
                metadata={"round": i, "light": []},
            ),
            iteration=i,
            target_island=0,
        )
    parent, _ = db.sample(num_inspirations=1)
    assert parent.id in ("p0", "p1")
    db.save(str(tmp_path), iteration=1)
    db2 = E.new_db(tmp_path)
    db2.load(str(tmp_path))
    assert sorted(db2.programs) == ["p0", "p1"]
    assert (
        max(db2.programs.values(), key=lambda p: p.metrics["combined_score"]).id == "p1"
    )
