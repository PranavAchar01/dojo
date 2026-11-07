"""The bridge: rules pass for generated builds, each rule fails for the fault it names, and the API behaves."""

from dataclasses import replace

import pytest
from fastapi.testclient import TestClient

from vla_forge import hero_bricks as hb
from vla_forge import hero_muse as hm
from vla_forge import hero_rules as hr
from vla_forge.server import app


def ids(results):
    return {r.id: r.ok for r in results}


@pytest.mark.parametrize(
    "rules",
    [
        hb.Rules(),
        hb.Rules(gap=2, pillar=3, deck=4, tier=False),
        hb.Rules(gap=4, pillar=1, deck=8),
    ],
)
def test_generated_bridge_passes_every_rule(rules):
    results = hr.check(hb.generate(rules), rules)
    assert all(r.ok for r in results), [r for r in results if not r.ok]


def test_rules_reject_impossible_numbers():
    assert hb.Rules(gap=6, deck=4).problems()
    assert hb.Rules(gap=1, deck=8).problems()
    with pytest.raises(ValueError):
        hb.generate(hb.Rules(gap=6, deck=4))


def _swap(bricks, name, **kw):
    return tuple(replace(b, **kw) if b.name == name else b for b in bricks)


def test_offset_deck_fails_rest_rule():
    rules = hb.Rules()
    bad = _swap(hb.generate(rules), "deck", col=4)
    assert not ids(hr.check(bad, rules, physics=False))["deck_rests"]


def test_uneven_pillars_fail_level_rule():
    rules = hb.Rules()
    bricks = tuple(b for b in hb.generate(rules) if b.name != "right2")
    res = ids(hr.check(bricks, rules, physics=False))
    assert not res["tops"]
    assert not res["shape"]


def test_intruder_in_gap_fails_gap_rule():
    rules = hb.Rules()
    extra = hb.Brick("x", "3003", "blue", 3, 0, 0, 0, "intruder")
    res = ids(hr.check((*hb.generate(rules), extra), rules, physics=False))
    assert not res["gap_empty"]


def test_misaligned_pillar_fails_stack_rule():
    rules = hb.Rules()
    bad = _swap(hb.generate(rules), "left2", col=1)
    assert not ids(hr.check(bad, rules, physics=False))["stacked"]


def test_settle_is_measured():
    s = hr.settle(hb.generate())
    assert s["max_mm"] <= hr.SETTLE_MAX_MM
    assert s["deck_tilt_deg"] <= hr.DECK_TILT_MAX_DEG


def test_validate_refuses_bad_rows():
    rows = hb.to_json(hb.generate())
    ok, problems = hm.validate(rows, "muse")
    assert not problems and len(ok) == len(rows)
    assert hm.validate([{**rows[0], "part": "9999"}], "muse")[1]
    assert hm.validate([{**rows[0], "colour": "pink"}], "muse")[1]
    assert hm.validate([{**rows[0], "col": 99}], "muse")[1]
    assert hm.validate([rows[0], rows[0]], "muse")[1]  # overlap
    assert hm.validate([{**rows[0], "lvl": 6}], "muse")[1]  # rests on nothing
    assert hm.validate([], "muse")[1]


def test_diff_names_each_change():
    a = hb.to_json(hb.generate())
    b = [dict(r) for r in a]
    b[0]["col"] += 1
    b.pop()
    out = hm.diff(a, b)
    assert any("step 1" in line and "col" in line for line in out)
    assert any("removed" in line for line in out)


def test_score_counts_exact_matches():
    rules = hb.Rules()
    ref = hb.generate(rules)
    s = hm.score(ref, ref, rules)
    assert s["exact_matches"] == len(ref) and s["rules_passed"] == s["rules_total"]


@pytest.fixture
def client(monkeypatch, tmp_path):
    monkeypatch.setattr(hb, "ACTIVE", tmp_path / "build.json")
    return TestClient(app)


def test_api_state_generate_check(client):
    s = client.get("/api/build/state").json()
    assert not s["muse"]["live"] or s["muse"]["live"] is True
    assert len(s["bricks"]) == 6
    r = client.post(
        "/api/build/check", json={"rules": s["rules"], "bricks": s["bricks"]}
    ).json()
    assert all(x["ok"] for x in r["results"])
    assert (
        client.post("/api/build/generate", json={"gap": 6, "deck": 4}).status_code
        == 422
    )


def test_api_mesh(client):
    assert len(client.get("/api/build/mesh/3003").json()["v"]) > 30
    assert client.get("/api/build/mesh/nope").status_code == 404


def test_api_use_saves_only_valid_builds(client):
    s = client.get("/api/build/state").json()
    assert (
        client.post(
            "/api/build/use", json={"rules": s["rules"], "bricks": s["bricks"]}
        ).status_code
        == 200
    )
    assert hb.load_active() is not None
    bad = [*s["bricks"], {**s["bricks"][0], "name": "dup"}]
    assert (
        client.post(
            "/api/build/use", json={"rules": s["rules"], "bricks": bad}
        ).status_code
        == 422
    )


def test_api_fix_with_stubbed_muse(client, monkeypatch):
    s = client.get("/api/build/state").json()
    monkeypatch.setattr(hm, "available", lambda: (True, ""))
    taller = hb.generate(hb.Rules(pillar=3))
    reply = {
        "summary": "one brick taller",
        "steps": [
            {k: v for k, v in r.items() if k in hm.STEP_PROPS}
            for r in hb.to_json(taller)
        ],
    }
    monkeypatch.setattr(
        hm,
        "_call",
        lambda user, images=None: (reply, {"prompt_tokens": 1, "completion_tokens": 2}),
    )
    r = client.post(
        "/api/build/fix",
        json={
            "rules": {"pillar": 3},
            "bricks": s["bricks"],
            "instruction": "make the pillars taller",
        },
    ).json()
    assert r["ok"] and r["diff"] and all(b["source"] == "muse" for b in r["bricks"])


def test_api_fix_refused_without_muse(client, monkeypatch):
    monkeypatch.setattr(hm, "available", lambda: (False, "off"))
    s = client.get("/api/build/state").json()
    r = client.post(
        "/api/build/fix",
        json={
            "rules": s["rules"],
            "bricks": s["bricks"],
            "instruction": "do something",
        },
    )
    assert r.status_code == 503


def test_flow_detail_names_failing_rules():
    from vla_forge import hero_flow as hf

    builds = [
        {"seated": 3, "total": 6, "rules_passed": 6, "rules_total": 8, "rules": [{"id": "shape", "ok": False}, {"id": "deck_rests", "ok": False}]},
        {"seated": 6, "total": 6, "rules_passed": 8, "rules_total": 8, "rules": [{"id": "shape", "ok": True}]},
    ]
    assert hf.failures(builds) == {"bricks seated": 1, "shape": 1, "deck_rests": 1}
    txt = hf.detail(builds, 1.0)
    assert "not a trained policy" in txt and "1/2 builds pass every rule" in txt and "shape (1/2)" in txt
