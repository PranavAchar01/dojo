"""A published model keeps its run: the record written at publish reads back the same after a reload."""

from __future__ import annotations

import json

from vla_forge import records
from vla_forge.pipeline import Round, Run


def _run() -> Run:
    run = Run(id="123456-abcd", prompt="push the blue block onto the orange square")
    run.route = {"decision": "train", "reason": "nothing close in the library"}
    run.layouts = [{}] * 20
    r = Round(0, demos=18, demos_asked=20, train={"steps": 900, "seconds": 41.5}, eval={"succeeded": 13, "n": 20},
              plan={"action": "stop", "rationale": "good enough"}, successes=[True] * 13 + [False] * 7,
              loss=[[i * 0.4, 2.0 / (i + 1)] for i in range(500)])
    run.rounds = [r]
    run.best_round = 0
    return run


def test_publish_then_reload_shows_the_same_fields(tmp_path, monkeypatch):
    monkeypatch.setattr(records, "RECORDS", tmp_path / "runs")
    monkeypatch.setattr(records, "SKILLS_DIR", tmp_path)
    records.save("a-abcd", records.from_run(_run()))
    got = records.load("a-abcd")
    assert got["route"]["decision"] == "train"
    assert got["demos"]["ok"] == 18 and got["demos"]["asked"] == 20
    assert got["layout_seed"] > 0 and got["n_scenes"] == 20
    assert got["train"]["steps"] == 900 and len(got["train"]["loss_curve"]) <= records.LOSS_POINTS
    assert got["eval"][0]["successes"].count(True) == 13
    assert got["critic"][0]["rationale"] == "good enough"


def test_missing_record_is_empty_and_ladder_backfill_only_matches(tmp_path, monkeypatch):
    monkeypatch.setattr(records, "RECORDS", tmp_path / "runs")
    monkeypatch.setattr(records, "SKILLS_DIR", tmp_path / "skills")
    (tmp_path / "skills").mkdir()
    assert records.load("none") == {}
    (tmp_path / "skills" / "registry.json").write_text(json.dumps([{"id": "x-7dde", "instruction": "i"}]))
    row = {"run": "0-7dde", "status": "published", "instruction": "i", "rounds": ["9/20", "15/20"], "init": "base"}
    ladder = tmp_path / "ladder.jsonl"
    ladder.write_text(json.dumps(row) + "\n" + json.dumps({**row, "run": "0-0000"}) + "\n")
    assert records.backfill(ladder) == ["x-7dde"]
    assert records.load("x-7dde")["eval"][1]["succeeded"] == 15
