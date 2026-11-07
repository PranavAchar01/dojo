"""A finished multi-arm line becomes one Library card whose numbers are only what the run measured."""

from __future__ import annotations

from types import SimpleNamespace

from vla_forge import multi, records


def _line(lid: str, completed: int, created: float = 1.0) -> dict:
    return {
        "id": lid,
        "prompt": "two arms",
        "name": "Housing",
        "created": created,
        "status": "done",
        "assembly": True,
        "stations": [
            {
                "index": 0,
                "name": "Seat",
                "robot": "so101",
                "controller": "vla",
                "instruction": "seat it",
                "skill_id": "s1",
            },
            {
                "index": 1,
                "name": "Crate",
                "robot": "vx300s.native.native",
                "controller": "scripted",
                "instruction": "crate it",
                "skill_id": "",
            },
        ],
        "training": [],
        "result": {
            "n_parts": 12,
            "completed": completed,
            "seed": 7,
            "makespan_s": 90.0,
            "per_station": [
                {"station": 0, "parts": 12, "succeeded": 9},
                {"station": 1, "parts": 9, "succeeded": 9},
            ],
        },
    }


def test_card_labels_each_arm_and_keeps_only_measured_numbers(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.setattr(records, "RECORDS", tmp_path)
    skills = {"s1": SimpleNamespace(n_eval=20, success_rate=0.8)}
    cid = multi.register(_line("a", 8), skills)
    t = multi.tiles()[0]
    assert (
        cid == t["id"] == "line-a" and t["success_rate"] == 8 / 12 and t["n_eval"] == 12
    )
    vla, scripted = t["arms"]
    assert vla["controller"] == "trained VLA" and vla["held_out"] == {
        "rate": 0.8,
        "n": 20,
        "k": 16,
    }
    assert scripted["controller"] == "scripted" and scripted["held_out"] is None


def test_unmeasured_arm_has_no_score_and_a_rerun_replaces_the_older_card(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.setattr(records, "RECORDS", tmp_path)
    multi.register(_line("a", 8), {})
    assert multi.cards()[0]["arms"][0]["held_out"] is None
    multi.register(_line("b", 10, created=2.0), {})
    assert [c["line"] for c in multi.cards()] == ["b"]


def test_a_run_that_finished_nothing_is_not_a_card() -> None:
    line = _line("c", 0)
    line["result"] = {}
    assert multi.record_from_line(line, {}) is None
