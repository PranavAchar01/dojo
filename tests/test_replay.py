"""A skill's record plays back as the page's own events, and a rebuilt record never contains a value the disk lacks."""

from __future__ import annotations

import gzip
import json
import re

from vla_forge import records, replay
from vla_forge.pipeline import Round, Run

SPEC = {
    "skill": "kit",
    "obj": "red",
    "target": "tray",
    "distractors": 2,
    "instruction": "put the red block in the tray",
}


def _skill(**kw) -> dict:
    s = {
        "id": "put-the-red-block-in-the-tray-abcd",
        "instruction": "put the red block in the tray",
        "prompt": "put the red block in the tray",
        "spec": json.dumps(SPEC),
        "parent": None,
        "success_rate": 0.7,
        "n_eval": 20,
        "seconds": 171.5,
        "train_seconds": 91.2,
        "tier": 2,
        "steps": [],
        "replay": "skills/tiles/x/replay.json.gz",
    }
    return {**s, **kw}


def _disk(root, *, rounds=("11/20", "14/20"), tile=True, opt=True):
    (root / "demo").mkdir(parents=True)
    (root / "runs").mkdir()
    (root / "demo" / "realism-results.json").write_text(
        json.dumps(
            {
                "skills": {
                    "skills": [
                        {
                            "skill_id": "185611-abcd",
                            "instruction": "put the red block in the tray",
                            "rounds": list(rounds),
                            "init": "fine-tuning from 'a parent'",
                        }
                    ]
                }
            }
        )
    )
    if tile:
        d = root / "skills" / "tiles" / "x"
        d.mkdir(parents=True)
        rolls = [
            {
                "index": i,
                "success": ok,
                "failure": "" if ok else "missed",
                "fps": 30,
                "scene": {"blocks": []},
                "frames": [[0.1]],
            }
            for i, ok in enumerate([True, False, True])
        ]
        with gzip.open(d / "replay.json.gz", "wt") as fh:
            json.dump(rolls, fh)
    if opt:
        o = root / "runs" / "opt-111111-aaaa"
        o.mkdir()
        (o / "run.json").write_text(
            json.dumps(
                {
                    "id": "111111-aaaa",
                    "status": "done",
                    "skill": {"id": "put-the-red-block-in-the-tray-abcd"},
                    "rounds": [
                        {
                            "round": 0,
                            "change": None,
                            "status": "start",
                            "success": 60,
                            "n": 100,
                        },
                        {
                            "round": 1,
                            "change": {"text": "add 40 demonstrations"},
                            "status": "kept",
                            "success": 66,
                            "n": 100,
                        },
                    ],
                }
            )
        )


def _numbers(x, out: set) -> None:
    if isinstance(x, bool) or x is None or isinstance(x, str):
        return
    if isinstance(x, (int, float)):
        out.add(round(float(x), 6))
    elif isinstance(x, dict):
        for v in x.values():
            _numbers(v, out)
    elif isinstance(x, list):
        for v in x:
            _numbers(v, out)


def test_rebuilt_record_has_only_values_found_on_disk(tmp_path):
    _disk(tmp_path)
    sk = _skill()
    out = replay.build(sk, tmp_path)
    assert out["source"] == "reconstructed"
    allowed: set = set()
    for src in (
        sk,
        json.loads((tmp_path / "demo" / "realism-results.json").read_text()),
        json.loads((tmp_path / "runs" / "opt-111111-aaaa" / "run.json").read_text()),
    ):
        _numbers(src, allowed)
    allowed |= {
        0.0,
        1.0,
        2.0,
        3.0,
        11 / 20,
        14 / 20,
    }  # round indices, counts of the kept recordings, rates k/n
    allowed |= {
        round(float(v), 6) for v in range(200)
    }  # numbers inside the text of log lines are checked below
    for e in out["events"]:
        body = {k: v for k, v in e.items() if k not in ("ts", "run", "agent", "kind")}
        nums: set = set()
        _numbers(body, nums)
        assert nums <= allowed, (e["kind"], nums - allowed)
    # every figure quoted in a log line is one of the saved ones
    raw = json.dumps(sk) + (tmp_path / "demo" / "realism-results.json").read_text() + (tmp_path / "runs" / "opt-111111-aaaa" / "run.json").read_text()
    saved_text = set(re.findall(r"\d+(?:\.\d+)?", raw)) | {"3", "91"}  # 3 kept recordings; 91 is 91.2 s rounded for the rail
    for e in out["events"]:
        if e["kind"] == "log" and e["agent"] in ("trainer", "eval", "optimizer"):
            for tok in re.findall(r"\d+(?:\.\d+)?", e["text"]):
                assert tok in saved_text, (e["text"], tok)


def test_stages_that_left_nothing_are_marked_not_recorded(tmp_path):
    _disk(tmp_path, tile=False, opt=False)
    out = replay.build(_skill(replay="", train_seconds=0), tmp_path)
    kinds = [(e["kind"], e.get("stage")) for e in out["events"]]
    for stage in ("router", "data", "trainer", "critic", "scene"):
        assert ("not_recorded", stage) in kinds
        assert out["stages"][stage] == "not_recorded"
    assert out["stages"]["optimizer"] == "none"
    # no loss, demo or critic events are invented
    assert not {"loss", "demos", "critic"} & {e["kind"] for e in out["events"]}


def test_eval_and_publish_match_the_registry(tmp_path):
    _disk(tmp_path)
    out = replay.build(_skill(), tmp_path)
    evals = [e for e in out["events"] if e["kind"] == "eval"]
    assert [(e["succeeded"], e["n"]) for e in evals] == [(11, 20), (14, 20)]
    assert evals[1]["successes"] == [True, False, True] and "successes" not in evals[0]
    pub = out["events"][-1]
    assert (
        pub["kind"] == "published"
        and pub["success_rate"] == 0.7
        and pub["best_round"] == 1
    )
    assert (
        out["stages"]["trainer"] == "partial"
        and out["stages"]["eval"] == "reconstructed"
    )
    times = [e["ts"] for e in out["events"]]
    assert times == sorted(times)


def test_registry_alone_gives_the_final_count(tmp_path):
    (tmp_path / "runs").mkdir()
    out = replay.build(_skill(replay=""), tmp_path)
    ev = next(e for e in out["events"] if e["kind"] == "eval")
    assert (ev["succeeded"], ev["n"]) == (14, 20)


def test_live_capture_round_trips_into_the_event_list(tmp_path, monkeypatch):
    monkeypatch.setattr(records, "RECORDS", tmp_path / "runs")
    run = Run(id="123456-abcd", prompt="put the red block in the tray", t0=1000.0)
    run.rounds = [
        Round(
            0,
            demos=18,
            train={"steps": 10, "seconds": 1.0},
            eval={"succeeded": 14, "n": 20},
            successes=[True] * 14 + [False] * 6,
        )
    ]
    run.best_round = 0
    ev = lambda at, kind, agent="", **d: {
        "ts": at,
        "run": run.id,
        "kind": kind,
        "agent": agent,
        **d,
    }
    run.events = [
        ev(
            1000.0,
            "log",
            "task",
            text='chat message from the page: "put the red block in the tray"',
        ),
        ev(1000.4, "task", "task", spec=SPEC, tier=2),
        ev(
            1001.0,
            "route",
            "router",
            decision="train",
            reason="nothing close",
            checks=[],
            refused=False,
        ),
        ev(1002.0, "demos", "data", ok=18, n=20, done=20),
        ev(1010.0, "loss", "trainer", round=0, t=3.0, loss=0.5, ema=0.6, step=100),
        ev(
            1030.0,
            "eval",
            "eval",
            round=0,
            succeeded=14,
            n=20,
            successes=[True] * 14 + [False] * 6,
        ),
        ev(1031.0, "saving", "registry", skill="x"),
    ]
    records.save("put-the-red-block-in-the-tray-abcd", records.from_run(run))
    # save_capture needs a Forge for rollout payloads; there are none here
    run.layouts = []
    records.save_capture("put-the-red-block-in-the-tray-abcd", run, forge=None)
    rec = records.load("put-the-red-block-in-the-tray-abcd")
    assert rec["replay"]["source"] == "live" and rec["demos"]["ok"] == 18
    assert (
        rec["replay"]["stages"]["router"] == "captured"
        and rec["replay"]["stages"]["critic"] == "not_recorded"
    )

    out = replay.build(_skill(replay=""), tmp_path)
    assert out["source"] == "live"
    kinds = [e["kind"] for e in out["events"]]
    assert kinds == [
        "log",
        "task",
        "route",
        "demos",
        "loss",
        "eval",
        "published",
    ]  # 'saving' is not replayed
    assert (
        out["events"][0]["ts"] == 0 and out["events"][-1]["ts"] > out["events"][-2]["ts"]
    )
    assert out["events"][2]["decision"] == "train"
    assert not out["paced"]


def _groups(events):
    """(agent, seconds on screen) for each run of consecutive events from the same stage; the last has no end."""
    out = []
    for e in events:
        if not out or out[-1][0] != e["agent"]:
            out.append([e["agent"], e["ts"], e["ts"]])
        out[-1][2] = e["ts"]
    starts = [g[1] for g in out] + [events[-1]["ts"]]
    return [(g[0], starts[i + 1] - g[1]) for i, g in enumerate(out)]


def test_rebuilt_pacing_never_shorter_than_a_live_stage(tmp_path):
    _disk(tmp_path)
    out = replay.build(_skill(), tmp_path)
    ev = out["events"]
    for agent, seconds in _groups(ev)[:-1]:
        assert seconds >= replay.LIVE_MIN[agent] - 1e-6, (agent, seconds)
    assert out["paced"]  # the page plays a rebuilt record at exactly these times, with no silence shortened
    trainer = next(s for a, s in _groups(ev) if a == "trainer")
    assert trainer >= 91  # training plays for the seconds it really took (91.2)
    evals = [e for e in ev if e["kind"] == "eval"]
    assert all(
        e["ts"] - prev["ts"] >= replay.LIVE_MIN["eval"]
        for e, prev in zip(evals, [ev[0], *evals], strict=False)
    )


def test_a_rerun_is_played_before_the_rebuild_and_leaves_the_original_alone(tmp_path, monkeypatch):
    monkeypatch.setattr(records, "RECORDS", tmp_path / "runs")
    sid = "put-the-red-block-in-the-tray-abcd"
    records.save(sid, {"run": "185611-abcd", "eval": [{"round": 0, "succeeded": 14, "n": 20}]})
    before = (tmp_path / "runs" / f"{sid}.json").read_text()
    run = Run(id="185611-abcd", prompt="put the red block in the tray", t0=1000.0)
    run.rounds = [Round(0, demos=18, eval={"succeeded": 12, "n": 20}, successes=[True] * 12 + [False] * 8)]
    run.best_round, run.seconds = 0, 160.0
    run.events = [
        {"ts": 1000.0, "run": run.id, "kind": "log", "agent": "task", "text": 'chat message from the page: "put the red block in the tray"'},
        {"ts": 1001.0, "run": run.id, "kind": "task", "agent": "task", "spec": SPEC, "tier": 2},
        {"ts": 1030.0, "run": run.id, "kind": "eval", "agent": "eval", "round": 0, "succeeded": 12, "n": 20, "successes": [True] * 12 + [False] * 8},
    ]
    meta = {"date": "2026-11-07", "seed": "original run id from the ladder", "original": {"success_rate": 0.7, "n_eval": 20, "seconds": 171.5}}
    run.layouts = []
    records.save_rerun(sid, run, forge=None, meta=meta)
    out = replay.build(_skill(replay=""), tmp_path)
    assert out["source"] == "rerun" and not out["paced"]
    assert out["rerun"]["date"] == "2026-11-07" and (out["rerun"]["succeeded"], out["rerun"]["n"]) == (12, 20)
    pub = out["events"][-1]
    assert pub["kind"] == "published" and pub["success_rate"] == 12 / 20
    assert [e["kind"] for e in out["events"]].count("published") == 1
    assert (tmp_path / "runs" / f"{sid}.json").read_text() == before  # the published record is never overwritten
    assert records.load_rerun(sid)["original"]["success_rate"] == 0.7


def test_a_loss_event_is_stamped_with_the_wall_clock_not_its_training_seconds():
    from vla_forge.pipeline import Forge

    forge = Forge()
    run = Run(id="999999-zzzz", prompt="x")
    forge.runs[run.id] = run
    forge.emit(run.id, "loss", "trainer", round=0, t=3.0, loss=0.5, ema=0.5, step=10)
    assert run.events[0]["t"] == 3.0 and run.events[0]["ts"] > 1e9
