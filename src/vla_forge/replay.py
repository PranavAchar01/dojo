"""The development of a saved skill as the event list the page already understands, so a Library card can play it back.

Two sources, never mixed up:
  * live: the stream the page received while the skill was trained (records.save_capture), played as it came;
  * reconstructed: older skills, rebuilt only from what is on disk (registry row, ladder log, realism table, the
    tile's held-out recordings, optimizer runs). A stage that left nothing behind is marked `not_recorded`.
Every value in a reconstructed event is copied from one of those files; nothing is estimated."""

from __future__ import annotations

import gzip
import json
from pathlib import Path

from . import records
from .registry import ROOT

STAGES = (
    "task",
    "router",
    "scene",
    "data",
    "trainer",
    "eval",
    "critic",
    "optimizer",
    "registry",
)
STEP = 1.0  # seconds between reconstructed events: playback pacing only, the client compresses it further
RUN = "main"
# Seconds each stage stays on screen in a live run, from the pipeline's own timing notes (pipeline.EVAL_SECONDS: 20
# rollouts take 12-14 s; the critic's budget: a batch of 30 demos takes 9 s). A rebuilt record never shows a stage for
# less. Training plays for the seconds it really took. A captured record keeps its own timing; only silences longer
# than IDLE_CAP are shortened, which no stage's animation spans.
LIVE_MIN = {"task": 0.8, "router": 1.5, "scene": 1.2, "data": 9.0, "trainer": 5.0, "eval": 12.0, "critic": 1.5, "optimizer": 1.0, "registry": 0.0}
IDLE_CAP = 4.0


def _json(path: Path):
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return None


def _rounds(strings: list) -> list[tuple[int, int]]:
    out = []
    for s in strings or []:
        k, n = (int(x) for x in str(s).split("/"))
        out.append((k, n))
    return out


class Sources:
    """What is on disk for one skill. Fields are None or empty when nothing was kept."""

    def __init__(self, skill: dict, root: Path = ROOT, parent_instruction: str = ""):
        self.skill = skill
        self.parent_instruction = parent_instruction
        sid, ins = skill["id"], skill["instruction"]
        tail = sid[-4:]
        self.rounds: list[tuple[int, int]] = []
        self.init_note = ""
        self.origin_file = ""
        self.run_id = ""
        ladder = root / "runs" / "ladder.jsonl"
        if ladder.exists():
            for line in ladder.read_text().splitlines():
                row = json.loads(line)
                if (
                    row.get("status") == "published"
                    and row["run"][-4:] == tail
                    and row["instruction"] == ins
                ):
                    self.rounds, self.init_note, self.origin_file = (
                        _rounds(row.get("rounds")),
                        row.get("init") or "",
                        "runs/ladder.jsonl",
                    )
                    self.run_id = row["run"]
        real = _json(root / "demo" / "realism-results.json")
        for row in ((real or {}).get("skills") or {}).get("skills") or []:
            if row["skill_id"][-4:] == tail and row["instruction"] == ins:
                self.rounds, self.init_note, self.origin_file = (
                    _rounds(row.get("rounds")),
                    row.get("init") or "",
                    "demo/realism-results.json",
                )
                self.run_id = row["skill_id"]
        self.rolls: list[bool] = []
        if skill.get("replay"):
            try:
                with gzip.open(root / skill["replay"], "rt") as fh:
                    self.rolls = [bool(r["success"]) for r in json.load(fh)]
            except (OSError, ValueError):
                self.rolls = []
        self.optimizer = self._optimizer(root, sid)

    @staticmethod
    def _optimizer(root: Path, sid: str) -> dict | None:
        best = None
        for f in sorted((root / "runs").glob("opt-*/run.json")):
            d = _json(f)
            if (
                not d
                or (d.get("skill") or {}).get("id") != sid
                or d.get("status") != "done"
                or not d.get("rounds")
            ):
                continue
            best = d
        return best


def best_round(rounds: list[tuple[int, int]]) -> int:
    """The round the run kept: the first one with the highest held-out rate (the pipeline only replaces on a strict gain)."""
    return max(range(len(rounds)), key=lambda i: rounds[i][0] / rounds[i][1]) if rounds else -1


def _ev(t: float, kind: str, agent: str = "", **data) -> dict:
    return {"ts": round(t, 3), "run": RUN, "kind": kind, "agent": agent, **data}


def _optimizer_events(opt: dict, t: float) -> tuple[list[dict], float]:
    evs = []
    rounds = opt["rounds"]
    for r in rounds:
        ch = r.get("change") or {}
        what = ch.get("text") or ("baseline" if r["round"] == 0 else "")
        line = f"Round {r['round']}: {what + ', ' if what else ''}{r['success']}/{r['n']} on the optimizer's scenes"
        if r.get("status") and r["round"]:
            line += f" ({r['status']})"
        t += STEP
        evs.append(_ev(t, "log", "optimizer", text=line))
    first = rounds[0]
    best = max(rounds, key=lambda r: r["success"])
    t += STEP
    evs.append(
        _ev(
            t,
            "stage_mark",
            "optimizer",
            stage="optimizer",
            state="done",
            metric=f"{first['success']}→{best['success']}/{first['n']}",
        )
    )
    t += 0.2
    evs.append(
        _ev(
            t,
            "log",
            "optimizer",
            text=f"Optimizer run opt-{opt['id']}: {len(rounds) - 1} rounds after the baseline, best {best['success']}/{best['n']} in round {best['round']}",
        )
    )
    return evs, t


def reconstruct(src: Sources) -> dict:
    """A record from the files an old skill left. Every stage stays on screen at least as long as it does in a live run
    (LIVE_MIN), and a stage whose real duration is known (training) runs for that long, ticking its seconds."""
    sk = src.skill
    prompt = sk.get("prompt") or sk["instruction"]
    stages = dict.fromkeys(STAGES, "not_recorded")
    ev: list[dict] = []
    t = 0.0
    group, group_t = "", 0.0

    def add(kind, agent="", **data):
        nonlocal t, group, group_t
        if agent != group:
            t = max(t, group_t + LIVE_MIN.get(group, 0.0))
            group, group_t = agent, t
        ev.append(_ev(t, kind, agent, **data))
        t += 0.3

    def tick(stage, seconds, label):
        """The stage working for `seconds`, one mark a second; the marks carry the time, never a value."""
        nonlocal t
        for k in range(1, int(seconds) + 1):
            t += STEP - 0.3 if k == 1 else STEP
            ev.append(_ev(t, "stage_mark", stage, stage=stage, state="live", metric=label(k)))
        t += 0.3

    def missing(stage, note):
        add("not_recorded", stage, stage=stage, note=note)

    add("log", "task", text=f'chat message from the page: "{prompt}"')
    spec = sk["spec"] if isinstance(sk["spec"], dict) else json.loads(sk["spec"] or "{}")
    chain = bool(sk.get("steps"))
    add("task", "task", spec=spec or {"instruction": sk["instruction"]}, tier=sk.get("tier", 1))
    stages["task"] = "reconstructed"
    missing("router", "Routing was not recorded for this skill: no feasibility checks or arm ranking were kept.")
    if src.rolls:
        add("scene", "scene")
        stages["scene"] = "reconstructed"
    else:
        missing("scene", "No scene was kept for this skill.")
    missing(
        "data",
        "The demonstrations were not recorded: how many were collected and how many passed is gone.",
    )
    note = src.init_note or (
        f"started from the skill '{src.parent_instruction}'" if src.parent_instruction else ""
    )
    if sk.get("train_seconds"):
        sec = sk["train_seconds"]
        if note:
            add("init", "trainer", reason=note)
        add("log", "trainer", text=f"Training time {sec:g} s. The loss curve was not recorded.")
        tick("trainer", sec, lambda k: f"{k}s")
        add("stage_mark", "trainer", stage="trainer", state="done", metric=f"{round(sec)}s")
        stages["trainer"] = "partial"
    else:
        missing("trainer", "Neither the loss curve nor the training time was recorded.")

    n_eval = sk.get("n_eval") or 0
    rounds = src.rounds
    if not rounds and n_eval:
        k = round(sk["success_rate"] * n_eval)
        if abs(k / n_eval - sk["success_rate"]) < 1e-3:
            rounds = [(k, n_eval)]
    best = best_round(rounds)
    for i, (k, n) in enumerate(rounds):
        extra: dict = {}
        if i == best and src.rolls:
            extra["successes"] = src.rolls
            extra["sampled"] = True
        add("stage_mark", "eval", stage="eval", state="live", metric="evaluating")
        tick("eval", LIVE_MIN["eval"], lambda k: "evaluating")
        add("eval", "eval", round=i, succeeded=k, n=n, success_rate=k / n, **extra)
    if rounds:
        stages["eval"] = "reconstructed"
        if src.rolls:
            add(
                "log",
                "eval",
                text=f"Held-out result {rounds[best][0]}/{rounds[best][1]}. {len(src.rolls)} of those scenes were kept as recordings; the other outcomes were not.",
            )
    else:
        missing("eval", "No held-out result was kept.")
    missing("critic", "The critic's notes were not recorded.")

    if src.optimizer and not chain:
        t = max(t, group_t + LIVE_MIN["critic"])
        evs, t2 = _optimizer_events(src.optimizer, t)
        ev.extend(evs)
        t, group, group_t = t2 + STEP, "optimizer", t
        stages["optimizer"] = "reconstructed"
    else:
        stages["optimizer"] = "none"
    add(
        "published",
        "registry",
        skill=sk["id"],
        address="",
        success_rate=sk["success_rate"],
        seconds=sk["seconds"],
        best_round=max(best, 0),
    )
    stages["registry"] = "reconstructed"
    return {
        "sid": sk["id"],
        "source": "reconstructed",
        "prompt": prompt,
        "paced": True,
        "stages": stages,
        "events": ev,
        "best_round": max(best, 0),
    }


def from_live(src: Sources, root: Path = ROOT, tag: str = "") -> dict | None:
    """A captured stream: the skill's own (tag ''), or the re-run of its pipeline (tag 'rerun.')."""
    sk = src.skill
    cap = _json(records.RECORDS / f"{sk['id']}.{tag}events.json")
    if not cap:
        return None
    rec = records.load(sk["id"])
    rerun = records.load_rerun(sk["id"]) if tag else {}
    if tag and not rerun:
        return None
    events = [e for e in cap["events"] if e["kind"] != "published"]
    t0 = events[0]["ts"] if events else cap["t0"]
    events = [{**e, "ts": round(e["ts"] - t0, 3)} for e in events]
    end = events[-1]["ts"] + 0.5 if events else 0.0
    stages = {**((rerun or rec.get("replay") or {}).get("stages") or {})}
    stages["optimizer"] = "none"
    if src.optimizer:
        evs, end = _optimizer_events(src.optimizer, end)
        events.extend(evs)
        stages["optimizer"] = "reconstructed"
    best = rerun["best_round"] if rerun else rec.get("best_round", 0)
    rate = (rerun["succeeded"] / rerun["n"]) if rerun else sk["success_rate"]
    events.append(
        _ev(
            end + STEP,
            "published",
            "registry",
            skill=sk["id"],
            address="",
            success_rate=rate,
            seconds=rerun["seconds"] if rerun else sk["seconds"],
            best_round=best,
        )
    )
    return {
        "sid": sk["id"],
        "source": "rerun" if tag else "live",
        "prompt": cap["prompt"],
        "paced": False,
        "stages": stages,
        "events": events,
        "best_round": best,
        **({"rerun": rerun} if rerun else {}),
    }


def build(skill: dict, root: Path = ROOT, parent_instruction: str = "") -> dict:
    """The fullest record there is: a re-run's capture, else the skill's own live capture, else a rebuild from disk."""
    src = Sources(skill, root, parent_instruction)
    return from_live(src, root, "rerun.") or from_live(src, root) or reconstruct(src)


def media(skill: dict, root: Path = ROOT) -> dict:
    """The scene and rollout payloads the replay's fetches are answered from: captured, else the tile's held-out recordings."""
    for tag in ("rerun.", ""):
        f = records.RECORDS / f"{skill['id']}.{tag}media.json.gz"
        if f.exists():
            with gzip.open(f, "rt") as fh:
                return json.load(fh)
    if not skill.get("replay"):
        return {"scene": None, "rollouts": {}}
    with gzip.open(root / skill["replay"], "rt") as fh:
        rolls = json.load(fh)
    src = Sources(skill, root)
    best = max(best_round(src.rounds), 0)
    payloads = [
        {
            "round": best,
            "index": i,
            **{k: r[k] for k in ("success", "failure", "fps", "scene", "frames")},
        }
        for i, r in enumerate(rolls)
    ]
    scene = {**rolls[0]["scene"], "frame": rolls[0]["frames"][0]} if rolls else None
    return {"scene": scene, "rollouts": {str(best): payloads}}
