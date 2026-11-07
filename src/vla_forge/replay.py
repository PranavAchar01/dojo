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
        real = _json(root / "demo" / "realism-results.json")
        for row in ((real or {}).get("skills") or {}).get("skills") or []:
            if row["skill_id"][-4:] == tail and row["instruction"] == ins:
                self.rounds, self.init_note, self.origin_file = (
                    _rounds(row.get("rounds")),
                    row.get("init") or "",
                    "demo/realism-results.json",
                )
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
    sk = src.skill
    prompt = sk.get("prompt") or sk["instruction"]
    stages = dict.fromkeys(STAGES, "not_recorded")
    ev: list[dict] = []
    t = 0.0

    def add(kind, agent="", **data):
        nonlocal t
        ev.append(_ev(t, kind, agent, **data))
        t += STEP

    def missing(stage, note):
        add("not_recorded", stage, stage=stage, note=note)

    add("log", "task", text=f'chat message from the page: "{prompt}"')
    spec = (
        sk["spec"] if isinstance(sk["spec"], dict) else json.loads(sk["spec"] or "{}")
    )
    chain = bool(sk.get("steps"))
    add(
        "task",
        "task",
        spec=spec or {"instruction": sk["instruction"]},
        tier=sk.get("tier", 1),
    )
    stages["task"] = "reconstructed"
    missing(
        "router",
        "Routing was not recorded for this skill: no feasibility checks or arm ranking were kept.",
    )
    note = src.init_note or (
        f"started from the skill '{src.parent_instruction}'"
        if src.parent_instruction
        else ""
    )
    if note:
        add("init", "trainer", reason=note)
    if src.rolls:
        add("scene", "scene")
        stages["scene"] = "reconstructed"
    else:
        missing("scene", "No scene was kept for this skill.")
    missing(
        "data",
        "The demonstrations were not recorded: how many were collected and how many passed is gone.",
    )
    if sk.get("train_seconds"):
        sec = sk["train_seconds"]
        add(
            "log",
            "trainer",
            text=f"Training time {sec:g} s. The loss curve was not recorded.",
        )
        add(
            "stage_mark",
            "trainer",
            stage="trainer",
            state="done",
            metric=f"{round(sec)}s",
        )
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
        evs, t2 = _optimizer_events(src.optimizer, t)
        ev.extend(evs)
        t = t2 + STEP
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


def from_live(src: Sources, root: Path = ROOT) -> dict | None:
    sk = src.skill
    cap = _json(records.RECORDS / f"{sk['id']}.events.json")
    rec = records.load(sk["id"])
    if not cap:
        return None
    events = cap["events"]
    t0 = events[0]["ts"] if events else cap["t0"]
    events = [{**e, "ts": round(e["ts"] - t0, 3)} for e in events]
    end = events[-1]["ts"] + 0.5 if events else 0.0
    stages = {**(rec.get("replay") or {}).get("stages", {})}
    stages["optimizer"] = "none"
    if src.optimizer:
        evs, end = _optimizer_events(src.optimizer, end)
        events.extend(evs)
        stages["optimizer"] = "reconstructed"
    events.append(
        _ev(
            end + STEP,
            "published",
            "registry",
            skill=sk["id"],
            address="",
            success_rate=sk["success_rate"],
            seconds=sk["seconds"],
            best_round=rec.get("best_round", 0),
        )
    )
    return {
        "sid": sk["id"],
        "source": "live",
        "prompt": cap["prompt"],
        "paced": False,
        "stages": stages,
        "events": events,
        "best_round": rec.get("best_round", 0),
    }


def build(skill: dict, root: Path = ROOT, parent_instruction: str = "") -> dict:
    src = Sources(skill, root, parent_instruction)
    return from_live(src, root) or reconstruct(src)


def media(skill: dict, root: Path = ROOT) -> dict:
    """The scene and rollout payloads the replay's fetches are answered from: captured, else the tile's held-out recordings."""
    f = records.RECORDS / f"{skill['id']}.media.json.gz"
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
