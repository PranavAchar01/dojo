"""What a published model keeps of the run that made it, so a restart does not empty the rail cards.

One JSON sidecar per skill under skills/runs/<id>.json. Fields that were never captured are simply absent."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from .registry import SKILLS_DIR

RECORDS = SKILLS_DIR / "runs"
LOSS_POINTS = 200


def downsample(points: list, k: int = LOSS_POINTS) -> list:
    if len(points) <= k:
        return points
    return [points[round(i * (len(points) - 1) / (k - 1))] for i in range(k)]


def from_run(run) -> dict:
    """The record of a finished single-skill run (pipeline.Run)."""
    rec: dict = {"run": run.id, "route": run.route or None, "parse": run.parse_info or None}
    rounds = run.rounds
    if any(r.demos or r.demos_asked for r in rounds):
        ok, asked = sum(r.demos for r in rounds), sum(r.demos_asked for r in rounds)
        rec["demos"] = {
            "ok": ok,
            "asked": asked or None,
            "per_round": [{"ok": r.demos, "asked": r.demos_asked or None} for r in rounds],
            "collected": "scripted expert in randomized scenes",
        }
    if run.layouts:
        rec["layout_seed"] = int(hashlib.sha1(run.id.encode()).hexdigest()[:8], 16)
        rec["n_scenes"] = len(run.layouts)
    if any(r.train for r in rounds):
        curve = [[i, round(t, 2), v] for i, r in enumerate(rounds) for t, v in r.loss]
        rec["train"] = {
            "rounds": len(rounds),
            "steps": sum(r.train.get("steps", 0) for r in rounds),
            "seconds": round(sum(r.train.get("seconds", 0.0) for r in rounds), 1),
            "final_loss": next((r.loss[-1][1] for r in reversed(rounds) if r.loss), None),
            "loss_curve": downsample(curve),
            "init": run.init_from or None,
        }
    rec["eval"] = [
        {"round": r.index, "succeeded": r.eval.get("succeeded"), "n": r.eval.get("n"), "successes": r.successes}
        for r in rounds
        if r.eval
    ]
    rec["best_round"] = run.best_round
    rec["critic"] = [
        {"round": r.index, "action": r.plan.get("action"), "rationale": r.plan.get("rationale")}
        for r in rounds
        if r.plan
    ]
    return {k: v for k, v in rec.items() if v not in (None, [], {})}


def save(sid: str, rec: dict) -> None:
    RECORDS.mkdir(parents=True, exist_ok=True)
    tmp = RECORDS / f"{sid}.json.tmp"
    tmp.write_text(json.dumps(rec))
    tmp.replace(RECORDS / f"{sid}.json")


def optimizer_runs(sid: str) -> list[dict]:
    """Optimizer rounds for a skill, read from the saved optimizer runs (runs/opt-*/run.json)."""
    out = []
    for f in sorted((SKILLS_DIR.parent / "runs").glob("opt-*/run.json")):
        try:
            d = json.loads(f.read_text())
        except (OSError, ValueError):
            continue
        if (d.get("skill") or {}).get("id") != sid:
            continue
        out.append(
            {
                "id": d.get("id"),
                "status": d.get("status"),
                "best_round": d.get("best_round"),
                "rounds": [
                    {"round": r.get("round"), "success": r.get("success"), "n": r.get("n")}
                    for r in d.get("rounds", [])
                ],
            }
        )
    return out


def load(sid: str) -> dict:
    """The saved record with the optimizer rounds found on disk; empty when nothing was kept."""
    f = RECORDS / f"{sid}.json"
    rec = json.loads(f.read_text()) if f.exists() else {}
    opt = optimizer_runs(sid)
    if opt:
        rec["optimizer"] = opt
    return rec


def backfill(ladder: Path | None = None) -> list[str]:
    """Records for older skills from what is honestly on disk: runs/ladder.jsonl rows (per-round held-out counts, init)."""
    ladder = ladder or SKILLS_DIR.parent / "runs" / "ladder.jsonl"
    done = []
    if not ladder.exists():
        return done
    skills = {s["id"]: s for s in json.loads((SKILLS_DIR / "registry.json").read_text())}
    for line in ladder.read_text().splitlines():
        row = json.loads(line)
        if row.get("status") != "published":
            continue
        sid = next((i for i, s in skills.items() if i.endswith(row["run"][-4:]) and s["instruction"] == row["instruction"]), None)
        if not sid or (RECORDS / f"{sid}.json").exists():
            continue
        ev = []
        for i, r in enumerate(row.get("rounds") or []):
            k, n = (int(x) for x in str(r).split("/"))
            ev.append({"round": i, "succeeded": k, "n": n})
        save(sid, {k: v for k, v in {"run": row["run"], "eval": ev, "init_note": row.get("init")}.items() if v})
        done.append(sid)
    return done
