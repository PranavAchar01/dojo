"""Re-run an old skill's pipeline in simulation and keep the whole run as its replay record.

    uv run python scripts/rerun.py <skill id> [<skill id> ...]

The run uses the skill's saved task and, where the original run id can be found on disk, the same id, so the 20
held-out scenes come from the same seed. Nothing is published: the registry, the checkpoint and the original
record are untouched. The result is skills/runs/<id>.rerun.* (events, media, numbers, date).
"""

from __future__ import annotations

import datetime as dt
import sys
import time

from vla_forge import records, replay
from vla_forge.pipeline import (
    EVAL_N,
    FIRST_DEMOS,
    FIRST_TRAIN,
    FORGE,
    Round,
    Run,
)
from vla_forge.registry import ROOT
from vla_forge.sim.tasks import TaskSpec


def say(run: Run, agent: str, text: str) -> None:
    FORGE.emit(run.id, "log", agent, text=text)
    print(f"  [{agent}] {text}", flush=True)


def seed_run_id(skill: dict, src: replay.Sources) -> tuple[str, str]:
    """The id the original run had (so its scenes repeat), and how it was found."""
    if src.run_id:
        return src.run_id, f"original run id from {src.origin_file}"
    rec = records.load(skill["id"])
    if rec.get("run"):
        return rec["run"], "original run id from its saved record"
    return f"rerun-{skill['id'][-4:]}", "original run id not on disk: new scenes"


def rerun(sid: str) -> dict:
    skill = next(s for s in FORGE.registry.skills.values() if s.id == sid)
    row = {
        "id": skill.id,
        "instruction": skill.instruction,
        "prompt": skill.prompt or skill.instruction,
        "spec": skill.spec,
        "replay": skill.replay,
        "steps": skill.steps,
    }
    if skill.composite:
        raise SystemExit(f"{sid} is a chain; only single skills are re-run")
    src = replay.Sources(row)
    rid, how = seed_run_id(row, src)
    run = FORGE.new_run(row["prompt"], mode="train")
    FORGE.runs.pop(run.id)
    FORGE.order.remove(run.id)
    run.id = rid
    run.robot, run.pinned = skill.robot, True
    FORGE.runs[rid] = run
    FORGE.order.append(rid)

    held = FORGE.registry.skills.pop(
        sid
    )  # the run must not start from, or reuse, the skill it re-makes
    try:
        run.t0 = time.time()
        run.status = "running"
        run.spec = TaskSpec.from_json(skill.spec)
        FORGE._judge(run, run.spec)
        run.parse_info = {
            "model": "saved spec",
            "seconds": 0.0,
            "note": "the task was read from the saved skill",
        }
        say(run, "task", f'chat message from the page: "{run.prompt}"')
        FORGE.emit(
            run.id,
            "task",
            "task",
            spec=run.spec.to_json(),
            parse=run.parse_info,
            tier=run.tier,
        )
        r = FORGE.route(run)
        FORGE.emit(run.id, "route", "router", **r.to_json())
        say(run, "router", f"{r.decision}: {r.reason}")
        info = FORGE.scene(run)
        FORGE.emit(run.id, "scene", "scene", **info)
        say(
            run,
            "scene",
            f"SO-101 + {len(info['layout']['blocks'])} blocks; {EVAL_N} held-out layouts sampled",
        )
        n, near, secs, lr = FIRST_DEMOS, [], FIRST_TRAIN, 3e-4
        while True:
            say(
                run,
                "data",
                f"generating {n} expert demos"
                + (f" focused near {len(near)} failure spots" if near else ""),
            )
            st = FORGE.collect(run, n, near or None, "data")
            run.rounds.append(
                Round(index=len(run.rounds), demos=st["ok"], demos_asked=st["n"])
            )
            say(
                run,
                "data",
                f"{st['ok']}/{st['n']} demos ok, {st['frames']} frames encoded in {st['seconds']}s",
            )
            say(
                run,
                "trainer",
                f"training on {run.data.episodes} episodes (+ replay) for {secs:.0f}s",
            )
            tr = FORGE.train(run, secs, lr, "trainer")
            run.rounds[-1].train = tr
            say(run, "trainer", f"{tr['steps']} steps, loss {tr['loss'] or 0:.3f}")
            say(run, "eval", f"rolling out the policy in {EVAL_N} held-out scenes")
            rep = FORGE.evaluate(run, "eval")
            rnd = run.rounds[-1]
            rnd.eval = {
                k: rep[k]
                for k in ("n", "succeeded", "success_rate", "failures", "seconds")
            }
            rnd.successes = [x.success for x in run.rollouts[rnd.index]]
            FORGE.emit(
                run.id,
                "eval",
                "eval",
                round=rnd.index,
                **rnd.eval,
                successes=rnd.successes,
            )
            say(
                run,
                "eval",
                f"{rep['succeeded']}/{rep['n']} succeeded in {rep['seconds']}s"
                + (
                    f"; failures: {', '.join(f'{v} {k}' for k, v in rep['failures'].items())}"
                    if rep["failures"]
                    else ""
                ),
            )
            plan, model = FORGE.critic(run, rep)
            rnd.plan = {**plan, "model": model}
            FORGE.emit(
                run.id, "critic", "critic", round=rnd.index, plan=plan, model=model
            )
            say(run, "critic", f"[{model}] {plan['action']}: {plan['rationale']}")
            if plan["action"] != "retrain":
                break
            near = (
                [
                    {
                        "target": f["target_xy"],
                        **({"goal": f["goal_xy"]} if f["goal_xy"] else {}),
                    }
                    for f in rep["failed_starts"]
                ]
                if plan["focus_failures"]
                else []
            )
            n, secs, lr = plan["extra_demos"], plan["train_seconds"], plan["lr"]
        run.seconds = round(run.elapsed(), 1)
        best = run.rounds[run.best_round].eval
        FORGE.emit(
            run.id,
            "published",
            "registry",
            skill=sid,
            address="",
            success_rate=best["success_rate"],
            seconds=run.seconds,
            best_round=run.best_round,
        )
        meta = {
            "date": dt.date.today().isoformat(),
            "seed": how,
            "original": {
                "success_rate": held.success_rate,
                "n_eval": held.n_eval,
                "seconds": held.seconds,
            },
            "init": run.init_from.get("reason"),
        }
        return records.save_rerun(sid, run, FORGE, meta)
    finally:
        FORGE.registry.skills[sid] = held
        del run.model, run.data, run.best_state


def main() -> None:
    FORGE.warm()
    for sid in sys.argv[1:]:
        t = time.time()
        rec = rerun(sid)
        print(
            f"{sid}: re-run {rec['succeeded']}/{rec['n']} (published {rec['original']['success_rate']:.2f}) "
            f"in {time.time() - t:.0f}s, rounds {rec['rounds']}, seed: {rec['seed']}",
            flush=True,
        )


if __name__ == "__main__":
    assert ROOT
    main()
