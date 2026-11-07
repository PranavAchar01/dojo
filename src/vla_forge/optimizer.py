"""The recursive optimizer: a language model improves a trained skill, one change per round, scored on the same scenes.

R0 scores the skill as trained on 100 held-out simulated scenes. Every later round: the model reads the failure
summary and proposes exactly one change to the policy's training (from a fixed menu, validated here), the change
is applied, the policy is fine-tuned from the current weights and scored again on the same 100 scenes. The change
is kept when the success count rises and reverted otherwise. Every round is written to runs/opt-<id>/run.json.

The proposing model is Llama through the Meta Llama API (llama.py). Until a key works, the stub backend replays
the changes recorded in a saved run.
"""

from __future__ import annotations

import contextlib
import copy
import json
import math
import os
import time
from collections.abc import Callable
from concurrent.futures import as_completed
from dataclasses import dataclass, field
from pathlib import Path
from uuid import uuid4

import numpy as np

from . import llm, montage
from .datagen import demo_job, pool
from .encoders import device
from .learn import SCOPES, Demos, train
from .policy import VLA
from .registry import ROOT, Skill
from .rollout import EVAL_SEED0, _goal_xy, evaluate
from .sim.tasks import sample_layout

N_SCENES = int(
    os.environ.get("VLA_FORGE_OPT_SCENES", "100")
)  # fewer only for smoke tests
SEED0 = (
    EVAL_SEED0 + 200_000
)  # held-out: never a training seed, never the 20 scenes a skill was published on
CHUNK = 25
KEEP_FRAMES = 32
START_DEMOS = 60  # fresh demonstrations every fine-tune starts from, before any change
RUNS = ROOT / "runs"

DEFAULTS = {
    "lr": 1e-4,
    "train_seconds": 40.0,
    "state_noise": 0.03,
    "replay_frac": 0.3,
    "batch": 256,
    "weight_decay": 1e-4,
    "scope": "all",
}
# knob -> (low, high, unit text); the demo knobs add demonstrations for this round and every later one
KNOBS = {
    "extra_demos_near_failures": (10, 60, "demonstrations at the failed starts"),
    "extra_demos_random": (10, 60, "demonstrations in fresh random scenes"),
    "lr": (5e-5, 6e-4, "learning rate"),
    "train_seconds": (20, 75, "seconds of training per round"),
    "state_noise": (0.0, 0.08, "noise added to the joint readings"),
    "replay_frac": (0.0, 0.7, "share of each batch replayed from the base data"),
    "batch": (64, 512, "batch size (64, 128, 256 or 512)"),
    "weight_decay": (1e-5, 1e-2, "weight decay"),
    "scope": (0, 0, "which layers learn: all, action_layers or perception_layers"),
}
SCHEMA = {
    "type": "object",
    "properties": {
        "knob": {"type": "string", "enum": list(KNOBS)},
        "value": {"type": "string"},
        "reason": {"type": "string"},
    },
    "required": ["knob", "value", "reason"],
    "additionalProperties": False,
}
SYSTEM = f"""You improve a trained robot policy by changing how it is fine-tuned. The policy is a small vision-language-action
model for a simulated SO-101 arm. It was scored on a fixed set of held-out simulated scenes; you read the failures and
propose exactly ONE change to the next fine-tuning round. The change is kept only if success on the same scenes rises.

Allowed knobs and ranges (value is a number written as text, except scope):
{json.dumps({k: {"low": v[0], "high": v[1], "meaning": v[2]} for k, v in KNOBS.items()})}
scope must be one of {list(SCOPES)}. Demonstration counts are whole numbers; batch is 64, 128, 256 or 512.
Use the failure tags and the failure rate by region to choose. Do not repeat a change that was reverted unless you
change its value clearly. reason: one plain sentence under 140 characters, no dashes, no claims beyond the numbers."""


def wilson(k: int, n: int, z: float = 1.96) -> list[float]:
    if n == 0:
        return [0.0, 0.0]
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [round(max(0.0, c - h), 3), round(min(1.0, c + h), 3)]


def clean_change(raw: dict) -> dict:
    """Validate a proposed change: a known knob, a value clamped into its range, a short plain reason."""
    knob = raw.get("knob")
    if knob not in KNOBS:
        raise ValueError(f"unknown knob {knob!r}")
    val = str(raw.get("value", "")).strip().lower()
    lo, hi, _ = KNOBS[knob]
    if knob == "scope":
        if val not in SCOPES:
            raise ValueError(f"unknown scope {val!r}")
        v: float | str = val
    else:
        x = float(val)
        if knob == "batch":
            v = min((64, 128, 256, 512), key=lambda b: abs(b - x))
        elif knob.startswith("extra_demos") or knob == "train_seconds":
            v = int(min(max(round(x), lo), hi))
        else:
            v = float(min(max(x, lo), hi))
    reason = " ".join(str(raw.get("reason", "")).replace("—", ",").split())[:140]
    return {"knob": knob, "value": v, "reason": reason}


def describe(ch: dict) -> str:
    k, v = ch["knob"], ch["value"]
    if k.startswith("extra_demos"):
        where = "at the failed starts" if k.endswith("failures") else "in fresh scenes"
        return f"add {v} demonstrations {where}"
    if k == "scope":
        return (
            "train all layers"
            if v == "all"
            else f"train only the {v.replace('_', ' ')}"
        )
    return {
        "lr": f"learning rate {v:.2g}",
        "train_seconds": f"train {v} s per round",
        "state_noise": f"joint noise {v:.3f}",
        "replay_frac": f"replay {round(v * 100)}% of each batch from base data",
        "batch": f"batch size {v}",
        "weight_decay": f"weight decay {v:.2g}",
    }[k]


def failure_report(scored: list[dict], cfg: dict) -> dict:
    fails = [s for s in scored if not s["success"]]
    tags: dict[str, int] = {}
    for s in fails:
        tags[s["failure"]] = tags.get(s["failure"], 0) + 1
    r = np.array([np.hypot(*s["target_xy"]) for s in scored])
    side = np.array([s["target_xy"][1] for s in scored])
    ok = np.array([s["success"] for s in scored])

    def rate(mask) -> dict:
        n = int(mask.sum())
        return {
            "scenes": n,
            "failure_rate": round(float((~ok[mask]).mean()), 2) if n else None,
        }

    q1, q2 = np.quantile(r, [1 / 3, 2 / 3])
    return {
        "scenes": len(scored),
        "succeeded": int(ok.sum()),
        "failure_tags": tags,
        "by_distance_from_base": {
            "near": rate(r <= q1),
            "middle": rate((r > q1) & (r <= q2)),
            "far": rate(r > q2),
        },
        "by_side": {
            "left": rate(side > 0.04),
            "centre": rate(np.abs(side) <= 0.04),
            "right": rate(side < -0.04),
        },
        "median_closest_gripper_distance_m_in_failures": round(
            float(np.median([s["stats"]["min_grip_dist"] for s in fails])), 3
        )
        if fails
        else None,
        "current_settings": cfg,
    }


def propose(report: dict, history: list[dict], rnd: int) -> tuple[dict, str]:
    """One change from the language model, validated. Returns (change, model that answered)."""
    user = json.dumps({"round": rnd, "failure_report": report, "history": history})
    try:
        raw, model = llm._chat_json(SYSTEM, user, SCHEMA, "optimizer")
        return clean_change(raw), model
    except (RuntimeError, ValueError, TypeError) as e:
        order = [
            {"knob": "extra_demos_near_failures", "value": "30"},
            {"knob": "lr", "value": "0.0001"},
            {"knob": "train_seconds", "value": "60"},
            {"knob": "replay_frac", "value": "0.3"},
            {"knob": "state_noise", "value": "0.05"},
            {"knob": "scope", "value": "action_layers"},
        ]
        pick = order[(rnd - 1) % len(order)]
        return clean_change(
            {**pick, "reason": f"fixed menu, no model answered: {e}"}
        ), "rules"


@dataclass
class Opt:
    id: str
    skill: dict
    n_rounds: int
    started: float = field(default_factory=time.time)
    status: str = "starting"
    phase: str = ""
    llm: dict = field(default_factory=dict)
    rounds: list = field(default_factory=list)
    best_round: int = 0
    hero: dict | None = None
    error: str = ""
    n_scenes: int = N_SCENES
    seconds: float = 0.0

    def to_json(self) -> dict:
        d = {k: v for k, v in self.__dict__.items()}
        d["seconds"] = round(
            (time.time() if self.status != "done" else self.started + self.seconds)
            - self.started,
            1,
        )
        return d


def _save(o: Opt) -> None:
    d = RUNS / f"opt-{o.id}"
    d.mkdir(parents=True, exist_ok=True)
    tmp = d / "run.json.tmp"
    tmp.write_text(json.dumps(o.to_json(), indent=1))
    tmp.replace(d / "run.json")


def _compact(r, nbody_frames: int = KEEP_FRAMES) -> dict:
    idx = montage.moment_index(len(r.frames), nbody_frames)
    return {
        "index": r.index,
        "success": bool(r.success),
        "failure": r.failure,
        "steps": r.steps,
        "stats": r.stats,
        "layout": r.layout,
        "target_xy": [float(v) for v in r.layout["blocks"]["target"][1:3]],
        "goal_xy": _goal_xy(r.layout),
        "frames": np.array([r.frames[i] for i in idx], np.float32),
    }


def score(
    model: VLA,
    spec,
    layouts: list[dict],
    robot: str,
    progress: Callable[[int], None],
    first: int = 0,
) -> list[dict]:
    """The same scenes every time: scene i always has seed SEED0 + i (so the same lighting and the same layout)."""
    out: list[dict] = []
    for c in range(0, len(layouts), CHUNK):
        part = layouts[c : c + CHUNK]
        rolls = evaluate(
            model,
            spec,
            len(part),
            seed0=SEED0 + first + c,
            layouts=part,
            record=True,
            robot=robot,
        )
        for r in rolls:
            r.index += c
            out.append(_compact(r))
        progress(len(out))
    return out


def collect(
    spec,
    n: int,
    near: list[dict] | None,
    seed_base: int,
    tmp: Path,
    robot: str,
    data: Demos,
) -> int:
    jobs = []
    for i in range(n):
        j = {
            "spec": spec.to_json(),
            "seed": seed_base + i,
            "out": str(tmp / f"ep{seed_base + i}.npz"),
            "noise": 0.05,
            "robot": robot,
        }
        if near:
            j["near"] = near[i % len(near)]
        jobs.append(j)
    ok = 0
    p = pool()
    for f in as_completed([p.submit(demo_job, j) for j in jobs]):
        r = f.result()
        if r["ok"]:
            data.add_episode(r["path"], spec.instruction)
            Path(r["path"]).unlink()
            ok += 1
    return ok


def eligible(skills: dict[str, Skill]) -> list[Skill]:
    """Single-arm skills with headroom: the optimizer can only help where the policy still fails."""
    return sorted(
        (
            s
            for s in skills.values()
            if not s.composite
            and s.robot == "so101"
            and s.model
            and s.success_rate < 0.9
        ),
        key=lambda s: s.success_rate,
    )


def run(
    skill: Skill,
    n_rounds: int,
    opt: Opt,
    replay: Demos | None,
    compute: contextlib.AbstractContextManager | None = None,
    emit: Callable[..., None] | None = None,
    train_seconds: float = DEFAULTS["train_seconds"],
) -> Opt:
    comp = compute if compute is not None else contextlib.nullcontext()
    ev = emit or (lambda *a, **k: None)
    out = RUNS / f"opt-{opt.id}"
    tmp = out / "tmp"
    tmp.mkdir(parents=True, exist_ok=True)
    spec = skill.task()
    dev = device()
    robot = skill.robot
    model, _ = VLA.load(ROOT / skill.model, dev)
    layouts = [
        sample_layout(spec, np.random.default_rng(SEED0 + i)).to_json()
        for i in range(N_SCENES)
    ]
    stage = montage.Stage(robot)
    cfg = {**DEFAULTS, "train_seconds": float(train_seconds)}
    history: list[dict] = []
    all_scored: dict[int, list[dict]] = {}
    opt.llm = {"asked": "Llama", "answered": None}

    def phase(name: str, **kw) -> None:
        opt.phase = name
        _save(opt)
        ev(opt.id, "opt_phase", phase=name, **kw)

    def evaluate_round(rnd: int, m: VLA) -> list[dict]:
        phase("scoring", round=rnd)
        with comp:
            sc = score(
                m,
                spec,
                layouts,
                robot,
                lambda n: ev(opt.id, "opt_scoring", round=rnd, done=n),
            )
        phase("rendering", round=rnd)
        montage.write_sheets(stage, sc, out / f"r{rnd}")
        all_scored[rnd] = sc
        return sc

    try:
        opt.status = "running"
        t0 = time.time()
        sc0 = evaluate_round(0, model)
        k0 = sum(s["success"] for s in sc0)
        best_k, best_scored = k0, sc0
        opt.rounds.append(
            {
                "round": 0,
                "change": None,
                "status": "start",
                "success": k0,
                "n": N_SCENES,
                "ci": wilson(k0, N_SCENES),
                "failures": failure_report(sc0, cfg)["failure_tags"],
                "outcomes": [int(s["success"]) for s in sc0],
                "seconds": round(time.time() - t0, 1),
            }
        )
        phase("collecting", round=0)
        data = Demos()
        with comp:
            collect(spec, START_DEMOS, None, 50_000, tmp, robot, data)
        for rnd in range(1, n_rounds + 1):
            t1 = time.time()
            phase("proposing", round=rnd)
            report = failure_report(best_scored, cfg)
            change, answered = propose(report, history, rnd)
            opt.llm["answered"] = answered
            opt.llm["real_llama"] = answered.lower().startswith("llama")
            cand_cfg, cand_data = dict(cfg), data
            text = describe(change)
            row = {
                "round": rnd,
                "change": {**change, "text": text, "model": answered},
                "status": "training",
            }
            opt.rounds.append(row)
            _save(opt)
            ev(opt.id, "opt_change", round=rnd, change=row["change"])
            knob, val = change["knob"], change["value"]
            if knob.startswith("extra_demos"):
                phase("collecting", round=rnd)
                cand_data = copy.copy(data)
                cand_data.vis, cand_data.states, cand_data.actions = (
                    list(data.vis),
                    list(data.states),
                    list(data.actions),
                )
                cand_data.aux, cand_data.lang_idx = list(data.aux), list(data.lang_idx)
                cand_data.instructions, cand_data._packed = (
                    list(data.instructions),
                    None,
                )
                near = None
                if knob == "extra_demos_near_failures":
                    near = [
                        {
                            "target": s["target_xy"],
                            **({"goal": s["goal_xy"]} if s["goal_xy"] else {}),
                        }
                        for s in best_scored
                        if not s["success"]
                    ] or None
                with comp:
                    collect(
                        spec,
                        int(val),
                        near,
                        60_000 + 1_000 * rnd,
                        tmp,
                        robot,
                        cand_data,
                    )
            else:
                cand_cfg[knob] = val
            phase("training", round=rnd)
            cand = copy.deepcopy(model)
            with comp:
                st = train(
                    cand,
                    cand_data,
                    float(cand_cfg["train_seconds"]),
                    lr=float(cand_cfg["lr"]),
                    batch=int(cand_cfg["batch"]),
                    state_noise=float(cand_cfg["state_noise"]),
                    on_log=lambda e, r=rnd: ev(opt.id, "opt_loss", round=r, **e),
                    log_every=0.5,
                    replay=replay,
                    replay_frac=float(cand_cfg["replay_frac"]),
                    weight_decay=float(cand_cfg["weight_decay"]),
                    scope=str(cand_cfg["scope"]),
                )
            sc = evaluate_round(rnd, cand)
            k = sum(s["success"] for s in sc)
            kept = k > best_k
            row.update(
                {
                    "status": "kept" if kept else "reverted",
                    "success": k,
                    "n": N_SCENES,
                    "ci": wilson(k, N_SCENES),
                    "failures": failure_report(sc, cfg)["failure_tags"],
                    "outcomes": [int(s["success"]) for s in sc],
                    "train": {
                        "steps": st["steps"],
                        "seconds": st["seconds"],
                        "frames": st["frames"],
                    },
                    "seconds": round(time.time() - t1, 1),
                    "vs": best_k,
                }
            )
            history.append(
                {
                    "round": rnd,
                    "change": change,
                    "success_before": best_k,
                    "success_after": k,
                    "kept": kept,
                }
            )
            if kept:
                model, cfg, data, best_k, best_scored = cand, cand_cfg, cand_data, k, sc
                opt.best_round = rnd
            del cand
            _save(opt)
            ev(opt.id, "opt_round", round=rnd, row=row)
        phase("replay")
        _hero(opt, stage, all_scored, out)
        model.save(
            out / "final.pt",
            {"optimizer": opt.id, "from": skill.id, "settings": cfg, "success": best_k},
        )
        opt.status = "done"
        opt.seconds = round(time.time() - t0, 1)
        opt.phase = ""
    except Exception as e:  # recorded on the run, then raised for the caller's log
        opt.status, opt.error = "error", f"{type(e).__name__}: {e}"
        raise
    finally:
        stage.close()
        _save(opt)
        ev(opt.id, "opt_done", status=opt.status)
    return opt


def _hero(
    opt: Opt, stage: montage.Stage, scored: dict[int, list[dict]], out: Path
) -> None:
    """One scene, the first try against the last kept policy: prefer a scene that failed then and passes now."""
    a, b = scored[0], scored[opt.best_round]
    pick = next(
        (i for i in range(N_SCENES) if not a[i]["success"] and b[i]["success"]), None
    )
    if pick is None:
        pick = next((i for i in range(N_SCENES) if not a[i]["success"]), 0)
    montage.write_hero(stage, a[pick], out / "hero_r0.jpg")
    montage.write_hero(stage, b[pick], out / "hero_final.jpg")
    opt.hero = {
        "scene": pick,
        "round_a": 0,
        "round_b": opt.best_round,
        "frames": montage.HERO_FRAMES,
        "w": montage.HERO_W,
        "h": montage.HERO_H,
        "a_success": a[pick]["success"],
        "b_success": b[pick]["success"],
        "a_failure": a[pick]["failure"],
        "b_failure": b[pick]["failure"],
    }


def new_opt(skill: Skill, n_rounds: int) -> Opt:
    return Opt(
        id=time.strftime("%H%M%S") + "-" + uuid4().hex[:4],
        skill={
            "id": skill.id,
            "instruction": skill.instruction,
            "success_rate": skill.success_rate,
            "n_eval": skill.n_eval,
        },
        n_rounds=n_rounds,
    )


def list_runs() -> list[dict]:
    out = []
    for p in sorted(RUNS.glob("opt-*/run.json"), reverse=True):
        d = json.loads(p.read_text())
        out.append(
            {
                k: d.get(k)
                for k in ("id", "skill", "status", "n_rounds", "started", "best_round")
            }
            | {
                "rounds_done": sum(
                    1 for r in d["rounds"] if r.get("success") is not None
                )
            }
        )
    return out


def load_run(oid: str) -> dict | None:
    p = RUNS / f"opt-{oid}" / "run.json"
    return (
        json.loads(p.read_text())
        if p.exists() and oid.replace("-", "").isalnum()
        else None
    )


def asset(oid: str, rel: str) -> Path | None:
    base = (RUNS / f"opt-{oid}").resolve()
    p = (base / rel).resolve()
    ok = (
        oid.replace("-", "").isalnum()
        and p.is_file()
        and base in p.parents
        and p.suffix == ".jpg"
    )
    return p if ok else None


def rerender_hero(oid: str) -> dict:
    """Replay the saved run's hero scene again (the first-try skill and final.pt) and rewrite the two filmstrips."""
    out = RUNS / f"opt-{oid}"
    d = json.loads((out / "run.json").read_text())
    from .registry import Registry

    sk = Registry().skills[d["skill"]["id"]]
    spec, dev = sk.task(), device()
    i = d["hero"]["scene"]
    lay = sample_layout(spec, np.random.default_rng(SEED0 + i)).to_json()
    stage = montage.Stage(sk.robot)
    res = {}
    for tag, path in (("a", ROOT / sk.model), ("b", out / "final.pt")):
        m, _ = VLA.load(path, dev)
        r = score(m, spec, [lay], sk.robot, lambda n: None, first=i)[0]
        res[tag] = r
        montage.write_hero(
            stage, r, out / ("hero_r0.jpg" if tag == "a" else "hero_final.jpg")
        )
    stage.close()
    d["hero"].update(
        a_success=res["a"]["success"],
        b_success=res["b"]["success"],
        a_failure=res["a"]["failure"],
        b_failure=res["b"]["failure"],
    )
    (out / "run.json").write_text(json.dumps(d, indent=1))
    return d["hero"]
