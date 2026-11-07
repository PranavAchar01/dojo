"""The optimizer's loop on an established framework: OpenEvolve's population database, the language model as the operator.

OpenEvolve (github.com/algorithmicsuperintelligence/openevolve, an open AlphaEvolve-style system) supplies the part that
has to keep working round after round: a ProgramDatabase with islands, a MAP-Elites grid, elite archive, parent and
inspiration sampling, migration between islands, and checkpoint save and load. Its program is text; here a program is the
fine-tuning recipe (settings plus the plan of extra demonstrations) of one trained policy. The mutation operator is the
same schema-constrained language-model call as before (optimizer.propose); evaluation is training on the parent's data
and scoring on the 100 SELECT scenes. docs/optimizer-framework.md has the comparison and the reasons.

One round:
  1. the database picks a parent (an elite) and a few inspirations from another cell of the grid;
  2. the model sees the parent's failures, the population and what was already tried under this parent, and proposes one change;
  3. the child trains from the parent's weights and data, is scored on SELECT (and drawn in Blender, trajectories saved);
  4. it is admitted to the population only if SELECT success beats its parent's;
  5. the control arm takes its own round: a random valid change on its own best, same rule, same budget;
  6. every `test_every` rounds, or when TEST_REQUEST appears in the run folder, the current best of each arm is scored on TEST.
TEST never feeds back into any choice. Every peek is counted and written to the run record.

Everything lives under runs/opt-<id>/ and the run resumes from it: run.json (what the page reads), evo/ (the database),
weights/ (one .pt per elite and the control's best), traj/ (full trajectories per round), state.json (control arm).
Create a file STOP in the folder (or send SIGINT to scripts/run_optimizer.py) to end a perpetual run after the round.
"""

from __future__ import annotations

import contextlib
import copy
import json
import signal
import threading
import time
from collections import OrderedDict
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from openevolve.config import DatabaseConfig
from openevolve.database import Program, ProgramDatabase

from . import blender_montage, lanes, llama, montage, subq
from . import optimizer as O
from .encoders import device
from .learn import Demos, train
from .policy import VLA
from .registry import ROOT, Skill
from .sim.tasks import sample_layout

TEST_EVERY = 4
ROUND_BUDGET_S = 480.0
MAX_ROUNDS_SEEDS = (
    400  # round numbers whose demonstration seeds are checked against SELECT and TEST
)
CACHE = 1  # trained policies kept in memory besides the control arm's
LIGHT = ("index", "success", "failure", "steps", "stats", "target_xy", "goal_xy")
PEEK_NOTE = (
    "TEST is the held-out set. Which policy counts as best is decided on SELECT only. TEST is looked at on the current "
    "best of each arm at R0, every few rounds and at the end; every look is counted here, the reported number is the "
    "latest one (not the highest), and more looks mean more chances for a lucky reading."
)


def light(s: dict) -> dict:
    d = {k: s[k] for k in LIGHT}
    d["success"] = bool(d["success"])
    d["stats"] = {k: float(v) for k, v in d["stats"].items()}
    d["target_xy"] = [float(v) for v in d["target_xy"]]
    d["goal_xy"] = [float(v) for v in d["goal_xy"]] if d["goal_xy"] else None
    return d


@dataclass
class State:
    model: VLA
    data: Demos
    scored: list[dict]


def recipe_text(rec: dict) -> str:
    cfg = {k: v for k, v in rec["cfg"].items() if O.DEFAULTS.get(k) != v}
    plan = [
        f"{e['n']} {'near' if e['knob'].endswith('failures') else 'fresh'}@R{e['round']}"
        for e in rec["plan"]
    ]
    return json.dumps({"changed_settings": cfg, "extra_demos": plan})


def metrics_of(k: int, n: int, rec: dict) -> dict:
    return {
        "combined_score": k / n,
        "demos": float(sum(e["n"] for e in rec["plan"])),
        "train": float(rec["cfg"]["train_seconds"]),
    }


def new_db(path: Path) -> ProgramDatabase:
    cfg = DatabaseConfig(
        db_path=str(path),
        in_memory=True,
        population_size=64,
        archive_size=12,
        num_islands=2,
        feature_dimensions=["demos", "train"],
        feature_bins=4,
        elite_selection_ratio=0.3,
        exploration_ratio=0.2,
        exploitation_ratio=0.6,
        migration_interval=6,
        migration_rate=0.2,
        random_seed=11,
        log_prompts=False,
    )
    return ProgramDatabase(cfg)


def run(
    skill: Skill,
    n_rounds: int,
    opt: O.Opt,
    replay: Demos | None,
    compute: contextlib.AbstractContextManager | None = None,
    emit: Callable[..., None] | None = None,
    train_seconds: float = O.DEFAULTS["train_seconds"],
    control: str | None = "random",
    *,
    test_every: int = TEST_EVERY,
    round_budget_s: float = ROUND_BUDGET_S,
    resume: bool = False,
    max_minutes: float | None = None,
) -> O.Opt:
    comp = compute if compute is not None else contextlib.nullcontext()
    ev = emit or (lambda *a, **k: None)
    perpetual = n_rounds <= 0
    out = O.RUNS / f"opt-{opt.id}"
    for sub in ("tmp", "weights", "traj", "evo"):
        (out / sub).mkdir(parents=True, exist_ok=True)
    tmp = out / "tmp"
    stop_file = out / "STOP"
    stop_file.unlink(missing_ok=True)
    stop = threading.Event()
    if threading.current_thread() is threading.main_thread():
        signal.signal(signal.SIGINT, lambda *_: stop.set())
        signal.signal(signal.SIGTERM, lambda *_: stop.set())
    spec = skill.task()
    dev = device()
    robot = skill.robot
    base, _ = VLA.load(ROOT / skill.model, dev)
    N, NT = O.N_SCENES, O.N_TEST
    layouts = [
        sample_layout(spec, np.random.default_rng(O.SEED0 + i)).to_json()
        for i in range(N)
    ]
    test_layouts = [
        sample_layout(spec, np.random.default_rng(O.TEST_SEED0 + i)).to_json()
        for i in range(NT)
    ]
    seeds = O.split(max_rounds=MAX_ROUNDS_SEEDS)
    sel_keys = {json.dumps(x, sort_keys=True) for x in layouts}
    if any(json.dumps(x, sort_keys=True) in sel_keys for x in test_layouts):
        raise ValueError("a TEST scene is also a SELECT scene")
    opt.split = {
        "train": {
            "seeds": f"{O.START_SEED0}.. (start), {O.ROUND_SEED0}+1000*round.. (extra)",
            "start_demos": O.START_DEMOS,
            "used_for": "demonstration generation only",
        },
        "select": {"seed0": O.SEED0, "n": N, "used_for": "keep or revert, every round"},
        "test": {
            "seed0": O.TEST_SEED0,
            "n": NT,
            "used_for": "never used to choose; scored on the current best at R0, every few rounds and at the end (each look counted)",
        },
        "disjoint": True,
        "sizes": {k: len(v) for k, v in seeds.items()},
    }
    stage = montage.Stage(robot)
    cfg0 = {**O.DEFAULTS, "train_seconds": float(train_seconds)}
    opt.llm = opt.llm or {
        "asked": f"{llama.NAMES.get(llama.OPT_MODEL, llama.OPT_MODEL)} and {subq.NAME}",
        "backend": None,
        "backends_used": [],
    }
    opt.mode = "perpetual" if perpetual else "bounded"
    opt.session_start = time.time()
    bl = opt.blender or {
        "engine": None,
        "renders": 0,
        "rounds": 0,
        "seconds": 0.0,
        "workers": 0,
    }
    use_blender = blender_montage.blender_bin() is not None
    db = new_db(out / "evo")
    cache: OrderedDict[str, State] = OrderedDict()
    data_root: dict[str, Demos] = {}
    ctl: dict = {}
    test0: list[int] = []
    last_peek_round = -1

    # ------------------------------------------------------------------ helpers
    def phase(name: str, **kw) -> None:
        opt.phase = name
        save()
        ev(opt.id, "opt_phase", phase=name, **kw)

    def save() -> None:
        opt.engine = engine_json()
        opt.seconds = round(time.time() - opt.session_start + opt.prior_seconds, 1)
        O._save(opt)

    def best_program() -> Program:
        return max(
            db.programs.values(),
            key=lambda p: (p.metrics["combined_score"], -p.metadata["round"]),
        )

    def engine_json() -> dict:
        progs = sorted(db.programs.values(), key=lambda p: p.metadata["round"])
        best = best_program() if progs else None
        return {
            "framework": "OpenEvolve ProgramDatabase 0.4.1 (islands, MAP-Elites, elite archive, checkpoint and resume)",
            "operator": "language model proposal, one schema-constrained change",
            "islands": db.config.num_islands,
            "feature_dimensions": ["extra demonstrations", "seconds of training"],
            "population": [
                {
                    "program": p.id,
                    "round": p.metadata["round"],
                    "parent": p.metadata.get("parent_round"),
                    "island": p.metadata.get("island"),
                    "success": p.metrics["combined_score"] * N,
                    "recipe": json.loads(recipe_text(json.loads(p.code))),
                }
                for p in progs
            ],
            "best": best.id if best else None,
            "next_round": len(opt.rounds),
            "test_every": test_every,
            "round_budget_s": round_budget_s,
            "stop_with": f"create {out.name}/STOP",
        }

    def data0() -> Demos:
        if "d" not in data_root:
            d = Demos()
            with comp:
                O.collect(spec, O.START_DEMOS, None, O.START_SEED0, tmp, robot, d)
            data_root["d"] = d
        return O._copy_demos(data_root["d"])

    def build_data(plan: list[dict]) -> Demos:
        d = data0()
        for e in plan:
            with comp:
                O.collect(
                    spec,
                    e["n"],
                    e["near"],
                    O.ROUND_SEED0 + 1000 * e["round"],
                    tmp,
                    robot,
                    d,
                )
        return d

    def remember(pid: str, st: State) -> None:
        cache[pid] = st
        cache.move_to_end(pid)
        while len(cache) > CACHE:
            cache.popitem(last=False)

    def state_of(p: Program) -> State:
        if p.id in cache:
            cache.move_to_end(p.id)
            return cache[p.id]
        rec = json.loads(p.code)
        w = out / "weights" / f"{p.id}.pt"
        m = copy.deepcopy(base) if p.metadata["round"] == 0 else VLA.load(w, dev)[0]
        st = State(m, build_data(rec["plan"]), p.metadata["light"])
        remember(p.id, st)
        return st

    def save_traj(rnd: int, traj: dict[int, np.ndarray], sc: list[dict]) -> None:
        np.savez_compressed(
            out / "traj" / f"r{rnd}.npz", **{f"s{i}": a for i, a in traj.items()}
        )
        (out / "traj" / f"r{rnd}.json").write_text(json.dumps([light(s) for s in sc]))

    def score_with_traj(rnd: int, m: VLA) -> tuple[list[dict], dict]:
        traj: dict[int, np.ndarray] = {}
        with comp:
            sc = O.score(
                m,
                spec,
                layouts,
                robot,
                lambda n: ev(opt.id, "opt_scoring", round=rnd, done=n),
                traj=traj,
            )
        return sc, traj

    def evaluate_round(
        rnd: int, m: VLA, sheets: bool, pre: tuple[list[dict], dict] | None = None
    ) -> list[dict]:
        """Score (or take a scoring already done, `pre`) and, with sheets, save the trajectories and render them."""
        phase("scoring", round=rnd)
        traj: dict[int, np.ndarray] = {}
        if pre is not None:
            sc, traj = pre
        else:
            with comp:
                sc = O.score(
                    m,
                    spec,
                    layouts,
                    robot,
                    lambda n: ev(opt.id, "opt_scoring", round=rnd, done=n),
                    traj=traj if sheets else None,
                )
        if sheets:
            save_traj(rnd, traj, sc)
            phase("rendering", round=rnd)
            st = None
            if use_blender:
                try:
                    st = blender_montage.render_sheets(stage, sc, out / f"r{rnd}")
                except RuntimeError:
                    st = None
            if st is not None:
                bl["engine"] = st["engine"]
                bl["renders"] += st["renders"]
                bl["rounds"] += 1
                bl["seconds"] = round(bl["seconds"] + st["seconds"], 1)
                bl["workers"] = max(bl["workers"], st["workers"])
                bl.setdefault("per_round", []).append(
                    {
                        "round": rnd,
                        **{k: st[k] for k in ("renders", "workers", "seconds")},
                    }
                )
            else:
                montage.write_sheets(stage, sc, out / f"r{rnd}")
                bl["engine"] = "mujoco"
            opt.blender = bl
        return sc

    def train_child(
        m: VLA, data: Demos, cfg: dict, kind: str, rnd: int
    ) -> tuple[VLA, dict]:
        cand = copy.deepcopy(m)
        with comp:
            st = train(
                cand,
                data,
                float(cfg["train_seconds"]),
                lr=float(cfg["lr"]),
                batch=int(cfg["batch"]),
                state_noise=float(cfg["state_noise"]),
                on_log=(lambda e, r=rnd: ev(opt.id, "opt_loss", round=r, **e))
                if kind == "optimizer"
                else None,
                log_every=0.5,
                replay=replay,
                replay_frac=float(cfg["replay_frac"]),
                weight_decay=float(cfg["weight_decay"]),
                scope=str(cfg["scope"]),
            )
        return cand, st

    def apply_change(
        change: dict | None, parent_st: State, rec: dict, scored: list[dict], rnd: int
    ):
        """(child recipe, child data): demonstration knobs add data for this round, every other knob edits the settings."""
        cfg, plan = dict(rec["cfg"]), list(rec["plan"])
        data = parent_st.data
        if change and change["knob"].startswith("extra_demos"):
            near = None
            if change["knob"] == "extra_demos_near_failures":
                near = [
                    {
                        "target": s["target_xy"],
                        **({"goal": s["goal_xy"]} if s["goal_xy"] else {}),
                    }
                    for s in scored
                    if not s["success"]
                ] or None
            entry = {
                "knob": change["knob"],
                "n": int(change["value"]),
                "round": rnd,
                "near": near,
            }
            phase("collecting", round=rnd)
            data = O._copy_demos(data)
            with comp:
                O.collect(
                    spec, entry["n"], near, O.ROUND_SEED0 + 1000 * rnd, tmp, robot, data
                )
            plan.append(entry)
        elif change:
            v = change["value"]
            if change["knob"] == "train_seconds":
                v = min(v, max(O.KNOBS["train_seconds"][0], 0.3 * round_budget_s))
            cfg[change["knob"]] = v
        return {"cfg": cfg, "plan": plan}, data

    def row_base(
        rnd: int, change: dict | None, backend: str, text: str, kind: str
    ) -> dict:
        ch = (
            {**change, "text": text, "model": llama.label(backend)}
            if change
            else {
                "knob": None,
                "value": None,
                "text": text,
                "reason": "control: no change",
            }
        )
        if kind == "control" and change:
            ch.pop("model", None)
        return {"round": rnd, "change": ch, "backend": backend, "status": "training"}

    def finish_row(
        row: dict, k: int, vs: int, sc: list[dict], cfg: dict, st: dict, t1: float
    ) -> None:
        row.update(
            {
                "status": "kept" if k > vs else "reverted",
                "success": k,
                "n": N,
                "ci": O.wilson(k, N),
                "failures": O.failure_report(sc, cfg)["failure_tags"],
                "outcomes": [int(s["success"]) for s in sc],
                "train": {
                    "steps": st["steps"],
                    "seconds": st["seconds"],
                    "frames": st["frames"],
                },
                "seconds": round(time.time() - t1, 1),
                "vs": vs,
                "budget_s": round_budget_s,
                "over_budget": time.time() - t1 > round_budget_s,
            }
        )

    # ------------------------------------------------------------------ the two arms, one round each
    history: list[dict] = []

    def change_of(arms: list, who: str) -> dict:
        return next(ch for w, ch, _ in arms if w == who)

    def subq_view(pround: int, report: dict, cfg: dict) -> dict:
        """Everything SubQ reads: every round, every scored scene of every round, then trajectories (parent's round
        first), each subsampled to 6 moments."""
        rounds = [
            {
                k: r.get(k)
                for k in ("round", "parent", "status", "success", "vs", "failures")
            }
            | {"change": (r.get("change") or {}).get("text")}
            for r in opt.rounds
            if r.get("status") != "training"
        ]
        scenes, traj = {}, {}
        for f in sorted((out / "traj").glob("r*.json")):
            with contextlib.suppress(OSError, ValueError):
                scenes[f.stem] = json.loads(f.read_text())
        order = [out / "traj" / f"r{pround}.npz"] + sorted(
            (out / "traj").glob("r*.npz"), key=lambda p: -p.stat().st_mtime
        )
        for f in dict.fromkeys(order):
            if not f.exists():
                continue
            with contextlib.suppress(OSError, ValueError), np.load(f) as z:
                traj[f.stem] = {
                    k: np.round(
                        z[k][np.linspace(0, len(z[k]) - 1, 6).astype(int)], 3
                    ).tolist()
                    for k in z.files
                }
        return O.subq_state(report, cfg, rounds, scenes, traj)

    def optimizer_round(rnd: int) -> None:
        t1 = time.time()
        phase("proposing", round=rnd)
        db.set_current_island(rnd % db.config.num_islands)
        parent, insp = db.sample(num_inspirations=3)
        prec = json.loads(parent.code)
        pround = parent.metadata["round"]
        pst = state_of(parent)
        k_parent = round(parent.metrics["combined_score"] * N)
        report = O.failure_report(pst.scored, prec["cfg"])
        ev_scenes = O.pick_evidence(pst.scored, rnd)
        png = O.evidence_png(out / f"r{pround}", ev_scenes)
        hist = [h for h in history if h["parent"] == pround or h["kept"]]
        extra = {
            "parent_program": parent.id,
            "evidence_scenes": ev_scenes,
            "population": [
                {
                    "program": p.id,
                    "round": p.metadata["round"],
                    "success": round(p.metrics["combined_score"] * N),
                    "recipe": json.loads(recipe_text(json.loads(p.code))),
                }
                for p in sorted(
                    db.programs.values(), key=lambda q: -q.metrics["combined_score"]
                )[:6]
            ],
            "inspirations": [p.id for p in insp],
        }
        (change, backend), (q_change, q_p, q_err) = O.propose_pair(
            report,
            hist,
            rnd,
            [png] if png else None,
            extra,
            subq_view(pround, report, prec["cfg"]),
            prec["cfg"],
            opt.id,
        )
        opt.llm["backend"] = backend
        if backend not in opt.llm["backends_used"]:
            opt.llm["backends_used"].append(backend)
        opt.llm["real_meta"] = backend.startswith("meta:")
        row = row_base(rnd, change, backend, O.describe(change), "optimizer")
        row.update({"parent": pround, "program": f"p{rnd}", "evidence": ev_scenes})
        opt.rounds.append(row)
        save()
        ev(opt.id, "opt_change", round=rnd, change=row["change"])
        # head to head: each distinct proposal trains from the same parent with the same budget, scored on the same SELECT
        arms = [("meta", change, backend)]
        if q_change and not O.same_change(q_change, change):
            arms.append(("subq", q_change, f"subq:{subq.MODEL}"))
        res: dict[str, dict] = {}
        for who, ch, _be in arms:
            crec, cdata = apply_change(ch, pst, prec, pst.scored, rnd)
            phase("training", round=rnd, arm=who)
            # the loss chart follows the first arm only, so two curves never interleave
            kind = "optimizer" if who == "meta" else "duel"
            cand, tst = train_child(pst.model, cdata, crec["cfg"], kind, rnd)
            phase("scoring", round=rnd, arm=who)
            sc, traj = score_with_traj(rnd, cand)
            res[who] = {
                "crec": crec,
                "cdata": cdata,
                "cand": cand,
                "tst": tst,
                "pre": (sc, traj),
                "k": sum(s["success"] for s in sc),
            }
        if "subq" not in res:
            winner = "agreed" if q_change else "solo"
            take = "meta"
        else:
            km, kq = res["meta"]["k"], res["subq"]["k"]
            winner = "meta" if km > kq else "subq" if kq > km else "tie"
            take = "subq" if winner == "subq" else "meta"
        w = res[take]
        if take == "subq":
            change, backend = q_change, f"subq:{subq.MODEL}"
            row["change"] = {
                **q_change,
                "text": O.describe(q_change),
                "model": subq.NAME,
            }
            row["backend"] = backend
        crec, cdata, cand, tst = w["crec"], w["cdata"], w["cand"], w["tst"]
        sc = evaluate_round(rnd, cand, sheets=True, pre=w["pre"])
        k = w["k"]
        finish_row(row, k, k_parent, sc, crec["cfg"], tst, t1)
        row["duel"] = {
            "round": rnd,
            "n_select": N,
            "parent_k": k_parent,
            "meta": {
                "change": {kk: change_of(arms, "meta")[kk] for kk in ("knob", "value")},
                "label": llama.label(arms[0][2]),
                "backend": arms[0][2],
                "k": res["meta"]["k"],
            },
            "subq": {
                "change": {kk: q_change[kk] for kk in ("knob", "value")}
                if q_change
                else None,
                "p": round(q_p, 3) if q_p is not None else None,
                "error": q_err,
                "k": res["subq"]["k"]
                if "subq" in res
                else (res["meta"]["k"] if q_change else None),
            },
            "winner": winner,
            "admitted": k > k_parent,
        }
        lanes.duel(opt.id, row["duel"])
        history.append(
            {
                "round": rnd,
                "parent": pround,
                "change": change,
                "success_before": k_parent,
                "success_after": k,
                "kept": k > k_parent,
            }
        )
        if k > k_parent:
            pid = f"p{rnd}"
            cand.save(
                out / "weights" / f"{pid}.pt",
                {
                    "optimizer": opt.id,
                    "program": pid,
                    "settings": crec["cfg"],
                    "success": k,
                },
            )
            child = Program(
                id=pid,
                code=json.dumps(crec),
                language="json",
                parent_id=parent.id,
                generation=parent.generation + 1,
                iteration_found=rnd,
                changes_description=O.describe(change),
                metrics=metrics_of(k, N, crec),
                metadata={
                    "round": rnd,
                    "parent_round": pround,
                    "light": [light(s) for s in sc],
                    "reason": change.get("reason", ""),
                },
            )
            db.add(child, iteration=rnd)
            remember(pid, State(cand, cdata, [light(s) for s in sc]))
        db.increment_island_generation()
        if db.should_migrate():
            db.migrate_programs()
        opt.best_round = best_program().metadata["round"]
        db.save(str(out / "evo"), iteration=rnd)
        save()
        ev(opt.id, "opt_round", round=rnd, row=row)

    def control_round(rnd: int) -> None:
        t1 = time.time()
        phase("control", round=rnd)
        rng = np.random.default_rng(7 + rnd)
        if control == "random":
            change, backend = O.random_change(rng, ctl["hist"]), "random"
        else:
            change, backend = None, "none"
        text = O.describe(change) if change else "no change, train again"
        row = row_base(rnd, change, backend, text, "control")
        ctl["rows"].append(row)
        opt.control["rounds"] = ctl["rows"]
        save()
        rec = {"cfg": ctl["cfg"], "plan": ctl["plan"]}
        pst = State(ctl["model"], ctl["data"], ctl["scored"])
        crec, cdata = apply_change(change, pst, rec, ctl["scored"], rnd)
        phase("control", round=rnd)
        cand, tst = train_child(ctl["model"], cdata, crec["cfg"], "control", rnd)
        sc = evaluate_round(rnd, cand, sheets=False)
        k = sum(s["success"] for s in sc)
        finish_row(row, k, ctl["best_k"], sc, crec["cfg"], tst, t1)
        ctl["hist"].append(
            {
                "round": rnd,
                "change": change or {"knob": "none", "value": "none", "reason": ""},
                "success_before": ctl["best_k"],
                "success_after": k,
                "kept": k > ctl["best_k"],
            }
        )
        if k > ctl["best_k"]:
            ctl.update(
                model=cand,
                cfg=crec["cfg"],
                plan=crec["plan"],
                data=cdata,
                best_k=k,
                scored=[light(s) for s in sc],
                best_round=rnd,
            )
            cand.save(
                out / "weights" / "control.pt",
                {"optimizer": opt.id, "control": True, "success": k},
            )
        (out / "state.json").write_text(
            json.dumps(
                {
                    "cfg": ctl["cfg"],
                    "plan": ctl["plan"],
                    "best_k": ctl["best_k"],
                    "best_round": ctl["best_round"],
                    "hist": ctl["hist"],
                    "scored": ctl["scored"],
                }
            )
        )
        opt.control.update(best_round=ctl["best_round"], select_final=ctl["best_k"])
        save()

    # ------------------------------------------------------------------ TEST peeks (counted, never fed back)
    def peek(rnd: int, why: str) -> None:
        nonlocal last_peek_round
        if (
            last_peek_round == rnd
            and opt.test.get("peeks")
            and opt.test["peeks"][-1]["why"] == why
        ):
            return
        phase("testing", round=rnd)
        best = best_program()
        bst = state_of(best)
        with comp:
            t = O.score_outcomes(bst.model, spec, test_layouts, robot, O.TEST_SEED0)
        peeks = list(opt.test.get("peeks", []))
        pk = {
            "round": rnd,
            "why": why,
            "best_round": best.metadata["round"],
            "optimizer": sum(t),
        }
        cmp_ = O.compare(test0, t)
        pk.update(delta=cmp_["delta"], p=cmp_["mcnemar"]["p"])
        if control:
            with comp:
                ct = O.score_outcomes(
                    ctl["model"], spec, test_layouts, robot, O.TEST_SEED0
                )
            pk.update(control=sum(ct), control_best_round=ctl["best_round"])
            cc = O.compare(test0, ct)
            opt.control.update(
                kind=control,
                test_final=ct,
                vs_r0=cc,
                optimizer_vs_control=O.compare(ct, t),
                renders="none: the control arm is scored without Blender",
            )
        peeks.append(pk)
        opt.test = {
            "n": NT,
            "r0": test0,
            "final": t,
            "final_round": best.metadata["round"],
            "compare": cmp_,
            "peeks": peeks,
            "peek_count": len(peeks),
            "note": PEEK_NOTE,
        }
        last_peek_round = rnd
        hero(best.metadata["round"])
        save()
        ev(opt.id, "opt_peek", round=rnd, peek=pk)

    def hero(best_round: int) -> None:
        if best_round == 0 or (opt.hero and opt.hero.get("round_b") == best_round):
            return
        phase("replay")
        a, b = (
            np.load(out / "traj" / "r0.npz"),
            np.load(out / "traj" / f"r{best_round}.npz"),
        )
        sa = json.loads((out / "traj" / "r0.json").read_text())
        sb = json.loads((out / "traj" / f"r{best_round}.json").read_text())
        pick = next(
            (i for i in range(N) if not sa[i]["success"] and sb[i]["success"]), None
        )
        if pick is None:
            pick = next((i for i in range(N) if not sa[i]["success"]), 0)
        for z, ss, name in ((a, sa, "hero_r0.jpg"), (b, sb, "hero_final.jpg")):
            scene = {"layout": layouts[pick], "frames": z[f"s{pick}"]}
            r = None
            if use_blender:
                try:
                    r = blender_montage.render_hero(stage, scene, out / name)
                except RuntimeError:
                    r = None
            if r is not None:
                bl["renders"] += r["renders"]
                bl["seconds"] = round(bl["seconds"] + r["seconds"], 1)
                bl["hero_renders"] = bl.get("hero_renders", 0) + r["renders"]
            else:
                montage.write_hero(stage, scene, out / name)
        opt.hero = {
            "scene": pick,
            "round_a": 0,
            "round_b": best_round,
            "frames": montage.HERO_FRAMES,
            "w": montage.HERO_W,
            "h": montage.HERO_H,
            "a_success": sa[pick]["success"],
            "b_success": sb[pick]["success"],
            "a_failure": sa[pick]["failure"],
            "b_failure": sb[pick]["failure"],
            "engine": "blender-eevee" if use_blender else "mujoco",
        }

    # ------------------------------------------------------------------ start or resume
    try:
        opt.status = "running"
        opt.error = ""
        t0 = time.time()
        resumed = (
            resume
            and (out / "evo").exists()
            and any((out / "evo").iterdir())
            and len(opt.rounds) > 0
        )
        if resumed:
            db.load(str(out / "evo"))
            test0 = list(opt.test["r0"])
            last_peek_round = max(
                (p["round"] for p in opt.test.get("peeks", [])), default=-1
            )
            for r in opt.rounds[1:]:
                if r.get("success") is not None and "parent" in r:
                    history.append(
                        {
                            "round": r["round"],
                            "parent": r["parent"],
                            "change": r["change"],
                            "success_before": r["vs"],
                            "success_after": r["success"],
                            "kept": r["status"] == "kept",
                        }
                    )
            opt.rounds = [
                r for r in opt.rounds if r.get("success") is not None
            ]  # drop a round cut off midway
            if control:
                s = json.loads((out / "state.json").read_text())
                done = [
                    r
                    for r in opt.control.get("rounds", [])
                    if r.get("success") is not None
                ]
                data = build_data(s["plan"])
                cm = (
                    copy.deepcopy(base)
                    if s["best_round"] == 0
                    else VLA.load(out / "weights" / "control.pt", dev)[0]
                )
                ctl.update(
                    model=cm,
                    cfg=s["cfg"],
                    plan=s["plan"],
                    data=data,
                    best_k=s["best_k"],
                    best_round=s["best_round"],
                    hist=s["hist"],
                    scored=s["scored"],
                    rows=done,
                )
                opt.control["rounds"] = done
            # a round that finished its optimizer arm but not its control arm: the control catches up first
            opt.llm["resumed"] = opt.llm.get("resumed", 0) + 1
        else:
            sc0 = evaluate_round(0, base, sheets=True)
            k0 = sum(s["success"] for s in sc0)
            opt.rounds = [
                {
                    "round": 0,
                    "change": None,
                    "status": "start",
                    "success": k0,
                    "n": N,
                    "ci": O.wilson(k0, N),
                    "failures": O.failure_report(sc0, cfg0)["failure_tags"],
                    "outcomes": [int(s["success"]) for s in sc0],
                    "seconds": round(time.time() - t0, 1),
                    "program": "p0",
                }
            ]
            phase("testing", round=0)
            with comp:
                test0 = O.score_outcomes(base, spec, test_layouts, robot, O.TEST_SEED0)
            opt.test = {
                "n": NT,
                "r0": test0,
                "peeks": [],
                "peek_count": 0,
                "note": PEEK_NOTE,
            }
            root = Program(
                id="p0",
                code=json.dumps({"cfg": cfg0, "plan": []}),
                language="json",
                iteration_found=0,
                changes_description="the skill as trained",
                metrics=metrics_of(k0, N, {"cfg": cfg0, "plan": []}),
                metadata={
                    "round": 0,
                    "parent_round": None,
                    "light": [light(s) for s in sc0],
                },
            )
            db.add(root, iteration=0, target_island=0)
            for i in range(
                1, db.config.num_islands
            ):  # every island starts from the same root
                clone = copy.deepcopy(root)
                clone.id = f"p0-i{i}"
                clone.metadata = {**root.metadata, "round": 0}
                db.add(clone, iteration=0, target_island=i)
            remember("p0", State(copy.deepcopy(base), data0(), [light(s) for s in sc0]))
            if control:
                opt.control = {"kind": control, "rounds": []}
                ctl.update(
                    model=copy.deepcopy(base),
                    cfg=dict(cfg0),
                    plan=[],
                    data=data0(),
                    best_k=k0,
                    best_round=0,
                    hist=[],
                    scored=[light(s) for s in sc0],
                    rows=opt.control["rounds"],
                )
            opt.best_round = 0
            save()
        rnd = len(opt.rounds)
        while True:
            if stop_file.exists() or stop.is_set():
                break
            if not perpetual and rnd > n_rounds:
                break
            if max_minutes and (time.time() - t0) / 60 > max_minutes:
                break
            optimizer_round(rnd)
            if control:
                control_round(rnd)
            req = out / "TEST_REQUEST"
            if req.exists():
                req.unlink()
                peek(rnd, "requested")
            elif test_every and rnd % test_every == 0:
                peek(rnd, f"every {test_every} rounds")
            rnd += 1
        peek(rnd - 1, "final")
        opt.status = "done" if (not perpetual and rnd > n_rounds) else "stopped"
        opt.phase = ""
    except Exception as e:  # recorded on the run, then raised for the caller's log
        opt.status, opt.error = "error", f"{type(e).__name__}: {e}"
        raise
    finally:
        stage.close()
        save()
        ev(opt.id, "opt_done", status=opt.status)
    return opt
