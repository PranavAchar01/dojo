"""One train/eval cycle from the command line, for measuring the pipeline without the agents.

uv run python scripts/experiment.py --skill pick_place --obj red --target white --demos 60 --seconds 90
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from vla_forge.datagen import demo_job, pool
from vla_forge.learn import Demos, train
from vla_forge.policy import VLA
from vla_forge.rollout import evaluate, summarize
from vla_forge.sim.tasks import TaskSpec

ROOT = Path(__file__).resolve().parents[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skill", default="pick_place")
    ap.add_argument("--obj", default="red")
    ap.add_argument("--target", default="white")
    ap.add_argument("--distractors", type=int, default=2)
    ap.add_argument("--demos", type=int, default=60)
    ap.add_argument("--seconds", type=float, default=90)
    ap.add_argument("--eval", type=int, default=20)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--init", default="")
    ap.add_argument("--eval-train", type=int, default=0)
    ap.add_argument("--save", default="")
    ap.add_argument("--look", type=float, default=1.0)
    ap.add_argument("--state-noise", type=float, default=0.03)
    ap.add_argument("--no-state", action="store_true")
    ap.add_argument("--noise", type=float, default=0.0)
    a = ap.parse_args()
    spec = TaskSpec(
        a.skill, a.obj, a.target if a.skill != "lift" else None, a.distractors
    )
    out = ROOT / "runs" / "exp" / f"{a.skill}-{a.obj}-{int(time.time())}"
    t0 = time.time()
    p = pool()
    print(f"pool ready {time.time() - t0:.1f}s")
    t1 = time.time()
    jobs = [
        {"spec": spec.to_json(), "seed": s, "out": str(out / f"ep{s:04d}.npz"), "look_strength": a.look, "noise": a.noise}
        for s in range(a.demos)
    ]
    res = list(p.map(demo_job, jobs))
    ok = [r for r in res if r["ok"]]
    print(f"demos {len(ok)}/{len(res)} in {time.time() - t1:.1f}s")
    t2 = time.time()
    data = Demos()
    for r in ok:
        data.add_episode(r["path"], spec.instruction)
    print(f"encoded {data.frames} frames in {time.time() - t2:.1f}s")
    if a.init:
        model, _ = VLA.load(Path(a.init))
    else:
        from vla_forge.policy import HeadConfig

        model = VLA(HeadConfig(use_state=not a.no_state), norm=data.norm())
    print(f"trainable params {model.trainable_params():,}")
    t3 = time.time()
    st = train(
        model,
        data,
        a.seconds,
        lr=a.lr,
        state_noise=a.state_noise,
        on_log=lambda e: print(json.dumps(e)) if e["step"] % 200 < 20 else None,
    )
    print(f"train {st} in {time.time() - t3:.1f}s")
    if a.save:
        model.save(Path(a.save), {"instruction": spec.instruction})
    if a.eval_train:
        tl = [r["layout"] for r in ok[: a.eval_train]]
        rt = evaluate(model, spec, len(tl), seed0=0, layouts=tl, record=False, look_strength=a.look)
        st_ = summarize(rt)
        print(f"eval on training layouts {st_['succeeded']}/{st_['n']} failures {st_['failures']}")
    t4 = time.time()
    rolls = evaluate(model, spec, a.eval, record=False, look_strength=a.look)
    s = summarize(rolls)
    print(
        f"eval {s['succeeded']}/{s['n']} in {time.time() - t4:.1f}s failures {s['failures']}"
    )
    print(f"total {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
