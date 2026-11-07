"""Run the optimizer on a skill and save it as runs/opt-<id>: R0, then rounds of the population loop plus the control arm,
with the held-out TEST set looked at every few rounds and at the end.

    scripts/heavy.sh .venv/bin/python scripts/run_optimizer.py --skill insert-the-green-block-into-the-socket-f611 --rounds 8
    ... --perpetual --max-minutes 45        # until stopped (STOP file in the run folder, or Ctrl-C), resumable
    ... --resume 123456-abcd --perpetual    # continue a saved run

VLA_FORGE_OPT_SCENES / VLA_FORGE_OPT_TEST_SCENES shrink the two scene sets for a smoke test.
"""

from __future__ import annotations

import argparse
import json

from vla_forge import optimizer, optimizer_evo
from vla_forge.pipeline import FORGE


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--skill")
    ap.add_argument("--rounds", type=int, default=5)
    ap.add_argument(
        "--perpetual", action="store_true", help="no round limit: run until stopped"
    )
    ap.add_argument("--resume", metavar="ID", help="continue a saved run")
    ap.add_argument("--max-minutes", type=float)
    ap.add_argument("--test-every", type=int, default=optimizer_evo.TEST_EVERY)
    ap.add_argument("--round-budget", type=float, default=optimizer_evo.ROUND_BUDGET_S)
    ap.add_argument(
        "--train-seconds", type=float, default=optimizer.DEFAULTS["train_seconds"]
    )
    ap.add_argument("--control", default="random", choices=["random", "none", "off"])
    a = ap.parse_args()
    FORGE.warm()
    if a.resume:
        opt = optimizer.load_opt(a.resume)
        sk = FORGE.registry.skills[opt.skill["id"]]
        rounds = 0 if a.perpetual else opt.n_rounds
        if a.perpetual:
            opt.n_rounds = 0
    else:
        if not a.skill:
            ap.error("--skill is required")
        sk = FORGE.registry.skills[a.skill]
        rounds = 0 if a.perpetual else a.rounds
        opt = optimizer.new_opt(sk, rounds)
    print("run", opt.id, flush=True)
    optimizer.run(
        sk,
        rounds,
        opt,
        FORGE.replay,
        FORGE.compute,
        lambda *x, **k: (
            print(json.dumps([x[1], k], default=str)[:240], flush=True)
            if x[1] in ("opt_round", "opt_change", "opt_phase", "opt_peek")
            else None
        ),
        a.train_seconds,
        None if a.control == "off" else a.control,
        test_every=a.test_every,
        round_budget_s=a.round_budget,
        resume=bool(a.resume),
        max_minutes=a.max_minutes,
    )
    print("done", opt.id, opt.status, flush=True)


if __name__ == "__main__":
    main()
