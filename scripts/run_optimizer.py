"""Run one optimizer experiment end to end (R0, optimizer arm, control arm, TEST) and save it as runs/opt-<id>.

    scripts/heavy.sh .venv/bin/python scripts/run_optimizer.py --skill insert-the-green-block-into-the-socket-f611 --rounds 5

VLA_FORGE_OPT_SCENES / VLA_FORGE_OPT_TEST_SCENES shrink the two scene sets for a smoke test.
"""

from __future__ import annotations

import argparse
import json

from vla_forge import optimizer
from vla_forge.pipeline import FORGE


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--skill", required=True)
    ap.add_argument("--rounds", type=int, default=5)
    ap.add_argument(
        "--train-seconds", type=float, default=optimizer.DEFAULTS["train_seconds"]
    )
    ap.add_argument("--control", default="random", choices=["random", "none", "off"])
    a = ap.parse_args()
    FORGE.warm()
    sk = FORGE.registry.skills[a.skill]
    opt = optimizer.new_opt(sk, a.rounds)
    print("run", opt.id, flush=True)
    optimizer.run(
        sk,
        a.rounds,
        opt,
        FORGE.replay,
        FORGE.compute,
        lambda *x, **k: (
            print(json.dumps([x[1], k], default=str)[:200], flush=True)
            if x[1] in ("opt_round", "opt_change", "opt_phase")
            else None
        ),
        a.train_seconds,
        None if a.control == "off" else a.control,
    )
    print("done", opt.id, opt.status, flush=True)


if __name__ == "__main__":
    main()
