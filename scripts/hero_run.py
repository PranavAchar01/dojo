"""Build the whole bridge on the two-arm table N times (seeds 0..N-1, two at a time): seated bricks and rule results per seed.

usage: hero_run.py N OUT.json [SHOT.png]
"""

import json
import sys
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context

from PIL import Image
from vla_forge import hero_build as hbd

WORKERS = 2


def one(s: int, shot: bool) -> dict:
    r = hbd.run(s, record=shot)
    r["thumb_ok"] = r["thumb"] is not None
    if shot and r["thumb"] is not None:
        Image.fromarray(r["thumb"]).save(sys.argv[3])
    r["n_frames"] = len(r["frames"])
    for k in ("frames", "marks", "table", "thumb"):
        r.pop(k)
    return r


if __name__ == "__main__":
    n, out = int(sys.argv[1]), sys.argv[2]
    shot = len(sys.argv) > 3
    with ProcessPoolExecutor(WORKERS, mp_context=get_context("spawn")) as pool:
        res = list(pool.map(one, range(n), [shot and s == 0 for s in range(n)]))
    for r in res:
        print(
            r["seed"],
            f"{r['seated']}/{r['total']} bricks, rules {r['rules_passed']}/{r['rules_total']}, {r['wall_seconds']}s, frames {r['n_frames']}",
            flush=True,
        )
    ok = sum(
        r["seated"] == r["total"] and r["rules_passed"] == r["rules_total"] for r in res
    )
    print(f"{ok}/{n} builds seat every brick and pass every rule", flush=True)
    with open(out, "w") as fh:
        json.dump(res, fh, indent=1)
