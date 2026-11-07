"""Rebuild the Brick bridge card from the held-out placements: records every build, then updates skill SKILL_ID in place.

usage: hero_publish.py SKILL_ID MAIN_CHECKOUT  (the checkout whose skills/ the server reads)
"""

import sys
import time
from pathlib import Path

from vla_forge import hero_flow, hero_replay
from vla_forge.registry import Registry

if __name__ == "__main__":
    sid, ROOT = sys.argv[1], Path(sys.argv[2])
    t0 = time.time()
    rules, bricks = hero_flow.plan()
    scene, _ = hero_flow.scene_and_start(bricks)
    builds = hero_flow.build_all(bricks, rules, lambda k, b: print(k, b["seed"], f"{b['seated']}/{b['total']}", hero_flow.success(b), flush=True))
    ok = sum(hero_flow.success(b) for b in builds)
    rolls = [hero_replay.payload(i, b, scene) for i, b in enumerate(builds)]
    best = hero_flow.best(builds)
    files = hero_replay.save(sid, ROOT, best["thumb"], rolls)
    reg = Registry(ROOT / "skills" / "registry.json")
    record = dict(reg.get(sid).record or {})
    record["detail"] = hero_flow.detail(builds, round(time.time() - t0, 1))
    record["builds"] = [
        {k: b[k] for k in ("seed", "bricks", "seated", "total", "rules", "rules_passed", "rules_total", "sim_seconds", "wall_seconds")}
        for b in builds
    ]
    reg.update(
        sid,
        name=hero_flow.NAME,
        success_rate=ok / len(builds),
        n_eval=len(builds),
        seconds=round(time.time() - t0, 1),
        record=record,
        **files,
    )
    print(f"{ok}/{len(builds)} builds pass every rule; thumbnail from seed {best['seed']}; {time.time() - t0:.0f}s")
