"""Publish "Pack the toolbox" as an in-progress Library card: the scripted expert on the arm the engine routes to.

The spec gets its feasibility and routing exactly as a prompt's does (pipeline._judge); every seed runs the expert
on that arm, recorded at the tiles' 30 fps, and the code-checked "packed" rule decides success. The card is a
scripted skill (no policy yet), so the Library shows it as in progress until a trained policy replaces it.

usage: uv run python scripts/toolbox_card.py [N] [WORKERS] [ROBOT_ID]
"""

from __future__ import annotations

import gzip
import json
import secrets
import sys
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from vla_forge import tile_stage
from vla_forge.registry import ROOT, Registry, Skill
from vla_forge.sim import feasibility, routing
from vla_forge.sim.expert import Expert
from vla_forge.sim.scene import Layout, Scene
from vla_forge.sim.tasks import TaskSpec, check, sample_layout
from vla_forge.sim.toolbox import packed
from vla_forge.tiles import REPLAY_KEEP, tile_dir
from vla_forge.viz import export_scene

NAME = "Pack the toolbox"


def routed_spec(robot: str = "") -> tuple[TaskSpec, str]:
    """The spec routed as a prompt's is; a robot id given on the command line (an earlier routing's pick) skips
    the routing probes, which take minutes."""
    spec = TaskSpec("pack", "red")
    if robot:
        spec.feasibility = feasibility.for_spec(spec, robot)
        return spec, robot
    spec.feasibility = feasibility.for_spec(spec, "so101")
    spec.routing = routing.for_spec(spec)
    robot = spec.routing["chosen"][0]
    spec.feasibility = feasibility.for_spec(spec, robot)
    return spec, robot


def one(args: tuple[dict, str, int]) -> dict:
    spec_json, robot, seed = args
    spec = TaskSpec.from_json(spec_json)
    t0 = time.time()
    layout = sample_layout(spec, np.random.default_rng(seed), robot=robot)
    sc = Scene(layout, robot)
    sc.reset(layout)
    frames = [sc.pose_frame()]
    Expert(sc, spec, on_frame=lambda s: frames.append(s.pose_frame())).run()
    ok = bool(check(spec, sc, layout))
    inside = sum(packed(sc, [n]) for n in layout.tools)
    sc.close()
    return {
        "seed": seed,
        "success": ok,
        "failure": "" if ok else f"{inside}/{len(layout.tools)} tools packed",
        "layout": layout.to_json(),
        "frames": frames,
        "seconds": round(time.time() - t0, 1),
    }


def main(n: int, workers: int, robot: str = "") -> None:
    t0 = time.time()
    spec, robot = routed_spec(robot)
    print("routed to", robot, flush=True)
    with ProcessPoolExecutor(workers) as pool:
        res = list(pool.map(one, [(spec.to_json(), robot, s) for s in range(n)]))
    for r in res:
        print(
            r["seed"],
            "ok" if r["success"] else r["failure"],
            f"{r['seconds']}s",
            flush=True,
        )
    good = sum(r["success"] for r in res)
    print(f"packed {good}/{n}", flush=True)
    keep = (
        [r for r in res if r["success"]][: REPLAY_KEEP - 1]
        + [r for r in res if not r["success"]][:1]
    )[:REPLAY_KEEP]
    sid = f"pack-the-toolbox-{secrets.token_hex(2)}"
    d = tile_dir(sid)
    d.mkdir(parents=True, exist_ok=True)
    rolls = [
        {
            "index": r["seed"],
            "success": r["success"],
            "failure": r["failure"],
            "fps": 30,
            "scene": export_scene(Layout.from_json(r["layout"]), robot),
            "frames": r["frames"],
        }
        for r in keep
    ]
    with gzip.open(d / "replay.json.gz", "wt") as fh:
        json.dump(rolls, fh, separators=(",", ":"))
    tile_stage.shoot([sid])
    secs = round(time.time() - t0, 1)
    skill = Skill(
        id=sid,
        instruction=spec.instruction,
        spec=spec.to_json(),
        model="",
        parent=None,
        success_rate=good / n,
        n_eval=n,
        seconds=secs,
        prompt=NAME,
        robot=robot,
        origin="scripted",
        name=NAME,
        thumb_png=str((d / "thumb.png").relative_to(ROOT)),
        replay=str((d / "replay.json.gz").relative_to(ROOT)),
        replay_source="scripted expert",
        record={
            "detail": f"scripted expert on {robot}: {good}/{n} seeds packed every tool; policy training next",
            "seeds": [
                {k: r[k] for k in ("seed", "success", "failure", "seconds")}
                for r in res
            ],
        },
    )
    Registry().add(skill)
    print("saved", sid, flush=True)


if __name__ == "__main__":
    main(
        int(sys.argv[1]) if len(sys.argv) > 1 else 20,
        int(sys.argv[2]) if len(sys.argv) > 2 else 2,
        sys.argv[3] if len(sys.argv) > 3 else "",
    )
