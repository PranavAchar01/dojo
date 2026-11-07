"""Per-brick diagnosis of the scripted bridge: usage hero_diag.py SEED [SEED...] (one JSON line per brick)."""

import json
import sys

import numpy as np

from vla_forge import hero_build as hbd
from vla_forge import hero_expert as he
from vla_forge import hero_sim as ds


def trace(e, stage, name):
    p, yaw = e.t.pose(name)
    tp = ds.target_pose(hbd.hb.brick_map()[name])[0]
    print(
        f"  {e.pre}{name:7s} {stage:9s} brick-target mm={np.round((p - tp) * 1e3, 1)} yaw={np.degrees(yaw):6.1f} ee={np.round(e.ee * 1e3, 0)} align={np.degrees(np.arccos(np.clip(e.ik.fk(e.q)[1][:, 2] @ e.ik.heading, -1, 1))) if e.ik.heading is not None else -1:.1f}deg",
        flush=True,
    )


if __name__ == "__main__":
    for s in map(int, sys.argv[1:]):
        he.Expert.__init__.__defaults__  # noqa: B018
        r = hbd.run(s, record=False, trace=trace)
        for b in r["bricks"]:
            print(json.dumps({"seed": s, **b}), flush=True)
        print(
            json.dumps(
                {
                    "seed": s,
                    "seated": r["seated"],
                    "rules": [(x["id"], x["ok"]) for x in r["rules"]],
                    "wall": r["wall_seconds"],
                }
            ),
            flush=True,
        )
