"""Step 1 of the bridge (the left pillar base) on the two-arm table: N randomized episodes, measured against 2 mm / 5 deg.

usage: duck_step1.py N OUT.json [SHOT.png]
"""

import json
import sys

import numpy as np

from vla_forge import hero_expert as de
from vla_forge import hero_sim as ds


def episode(seed: int, shot: str | None = None) -> dict:
    t = ds.Table.new(960)
    rng = np.random.default_rng(seed)
    t.reset(rng, jitter=0.02)
    t.step(30)
    e = de.Expert(t, "a_")
    try:
        e.pick("left1")
        lifted = t.pose("left1")[0][2] > 0.02
        ok = e.place("left1")
        e.home()
    except Exception as ex:  # an IK or contact failure is a failed episode, recorded as such
        return {"seed": seed, "success": False, "error": repr(ex)[:120]}
    pe, ye, dz = e.pre_snap
    if shot:
        from PIL import Image

        Image.fromarray(t.render("overview", 1280, 960)).save(shot)
    return {"seed": seed, "success": bool(ok), "lifted": bool(lifted), "pos_err_mm": round(pe * 1e3, 2),
            "yaw_err_deg": round(float(np.degrees(ye)), 2), "dz_mm": round(dz * 1e3, 2), "ik_err_mm": round(e.max_ik_err * 1e3, 2)}


if __name__ == "__main__":
    n, out = int(sys.argv[1]), sys.argv[2]
    shot = sys.argv[3] if len(sys.argv) > 3 else None
    res = [episode(s, shot if s == 0 else None) for s in range(n)]
    json.dump({"episodes": res, "success_rate": sum(r["success"] for r in res) / n}, open(out, "w"), indent=1)
    for r in res:
        print(r)
