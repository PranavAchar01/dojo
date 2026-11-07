"""Build the whole bridge on the two-arm table N times: seated bricks and rule results per seed.

usage: hero_run.py N OUT.json [SHOT.png]
"""

import json
import sys

from PIL import Image

from vla_forge import hero_build as hbd

if __name__ == "__main__":
    n, out = int(sys.argv[1]), sys.argv[2]
    res = []
    for s in range(n):
        r = hbd.run(s, record=len(sys.argv) > 3 and s == 0)
        if len(sys.argv) > 3 and s == 0 and r["thumb"] is not None:
            Image.fromarray(r["thumb"]).save(sys.argv[3])
        print(s, f"{r['seated']}/{r['total']} bricks, rules {r['rules_passed']}/{r['rules_total']}, {r['wall_seconds']}s, frames {len(r['frames'])}", flush=True)
        for k in ("frames", "marks", "table", "thumb"):
            r.pop(k)
        res.append(r)
    json.dump(res, open(out, "w"), indent=1)
