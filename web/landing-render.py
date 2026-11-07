"""Renders the landing hero loop: a recorded successful rollout of the SO-101 (a skills/tiles/<id>/replay.json.gz),
put back into the compiled MuJoCo scene frame by frame. Run from the repo root:
  uv run python web/landing-render.py REPLAY.json.gz OUT_DIR [W H DIST AZ EL]
Writes OUT_DIR/f%04d.jpg; encode with ffmpeg afterwards."""

import gzip
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

from vla_forge import montage
from vla_forge.montage import Stage

_build = montage.build_spec


def _big(lay, robot="so101"):  # the default offscreen buffer is 640x480
    spec = _build(lay, robot=robot)
    spec.visual.global_.offwidth, spec.visual.global_.offheight = 2400, 1400
    return spec


montage.build_spec = _big

sid, out = sys.argv[1], Path(sys.argv[2])
w, h = (int(sys.argv[3]), int(sys.argv[4])) if len(sys.argv) > 4 else (1600, 800)
dist, az, el = (
    (float(x) for x in sys.argv[5:8]) if len(sys.argv) > 7 else (0.5, 148.0, -30.0)
)
with gzip.open(Path(sid), "rt") as fh:
    rolls = json.load(fh)
r = next(x for x in rolls if x["success"])
out.mkdir(parents=True, exist_ok=True)
st = Stage()
frames = np.asarray(r["frames"], float)
# the replay stores poses, not the layout; rebuild it from the first frame (block yaw from its quaternion)
f0 = frames[0].reshape(-1, 7)


def yaw(q):
    return float(
        np.arctan2(2 * (q[0] * q[3] + q[1] * q[2]), 1 - 2 * (q[2] ** 2 + q[3] ** 2))
    )


sc = r["scene"]
layout = {
    "blocks": {
        b["name"]: [
            b["color"],
            float(f0[b["body"]][0]),
            float(f0[b["body"]][1]),
            yaw(f0[b["body"]][3:]),
        ]
        for b in sc["blocks"]
    },
    "bowls": {
        b["name"]: [b["color"], float(f0[b["body"]][0]), float(f0[b["body"]][1])]
        for b in sc["bowls"]
    },
    "pads": {
        b["name"]: [b["color"], float(f0[b["body"]][0]), float(f0[b["body"]][1])]
        for b in sc["pads"]
    },
}
print(layout)
n = len(frames)
for i0 in range(0, n, 20):
    idx = list(range(i0, min(n, i0 + 20)))
    for i, img in zip(idx, st.render(layout, frames, idx, w, h, dist, az, el)):
        Image.fromarray(img).save(out / f"f{i:04d}.jpg", quality=92)
st.close()
print(n)
