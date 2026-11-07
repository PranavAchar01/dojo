"""Trained-model tile thumbnails, shot on the shared stage from the engine's own recorded rollout.

A tile's saved replay (tiles.load_replay: the exported scene plus body-pose frames) gives the scene state in the
grasp-to-place stretch of its first successful rollout; web/tile-thumb.html draws that exact state in the Sim tab's
light look and scripts/tile_thumbs.mjs stamps the PNG with the stage tag.
"""

import gzip
import json
import os
import subprocess
import tempfile
from pathlib import Path

from .registry import ROOT, TILES_DIR
from .sim.robots import DEFAULT, parse_id
from .viz import robot_meshes

PHASE = 0.55  # the stretch of the rollout where the arm is working on the objects


def shoot(ids: list[str] | None = None) -> list[str]:
    """Shoot thumb.png for the given tiles (default: every tile with a replay); returns the ids shot."""
    found = [d.name for d in sorted(TILES_DIR.iterdir()) if (d / "replay.json.gz").exists()]
    ids = [i for i in found if not ids or i in ids]
    robots: set[str] = set()
    with tempfile.TemporaryDirectory() as tmp:
        for tid in ids:
            with gzip.open(TILES_DIR / tid / "replay.json.gz", "rt") as fh:
                rolls = json.load(fh)
            r = next((x for x in rolls if x["success"]), rolls[0])
            robot = r["scene"].get("robot_id", "so101")
            try:
                parse_id(robot)
            except ValueError:  # an older tile labelled by its rig ("so101-duo"): its meshes are the default arm's
                robot = DEFAULT
            robots.add(robot)
            job = {
                "id": tid,
                "robot": robot,
                "scene": r["scene"],
                "frame": r["frames"][int(len(r["frames"]) * PHASE)],
                "out": str(TILES_DIR / tid / "thumb.png"),
            }
            Path(tmp, f"{tid}.json").write_text(json.dumps(job))
        for robot in robots:
            Path(tmp, f"meshes-{robot}.json").write_text(json.dumps(robot_meshes(robot)))
        pw = os.environ.get("PW_DIR", str(Path.home() / "helloworld/qloo-inspo/node_modules"))
        subprocess.run(["node", str(ROOT / "scripts/tile_thumbs.mjs"), tmp], check=True, env={**os.environ, "PW_DIR": pw})
    return ids
