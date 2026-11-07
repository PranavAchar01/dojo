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

from . import hero_replay
from .registry import ROOT, TILES_DIR
from .sim.robots import DEFAULT, parse_id
from .viz import robot_meshes

BUILD_PHASE = 0.9  # the bridge build: late enough that the bricks stand in their places
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
                if robot != hero_replay.ROBOT_ID:  # the two-arm bridge scene draws from its own mesh set
                    parse_id(robot)
            except ValueError:  # an older tile labelled by its rig ("so101-duo"): its meshes are the default arm's
                robot = DEFAULT
            robots.add(robot)
            job = {
                "id": tid,
                "robot": robot,
                "scene": r["scene"],
                "frame": r["frames"][int(len(r["frames"]) * (BUILD_PHASE if robot == hero_replay.ROBOT_ID else PHASE))],
                "out": str(TILES_DIR / tid / "thumb.png"),
            }
            Path(tmp, f"{tid}.json").write_text(json.dumps(job))
        for robot in robots:
            meshes = hero_replay.meshes() if robot == hero_replay.ROBOT_ID else robot_meshes(robot)
            Path(tmp, f"meshes-{robot}.json").write_text(json.dumps(meshes))
        pw = os.environ.get("PW_DIR", str(Path.home() / "helloworld/qloo-inspo/node_modules"))
        subprocess.run(["node", str(ROOT / "scripts/tile_thumbs.mjs"), tmp], check=True, env={**os.environ, "PW_DIR": pw})
    return ids


PITCH = 0.4  # arms stand this far apart along y, as the Sim's multi-arm view stands them


def busiest(episodes: list[dict], n_arms: int) -> tuple[float, int]:
    """The second of the recorded schedule with the most arms mid-motion (a second in from either end of an episode)."""
    end = max(e["start"] + e["duration"] for e in episodes)
    best, t = (0.0, 0), 0.0
    while t < end:
        n = sum(
            any(e["station"] == k and e["start"] + 1 <= t < e["start"] + e["duration"] - 1 for e in episodes)
            for k in range(n_arms)
        )
        if n > best[1]:
            best = (t, n)
        t += 0.5
    return best


def shoot_multi(card_ids: list[str] | None = None) -> list[str]:
    """Shoot thumb.png of the multi-arm Library cards: every arm of the saved line run, at its busiest second."""
    from . import factory, multi

    colors = factory.block_colors()
    lines = factory.load_all()
    wanted = [c for c in multi.cards() if not card_ids or multi.card_id(c["line"]) in card_ids]
    done = []
    with tempfile.TemporaryDirectory() as tmp:
        for rec in wanted:
            ln = lines[rec["line"]]
            pb = ln.result["playback"]
            sec, _n = busiest(pb["episodes"], len(ln.stations))
            cells = []
            for k, sc in enumerate(factory.cell_scenes(ln)):
                eps = [e for e in pb["episodes"] if e["station"] == k]
                now = next((e for e in eps if e["start"] <= sec < e["start"] + e["duration"]), None)
                past = [e for e in eps if e["start"] + e["duration"] <= sec]
                if now:
                    frame = now["frames"][min(len(now["frames"]) - 1, int((sec - now["start"]) * pb["fps"]))]
                elif past:
                    frame = max(past, key=lambda e: e["start"])["frames"][-1]
                else:
                    frame = sc["home"]
                shown = now or (max(past, key=lambda e: e["start"]) if past else None)
                if shown and sc["blocks"]:
                    sc["blocks"][0]["rgb"] = colors[shown["color"]]
                cells.append({"scene": sc, "frame": frame, "robot": sc["robot_id"]})
            cid = multi.card_id(rec["line"])
            (TILES_DIR / cid).mkdir(parents=True, exist_ok=True)
            Path(tmp, f"{cid}.json").write_text(
                json.dumps({"id": cid, "pitch": PITCH, "cells": cells, "out": str(TILES_DIR / cid / "thumb.png")})
            )
            done.append(cid)
        for robot in {c["robot"] for f in Path(tmp).glob("*.json") for c in json.loads(f.read_text())["cells"]}:
            Path(tmp, f"meshes-{robot}.json").write_text(json.dumps(robot_meshes(robot)))
        pw = os.environ.get("PW_DIR", str(Path.home() / "helloworld/qloo-inspo/node_modules"))
        subprocess.run(["node", str(ROOT / "scripts/tile_thumbs.mjs"), tmp], check=True, env={**os.environ, "PW_DIR": pw})
    return done
