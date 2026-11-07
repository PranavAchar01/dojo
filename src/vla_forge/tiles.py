"""Library tiles: what a finished skill leaves on disk besides its checkpoint.

skills/tiles/<skill id>/thumb.png        one real frame of one of its own rollouts, drawn on the shared stage (tile_stage.py)
skills/tiles/<skill id>/replay.json.gz   recorded rollouts (scene + body poses at 30 Hz) for the tile's player

The thumbnail is the final state of a recorded rollout: the blocks' recorded poses plus the arm's recorded joint
positions, put back into the same compiled scene and rendered from a fixed three-quarter camera.
"""

from __future__ import annotations

import gzip
import json
from pathlib import Path

from .registry import ROOT, TILES_DIR, Skill
from .sim.scene import Layout
from .viz import export_scene

THUMB_W, THUMB_H = 480, 360
REPLAY_KEEP = 3  # rollouts kept per tile: successes first, then one failure if any


def tile_dir(skill_id: str) -> Path:
    return TILES_DIR / skill_id


def pick(rolls: list) -> list:
    """Which rollouts a tile keeps: up to REPLAY_KEEP - 1 successes, then the first failure."""
    ok = [r for r in rolls if r.success][: REPLAY_KEEP - 1]
    bad = [r for r in rolls if not r.success][:1]
    return (ok + bad)[:REPLAY_KEEP] or rolls[:1]


def payload(r, instructions: list[str] | None = None, robot: str = "so101") -> dict:
    p = {
        "index": r.index,
        "success": r.success,
        "failure": r.failure,
        "fps": 30,
        "scene": export_scene(Layout.from_json(r.layout), robot),
        "frames": r.frames,
    }
    if instructions:  # a chained rollout
        p.update(
            phase_frames=r.phase_frames, instructions=instructions, steps_ok=r.steps_ok
        )
    return p


def save_assets(
    skill: Skill, rolls: list, source: str, instructions: list[str] | None = None
) -> dict:
    """Write the tile's thumbnail and replay; returns the registry fields that point at them."""
    keep = pick(rolls)
    d = tile_dir(skill.id)
    d.mkdir(parents=True, exist_ok=True)
    replay = d / "replay.json.gz"
    with gzip.open(replay, "wt") as fh:
        json.dump(
            [payload(r, instructions, skill.robot) for r in keep],
            fh,
            separators=(",", ":"),
        )
    from . import tile_stage

    tile_stage.shoot([skill.id])  # the thumbnail is a frame of that replay, drawn on the shared stage
    return {
        "thumb_png": str(d.relative_to(ROOT) / "thumb.png"),
        "replay": str(replay.relative_to(ROOT)),
        "replay_source": source,
    }


def load_replay(skill: Skill) -> list[dict]:
    with gzip.open(ROOT / skill.replay, "rt") as fh:
        return json.load(fh)


def tile(skill: Skill) -> dict:
    """The tile as the library page sees it (no pose arrays)."""
    tests = skill.tests[-5:]
    return {
        "id": skill.id,
        "name": skill.name or skill.instruction,
        "instruction": skill.instruction,
        "prompt": skill.prompt or skill.instruction,
        "tier": skill.tier,
        "robot": skill.robot,
        "origin": "chain" if skill.composite else skill.origin,
        "parent": skill.parent,
        "success_rate": skill.success_rate,
        "n_eval": skill.n_eval,
        "seconds": skill.seconds,
        "train_seconds": skill.train_seconds,
        "created": skill.created,
        "steps": skill.steps,
        "thumb": bool(skill.thumb_png),
        "thumb_v": int((ROOT / skill.thumb_png).stat().st_mtime) if skill.thumb_png and (ROOT / skill.thumb_png).exists() else 0,
        "replay": bool(skill.replay),
        "replay_source": skill.replay_source,
        "detail": skill.record.get("detail", ""),
        "tests": tests,
        "agent_address": skill.agent_address,
    }
