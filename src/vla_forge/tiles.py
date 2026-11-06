"""Library tiles: what a finished skill leaves on disk besides its checkpoint.

skills/tiles/<skill id>/thumb.png        one real frame of one of its own rollouts, rendered by MuJoCo
skills/tiles/<skill id>/replay.json.gz   recorded rollouts (scene + body poses at 30 Hz) for the tile's player

The thumbnail is the final state of a recorded rollout: the blocks' recorded poses plus the arm's recorded joint
positions, put back into the same compiled scene and rendered from a fixed three-quarter camera.
"""

from __future__ import annotations

import gzip
import json
from pathlib import Path

import mujoco
import numpy as np
from PIL import Image

from .registry import ROOT, TILES_DIR, Skill
from .sim.robots import calibrate, fix_qpos0
from .sim.scene import JOINTS, Layout, build_spec
from .viz import export_scene

THUMB_W, THUMB_H = 480, 360
REPLAY_KEEP = 3  # rollouts kept per tile: successes first, then one failure if any


def tile_dir(skill_id: str) -> Path:
    return TILES_DIR / skill_id


def render_thumb(
    layout_json: dict, frame: list, q: list, out: Path, robot: str = "so101"
) -> Path:
    """Render one recorded state of a rollout to a PNG, on the robot that ran it."""
    lay = Layout.from_json(layout_json)
    m = build_spec(lay, robot=robot).compile()
    fix_qpos0(m)
    d = mujoco.MjData(m)
    rb = calibrate(robot)
    joints = (
        JOINTS if rb.default else rb.arm_joints
    )  # a custom robot's q ends with its grip length
    for j, v in zip(joints, q, strict=False):
        d.qpos[m.joint(j).qposadr[0]] = v
    f = np.asarray(frame, float).reshape(-1, 7)
    for name in lay.blocks:
        bid = m.body(name).id
        adr = m.joint(f"{name}_free").qposadr[0]
        d.qpos[adr : adr + 3] = f[bid - 1, :3]
        d.qpos[adr + 3 : adr + 7] = f[bid - 1, 3:]
    for name, (_c, x, y) in lay.fixtures().items():
        d.mocap_pos[m.body(name).mocapid[0]] = [x, y, 0.0]
    mujoco.mj_forward(m, d)
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.lookat[:] = [0.17, 0.0, 0.05]
    cam.distance = 0.68
    cam.azimuth = 148.0
    cam.elevation = -34.0
    with mujoco.Renderer(m, THUMB_H, THUMB_W) as r:
        r.update_scene(d, camera=cam)
        img = r.render()
    out.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(img).save(out, optimize=True)
    return out


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
    shot = keep[0]
    thumb = render_thumb(
        shot.layout, shot.frames[-1], shot.final_q, d / "thumb.png", skill.robot
    )
    replay = d / "replay.json.gz"
    with gzip.open(replay, "wt") as fh:
        json.dump(
            [payload(r, instructions, skill.robot) for r in keep],
            fh,
            separators=(",", ":"),
        )
    return {
        "thumb_png": str(thumb.relative_to(ROOT)),
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
        "replay": bool(skill.replay),
        "replay_source": skill.replay_source,
        "tests": tests,
        "agent_address": skill.agent_address,
    }
