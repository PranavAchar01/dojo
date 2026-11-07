"""The skill registry: every trained skill is a checkpoint plus a record, and later prompts start from the nearest.

The root entry is the multi-task base policy (scripts/pretrain_base.py). Each prompt fine-tunes from the skill
whose instruction is closest in SigLIP2 text space (plus a bonus for the same motor skill), so the library
grows by recursion: a new skill is always a child of an existing one.
"""

from __future__ import annotations

import json
import os
import threading
import time
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path

import torch

from .encoders import encode_text
from .sim.tasks import TaskSpec

ROOT = Path(__file__).resolve().parents[2]
SKILLS_DIR = ROOT / "skills"
HIDDEN = SKILLS_DIR / "hidden.json"
BASE_PATH = ROOT / "models" / "base.pt"
TILES_DIR = (
    SKILLS_DIR / "tiles"
)  # per skill: thumb.png (a frame of its own rollout) + replay.json.gz
SAME_SKILL_BONUS = 0.15
ROBOT = os.environ.get(
    "VLA_FORGE_ROBOT", "so101"
)  # robot config id; the customizer adds others


@dataclass
class Skill:
    id: str
    instruction: str
    spec: str  # TaskSpec json
    model: str  # checkpoint path, relative to ROOT
    parent: str | None
    success_rate: float
    n_eval: int
    created: float = field(default_factory=time.time)
    seconds: float = 0.0  # wall clock of the run that produced it
    agent_address: str = ""
    thumb: list = field(default_factory=list)  # one pose frame for the gallery card
    tier: int = 1
    steps: list = field(
        default_factory=list
    )  # composed skills: [{instruction, skill_id}, ...]; no checkpoint
    prompt: str = ""
    # library tile
    name: str = ""  # what the owner calls it; defaults to the instruction
    robot: str = "so101"  # the robot id it was trained on (sim/robots.py); a policy only runs on that robot
    origin: str = (
        "prompt"  # prompt | remix | scratch | step (trained for a chain) | chain
    )
    train_seconds: float = 0.0  # time spent in the training loop itself (all rounds)
    thumb_png: str = ""  # relative to ROOT; a real frame of one of its own rollouts
    replay: str = ""  # relative to ROOT; recorded rollout(s) for the tile's player
    replay_source: str = ""  # "held-out eval" or "test"
    tests: list = field(
        default_factory=list
    )  # [{t, n, succeeded}], every Test run on fresh scenes

    @property
    def composite(self) -> bool:
        return bool(self.steps)

    def task(self) -> TaskSpec:
        return TaskSpec.from_json(self.spec)


class Registry:
    def __init__(self, path: Path = SKILLS_DIR / "registry.json"):
        self.path = path
        self.lock = threading.Lock()
        self.skills: dict[str, Skill] = {}
        if path.exists():
            known = {f.name for f in fields(Skill)}
            for d in json.loads(path.read_text()):
                self.skills[d["id"]] = Skill(
                    **{k: v for k, v in d.items() if k in known}
                )
        self._emb: dict[str, torch.Tensor] = {}

    def _save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps([asdict(s) for s in self.skills.values()], indent=1))
        tmp.replace(self.path)

    def add(self, skill: Skill) -> None:
        with self.lock:
            self.skills[skill.id] = skill
            self._save()

    def set_address(self, skill_id: str, address: str) -> None:
        with self.lock:
            self.skills[skill_id].agent_address = address
            self._save()

    def update(self, skill_id: str, **changes) -> Skill:
        with self.lock:
            sk = self.skills[skill_id]
            for k, v in changes.items():
                setattr(sk, k, v)
            self._save()
            return sk

    def add_test(self, skill_id: str, result: dict) -> Skill:
        with self.lock:
            sk = self.skills[skill_id]
            sk.tests.append(result)
            self._save()
            return sk

    def ordered(self) -> list[Skill]:
        """Shown skills, newest first. skills/hidden.json lists older renders and repeats kept on disk but not shown."""
        hide = (
            HIDDEN.exists() and set(json.loads(HIDDEN.read_text())["hidden"]) or set()
        )
        return sorted(
            (s for s in self.skills.values() if s.id not in hide),
            key=lambda s: -s.created,
        )

    def exact(self, instruction: str, robot: str = "so101") -> Skill | None:
        """The best published single skill for exactly this instruction on this robot (what a composed task reuses)."""
        hits = [
            s
            for s in self.skills.values()
            if not s.composite and s.instruction == instruction and s.robot == robot
        ]
        return max(hits, key=lambda s: (s.success_rate, s.created)) if hits else None

    def nearest(
        self, spec: TaskSpec, robot: str = "so101"
    ) -> tuple[str | None, Path, float, str]:
        """(skill id or None for the base, checkpoint, similarity, human-readable reason). Only skills trained on
        the same robot count: another robot's policy speaks a different joint space."""
        singles = [
            s for s in self.skills.values() if not s.composite and s.robot == robot
        ]
        if not singles:
            return (
                None,
                BASE_PATH,
                0.0,
                "no skills yet: starting from the multi-task base policy",
            )
        names = [s.instruction for s in singles]
        missing = [n for n in [*names, spec.instruction] if n not in self._emb]
        if missing:
            for n, e in zip(missing, encode_text(missing), strict=True):
                self._emb[n] = e.float().cpu()
        q = self._emb[spec.instruction]
        best, best_sim = None, -1.0
        for s in singles:
            sim = float(q @ self._emb[s.instruction]) + (
                SAME_SKILL_BONUS if s.task().skill == spec.skill else 0.0
            )
            if sim > best_sim:
                best, best_sim = s, sim
        assert best is not None
        if best.task().skill != spec.skill:
            return (
                None,
                BASE_PATH,
                best_sim,
                f"closest skill '{best.instruction}' is a different motion: starting from the base policy",
            )
        return (
            best.id,
            ROOT / best.model,
            best_sim,
            f"fine-tuning from '{best.instruction}' (similarity {best_sim:.2f})",
        )
