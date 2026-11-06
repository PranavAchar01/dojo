"""The skill registry: every trained skill is a checkpoint plus a record, and later prompts start from the nearest.

The root entry is the multi-task base policy (scripts/pretrain_base.py). Each prompt fine-tunes from the skill
whose instruction is closest in SigLIP2 text space (plus a bonus for the same motor skill), so the library
grows by recursion: a new skill is always a child of an existing one.
"""

from __future__ import annotations

import json
import threading
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import torch

from .encoders import encode_text
from .sim.tasks import TaskSpec

ROOT = Path(__file__).resolve().parents[2]
SKILLS_DIR = ROOT / "skills"
BASE_PATH = ROOT / "models" / "base.pt"
SAME_SKILL_BONUS = 0.15


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
            for d in json.loads(path.read_text()):
                self.skills[d["id"]] = Skill(**d)
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

    def ordered(self) -> list[Skill]:
        return sorted(self.skills.values(), key=lambda s: -s.created)

    def exact(self, instruction: str) -> Skill | None:
        """The best published single skill for exactly this instruction (what a composed task reuses)."""
        hits = [
            s
            for s in self.skills.values()
            if not s.composite and s.instruction == instruction
        ]
        return max(hits, key=lambda s: (s.success_rate, s.created)) if hits else None

    def nearest(self, spec: TaskSpec) -> tuple[str | None, Path, float, str]:
        """(skill id or None for the base, checkpoint, similarity, human-readable reason)."""
        singles = [s for s in self.skills.values() if not s.composite]
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
