"""Pretrain the multi-task base policy that every prompt fine-tunes from (the root of the skill library).

Generates expert demonstrations for random (skill, colour) tasks with domain randomization and DART noise,
encodes them with the frozen DINOv2 / SigLIP2 towers, stores the features in shards under models/base_data/,
trains the action head for a wall-clock budget and saves models/base.pt. Per-skill success is measured on
held-out random layouts and written to models/base.json.

    VLA_FORGE_TIMED=1 HF_HUB_OFFLINE=1 scripts/heavy.sh uv run python -u scripts/pretrain_base.py --episodes 1600
"""

from __future__ import annotations

import argparse
import json
import shutil
import time
from concurrent.futures import as_completed
from pathlib import Path

import numpy as np

from vla_forge.datagen import demo_job, pool
from vla_forge.learn import Demos, load_shards, save_shard, train
from vla_forge.policy import VLA
from vla_forge.rollout import evaluate, summarize
from vla_forge.sim.scene import BLOCK_COLORS
from vla_forge.sim.tasks import SKILLS, TaskSpec, valid_targets

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "models" / "base_data"
SHARD = 100  # episodes per feature shard


# stacking is the hardest motion (a 3.6 cm landing zone on a block that can be knocked over), so it gets the
# biggest share of the demonstrations; reach is the easiest
WEIGHTS = {"reach": 0.12, "push": 0.18, "lift": 0.14, "pick_place": 0.24, "stack": 0.32}


def random_spec(rng: np.random.Generator) -> TaskSpec:
    skill = SKILLS[int(rng.choice(len(SKILLS), p=[WEIGHTS[s] for s in SKILLS]))]
    obj = list(BLOCK_COLORS)[int(rng.integers(len(BLOCK_COLORS)))]
    targets = [t for t in valid_targets(skill) if t != obj or skill != "stack"]
    target = targets[int(rng.integers(len(targets)))] if targets else None
    return TaskSpec(skill, obj, target, 2)


def collect(n: int, seed0: int, noise: float) -> None:
    """Generate n episodes, encode them as they arrive, write feature shards; raw images are deleted."""
    rng = np.random.default_rng(seed0)
    tmp = ROOT / "runs" / "base_raw"
    jobs = []
    for i in range(n):
        spec = random_spec(rng)
        jobs.append(
            {
                "spec": spec.to_json(),
                "seed": seed0 + i,
                "out": str(tmp / f"ep{seed0 + i:06d}.npz"),
                "noise": noise,
            }
        )
    p = pool()
    t0 = time.time()
    buf = Demos()
    shard = len(list(DATA.glob("shard_*.pt")))
    ok = 0
    futs = {p.submit(demo_job, j): j for j in jobs}
    for done, f in enumerate(as_completed(futs), 1):
        r = f.result()
        if r["ok"]:
            ok += 1
            spec = TaskSpec.from_json(futs[f]["spec"])
            buf.add_episode(r["path"], spec.instruction)
            Path(r["path"]).unlink()
        if buf.episodes >= SHARD or (done == n and buf.episodes):
            save_shard(buf, DATA / f"shard_{shard:03d}.pt")
            shard += 1
            buf = Demos()
        if done % 50 == 0 or done == n:
            print(f"collect {done}/{n} ok {ok} {time.time() - t0:.0f}s", flush=True)
    shutil.rmtree(tmp, ignore_errors=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=1600)
    ap.add_argument("--seconds", type=float, default=1800)
    ap.add_argument("--noise", type=float, default=0.05)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--skip-collect", action="store_true")
    ap.add_argument("--eval", type=int, default=20)
    a = ap.parse_args()
    if not a.skip_collect:
        have = len(list(DATA.glob("shard_*.pt"))) * SHARD
        if a.episodes > have:
            collect(a.episodes - have, 100_000 + have * 3, a.noise)
    shards = sorted(DATA.glob("shard_*.pt"))
    t0 = time.time()
    data = load_shards(shards)
    print(
        f"loaded {data.episodes} episodes / {data.frames} frames in {time.time() - t0:.0f}s",
        flush=True,
    )
    model = VLA(norm=data.norm())
    print(f"trainable params {model.trainable_params():,}", flush=True)

    def log(e):
        if e["step"] % 2000 < 25:
            print(json.dumps(e), flush=True)

    st = train(model, data, a.seconds, lr=a.lr, on_log=log, log_every=1.0)
    print(f"train {st}", flush=True)
    out = ROOT / "models" / "base.pt"
    model.save(out, {"episodes": data.episodes, "frames": data.frames, "train": st})
    rng = np.random.default_rng(7)
    report = {
        "episodes": data.episodes,
        "frames": data.frames,
        "train": st,
        "skills": {},
    }
    for skill in SKILLS:
        spec = random_spec(rng)
        while spec.skill != skill:
            spec = random_spec(rng)
        n = a.eval * 2 if skill == "stack" else a.eval
        s = summarize(evaluate(model, spec, n, seed0=2_000_000, record=False))
        report["skills"][spec.instruction] = {
            "succeeded": s["succeeded"],
            "n": s["n"],
            "failures": s["failures"],
        }
        print(
            f"eval '{spec.instruction}': {s['succeeded']}/{s['n']} {s['failures']}",
            flush=True,
        )
    (ROOT / "models" / "base.json").write_text(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
