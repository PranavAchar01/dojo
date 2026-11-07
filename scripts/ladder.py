"""Run prompts through the live server one at a time and print what each one measured.

    uv run python scripts/ladder.py "push the green block onto the blue square" "put the blue block in the blue bowl"

Appends one JSON line per prompt to runs/ladder.jsonl (tier, wall clock, rounds, success, plan for tier 3).
"""

import json
import os
import sys
import time
import urllib.request
from pathlib import Path

API = f"http://127.0.0.1:{os.environ.get('VLA_FORGE_PORT', '8100')}/api"
OUT = Path(__file__).resolve().parents[1] / "runs" / "ladder.jsonl"


def call(path: str, body: dict | None = None) -> dict:
    req = urllib.request.Request(
        API + path,
        data=json.dumps(body).encode() if body is not None else None,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.loads(r.read())


def run(prompt: str) -> dict:
    while call("/state")["active"]:
        time.sleep(1)
    rid = call("/prompt", {"prompt": prompt})["run"]
    while True:
        s = call(f"/runs/{rid}")
        if s["status"] in ("published", "error", "rejected"):
            return s
        time.sleep(1)


def main() -> None:
    for p in sys.argv[1:]:
        s = run(p)
        rounds = [r["eval"] for r in s["rounds"] if r.get("eval")]
        best = s["rounds"][s["best_round"]]["eval"] if s["best_round"] >= 0 else None
        row = {
            "prompt": p,
            "run": s["id"],
            "status": s["status"],
            "tier": s["tier"],
            "seconds": s["elapsed"],
            "instruction": s["spec"]["instruction"] if s["spec"] else None,
            "init": (s.get("init_from") or {}).get("reason"),
            "rounds": [f"{r['succeeded']}/{r['n']}" for r in rounds],
            "success": f"{best['succeeded']}/{best['n']}" if best else None,
            "per_step": best.get("per_step") if best else None,
            "plan": [
                {k: p_[k] for k in ("instruction", "source", "skill_id", "seconds")}
                for p_ in s["plan"]
            ],
            "error": s["error"],
        }
        with OUT.open("a") as f:
            f.write(json.dumps(row) + "\n")
        print(json.dumps(row), flush=True)


if __name__ == "__main__":
    main()
