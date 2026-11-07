"""Per-arm table workspace: fitted from measured reach, and the one the samplers draw from.
Run: HF_HUB_OFFLINE=1 uv run --with pytest pytest tests/test_workspace.py -s"""

import hashlib
import json
import math

import numpy as np
import pytest

from vla_forge.compose import Step
from vla_forge.compose import sample_layout as compose_layout
from vla_forge.sim import robots as R
from vla_forge.sim.tasks import (
    DEFAULT_WS,
    TaskSpec,
    Workspace,
    required_workspace,
    sample_layout,
)

# sha256 prefixes of 40 seeded layouts per skill, taken from the engine before workspaces were per-arm
GOLDEN = {
    "reach": "ec8fd90c06fe333f",
    "push": "6a869f98b9fdf95d",
    "lift": "ec8fd90c06fe333f",
    "pick_place": "c25d012fd15c2868",
    "stack": "487bfedf8db1fd76",
    "kit": "6a3617bdc99be477",
    "insert": "d8c49467797c1394",
    "rotate": "48f596fb0dca6e93",
}
GOLDEN_COMPOSE = "8fcd7ecf5f0932e5"
SPECS = {
    "reach": TaskSpec("reach", "red"),
    "push": TaskSpec("push", "red", "blue", 1),
    "lift": TaskSpec("lift", "red"),
    "pick_place": TaskSpec("pick_place", "red", "white", 2),
    "stack": TaskSpec("stack", "red", "blue", 1),
    "kit": TaskSpec("kit", "red", "front-left", 1),
    "insert": TaskSpec("insert", "red", None, 1),
    "rotate": TaskSpec("rotate", "red", None, 1),
}


def _digest(skill, robot):
    h = hashlib.sha256()
    for seed in range(40):
        lay = sample_layout(SPECS[skill], np.random.default_rng(seed), robot=robot)
        h.update(json.dumps(lay.to_json(), sort_keys=True).encode())
    return h.hexdigest()[:16]


@pytest.mark.parametrize("robot", [None, "so101"])
def test_so101_layouts_are_unchanged(robot):
    assert {k: _digest(k, robot) for k in SPECS} == GOLDEN
    h = hashlib.sha256()
    steps = [Step("pick_place", "red", "white"), Step("push", "blue", "green")]
    for seed in range(40):
        lay = compose_layout(steps, np.random.default_rng(seed), robot=robot)
        h.update(json.dumps(lay.to_json(), sort_keys=True).encode())
    assert h.hexdigest()[:16] == GOLDEN_COMPOSE


def test_so101_workspace_is_the_stock_sector():
    ws = R.workspace("so101")
    assert (ws["r_lo"], ws["r_hi"], ws["ang"]) == (*R.WS_R, R.WS_ANG)
    assert (DEFAULT_WS.r_lo, DEFAULT_WS.r_hi, DEFAULT_WS.ang) == (*R.WS_R, R.WS_ANG)
    assert R.report("so101")["workspace"]["r_hi"] == R.WS_R[1]


def test_so100_workspace_excludes_the_near_base_misses():
    rid = "so100.native.native"
    ws = R.workspace(rid)
    assert ws["r_lo"] >= 0.15  # every old miss sat at 12.5-15.2 cm
    ok, detail = R.workspace_check(rid)
    print("so100:", ok, detail)
    # the preview dots are the fitted sector's, and they all land inside it
    for p in R.reachability(rid)["points"]:
        assert ws["r_lo"] - 1e-3 <= math.hypot(p["x"], p["y"]) <= ws["r_hi"] + 1e-3
    if ok:  # the tasks then sample inside that same sector
        for spec in SPECS.values():
            for seed in range(10):
                lay = sample_layout(spec, np.random.default_rng(seed), robot=rid)
                for _c, x, y, *_ in lay.blocks.values():
                    assert ws["r_lo"] <= math.hypot(x, y) <= ws["r_hi"] + 0.02


def test_sampler_follows_the_workspace_it_is_given(monkeypatch):
    ws = Workspace(0.17, 0.31, math.radians(30))
    monkeypatch.setattr("vla_forge.sim.robots.task_workspace", lambda rid: ws)
    for spec in SPECS.values():
        for seed in range(15):
            lay = sample_layout(spec, np.random.default_rng(seed), robot="x")
            for _c, x, y, *_ in lay.blocks.values():
                assert ws.r_lo - 0.01 <= math.hypot(x, y) <= ws.r_hi + 0.02
                assert abs(math.atan2(y, x)) <= ws.ang + 0.05


def test_required_workspace_comes_from_the_task_constraints():
    depth, ang = required_workspace(0.125, 0.285)
    assert depth == pytest.approx(0.16)  # 8 cm push lane + 5 cm standoff + 2 x 1.5 cm
    assert 0.2 < ang < 0.4


def test_every_arm_reports_a_workspace_and_prints_the_table():
    print(f"\n{'arm':10} {'r_lo':>5} {'r_hi':>5} {'ang':>4}  check")
    for a in R.ARMS:
        rid = R.make_id(a)
        rep = R.report(rid)
        w = rep["workspace"]
        c = next(c for c in rep["checks"] if c["name"] == "Reaches the table workspace")
        print(
            f"{rid:10} {w['r_lo'] * 100:5.0f} {w['r_hi'] * 100:5.0f} {w['ang_deg']:4.0f}  "
            f"{'pass' if c['ok'] else 'FAIL'}: {c['detail']}"
        )
        assert {"r_lo", "r_hi", "ang"} <= set(w)
