"""Robot builder: ids, assembly, checks and the router hook.  Run: uv run --with pytest pytest tests"""

import pytest

from vla_forge.sim import robots as R
from vla_forge.sim.scene import Layout, Scene


def test_ids_round_trip():
    assert R.make_id("so101") == "so101"
    assert R.parse_id("so101") == ("so101", "native", "native")
    assert R.make_id("vx300s", "native", "franka_hand") == "vx300s.native.franka_hand"
    with pytest.raises(ValueError):
        R.parse_id("so101.native.suction")


def test_stock_arm_passes_and_keeps_six_actions():
    rep = R.report("so101")
    assert rep["ok"] and rep["failure"] is None
    assert rep["act_dim"] == 6
    assert R.robot_feasibility("so101", "pick_place")[0]


def test_seven_joint_arm_compiles_and_reaches():
    rid = "vx300s.native.native"
    rep = R.report(rid)
    assert rep["ok"], rep["failure"]
    assert rep["act_dim"] == 7
    assert rep["reach"]["fraction"]["grasp"] >= 0.9
    sc = Scene(Layout(), rid)
    assert sc.robot.act_dim == 7
    sc.close()


def test_failures_explain_themselves():
    ok, reason = R.robot_feasibility("wx250s.native.native", "pick_place")
    assert not ok and "open only" in reason
    ok, reason = R.robot_feasibility("koch.native.native", {"skill": "lift"})
    assert not ok and "too shallow" in reason
    # pushing needs no grip, so a jaw too small for a block is fine for it
    assert R.robot_feasibility("wx250s.native.native", "push")[0]


def test_usable_keeps_the_skills_a_robot_can_do():
    ok, why, feas = R.usable("wx250s.native.native")
    assert ok and feas["push"]["ok"] and not feas["pick_place"]["ok"]
    assert "refuses" in why
    ok, why, _ = R.usable("koch.native.native")
    assert not ok and "workspace" in why
