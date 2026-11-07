"""The engine routes a task to the arm(s) whose measured checks pass.  Names are never matched."""

from vla_forge.sim import routing, toolbox

XY = toolbox.box_centre()
TOOLS = ["ratchet_wrench", "pliers", "screwdriver"]
ALL = list(toolbox.ALL_TOOLS)


def test_toolbox_pack_routes_to_so101():
    r = routing.route(TOOLS, toolbox.TOOLBOX, XY)
    assert r["chosen"] == ["so101"]
    assert r["assignments"][0]["objects"] == TOOLS
    assert r["excluded"] == [] and r["pinned"] is None
    top = r["ranked"][0]
    assert top["robot"] == "so101" and top["score"] == 1.0 and top["fails"] == []


def test_every_arm_is_ranked_with_numbers():
    r = routing.route(ALL, toolbox.TOOLBOX, XY)
    assert {x["robot"] for x in r["ranked"]} == set(routing.available())
    assert all(0 <= x["score"] <= 1 and x["cost_nm"] > 0 for x in r["ranked"])


def test_pipe_wrench_is_routed_by_measurement_not_name():
    r = routing.route(ALL, toolbox.TOOLBOX, XY)
    so101 = next(x for x in r["ranked"] if x["robot"] == "so101")
    assert "pipe_wrench" in [f["object"] for f in so101["fails"]]
    if r["excluded"]:
        assert r["excluded"][0]["name"] == "pipe_wrench" and r["excluded"][0]["reasons"]
    else:
        assert any("pipe_wrench" in a["objects"] and a["robot"] != "so101" for a in r["assignments"])
    placed = [o for a in r["assignments"] for o in a["objects"]] + [e["name"] for e in r["excluded"]]
    assert sorted(placed) == sorted(ALL)


def test_pinned_infeasible_robot_gets_a_clear_message():
    r = routing.route(ALL, toolbox.TOOLBOX, XY, pinned="so101")
    assert r["pinned"]["ok"] is False
    assert "cannot do pipe_wrench" in r["pinned"]["message"]
    ok = routing.route(TOOLS, toolbox.TOOLBOX, XY, pinned="so101")
    assert ok["pinned"]["ok"] is True


def test_unpinned_run_takes_the_chosen_arm_and_pinned_run_keeps_its_own():
    from types import SimpleNamespace

    from vla_forge.pipeline import Forge
    from vla_forge.sim.tasks import TaskSpec

    free = SimpleNamespace(robot="so101", pinned=False)
    spec = TaskSpec("pack", "")
    Forge._judge(None, free, spec)
    assert spec.routing["chosen"] == ["so101"] and free.robot == "so101"
    assert spec.routing["pinned"] is None

    held = SimpleNamespace(robot="so101", pinned=True)
    spec = TaskSpec("pack", "")
    Forge._judge(None, held, spec)
    assert spec.routing["pinned"]["ok"] is True
    assert TaskSpec.from_json(spec.to_json()).routing == spec.routing
