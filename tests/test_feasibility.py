"""The engine's own feasibility reasoning: SO-101 + Metal Toolbox + four library tools."""

import json

from vla_forge.sim import feasibility
from vla_forge.sim.tasks import TaskSpec

TOOLS = ["pipe_wrench", "ratchet_wrench", "pliers", "screwdriver"]


def _result():
    return feasibility.feasibility("so101", TOOLS, "metal_toolbox", container_xy=(0.25, -0.15))


def test_pipe_wrench_excluded_for_length_grasp_and_mass():
    r = _result()
    assert [e["name"] for e in r["excluded"]] == ["pipe_wrench"]
    text = " ".join(r["excluded"][0]["reasons"])
    assert "long and the container's cavity" in text
    assert "jaws close" in text
    assert "weighs" in text
    assert len(r["excluded"][0]["reasons"]) == 3
    assert r["included"] == ["ratchet_wrench", "pliers", "screwdriver"]


def test_checks_carry_the_numbers():
    c = _result()["checks"]
    wrench = c["objects"]["pipe_wrench"]
    assert not wrench["fits_container"] and not wrench["graspable"] and not wrench["liftable"]
    assert wrench["reachable"]
    for t in ("ratchet_wrench", "pliers", "screwdriver"):
        assert all(c["objects"][t][k] for k in ("fits_container", "graspable", "liftable", "reachable"))
    assert c["packing"]["unplaced"] == []
    json.dumps(c)


def test_packing_runs_out_of_room():
    # a small set packs together into the cavity
    r = feasibility.feasibility("so101", ["pliers"] * 1 + ["screwdriver"], "metal_toolbox", (0.25, -0.15))
    assert r["included"] == ["pliers", "screwdriver"]


def test_spec_stores_result():
    spec = TaskSpec("pack", "")
    spec.feasibility = feasibility.for_spec(spec, "so101")
    assert TaskSpec.from_json(spec.to_json()).feasibility["included"]
    assert feasibility.for_spec(TaskSpec("lift", "red"), "so101") == {}
