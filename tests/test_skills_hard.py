"""The harder skills: kit (named tray slot), insert (square socket), rotate (diamond socket).
Run: uv run pytest tests/test_skills_hard.py"""

import math

import numpy as np
import pytest

from vla_forge.llm import rule_motion, rule_parse
from vla_forge.sim import tasks as T
from vla_forge.sim.expert import Expert
from vla_forge.sim.props import KIT_SLOTS, prop_parts, slot_offset
from vla_forge.sim.scene import Layout, Look, Scene
from vla_forge.viz import export_scene

HARD = ("kit", "insert", "rotate")


def _spec(skill):
    return T.TaskSpec(skill, "red", "front-left" if skill == "kit" else None)


def test_specs_round_trip_and_tiers():
    for sk in HARD:
        s = _spec(sk)
        assert T.TaskSpec.from_json(s.to_json()) == s
        assert sk in T.SKILLS and sk in T.TIER
    assert T.TIER["kit"] == 2 and T.TIER["insert"] == T.TIER["rotate"] == 3
    assert T.valid_targets("kit") == list(KIT_SLOTS)
    assert T.valid_targets("insert") == []


def test_instructions_and_rule_parse():
    assert (
        _spec("kit").instruction
        == "put the red block in the front-left slot of the tray"
    )
    assert _spec("insert").instruction == "insert the red block into the socket"
    for sk in HARD:
        got = rule_parse(_spec(sk).instruction)
        assert got.skill == sk and got.obj == "red"
    got = rule_parse("put the blue block in the back right slot of the tray")
    assert (got.skill, got.obj, got.target) == ("kit", "blue", "back-right")
    assert (
        rule_motion("rotate the green block and seat it in the diamond socket")[0]
        == "rotate"
    )


def test_layout_sampler_and_json():
    for sk in HARD:
        spec = _spec(sk)
        for i in range(10):
            lay = T.sample_layout(spec, np.random.default_rng(i))
            assert lay.props and "target" in lay.blocks
            back = Layout.from_json(lay.to_json())
            assert back.props == lay.props and back.key() == lay.key()
            g = lay.fixtures()["goal"]
            assert 0.15 < math.hypot(g[1], g[2]) < 0.30
    lay = T.sample_layout(_spec("kit"), np.random.default_rng(3))
    _kind, _c, tx, ty, yaw = lay.props["tray"]
    ox, oy = slot_offset("front-left")
    gx, gy = lay.props["goal"][2:4]
    assert abs(gx - (tx + ox * math.cos(yaw) - oy * math.sin(yaw))) < 1e-9
    assert abs(gy - (ty + ox * math.sin(yaw) + oy * math.cos(yaw))) < 1e-9


def test_old_layouts_unchanged():
    lay = Layout(blocks={"target": ("red", 0.2, 0.0, 0.0)})
    assert lay.key() == (
        (("target", "red"),),
        (),
        (),
        (),
        False,
    )
    assert "props" not in lay.to_json()
    assert Layout.from_json({"blocks": {}, "bowls": {}, "pads": {}}).props == {}


def _scene_with(block_xy, block_z, skill="insert", socket=(0.22, 0.0, 0.0)):
    spec = _spec(skill)
    lay = Layout(props={"goal": ("socket", "gray", *socket)})
    lay.blocks["target"] = ("red", 0.2, 0.1, 0.0)
    sc = Scene(lay)
    sc.reset(lay)
    adr = sc.model.joint("target_free").qposadr[0]
    sc.data.qpos[adr : adr + 3] = [*block_xy, block_z]
    yaw = block_xy and 0.0
    sc.data.qpos[adr + 3 : adr + 7] = [math.cos(yaw / 2), 0, 0, math.sin(yaw / 2)]
    sc.data.qvel[:] = 0
    import mujoco

    mujoco.mj_forward(sc.model, sc.data)
    return spec, sc


def test_seated_predicate():
    spec, sc = _scene_with((0.22, 0.001), 0.015)
    assert T.check(spec, sc)
    spec, sc = _scene_with((0.22, 0.001), 0.035)  # resting on the plate rim
    assert not T.check(spec, sc)
    spec, sc = _scene_with((0.25, 0.0), 0.015)  # on the table beside the socket
    assert not T.check(spec, sc)


def test_in_slot_predicate():
    spec = _spec("kit")
    lay = Layout(
        props={
            "tray": ("tray", "gray", 0.22, 0.0, 0.0),
            "goal": ("marker", "gray", 0.22 + 0.0305, -0.0305, 0.0),
        }
    )
    lay.blocks["target"] = ("red", 0.2, 0.1, 0.0)
    sc = Scene(lay)
    sc.reset(lay)
    adr = sc.model.joint("target_free").qposadr[0]
    import mujoco

    def put(x, y, z):
        sc.data.qpos[adr : adr + 3] = [x, y, z]
        sc.data.qvel[:] = 0
        mujoco.mj_forward(sc.model, sc.data)
        return T.check(spec, sc)

    assert put(0.22 + 0.0305, -0.0305, 0.019)
    assert not put(0.22 - 0.0305, -0.0305, 0.019)  # the back-left slot
    assert not put(0.22 + 0.0305, -0.0305, 0.034)  # on top of a divider


def test_props_export_interface():
    lay = T.sample_layout(_spec("kit"), np.random.default_rng(0))
    ex = export_scene(lay)
    assert [p["name"] for p in ex["props"]] == [
        "tray"
    ]  # the goal marker has no geometry
    p = ex["props"][0]
    assert p["kind"] == "tray" and len(p["rgb"]) == 3 and isinstance(p["body"], int)
    assert p["body"] == ex["bodies"].index("tray")
    for part in p["parts"]:
        assert part["type"] == "box" and len(part["size"]) == 3
        assert len(part["pos"]) == 3 and len(part["quat"]) == 4
    assert export_scene(Layout())["props"] == []
    assert len(prop_parts("socket")) == 12


@pytest.mark.parametrize("skill", HARD)
def test_expert_succeeds(skill):
    spec = _spec(skill)
    wins = 0
    for seed in (0, 1):
        rng = np.random.default_rng(seed)
        lay = T.sample_layout(spec, rng)
        sc = Scene(lay)
        sc.reset(lay, Look.sample(rng))
        Expert(sc, spec).run()
        wins += T.check(spec, sc, lay)
    assert wins == 2
