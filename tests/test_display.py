import mujoco

from vla_forge.sim.display import CONCRETE, SERVO, SHELL, restyle, robot_role
from vla_forge.sim.scene import Layout, build_spec


def test_roles_follow_names_then_colour():
    assert robot_role((0.9, 0.9, 0.9, 1), "base_shell") == "shell"
    assert robot_role((0.9, 0.9, 0.9, 1), "sts3215_03a_v1") == "servo"
    assert robot_role((0.1, 0.1, 0.1, 1), "plain") == "servo"
    assert robot_role((0.45, 0.45, 0.45, 1), "plain") == "steel"
    assert robot_role((0.1, 0.1, 0.1, 1), "moving_jaw_so101_v1") == "servo"
    assert robot_role((0.9, 0.9, 0.9, 1), "moving_jaw_so101_v1") == "shell"


def test_restyle_changes_colours_not_physics():
    m = build_spec(
        Layout.from_json({"blocks": {"b0": ["red", 0.2, 0.0, 0.0]}})
    ).compile()
    before = (
        m.geom_pos.copy(),
        m.geom_size.copy(),
        m.geom_contype.copy(),
        m.body_mass.copy(),
    )
    restyle(m)
    after = (m.geom_pos, m.geom_size, m.geom_contype, m.body_mass)
    assert all((a == b).all() for a, b in zip(before, after, strict=True))
    table = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "table")
    assert tuple(m.geom_rgba[table]) == CONCRETE
    rgba = {tuple(round(float(x), 3) for x in c) for c in m.geom_rgba}
    assert SHELL in rgba and SERVO in rgba
