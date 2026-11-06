"""Factory line: conveyor, totes, station cells, the planner's offline fallback and the zone schedule.
Run: uv run --with pytest pytest tests"""

import mujoco
import numpy as np

from vla_forge import factory as F
from vla_forge.sim import line as L
from vla_forge.sim.scene import Layout, Scene
from vla_forge.sim.tasks import check


def test_tote_is_robosuite_bin_scaled():
    boxes = L.bin_geoms()
    assert len(boxes) == 5  # floor + four walls of robosuite's bin1
    floor = min(boxes, key=lambda b: b["pos"][2])
    assert abs(floor["pos"][2] - floor["size"][2]) < 1e-6  # floor's underside on z = 0
    top = max(b["pos"][2] + b["size"][2] for b in boxes)
    assert 0.025 < top < 0.04  # a low tote rim the arm reaches over


def test_in_bin():
    b = np.array([0.15, 0.08, 0.0])
    assert L.in_bin(np.array([0.16, 0.09, 0.02]), b)
    assert not L.in_bin(np.array([0.25, 0.08, 0.02]), b)  # beside the tote
    assert not L.in_bin(np.array([0.15, 0.08, 0.2]), b)  # held above it


def test_station_layout_round_trips_and_sits_on_belt():
    lay = L.station_layout("red", "red", np.random.default_rng(0))
    assert lay.belt and "goal" in lay.bins
    _c, x, _y, _yaw = lay.blocks["target"]
    assert L.on_belt(x)
    back = Layout.from_json(lay.to_json())
    assert back.key() == lay.key() and back.bins == lay.bins


def test_belt_carries_a_part_to_the_photo_eye():
    lay = L.station_layout("blue", "blue", np.random.default_rng(1), at_window=False)
    sc = Scene(lay)
    sc.reset(lay)
    y0 = sc.body_pos("target")[1]
    frames = F._belt_phase(sc, L.WINDOW_Y + 0.011)
    p = sc.body_pos("target")
    assert frames and p[1] > y0 + 0.2  # carried along y
    assert abs(p[1] - L.WINDOW_Y) < 0.03  # and stopped near the photo-eye
    assert abs(p[0] - L.BELT_X) < L.BELT_HALF_W  # still on the belt
    assert abs(sc.data.qvel[L.belt_dof(sc.model)]) < 0.02  # belt stopped
    sc.close()


def test_expert_sorts_a_part_into_the_tote():
    from vla_forge.sim.expert import Expert

    lay = L.station_layout("red", "red", np.random.default_rng(2))
    st = F.Station(0, "Sort red", "red", "red")
    spec = st.spec()
    sc = Scene(lay)
    sc.reset(lay)
    Expert(sc, spec, rng=np.random.default_rng(0)).run()
    assert check(spec, sc, lay)
    sc.close()


def test_rule_planner_reads_stations_in_order():
    raw = F._rule_plan(
        "Red parts into the red bin, then blue parts into the blue tote; everything else at the end of the line."
    )
    assert [(s["part"], s["bin"]) for s in raw["stations"]] == [
        ("red", "red"),
        ("blue", "blue"),
        ("any", "gray"),
    ]
    assert raw["stations"][-1]["controller"] == "scripted"
    assert "green" in raw["parts"]  # something for the end of line to catch


def test_schedule_one_part_per_zone():
    line = F.Line(
        "t",
        "",
        "t",
        [F.Station(0, "a", "red", "red"), F.Station(1, "b", F.ANY, "gray")],
        ["red", "blue"],
    )
    eps = [
        {"station": 0, "part": 0, "kind": "pick", "frames": [0] * 300},  # 10 s at 30 Hz
        {"station": 0, "part": 1, "kind": "pass", "frames": [0] * 150},  # 5 s
        {"station": 1, "part": 1, "kind": "pick", "frames": [0] * 300},
    ]
    s = F._schedule(line, eps, 2)
    t = s["times"]
    assert t[(0, 0)] == {"start": 0.0, "duration": 10.0}
    assert t[(0, 1)]["start"] == 10.0  # waits for zone 1 to clear
    assert t[(1, 1)]["start"] == 15.0
    assert s["makespan"] == 25.0


def test_belt_does_not_touch_the_table():
    lay = L.station_layout("red", "red", np.random.default_rng(3))
    sc = Scene(lay)
    sc.reset(lay)
    m, d = sc.model, sc.data
    for _ in range(30):
        mujoco.mj_step(m, d)
    belt = m.body("belt").id
    pairs = [
        {int(m.geom_bodyid[c.geom1]), int(m.geom_bodyid[c.geom2])}
        for c in d.contact[: d.ncon]
    ]
    assert {0, belt} not in pairs
    sc.close()
