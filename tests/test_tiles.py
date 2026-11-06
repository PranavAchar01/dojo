"""Library tiles persist: new fields round-trip, old registries still load, and the thumbnail is a real render."""

from __future__ import annotations

import json

from PIL import Image

from vla_forge import tiles
from vla_forge.registry import Registry, Skill
from vla_forge.sim.scene import Layout, Scene
from vla_forge.sim.tasks import TaskSpec, sample_layout


def test_new_fields_survive_a_reload(tmp_path):
    path = tmp_path / "registry.json"
    reg = Registry(path)
    spec = TaskSpec("reach", "red")
    reg.add(
        Skill(
            "a",
            spec.instruction,
            spec.to_json(),
            "skills/a.pt",
            None,
            0.8,
            20,
            robot="so101",
            origin="scratch",
        )
    )
    reg.update(
        "a", name="poke red", train_seconds=61.5, thumb_png="skills/tiles/a/thumb.png"
    )
    reg.add_test("a", {"n": 5, "succeeded": 4})
    again = Registry(path).skills["a"]
    assert (again.name, again.origin, again.train_seconds) == (
        "poke red",
        "scratch",
        61.5,
    )
    assert again.tests == [{"n": 5, "succeeded": 4}]
    t = tiles.tile(again)
    assert t["name"] == "poke red" and t["thumb"] and not t["replay"]


def test_registry_from_before_the_library_loads(tmp_path):
    path = tmp_path / "registry.json"
    spec = TaskSpec("lift", "yellow").to_json()
    old = {"id": "b", "instruction": "x", "spec": spec, "model": "m", "parent": None,
           "success_rate": 0.5, "n_eval": 20, "retired_field": 1}  # fmt: skip
    path.write_text(json.dumps([old]))
    sk = Registry(path).skills["b"]
    assert sk.robot == "so101" and sk.tests == [] and sk.replay == ""


def test_thumbnail_is_rendered_from_a_recorded_state(tmp_path):
    import numpy as np

    lay = sample_layout(TaskSpec("reach", "red"), np.random.default_rng(0))
    sc = Scene(lay)
    sc.reset(lay)
    frame = sc.pose_frame()
    sc.close()
    out = tiles.render_thumb(lay.to_json(), frame, [0.0] * 6, tmp_path / "t.png")
    img = Image.open(out)
    assert img.size == (tiles.THUMB_W, tiles.THUMB_H)
    px = np.asarray(img)
    assert px.std() > 10  # a scene, not a blank frame
    assert Layout.from_json(lay.to_json()).blocks
