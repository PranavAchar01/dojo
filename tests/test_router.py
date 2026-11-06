"""The feasibility router: every decision comes from a real check (IK, the scene palette, measured times)."""

from __future__ import annotations

import pytest

from vla_forge import router
from vla_forge.compose import Step
from vla_forge.llm import rule_parse
from vla_forge.registry import Registry, Skill
from vla_forge.sim.tasks import TaskSpec

RULES = {"model": "rules", "kind": "single", "raw_steps": 1}


def rules_info(prompt: str) -> dict:
    from vla_forge.llm import rule_motion

    return {**RULES, "motion_word": rule_motion(prompt)[1]}


def tile(
    reg: Registry, spec: TaskSpec, rate: float, seconds: float = 150.0, **kw
) -> Skill:
    sk = Skill(
        id=f"{spec.skill}-{spec.obj}-{len(reg.skills)}",
        instruction=spec.instruction,
        spec=spec.to_json(),
        model="skills/none.pt",
        parent=None,
        success_rate=rate,
        n_eval=20,
        seconds=seconds,
        **kw,
    )
    reg.add(sk)
    return sk


@pytest.fixture
def reg(tmp_path) -> Registry:
    return Registry(tmp_path / "registry.json")


def decide(prompt: str, reg: Registry, **kw) -> router.Route:
    return router.route(prompt, rule_parse(prompt), rules_info(prompt), reg, **kw)


def test_workspace_is_measured():
    ws = router.workspace()
    assert 0.3 < ws["max_radius"] < 0.6
    assert 0.3 < ws["max_height"] < 0.7
    assert ws["sector_span"] > 0.1


def test_reach_passes_for_every_motion():
    for spec in (
        TaskSpec("reach", "red"),
        TaskSpec("push", "green", "blue"),
        TaskSpec("lift", "yellow"),
        TaskSpec("pick_place", "blue", "blue"),
        TaskSpec("stack", "orange", "red"),
    ):
        c = router.check_reach(spec)
        assert c.ok, (spec.skill, c.detail)


def test_reach_passes_for_a_chain():
    steps = [Step("pick_place", "blue", "blue"), Step("pick_place", "green", "green")]
    assert router.check_reach(steps).ok


def test_object_not_on_the_table_is_refused(reg):
    r = decide("touch the red mug", reg)
    assert r.decision == "not_feasible"
    assert "mug" in r.reason
    assert r.suggestion


def test_colour_the_scene_does_not_have_is_refused(reg):
    r = decide("put the red block in the purple bowl", reg)
    assert r.decision == "not_feasible"
    assert "purple bowl" in r.reason


def test_height_above_the_arm_is_refused(reg):
    r = decide("lift the red block 60 cm up", reg)
    assert r.decision == "not_feasible"
    assert "out of reach" in r.reason


def test_distance_wider_than_the_workspace_is_refused(reg):
    r = decide("push the red block 80 cm onto the blue square", reg)
    assert r.decision == "not_feasible"
    assert "workspace" in r.reason


def test_small_height_passes(reg):
    r = decide("lift the red block 5 cm up", reg)
    assert r.decision == "train"
    assert any(c.name == "reach" and c.ok and "5 cm" in c.detail for c in r.checks)


def test_no_known_motion_is_refused_in_rules_mode(reg):
    r = decide("fold the laundry", reg)
    assert r.refused


def test_unparsed_prompt_is_refused(reg):
    r = router.route("write me a poem", None, {"model": "rules"}, reg)
    assert r.decision == "not_feasible"


def test_new_prompt_trains_with_default_estimate(reg):
    r = decide("touch the purple block", reg)
    assert r.decision == "train"
    assert r.estimate_s == router.DEFAULT_TRAIN_S


def test_estimate_uses_measured_tiles(reg):
    tile(reg, TaskSpec("reach", "red"), 0.5, seconds=100)
    tile(reg, TaskSpec("reach", "blue"), 0.5, seconds=140)
    r = decide("touch the purple block", reg)
    assert r.estimate_s == pytest.approx(120)


def test_good_exact_tile_is_reused(reg):
    sk = tile(reg, TaskSpec("reach", "red"), 0.9)
    r = decide("touch the red block", reg)
    assert r.decision == "reuse"
    assert r.skill_id == sk.id


def test_weak_exact_tile_is_retrained(reg):
    tile(reg, TaskSpec("reach", "red"), 0.5)
    assert decide("touch the red block", reg).decision == "train"


def test_tile_for_another_robot_is_not_reused(reg):
    tile(reg, TaskSpec("reach", "red"), 0.95, robot="custom-arm")
    assert decide("touch the red block", reg).decision == "train"


def test_scratch_and_train_modes_never_reuse(reg):
    tile(reg, TaskSpec("reach", "red"), 0.95)
    assert decide("touch the red block", reg, mode="scratch").decision == "train"
    assert decide("touch the red block", reg, mode="train").decision == "train"


def test_remix_trains_from_the_tile(reg):
    src = tile(reg, TaskSpec("reach", "red"), 0.9)
    r = decide("touch the blue block", reg, mode="remix", from_skill=src.id)
    assert r.decision == "train"
    assert r.skill_id == src.id
    assert "remix" in r.reason


def chain(prompt_steps: list[Step], raw: int | None = None, dropped: str = "") -> tuple:
    info = {
        "model": "test",
        "kind": "composite",
        "raw_steps": raw or len(prompt_steps),
        "dropped": dropped,
    }
    return prompt_steps, info


def test_chain_of_known_tiles_routes_to_chain(reg):
    a = TaskSpec("pick_place", "blue", "blue")
    tile(reg, a, 0.9)
    steps, info = chain(
        [Step("pick_place", "blue", "blue"), Step("pick_place", "green", "green")]
    )
    r = router.route("sort blue and green", steps, info, reg)
    assert r.decision == "chain"
    assert [p["source"] for p in r.steps] == ["reuse", "train"]


def test_too_many_steps_is_too_complex(reg):
    steps, info = chain(
        [
            Step("pick_place", "blue", "blue"),
            Step("pick_place", "green", "green"),
            Step("stack", "red", "orange"),
        ],
        raw=5,
    )
    r = router.route("do five things", steps, info, reg)
    assert r.decision == "too_complex"
    assert r.suggestion


def test_chain_over_budget_is_too_complex(reg):
    steps, info = chain(
        [
            Step("pick_place", "blue", "blue"),
            Step("pick_place", "green", "green"),
            Step("stack", "red", "orange"),
        ]
    )
    r = router.route("three new skills", steps, info, reg)
    assert r.decision == "too_complex"
    assert r.estimate_s > router.CHAIN_BUDGET_S
    assert r.suggestion == steps[0].instruction


def test_route_json_round_trips(reg):
    js = decide("touch the red mug", reg).to_json()
    assert js["decision"] == "not_feasible"
    assert all({"name", "ok", "detail"} <= set(c) for c in js["checks"])
