"""Every thumbnail Dojo shows comes out of one pipeline: the browser stage (web/stage.js, the Sim tab's lab).

The model-library cards are shot by scripts/assets_thumbs.mjs through web/assets-thumb.html, the trained-model tiles
by scripts/refresh_thumbs.mjs through the Sim tab; both stamp the PNG with the stage tag (scripts/stamp.mjs and
scripts/refresh_thumbs.py). A thumbnail without the tag was made some other way and fails here.
"""

from pathlib import Path

import pytest
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
TAG = "dojo-stage-v1"


def _thumbs() -> list[Path]:
    found = []
    for pattern in ("assets/library/*/thumb.png", "assets/user/*/thumb.png", "skills/tiles/*/thumb.png"):
        found += sorted(ROOT.glob(pattern))
    return found


@pytest.mark.parametrize("path", _thumbs(), ids=lambda p: f"{p.parent.parent.name}/{p.parent.name}")
def test_thumbnail_comes_from_the_stage(path: Path) -> None:
    with Image.open(path) as im:
        assert im.text.get("dojo-stage") == TAG, f"{path} was not rendered by the stage pipeline"


def test_stage_tag_is_one_value() -> None:
    assert TAG in (ROOT / "web" / "stage.js").read_text()
    assert TAG in (ROOT / "scripts" / "stamp.mjs").read_text()
    assert TAG in (ROOT / "scripts" / "refresh_thumbs.py").read_text()


def test_no_other_renderer_writes_library_thumbnails() -> None:
    """The MuJoCo still renderer for tiles (render_thumb) is the old look; nothing may call it any more."""
    for py in (ROOT / "src" / "vla_forge").glob("*.py"):
        if py.name == "tiles.py":
            continue
        assert "render_thumb" not in py.read_text(), f"{py.name} renders a thumbnail outside the stage"
