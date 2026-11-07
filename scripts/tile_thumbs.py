"""Re-shoot tile thumbnails on the shared stage: uv run python scripts/tile_thumbs.py [--multi] [tile-id ...]

Trained-model tiles come from their saved rollout; --multi shoots the multi-arm cards from their saved line run."""

import sys

from vla_forge import tile_stage

args = sys.argv[1:]
if args[:1] == ["--multi"]:
    print(len(tile_stage.shoot_multi(args[1:])), "multi-arm cards")
else:
    print(len(tile_stage.shoot(args)), "tiles")
