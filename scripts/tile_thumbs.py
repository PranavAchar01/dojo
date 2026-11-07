"""Re-shoot trained-model tile thumbnails on the shared stage: uv run python scripts/tile_thumbs.py [tile-id ...]"""

import sys

from vla_forge import tile_stage

print(len(tile_stage.shoot(sys.argv[1:])), "tiles")
