"""Render the instruction photo and the exploded sheet. Usage: duck_shots.py OUTDIR"""

import sys
from pathlib import Path

from vla_forge import hero_render as r

out = Path(sys.argv[1])
out.mkdir(parents=True, exist_ok=True)
r.photo(out / "duck-photo.png")
r.exploded(out / "duck-exploded.png")
