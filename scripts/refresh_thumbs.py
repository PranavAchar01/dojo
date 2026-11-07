"""Crop the browser shots from refresh_thumbs.mjs to tile thumbnails (480x360, centred on the arm and work area)."""

import sys
from pathlib import Path

from PIL import Image

from vla_forge.tiles import THUMB_H, THUMB_W, tile_dir

src = Path(sys.argv[1])
for png in sorted(src.glob("*.png")):
    im = Image.open(png).convert("RGB")
    w, h = im.size
    cw, ch = min(w, int(h * THUMB_W / THUMB_H)), h
    x0 = (w - cw) // 2
    im.crop((x0, 0, x0 + cw, ch)).resize((THUMB_W, THUMB_H), Image.LANCZOS).save(
        tile_dir(png.stem) / "thumb.png", optimize=True
    )
    print(png.stem)
