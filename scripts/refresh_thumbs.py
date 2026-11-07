"""Crop the browser shots from refresh_thumbs.mjs to tile thumbnails (480x360, centred on the arm and work area)."""

import sys
from pathlib import Path

from PIL import Image

from vla_forge.tiles import THUMB_H, THUMB_W, tile_dir

src = Path(sys.argv[1])
for png in sorted(src.glob("*.png")):
    im = Image.open(png).convert("RGB")
    w, h = im.size
    zoom = 1.25  # the arm and its work area fill the tile
    cw = int(min(w, h * THUMB_W / THUMB_H) / zoom)
    ch = cw * THUMB_H // THUMB_W
    x0, y0 = (w - cw) // 2, int((h - ch) * 0.6)
    im.crop((x0, y0, x0 + cw, y0 + ch)).resize((THUMB_W, THUMB_H), Image.LANCZOS).save(
        tile_dir(png.stem) / "thumb.png", optimize=True
    )
    print(png.stem)
