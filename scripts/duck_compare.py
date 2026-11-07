"""Side-by-side of the published source photo crop and our render. Usage: duck_compare.py SOURCE.png OURS.png OUT.png"""

import sys

from PIL import Image, ImageDraw

from vla_forge.duck_render import _font

src = Image.open(sys.argv[1]).convert("RGB").crop((300, 320, 560, 560))  # the duck at lower middle of the photo
src = src.resize((780, 720), Image.LANCZOS)
ours = Image.open(sys.argv[2]).convert("RGB").crop((230, 200, 830, 800)).resize((720, 720), Image.LANCZOS)
sheet = Image.new("RGB", (780 + 720 + 30, 800), (245, 245, 243))
sheet.paste(src, (10, 70))
sheet.paste(ours, (800, 70))
d = ImageDraw.Draw(sheet)
f = _font(24)
d.text((14, 20), "Source: kit build, published photo", fill=(30, 30, 30), font=f)
d.text((804, 20), "Ours: LDraw meshes, same parts", fill=(30, 30, 30), font=f)
sheet.save(sys.argv[3])
