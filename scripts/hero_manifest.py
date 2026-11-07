"""Write assets/lego/manifest.json: provenance of every LDraw part (read from the part files themselves) and the
converted mesh stats. Usage: hero_manifest.py"""

import json
from pathlib import Path

from vla_forge import hero_bricks as hb

parts = {}
for part, name in hb.PART_NAME.items():
    src = hb.LIB / "parts" / f"{part}.dat"
    head = {}
    for line in src.read_text(errors="ignore").splitlines()[:12]:
        for key in ("Name", "Author", "!LICENSE"):
            if line.startswith((f"0 {key}:", f"0 {key} ")):
                head[key] = line.split(None, 2)[2].lstrip(": ").strip()
    v, f = hb.part_mesh_cached(part)
    parts[part] = {
        "title": name,
        "source_url": f"https://library.ldraw.org/library/official/parts/{part}.dat",
        "author": head.get("Author"),
        "licence": "CC BY 4.0 (" + head.get("!LICENSE", "") + ")",
        "licence_url": "https://creativecommons.org/licenses/by/4.0/",
        "footprint_studs": hb.PART_FOOTPRINT[part],
        "mesh": f"meshes/{part}.npz",
        "vertices": len(v),
        "triangles": len(f),
    }
out = {
    "build": "Rule-generated brick bridge (src/vla_forge/hero_bricks.py). Every placement follows from the three rules "
    "(gap, pillar height, deck length); nothing is placed by eye. Parts are plain LDraw bricks.",
    "library": "LDraw Parts Library, official complete.zip",
    "library_url": "https://library.ldraw.org/library/updates/complete.zip",
    "library_licence": "CC BY 4.0 (parts); the zip's CAlicense.txt text is Creative Commons Attribution 2.0",
    "attribution": "LDraw is a trademark of the Estate of James Jessiman. Parts are from the LDraw Parts Library.",
    "conversion": "src/vla_forge/hero_ldraw.py: sub-file recursion with BFC winding, LDU 0.4 mm, Y-down to Z-up, "
    "origin at the footprint bottom centre. Studs and underside tubes are kept in the visual mesh; physics uses boxes.",
    "parts": parts,
}
Path(hb.LEGO / "manifest.json").write_text(json.dumps(out, indent=2) + "\n")
