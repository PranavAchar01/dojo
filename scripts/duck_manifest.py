"""Write assets/lego/manifest.json: provenance of every LDraw part (read from the part files themselves) and the
converted mesh stats. Usage: duck_manifest.py"""

import json
from pathlib import Path

from vla_forge import duck_bricks as db

parts = {}
for part, name in db.PART_NAME.items():
    if part == "3039":
        continue
    src = db.LIB / "parts" / f"{part}.dat"
    head = {}
    for line in src.read_text(errors="ignore").splitlines()[:12]:
        for key in ("Name", "Author", "!LICENSE"):
            if line.startswith(f"0 {key}:") or line.startswith(f"0 {key} "):
                head[key] = line.split(None, 2)[2].lstrip(": ").strip()
    v, f = db.part_mesh_cached(part)
    parts[part] = {
        "title": name,
        "source_url": f"https://library.ldraw.org/library/official/parts/{part}.dat",
        "author": head.get("Author"),
        "licence": "CC BY 4.0 (" + head.get("!LICENSE", "") + ")",
        "licence_url": "https://creativecommons.org/licenses/by/4.0/",
        "footprint_studs": db.PART_FOOTPRINT[part],
        "mesh": f"meshes/{part}.npz",
        "vertices": len(v),
        "triangles": len(f),
    }
duck = [
    {"name": b.name, "part": b.part, "colour": {"3021": "red", "3004": "yellow"}.get(b.part, "red" if b.rgba[0] > 0.7 and b.rgba[1] < 0.1 else "yellow"),
     "stud_col": b.col, "stud_row": b.row, "plate_level": b.lvl, "yaw_deg": b.yaw_deg, "role": b.role,
     "pos_m": [round(float(x), 5) for x in b.target_pos()]}
    for b in db.DUCK
]
out = {
    "duck_source": {
        "what": "LEGO Serious Play 'Duck' kit, set 2000416 (6 parts). The kit has no instructions; the exercise is "
        "freeform. The assembly copies one duck from a published photo of builds made from the kit.",
        "kit_parts": "Brickset inventory https://brickset.com/inventories/2000416-1 : Plate 2x3 bright red x2, Brick 2x4 yellow, "
        "Brick 2x2 yellow, Brick 1x2 yellow, Brick 2x2 yellow with printed eye",
        "photo": "https://brickresales.com.au/blogs/blog/lego-serious-play-what-is-the-duck-challenge "
        "(feature image; duck at lower middle; used only for a local side-by-side, not redistributed)",
        "not_visible_in_photo": "the yellow 1x2 (placed as a low tail, our choice); the printed eye is drawn as a decal",
        "parts": duck,
    },
    "library": "LDraw Parts Library, official complete.zip",
    "library_url": "https://library.ldraw.org/library/updates/complete.zip",
    "library_licence": "CC BY 4.0 (parts); the zip's CAlicense.txt text is Creative Commons Attribution 2.0",
    "attribution": "LDraw is a trademark of the Estate of James Jessiman. Parts are from the LDraw Parts Library.",
    "conversion": "src/vla_forge/duck_ldraw.py: sub-file recursion with BFC winding, LDU 0.4 mm, Y-down to Z-up, "
    "origin at the footprint bottom centre. Studs and underside tubes are kept in the visual mesh; physics uses boxes.",
    "parts": parts,
}
Path(db.LEGO / "manifest.json").write_text(json.dumps(out, indent=2) + "\n")
