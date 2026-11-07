"""Build the model library from the curated catalog (assets_sources.CATALOG).

    uv run python -m vla_forge.assets_build [--fetch] [ids ...]

Fetches (with --fetch) into assets/_raw/, converts each model into assets/library/<id>/ (assets_convert), proves the
MJCF compiles in MuJoCo, and rewrites assets/library/manifest.json with provenance for every entry.
Heavy: run it through scripts/heavy.sh under mem-cap (two workers at most).
"""

from __future__ import annotations

import json
import sys
from concurrent.futures import ThreadPoolExecutor

from . import assets_convert as conv
from . import assets_sources as src

LIB = src.REPO / "assets" / "library"
MANIFEST = LIB / "manifest.json"

# Gazebo Fuel gives no dimensions in its metadata. Where a mesh was exported in cm / mm / ambiguous units, the real
# object's longest side in metres (from its catalogue size) fixes the scale. Poly Haven publishes true sizes.
LONGEST_M = {
    "picking_bin": 0.6,
    "generator": 1.1,
    "pump": 1.5,
    "euro_pallet": 1.2,
    "toolbox": 0.51,
    "tube_light": 1.9,
    "fire_extinguisher": 0.61,
    "construction_cone": 0.43,
    "jersey_barrier": 1.6,
    "large_crate": 1.08,
}
# Fuel GLBs are not consistent: most are Y-up as glTF says, a few were exported Z-up as Gazebo draws them.
# Found from the rendered thumbnails (a pallet lying on its side, a bin on end).
Z_UP = {"pallet_standard", "picking_bin"}
MAX_HULLS = {"static": 14, "dynamic": 8}
CATEGORY_TAGS = {
    "conveyor": ["conveyor", "belt", "line"],
    "pallet": ["pallet", "logistics"],
    "container": ["tote", "bin", "crate", "box"],
    "shelving": ["shelf", "rack", "storage"],
    "workbench": ["bench", "table", "cart"],
    "handling": ["logistics", "vehicle"],
    "machine": ["machine", "tool"],
    "part": ["part", "component"],
    "tool": ["tool", "hand tool"],
    "lighting": ["light", "lamp", "fixture"],
    "safety": ["safety", "guard"],
    "utility": ["utility", "industrial"],
    "structure": ["structure"],
}


def build_one(p: src.Pick) -> dict:
    raw = src.RAW / p.id
    prov = json.loads((raw / "provenance.json").read_text())
    s = conv.Settings(
        kind=p.kind,
        max_hulls=MAX_HULLS[p.kind],
        target_longest_m=LONGEST_M.get(p.id),
        y_up=False if p.id in Z_UP else None,
    )
    if p.density:
        s.density = p.density
    elif p.kind == "static":
        s.density = 1000.0
    dims = prov.get("dimensions_mm")
    known = [d / 1000.0 for d in dims] if dims else None
    scene = conv.load_folder(raw, s.y_up)
    pieces = conv.normalise(scene, s, known)
    out = LIB / p.id
    meta = conv.build_package(pieces, out, p.id, s)
    meta["mujoco"] = conv.check_mjcf(out)
    return entry(p, prov, meta)


def entry(p: src.Pick, prov: dict, meta: dict) -> dict:
    return {
        "id": p.id,
        "name": prov["name"],
        "category": p.category,
        "kind": p.kind,
        "description": p.note,
        "tags": sorted(
            {p.category, p.kind, *CATEGORY_TAGS.get(p.category, []), *p.id.split("_")}
        ),
        "origin": "library",
        "source": {
            "name": prov["source_name"],
            "url": prov["source_url"],
            "upstream": prov["upstream"],
            "author": prov["author"],
            "license": prov["license"],
            "license_url": prov["license_url"],
        },
        "size_m": meta["size_m"],
        "mass_kg": meta["mujoco"]["mass_kg"] if p.kind == "dynamic" else None,
        "faces": meta["faces"],
        "hulls": meta["hulls"],
        "collision": meta["collision"],
    }


def write_manifest(entries: list[dict]) -> None:
    by_id = {e["id"]: e for e in entries}
    LIB.mkdir(parents=True, exist_ok=True)
    MANIFEST.write_text(
        json.dumps(
            {
                "version": 1,
                "models": sorted(
                    by_id.values(), key=lambda e: (e["category"], e["id"])
                ),
            },
            indent=1,
        )
    )


def main(argv: list[str]) -> int:
    fetch = "--fetch" in argv
    ids = [a for a in argv if not a.startswith("--")]
    picks = [src.BY_ID[i] for i in ids] if ids else src.CATALOG
    if fetch:
        for p in picks:
            src.fetch(p)
    existing = {}
    if MANIFEST.exists():
        existing = {e["id"]: e for e in json.loads(MANIFEST.read_text())["models"]}

    def one(p: src.Pick):
        try:
            return p.id, build_one(p), None
        except Exception as e:  # noqa: BLE001 - report per model, keep building the rest
            return p.id, None, f"{type(e).__name__}: {e}"

    failed = []
    with ThreadPoolExecutor(2) as ex:
        for pid, ent, err in ex.map(one, picks):
            if err:
                failed.append((pid, err))
                print("FAIL", pid, err[:200], flush=True)
            else:
                existing[pid] = ent
                print(
                    "ok", pid, ent["size_m"], ent["hulls"], ent["collision"], flush=True
                )
                write_manifest(list(existing.values()))
    write_manifest(list(existing.values()))
    print(f"{len(existing)} in library, {len(failed)} failed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
