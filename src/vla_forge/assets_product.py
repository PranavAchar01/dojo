"""The factory's hero product: a real portable cassette player (Poly Haven, CC0, Mateusz Sadek) in its three
real parts (body, cassette, lid). The model's own named nodes are the split; no geometry is made or edited here.

    uv run --group assets python -m vla_forge.assets_product

Fetches the model into assets/_raw/cassette_player_src/, writes one GLB per part under assets/_raw/<part id>/
with the upstream provenance, records where each part sits in the assembled product (assets/library/product.json),
converts the parts with the library pipeline and lists them in assets/library/manifest_products.json.
"""

from __future__ import annotations

import json
import shutil

import numpy as np
import trimesh

from . import assets_build as build
from . import assets_convert as conv
from . import assets_sources as src

UPSTREAM = "portable_cassette_player"
PARTS = (  # part id, glTF node, role, what it is, physics kind
    (
        "cassette_body",
        "portable_cassette_player",
        "body",
        "Cassette player body with its tape well",
        "static",
    ),
    (
        "cassette_tape",
        "portable_cassette_player_casette",
        "cassette",
        "Compact cassette tape",
        "dynamic",
    ),
    (
        "cassette_lid",
        "portable_cassette_player_lid",
        "lid",
        "Cassette player lid",
        "dynamic",
    ),
)
PRODUCT = src.REPO / "assets" / "library" / "product.json"
PRODUCT_MANIFEST = src.REPO / "assets" / "library" / "manifest_products.json"
SRC_PICK = src.Pick(
    "cassette_player_src",
    src.PH,
    UPSTREAM,
    "product",
    "dynamic",
    "Portable cassette player, whole",
)


def _part_of(node_name: str) -> str:
    """Which named part a primitive's node belongs to (the exporter appended hashes to split primitives)."""
    if node_name.startswith(f"{UPSTREAM}_lid"):
        return f"{UPSTREAM}_lid"
    if node_name.startswith(f"{UPSTREAM}_casette"):
        return f"{UPSTREAM}_casette"
    return UPSTREAM


def prepare() -> dict:
    raw = src.fetch(SRC_PICK)
    prov = json.loads((raw / "provenance.json").read_text())
    gltf = next(raw.glob("*.gltf"))
    scene = trimesh.load(gltf, process=False)
    # glTF is Y-up; the sim is Z-up. Lengths are metres.
    rot = trimesh.transformations.rotation_matrix(np.pi / 2, [1, 0, 0])
    scene.apply_transform(rot)
    lo, hi = scene.bounds
    origin = np.array([(lo[0] + hi[0]) / 2, (lo[1] + hi[1]) / 2, lo[2]])
    layout = []
    for pid, node, role, note, kind in PARTS:
        # the glTF splits each named part into primitives (one per material); gather them by node name
        names = [n for n in scene.graph.nodes_geometry if _part_of(n) == node]
        one = trimesh.Scene()
        for n in names:
            world, gname = scene.graph[n]
            one.add_geometry(
                scene.geometry[gname].copy(), node_name=n, geom_name=gname, transform=world
            )
        b = one.bounds
        base = np.array([(b[0][0] + b[1][0]) / 2, (b[0][1] + b[1][1]) / 2, b[0][2]])
        layout.append(
            {
                "id": pid,
                "role": role,
                "node": node,
                "offset": (base - origin).round(5).tolist(),
                "size_m": (b[1] - b[0]).round(5).tolist(),
            }
        )
        d = src.RAW / pid
        if d.exists():
            shutil.rmtree(d)
        d.mkdir(parents=True)
        one.apply_transform(np.linalg.inv(rot))  # back to glTF's Y-up for the loader; textures kept
        one.export(d / "model.glb")
        p = dict(prov)
        p["name"] = f"{prov['name']} ({role})"
        p["upstream"] = f"{UPSTREAM}#{node}"
        p["dimensions_mm"] = None
        (d / "provenance.json").write_text(json.dumps(p, indent=1))
    prod = {
        "id": "cassette_player",
        "name": prov["name"],
        "size_m": (hi - lo).round(5).tolist(),
        "source": {
            k: prov[k]
            for k in ("author", "license", "license_url", "source_name", "source_url")
        },
        "parts": layout,
    }
    PRODUCT.write_text(json.dumps(prod, indent=1))
    return prod


def convert() -> None:
    entries = []
    for pid, _node, _role, note, kind in PARTS:
        p = src.Pick(pid, src.PH, UPSTREAM, "product", kind, note)
        meta = build.build_one(p)
        entries.append(meta)
        print("ok", pid, meta["size_m"], meta["hulls"], meta["collision"], flush=True)
    PRODUCT_MANIFEST.write_text(
        json.dumps({"version": 1, "models": entries}, indent=1)
    )


if __name__ == "__main__":
    print(json.dumps(prepare(), indent=1)[:1500])
    convert()
