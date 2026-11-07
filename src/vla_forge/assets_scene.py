"""Dress a station cell as a real factory: library models around the cell, a PBR concrete floor, overhead lighting,
and the hero product riding on the part.

`dress(spec, "factory")` is called by sim/scene.build_spec when a Layout names a dressing. It reads
scenes/<name>.json and adds, to the same MjSpec the arms and the belt live in:

* the floor: Poly Haven concrete, diffuse as a MuJoCo texture, with specular and reflectance so the lamps shine on it;
* scenery: library models (assets/library/<id>/model.xml, visual meshes) at the placements in the description.
  Scenery is real scale on the factory floor, outside the arms' reach, and carries no contacts;
* overhead lamps: MuJoCo lights over the cell, one casting a shadow;
* the product (see `Product`): the real cassette player's parts attached to the part block of the cell. The block
  is a 3 cm workpiece carrier; the product on it is a 1:6 model so it fits the SO-101 cell. The parts' positions
  follow the physics: the cassette seats when the block is seated in the pallet nest, the lid closes when the block
  is inserted in the socket. Nothing in the product is made here: meshes are the library's cassette_* parts.

Visual geoms go to group 1: MuJoCo draws groups 0-2 by default, and viz.export_scene (the browser's robot
export) only takes group 2, so scenery never inflates it.
"""

from __future__ import annotations

import json
import math
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

from .assets_sources import REPO

LIB = REPO / "assets" / "library"
SCENES = REPO / "scenes"
VIS_GROUP = 1


def load_description(name: str) -> dict:
    return json.loads((SCENES / f"{name}.json").read_text())


def _quat_z(yaw: float) -> list[float]:
    return [math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2)]


class _Assets:
    """Adds a library model's visual meshes, textures and materials to a spec once per (model, scale)."""

    def __init__(self, spec: mujoco.MjSpec):
        self.spec = spec
        self.done: dict[tuple[str, float], list[dict]] = {}

    def prepare(self, model_id: str, scale: float) -> list[dict]:
        key = (model_id, round(scale, 5))
        if key in self.done:
            return self.done[key]
        folder = LIB / model_id
        root = ET.parse(folder / "model.xml").getroot()
        tag = f"{model_id}@{scale:g}"
        asset = root.find("asset")
        for t in asset.findall("texture"):
            self.spec.add_texture(
                name=f"{tag}/{t.get('name')}",
                type=mujoco.mjtTexture.mjTEXTURE_2D,
                file=str(folder / t.get("file")),
            )
        for m in asset.findall("material"):
            kw = {}
            if m.get("texture"):
                kw["textures"] = ["", f"{tag}/{m.get('texture')}"]
            self.spec.add_material(
                name=f"{tag}/{m.get('name')}",
                rgba=[float(v) for v in m.get("rgba", "1 1 1 1").split()],
                specular=float(m.get("specular", 0.3)),
                shininess=float(m.get("shininess", 0.5)),
                **kw,
            )
        visual = []
        for g in root.find("worldbody").iter("geom"):
            if g.get("group") != "2":  # the collision hulls are group 3
                continue
            mesh = g.get("mesh")
            file = next(
                m.get("file") for m in asset.findall("mesh") if m.get("name") == mesh
            )
            self.spec.add_mesh(
                name=f"{tag}/{mesh}",
                file=str(folder / file),
                scale=[scale] * 3,
            )
            visual.append(
                {"mesh": f"{tag}/{mesh}", "material": f"{tag}/{g.get('material')}"}
            )
        self.done[key] = visual
        return visual

    def attach(
        self,
        body: mujoco.MjsBody,
        model_id: str,
        scale: float,
        pos=(0.0, 0.0, 0.0),
        yaw: float = 0.0,
    ) -> None:
        for g in self.prepare(model_id, scale):
            body.add_geom(
                type=mujoco.mjtGeom.mjGEOM_MESH,
                meshname=g["mesh"],
                material=g["material"],
                pos=list(pos),
                quat=_quat_z(yaw),
                group=VIS_GROUP,
                contype=0,
                conaffinity=0,
                density=0,
            )


# ---------------------------------------------------------------- the hero product


def _add_product(spec: mujoco.MjSpec, assets: _Assets, cell: dict) -> None:
    """Attach the three real parts, at the poses they have in the source model, to the cell's part block."""
    block = next((b for b in spec.bodies if b.name == "target"), None)
    if block is None or "product" not in cell:
        return
    desc = cell["product"]
    s = desc["scale"]
    parts = json.loads((LIB / "product.json").read_text())["parts"]
    holder = block.add_body(
        name="product",
        pos=[0.0, 0.0, 0.015],  # the carrier's top face, in the block frame
        quat=_quat_z(math.radians(desc.get("yaw", 0.0))),
    )
    for p in parts:
        body = holder.add_body(
            name=f"product_{p['role']}", pos=[v * s for v in p["offset"]]
        )
        assets.attach(body, p["id"], s)


# ---------------------------------------------------------------- the cell


def dress(spec: mujoco.MjSpec, name: str) -> None:
    cell = load_description(name)
    world = spec.worldbody
    assets = _Assets(spec)

    floor = cell["floor"]
    env = LIB / "env"
    spec.add_texture(
        name="concrete_tex",
        type=mujoco.mjtTexture.mjTEXTURE_2D,
        file=str(env / "floor_diff.png"),
    )
    spec.add_material(
        name="concrete_mat",
        textures=["", "concrete_tex"],
        texuniform=True,
        texrepeat=[floor["repeat"]] * 2,
        specular=floor.get("specular", 0.25),
        shininess=floor.get("shininess", 0.4),
        reflectance=floor.get("reflectance", 0.04),
    )
    next(g for g in spec.geoms if g.name == "floor").material = "concrete_mat"

    for i, lamp in enumerate(cell.get("lights", [])):
        world.add_light(
            name=f"lamp{i}",
            pos=lamp["pos"],
            dir=[0.0, 0.0, -1.0],
            diffuse=[lamp["intensity"]] * 3,
            specular=[lamp["intensity"] * 0.3] * 3,
            castshadow=bool(lamp.get("shadow", False)),
            cutoff=lamp.get("cutoff", 80.0),
        )

    for i, p in enumerate(cell["scenery"]):
        body = world.add_body(name=f"scn{i}_{p['id']}", pos=p["pos"])
        assets.attach(
            body, p["id"], p.get("scale", 1.0), yaw=math.radians(p.get("yaw", 0.0))
        )
    _add_product(spec, assets, cell)


def bind_product(scene, name: str) -> ProductDriver | None:
    """Link the product's parts to a compiled scene so they follow the physics (see ProductDriver)."""
    cell = load_description(name)
    if "product" not in cell:
        return None
    m = scene.model
    try:
        cas = m.body("product_cassette").id
        lid = m.body("product_lid").id
    except KeyError:
        return None
    return ProductDriver(scene, cas, lid, cell["product"])


class ProductDriver:
    """Moves the cassette and lid between open and closed poses from the cell's own success predicates.

    kit cell:    the cassette drops into the well when the block is seated in the pallet nest.
    insert cell: the cassette is already in; the lid closes when the block is inserted in the socket.
    A pose eases over ~0.5 s so it reads on video. The block and the arms are untouched: this only moves visuals.
    """

    def __init__(self, scene, cas: int, lid: int, cfg: dict):
        self.sc, self.cas, self.lid = scene, cas, lid
        m = scene.model
        s = cfg["scale"]
        self.cas_closed = m.body_pos[cas].copy()
        self.lid_closed = m.body_pos[lid].copy()
        self.cas_open = self.cas_closed + np.array(cfg["cassette_lift"], float) * s
        self.lid_open = self.lid_closed + np.array(cfg["lid_lift"], float) * s
        self.cas_t = 0.0  # 0 open, 1 closed
        self.lid_t = 0.0

    def reset(self, cassette_in: bool) -> None:
        self.cas_t = 1.0 if cassette_in else 0.0
        self.lid_t = 0.0
        self._apply()

    def step(self, cassette_in: bool, lid_closed: bool, dt: float = 0.1) -> None:
        k = dt / 0.5
        self.cas_t = float(np.clip(self.cas_t + (k if cassette_in else -k), 0, 1))
        self.lid_t = float(np.clip(self.lid_t + (k if lid_closed else -k), 0, 1))
        self._apply()

    def _apply(self) -> None:
        m = self.sc.model
        m.body_pos[self.cas] = (
            self.cas_open + (self.cas_closed - self.cas_open) * self.cas_t
        )
        m.body_pos[self.lid] = (
            self.lid_open + (self.lid_closed - self.lid_open) * self.lid_t
        )
        mujoco.mj_forward(m, self.sc.data)
