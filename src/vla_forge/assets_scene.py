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
from .sim.display import CONCRETE, FLOOR, STEEL

LIB = REPO / "assets" / "library"
SCENES = REPO / "scenes"
VIS_GROUP = 1


def load_description(name: str) -> dict:
    return json.loads((SCENES / f"{name}.json").read_text())


def _quat_z(yaw: float) -> list[float]:
    return [math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2)]


def _has_cutout(path) -> bool:
    from PIL import Image

    """A texture drawn with transparent holes (alpha), or one whose holes were flattened to black by the converter."""
    with Image.open(path) as im:
        if im.mode in ("RGBA", "LA"):
            return im.getchannel("A").getextrema()[0] < 200
        grey = np.asarray(im.convert("L"))
        return bool((grey < 14).mean() > 0.5)


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
        cutout = {
            t.get("name")
            for t in asset.findall("texture")
            if _has_cutout(folder / t.get("file"))
        }
        for m in asset.findall("material"):
            kw = {}
            if m.get("texture") in cutout:
                # chain-link and the like: MuJoCo draws no alpha cut-outs, so the screen becomes a clear panel
                self.spec.add_material(
                    name=f"{tag}/{m.get('name')}",
                    rgba=[0.78, 0.84, 0.9, 0.22],
                    specular=0.5,
                )
                continue
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
                inertia=mujoco.mjtMeshInertia.mjMESH_INERTIA_SHELL,
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
    spec.add_material(
        name="concrete_mat",
        rgba=list(FLOOR),
        specular=floor.get("specular", 0.3),
        shininess=floor.get("shininess", 0.5),
        reflectance=floor.get("reflectance", 0.12),
    )
    ground = next(g for g in spec.geoms if g.name == "floor")
    ground.material = "concrete_mat"
    ground.size = [8.0, 8.0, 0.1]  # out to the walls

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
    _add_frame(spec, assets, cell)
    _add_product(spec, assets, cell)
    _add_belt_items(spec, assets, cell)


# ---------------------------------------------------------------- the workcell frame

PITCH = 1.2  # metres between the two arms along the belt


def _box(parent, pos, half, rgba, mat=None, group=VIS_GROUP):
    kw = {"material": mat} if mat else {"rgba": list(rgba)}
    parent.add_geom(
        type=mujoco.mjtGeom.mjGEOM_BOX,
        pos=list(pos),
        size=list(half),
        group=group,
        contype=0,
        conaffinity=0,
        density=0,
        **kw,
    )


def _cyl(parent, pos, radius, half_len, axis, mat):
    q = {"x": [0.7071, 0, 0.7071, 0], "y": [0.7071, -0.7071, 0, 0]}[axis]
    parent.add_geom(
        type=mujoco.mjtGeom.mjGEOM_CYLINDER,
        pos=list(pos),
        quat=q,
        size=[radius, half_len, 0],
        material=mat,
        group=VIS_GROUP,
        contype=0,
        conaffinity=0,
        density=0,
    )


def _add_frame(spec: mujoco.MjSpec, assets: _Assets, cell: dict) -> None:
    """Aluminium-extrusion table, conveyor frame with rollers, tabletop guard panels, a parts bin and the walls.

    The table and the belt in the physics are unchanged; this adds the structure around them for a line of two
    cells (the table and belt visuals run on to the second arm). Plain boxes and cylinders for the extrusions and
    rollers; the guard panels and bin are library models.
    """
    fr = cell.get("frame")
    if not fr:
        return
    world = spec.worldbody
    y0, y1 = -0.6, PITCH * (fr.get("cells", 2) - 1) + 0.6
    x0, x1 = -0.25, 0.65
    spec.add_material(
        name="alu", rgba=[0.80, 0.815, 0.835, 1], specular=0.7, shininess=0.55
    )
    spec.add_material(
        name="wall_mat", rgba=[0.86, 0.87, 0.885, 1], specular=0.05, emission=0.55
    )
    spec.add_material(name="guard_glass", rgba=[0.78, 0.84, 0.9, 0.13], specular=0.4)
    spec.add_material(name="guard_frame", rgba=[0.08, 0.082, 0.094, 1])
    spec.add_material(name="shelf_mat", rgba=[0.5, 0.51, 0.53, 1], specular=0.1)
    spec.add_material(name="top_mat", rgba=list(CONCRETE), specular=0.15)

    # the table top runs on past the first cell
    _box(
        world,
        [(x0 + x1) / 2, (0.6 + y1) / 2, -0.02],
        [(x1 - x0) / 2, (y1 - 0.6) / 2, 0.02],
        CONCRETE,
        "top_mat",
    )
    # extrusion frame: square legs, a solid apron under the top on all four sides, cross rails and a low shelf
    leg, ap = 0.03, 0.05
    cx, cy0 = (x0 + x1) / 2, (y0 + y1) / 2
    for x in (x0 + leg, x1 - leg):
        for y in (y0 + leg, cy0, y1 - leg):
            _box(world, [x, y, -0.40], [leg, leg, 0.35], STEEL, "alu")
    for x in (x0 + 0.012, x1 - 0.012):  # apron: opaque aluminium plates
        _box(world, [x, cy0, -0.04 - ap], [0.012, (y1 - y0) / 2, ap], STEEL, "alu")
    for y in (y0 + 0.012, y1 - 0.012):
        _box(world, [cx, y, -0.04 - ap], [(x1 - x0) / 2, 0.012, ap], STEEL, "alu")
    for x in (x0 + leg, x1 - leg):  # low cross rails and a solid shelf
        _box(world, [x, cy0, -0.52], [0.02, (y1 - y0) / 2, 0.02], STEEL, "alu")
    _box(world, [cx, cy0, -0.54], [(x1 - x0) / 2 - 0.03, (y1 - y0) / 2 - 0.03, 0.01], STEEL, "shelf_mat")

    # the conveyor: side rails, end rollers, a roller bed, legs to the floor past the table
    bx, hw = 0.245, 0.035
    by0, by1 = -0.95, PITCH * (fr.get("cells", 2) - 1) + 0.95
    cy, hl = (by0 + by1) / 2, (by1 - by0) / 2
    for side in (-1, 1):
        _box(
            world,
            [bx + side * (hw + 0.007), cy, 0.006],
            [0.007, hl, 0.014],
            STEEL,
            "alu",
        )
    _box(world, [bx, cy, -0.0035], [hw, hl, 0.0015], STEEL, "alu")
    for y in np.arange(by0 + 0.03, by1, 0.06):
        _cyl(world, [bx, float(y), -0.0045], 0.004, hw, "x", "alu")
    for y in (by0, by1):
        _cyl(world, [bx, y, -0.001], 0.009, hw + 0.002, "x", "alu")
    for y in (by0 + 0.03, by1 - 0.03):
        for side in (-1, 1):
            _box(
                world,
                [bx + side * (hw + 0.007), y, -0.38],
                [0.008, 0.008, 0.37],
                STEEL,
                "alu",
            )
    # the belt's visible surface continues over the whole run (the physics belt is the stretch in each cell)
    spec.add_texture(
        name="belt_tex",
        type=mujoco.mjtTexture.mjTEXTURE_2D,
        builtin=mujoco.mjtBuiltin.mjBUILTIN_CHECKER,
        rgb1=[0.10, 0.105, 0.115],
        rgb2=[0.15, 0.155, 0.165],
        width=16,
        height=16,
    )
    spec.add_material(
        name="belt_vis", textures=["", "belt_tex"], texrepeat=[1, 60], specular=0.1
    )
    _box(world, [bx, cy, -0.0008], [hw, hl, 0.0012], None, "belt_vis")

    _guard(world, x0, x1, y0, y1, fr.get("guard", {}))
    # the same key and fill lights as sim/scene.py gives the first cell, repeated over each further arm
    for k in range(1, fr.get("cells", 2)):
        dy = PITCH * k
        world.add_light(
            name=f"key{k}",
            pos=[0.4, -0.4 + dy, 1.4],
            dir=[-0.3, 0.3, -1.0],
            diffuse=[0.8] * 3,
            specular=[0.2] * 3,
            castshadow=True,
        )
        world.add_light(
            name=f"fill{k}",
            pos=[-0.3, 0.5 + dy, 1.2],
            dir=[0.3, -0.4, -1.0],
            diffuse=[0.35] * 3,
            castshadow=False,
        )
    # walls and a back drop, so the room is lit and closed instead of a black void
    wz = 1.25
    for pos, half in (
        ([-2.2, 0.6, wz], [0.05, 6.0, 2.0]),
        ([4.0, 0.6, wz], [0.05, 6.0, 2.0]),
        ([0.9, -3.0, wz], [4.5, 0.05, 2.0]),
        ([0.9, 4.2, wz], [4.5, 0.05, 2.0]),
    ):
        _box(world, pos, half, None, "wall_mat")


def _guard(world, x0, x1, y0, y1, cfg: dict) -> None:
    """A rectangular guard around the line on the table: posts every `panel` metres, clear panels at one height,
    a top rail, an opening where the belt runs through each end, and one gate in the front."""
    h, w = cfg.get("height", 0.28), cfg.get("panel", 0.3)
    gx, gy = 0.0, 0.0
    spec_mat = {"rgba": [0.8, 0.84, 0.9, 0.12]}
    world_mat = "guard_glass"
    posts = []

    def run(ax: str, fixed: float, lo: float, hi: float, skip=(), gate=None) -> None:
        n = round((hi - lo) / w)
        for i in range(n):
            a, b = lo + i * w, lo + (i + 1) * w
            mid = (a + b) / 2
            posts.append((fixed, a) if ax == "y" else (a, fixed))
            if i in skip:
                continue
            if ax == "y":
                pos, half = [fixed, mid, h / 2], [0.003, w / 2 - 0.008, h / 2 - 0.012]
            else:
                pos, half = [mid, fixed, h / 2], [w / 2 - 0.008, 0.003, h / 2 - 0.012]
            if i == gate:
                _box(world, pos, half, None, world_mat)
                t = 0.008  # dark gate frame: four bars round the glass, and a handle
                if ax == "y":
                    ex, ez = [0.006, half[1], t], [0.006, t, half[2]]
                    for dz in (-half[2], half[2]):
                        _box(world, [pos[0], pos[1], pos[2] + dz], ex, None, "guard_frame")
                    for dy in (-half[1], half[1]):
                        _box(world, [pos[0], pos[1] + dy, pos[2]], ez, None, "guard_frame")
                    _box(world, [pos[0] - 0.012, pos[1] + half[1] - 0.04, pos[2]], [0.006, 0.006, 0.03], None, "guard_frame")
                else:
                    ex, ez = [half[0], 0.006, t], [t, 0.006, half[2]]
                    for dz in (-half[2], half[2]):
                        _box(world, [pos[0], pos[1], pos[2] + dz], ex, None, "guard_frame")
                    for dx in (-half[0], half[0]):
                        _box(world, [pos[0] + dx, pos[1], pos[2]], ez, None, "guard_frame")
                    _box(world, [pos[0] + half[0] - 0.04, pos[1] - 0.012, pos[2]], [0.006, 0.006, 0.03], None, "guard_frame")
            else:
                _box(world, pos, half, None, world_mat)
        posts.append((fixed, hi) if ax == "y" else (hi, fixed))
        mid_e = (lo + hi) / 2
        rail = [fixed, mid_e, h] if ax == "y" else [mid_e, fixed, h]
        rh = [0.008, (hi - lo) / 2, 0.008] if ax == "y" else [(hi - lo) / 2, 0.008, 0.008]
        _box(world, rail, rh, STEEL, "alu")

    bx_open = (0, 1, 2)  # side panels: x 0.05..0.35 holds the belt, the one at index 1
    run("y", x0, y0, y1)  # back
    run("y", x1, y0, y1, gate=3)  # front, with its gate
    run("x", y0, x0, x1, skip=(1,))  # near end (belt opening)
    run("x", y1, x0, x1, skip=(1,))  # far end (belt opening)
    for px, py in set(posts):
        _box(world, [px, py, h / 2], [0.01, 0.01, h / 2], STEEL, "alu")
    del gx, gy, spec_mat, bx_open


def _add_belt_items(spec: mujoco.MjSpec, assets: _Assets, cell: dict) -> None:
    """Cassette players riding the belt: unassembled upstream of the cells, finished downstream (visual only)."""
    items = (cell.get("frame") or {}).get("belt_items")
    belt = next((b for b in spec.bodies if b.name == "belt"), None)
    if not items or belt is None:
        return
    desc = cell["product"]
    sc = desc["scale"]
    parts = json.loads((LIB / "product.json").read_text())["parts"]
    for i, it in enumerate(items):
        holder = belt.add_body(
            name=f"belt_item{i}",
            pos=[0.0, it["y"], 0.003 + 0.0015 + it.get("z", 0.0)],
            quat=_quat_z(math.radians(it.get("yaw", 0.0))),
        )
        for p in parts:
            off = np.array(p["offset"], float)
            if not it.get("done", True):
                off = off + np.array(
                    {"cassette": desc["cassette_lift"], "lid": desc["lid_lift"]}.get(
                        p["role"], [0, 0, 0]
                    ),
                    float,
                )
            body = holder.add_body(name=f"belt_item{i}_{p['role']}", pos=list(off * sc))
            assets.attach(body, p["id"], sc)


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
