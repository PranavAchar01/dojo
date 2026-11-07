"""The curated list of real, openly licensed models the library is built from, and how to fetch them.

Nothing here is modelled by hand. Two upstream libraries are used:

* Poly Haven (https://polyhaven.com), all CC0. Files come from its public API (api.polyhaven.com), the 1k glTF
  with its textures.
* Gazebo Fuel (https://app.gazebosim.org), the open model server of the Gazebo project. Each model keeps its own
  licence (CC0 or CC BY 4.0 here) and author, both read from the model's own metadata. Only models whose licence
  allows reuse with attribution are listed; the non-commercial and no-derivatives ones were left out.

Each entry: id, source, upstream name, category, kind (static fixture or free-moving object) and a plain
description. Authors and licences are read from the upstream metadata at fetch time, not typed here.

Downloads are untrusted data: `fetch` writes each model into its own empty directory and only reads the files
(zip members are checked for path traversal and links). Nothing from a download is ever executed.
"""

from __future__ import annotations

import io
import json
import shutil
import stat
import urllib.parse
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
RAW = REPO / "assets" / "_raw"
UA = {"User-Agent": "dojo-asset-fetch/1.0"}
MAX_DOWNLOAD = 120 * 1024 * 1024


@dataclass(frozen=True)
class Pick:
    id: str
    source: str  # "polyhaven" | "fuel"
    upstream: str  # polyhaven asset id, or "Owner/Model name"
    category: str
    kind: str  # "static" | "dynamic"
    note: str
    density: float = 0.0  # kg/m3 for dynamic models, 0 = use the category default


PH, FU = "polyhaven", "fuel"
CATALOG: list[Pick] = [
    # conveyors
    Pick(
        "conveyor_module",
        FU,
        "Open-RMF/conveyor_block",
        "conveyor",
        "static",
        "Roller-belt conveyor module, steel frame and black belt",
    ),
    # pallets
    Pick(
        "euro_pallet",
        FU,
        "OpenRobotics/Euro pallet",
        "pallet",
        "dynamic",
        "EUR/EPAL wooden pallet, 1200 x 800 mm",
        150,
    ),
    Pick(
        "pallet_standard",
        FU,
        "OpenRobotics/Pallet_Standard",
        "pallet",
        "dynamic",
        "Standard wooden shipping pallet",
        150,
    ),
    # totes, bins, crates, boxes
    Pick(
        "grey_tote",
        FU,
        "OpenRobotics/Grey tote",
        "container",
        "dynamic",
        "Grey stackable plastic tote",
        120,
    ),
    Pick(
        "picking_bin",
        FU,
        "OpenRobotics/Picking_Bin",
        "container",
        "dynamic",
        "Warehouse picking bin",
        120,
    ),
    Pick(
        "large_crate",
        FU,
        "OpenRobotics/Large Crate",
        "container",
        "dynamic",
        "Large shipping crate",
        200,
    ),
    Pick(
        "plastic_crate_a",
        PH,
        "plastic_crate_01",
        "container",
        "dynamic",
        "Plastic stacking crate",
        120,
    ),
    Pick(
        "plastic_crate_b",
        PH,
        "plastic_crate_02",
        "container",
        "dynamic",
        "Plastic crate with handles",
        120,
    ),
    Pick(
        "plastic_crate_c",
        PH,
        "plastic_crate_03",
        "container",
        "dynamic",
        "Industrial plastic crate",
        120,
    ),
    Pick(
        "industrial_container",
        PH,
        "industrial_pastic_container",
        "container",
        "dynamic",
        "Industrial plastic storage container",
        120,
    ),
    Pick(
        "cardboard_box",
        PH,
        "cardboard_box_01",
        "container",
        "dynamic",
        "Cardboard shipping box",
        60,
    ),
    Pick(
        "military_crate",
        PH,
        "old_military_crate",
        "container",
        "dynamic",
        "Steel-banded crate",
        200,
    ),
    Pick(
        "wooden_crate",
        PH,
        "wooden_crate_01",
        "container",
        "dynamic",
        "Wooden shipping crate",
        200,
    ),
    Pick(
        "metal_trash_can",
        PH,
        "metal_trash_can",
        "container",
        "dynamic",
        "Steel waste bin",
        80,
    ),
    # shelving, racks, cabinets, benches
    Pick(
        "steel_shelves_a",
        PH,
        "steel_frame_shelves_01",
        "shelving",
        "static",
        "Steel frame storage shelves",
    ),
    Pick(
        "steel_shelves_b",
        PH,
        "steel_frame_shelves_02",
        "shelving",
        "static",
        "Steel frame shelves with boards",
    ),
    Pick(
        "steel_shelves_c",
        PH,
        "steel_frame_shelves_03",
        "shelving",
        "static",
        "Steel frame shelf unit",
    ),
    Pick(
        "metal_rack",
        PH,
        "worn_metal_rack",
        "shelving",
        "static",
        "Worn metal storage rack",
    ),
    Pick(
        "pallet_rack",
        FU,
        "OpenRobotics/Pallet_Rack",
        "shelving",
        "static",
        "Warehouse pallet racking bay",
    ),
    Pick(
        "picking_shelves",
        FU,
        "OpenRobotics/Picking_Shelves",
        "shelving",
        "static",
        "Warehouse picking shelves",
    ),
    Pick(
        "metal_cabinet",
        FU,
        "OpenRobotics/MetalCabinet",
        "shelving",
        "static",
        "Steel storage cabinet",
    ),
    Pick(
        "locker_bank",
        FU,
        "OpenRobotics/LockerBig",
        "shelving",
        "static",
        "Steel locker bank",
    ),
    Pick(
        "metal_office_desk",
        PH,
        "metal_office_desk",
        "workbench",
        "static",
        "Steel work desk",
    ),
    Pick(
        "tool_cart", PH, "tool_cart", "workbench", "dynamic", "Wheeled tool cart", 150
    ),
    Pick(
        "storage_cart",
        PH,
        "industrial_storage_cart",
        "workbench",
        "dynamic",
        "Industrial storage cart",
        150,
    ),
    Pick(
        "welding_cart",
        PH,
        "portable_welding_cart",
        "workbench",
        "static",
        "Portable welding cart",
    ),
    Pick(
        "packing_station",
        FU,
        "OpenRobotics/Packing_Station",
        "workbench",
        "static",
        "Packing station with table",
    ),
    Pick(
        "drawer_cabinet", PH, "drawer_cabinet", "workbench", "static", "Drawer cabinet"
    ),
    # machines, handling
    Pick(
        "hand_truck",
        PH,
        "hand_truck",
        "handling",
        "dynamic",
        "Two-wheel hand truck",
        100,
    ),
    Pick(
        "pallet_jack",
        FU,
        "OpenRobotics/Pallet_Jack",
        "handling",
        "static",
        "Manual pallet jack",
    ),
    Pick(
        "forklift",
        FU,
        "OpenRobotics/Forklift",
        "handling",
        "static",
        "Warehouse forklift",
    ),
    Pick("drill_press", PH, "drill_press_01", "machine", "static", "Bench drill press"),
    Pick("bench_vice", PH, "bench_vice_01", "machine", "static", "Bench vice"),
    Pick(
        "overhead_crane",
        PH,
        "overhead_crane",
        "machine",
        "static",
        "Overhead hoist crane",
    ),
    Pick(
        "generator",
        FU,
        "OpenRobotics/Generator",
        "machine",
        "static",
        "Portable generator",
    ),
    Pick("pump", FU, "OpenRobotics/Pump", "machine", "static", "Industrial pump"),
    # parts
    Pick(
        "gear_part",
        FU,
        "OpenRobotics/Gear Part",
        "part",
        "dynamic",
        "Machined gear (ARIAC)",
        7800,
    ),
    Pick(
        "pulley_part",
        FU,
        "OpenRobotics/Pulley Part",
        "part",
        "dynamic",
        "Machined pulley (ARIAC)",
        7800,
    ),
    Pick(
        "piston_rod_part",
        FU,
        "OpenRobotics/Piston Rod Part",
        "part",
        "dynamic",
        "Piston rod (ARIAC)",
        7800,
    ),
    Pick(
        "gasket_part",
        FU,
        "OpenRobotics/Gasket Part",
        "part",
        "dynamic",
        "Gasket (ARIAC)",
        1200,
    ),
    Pick(
        "circuit_board",
        PH,
        "circuit_board",
        "part",
        "dynamic",
        "Computer circuit board",
        1800,
    ),
    Pick(
        "cable_spool",
        FU,
        "OpenRobotics/Cable_Spool",
        "part",
        "dynamic",
        "Cable drum",
        400,
    ),
    # tools
    Pick(
        "adjustable_wrench",
        PH,
        "adjustable_wrench",
        "tool",
        "dynamic",
        "Adjustable wrench",
        7800,
    ),
    Pick(
        "combination_wrench",
        PH,
        "combination_wrench",
        "tool",
        "dynamic",
        "Combination wrench",
        7800,
    ),
    Pick("pliers", PH, "pliers", "tool", "dynamic", "Pliers", 7800),
    Pick("screwdriver", PH, "screwdriver", "tool", "dynamic", "Screwdriver", 2000),
    Pick(
        "ratchet_wrench",
        PH,
        "ratchet_wrench",
        "tool",
        "dynamic",
        "Ratchet wrench",
        7800,
    ),
    Pick("pipe_wrench", PH, "pipe_wrench", "tool", "dynamic", "Pipe wrench", 7800),
    Pick(
        "bolt_cutters", PH, "bolt_cutters_01", "tool", "dynamic", "Bolt cutters", 7800
    ),
    Pick(
        "measuring_tape",
        PH,
        "measuring_tape_01",
        "tool",
        "dynamic",
        "Tape measure",
        1200,
    ),
    Pick("metal_toolbox", PH, "metal_toolbox", "tool", "dynamic", "Steel toolbox", 300),
    Pick(
        "tool_chest",
        PH,
        "metal_tool_chest",
        "tool",
        "static",
        "Rolling steel tool chest",
    ),
    Pick("toolbox", FU, "OpenRobotics/Toolbox", "tool", "dynamic", "Red toolbox", 300),
    # lighting
    Pick(
        "fluorescent_light",
        PH,
        "mounted_fluorescent_lights",
        "lighting",
        "static",
        "Mounted fluorescent tube fixture",
    ),
    Pick(
        "caged_hanging_light",
        PH,
        "caged_hanging_light",
        "lighting",
        "static",
        "Caged hanging work light",
    ),
    Pick(
        "hanging_industrial_lamp",
        PH,
        "hanging_industrial_lamp",
        "lighting",
        "static",
        "Hanging industrial lamp",
    ),
    Pick(
        "caged_sconce",
        PH,
        "industrial_caged_sconce",
        "lighting",
        "static",
        "Caged wall light",
    ),
    Pick(
        "security_light",
        PH,
        "security_light",
        "lighting",
        "static",
        "Wall security floodlight",
    ),
    Pick(
        "tube_light",
        FU,
        "OpenRobotics/Tube Light",
        "lighting",
        "static",
        "Tube light fixture",
    ),
    # safety, structure, utilities
    Pick(
        "chainlink_fence",
        PH,
        "modular_chainlink_fence",
        "safety",
        "static",
        "Chain-link fence section",
    ),
    Pick(
        "jersey_barrier",
        FU,
        "OpenRobotics/Jersey Barrier",
        "safety",
        "static",
        "Concrete jersey barrier",
    ),
    Pick(
        "fire_extinguisher",
        FU,
        "OpenRobotics/FireExtinguisher",
        "safety",
        "dynamic",
        "Fire extinguisher",
        600,
    ),
    Pick(
        "wet_floor_sign",
        PH,
        "WetFloorSign_01",
        "safety",
        "dynamic",
        "Wet floor sign",
        150,
    ),
    Pick(
        "construction_cone",
        FU,
        "OpenRobotics/Construction Cone",
        "safety",
        "dynamic",
        "Traffic cone",
        250,
    ),
    Pick("fire_alarm", PH, "fire_alarm", "safety", "static", "Fire alarm pull station"),
    Pick(
        "security_camera",
        PH,
        "security_camera_01",
        "safety",
        "static",
        "Security camera",
    ),
    Pick("power_box", PH, "power_box_01", "utility", "static", "Electrical power box"),
    Pick("utility_box", PH, "utility_box_01", "utility", "static", "Utility cabinet"),
    Pick(
        "steel_drum",
        FU,
        "OpenRobotics/55gal_Drum",
        "utility",
        "dynamic",
        "55 gallon steel drum",
        80,
    ),
    Pick("oil_barrel", PH, "barrel_03", "utility", "dynamic", "Oil barrel", 80),
    Pick("propane_tank", PH, "propane_tank", "utility", "dynamic", "Propane tank", 250),
    Pick(
        "jerrycan",
        PH,
        "metal_jerrycan_green",
        "utility",
        "dynamic",
        "Metal jerrycan",
        150,
    ),
    Pick("cement_bag", PH, "cement_bag", "utility", "dynamic", "Cement bag", 1400),
    Pick(
        "airduct",
        PH,
        "modular_airduct_rectangular_01",
        "utility",
        "static",
        "Rectangular air duct run",
    ),
    Pick("pipes", PH, "modular_pipes", "utility", "static", "Modular steel pipework"),
    Pick(
        "rollershutter_door",
        PH,
        "rollershutter_door",
        "structure",
        "static",
        "Roller shutter door",
    ),
    Pick("ladder", FU, "OpenRobotics/Ladder_01", "structure", "static", "Step ladder"),
]
BY_ID = {p.id: p for p in CATALOG}


def _get(url: str, limit: int = MAX_DOWNLOAD) -> bytes:
    req = urllib.request.Request(url, headers=UA)
    with urllib.request.urlopen(req, timeout=90) as r:
        data = r.read(limit + 1)
    if len(data) > limit:
        raise ValueError(f"download over {limit} bytes: {url}")
    return data


def safe_extract(zf: zipfile.ZipFile, dest: Path) -> None:
    """Write zip members under dest. Absolute paths, '..' and links are refused; nothing is executed."""
    root = dest.resolve()
    for info in zf.infolist():
        if info.is_dir():
            continue
        mode = info.external_attr >> 16
        if stat.S_ISLNK(mode):
            raise ValueError(f"link in archive: {info.filename}")
        target = (dest / info.filename).resolve()
        if root != target and root not in target.parents:
            raise ValueError(f"path escapes the archive folder: {info.filename}")
        target.parent.mkdir(parents=True, exist_ok=True)
        with zf.open(info) as src, open(target, "wb") as out:
            shutil.copyfileobj(src, out)


def fetch_polyhaven(p: Pick, dest: Path) -> dict:
    base = "https://api.polyhaven.com"
    info = json.loads(_get(f"{base}/info/{p.upstream}"))
    files = json.loads(_get(f"{base}/files/{p.upstream}"))["gltf"]["1k"]["gltf"]
    (dest / "textures").mkdir(parents=True, exist_ok=True)
    (dest / Path(files["url"]).name).write_bytes(_get(files["url"]))
    for rel, f in files["include"].items():
        out = (dest / rel).resolve()
        if dest.resolve() not in out.parents:
            raise ValueError(rel)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_bytes(_get(f["url"]))
    authors = ", ".join(info.get("authors", {}).keys()) or "Poly Haven"
    return {
        "name": info["name"],
        "author": authors,
        "license": "CC0-1.0",
        "license_url": "https://creativecommons.org/publicdomain/zero/1.0/",
        "source_name": "Poly Haven",
        "source_url": f"https://polyhaven.com/a/{p.upstream}",
        "dimensions_mm": info.get("dimensions"),
    }


_FUEL_LICENSES = {
    "Creative Commons Zero v1.0 Universal": (
        "CC0-1.0",
        "https://creativecommons.org/publicdomain/zero/1.0/",
    ),
    "Creative Commons Attribution 4.0 International": (
        "CC-BY-4.0",
        "https://creativecommons.org/licenses/by/4.0/",
    ),
}


def fetch_fuel(p: Pick, dest: Path) -> dict:
    owner, name = p.upstream.split("/", 1)
    q = urllib.parse.quote
    meta = json.loads(
        _get(f"https://fuel.gazebosim.org/1.0/{q(owner)}/models/{q(name)}")
    )
    lic = _FUEL_LICENSES.get(meta.get("license_name", ""))
    if lic is None:
        raise ValueError(
            f"{p.id}: licence {meta.get('license_name')!r} is not on the allowed list"
        )
    blob = _get(f"https://fuel.gazebosim.org/1.0/{q(owner)}/models/{q(name)}.zip")
    with zipfile.ZipFile(io.BytesIO(blob)) as zf:
        safe_extract(zf, dest)
    author = owner
    cfg = dest / "model.config"
    if cfg.exists():
        import xml.etree.ElementTree as ET

        try:
            a = ET.parse(cfg).getroot().find("author/name")
            if a is not None and a.text and a.text.strip():
                author = f"{a.text.strip()} ({owner} on Gazebo Fuel)"
        except ET.ParseError:
            pass
    return {
        "name": meta["name"],
        "author": author,
        "license": lic[0],
        "license_url": lic[1],
        "source_name": "Gazebo Fuel",
        "source_url": f"https://app.gazebosim.org/{q(owner)}/fuel/models/{q(name)}",
        "dimensions_mm": None,
    }


def fetch(p: Pick, force: bool = False) -> Path:
    """Download one model into assets/_raw/<id>/ (a new empty folder) and write its provenance there."""
    dest = RAW / p.id
    if (dest / "provenance.json").exists() and not force:
        return dest
    if dest.exists():
        shutil.rmtree(dest)
    dest.mkdir(parents=True)
    prov = (fetch_polyhaven if p.source == PH else fetch_fuel)(p, dest)
    prov["upstream"] = p.upstream
    (dest / "provenance.json").write_text(json.dumps(prov, indent=1))
    return dest
