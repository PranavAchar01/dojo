"""Turn a source model (glTF, glb, DAE, OBJ, STL) into the library's form, for catalog models and user imports alike.

A library entry is a folder:

    model.glb          web viewer file (glTF Y-up, metres, base on y = 0, PBR kept, textures capped at 1024 px)
    model.xml          MuJoCo MJCF fragment (`<mujoco>` with its own assets): textured visual meshes (group 2, no
                       contacts) plus convex collision hulls (group 3) and, for dynamic models, a free joint
    visual/*.obj|png   the visual meshes and their base-colour textures (PNG: MuJoCo reads no JPEG)
    collision/*.obj    the convex hulls from the decomposition
    thumb.png          rendered once by scripts/assets_thumbs.mjs (three.js, same look as the viewer)
    meta.json          size, mass, hull count, faces, and the conversion settings that were used

Pipeline: load (Z-up, metres) -> normalise (scale, centre in xy, base on z = 0) -> bake the scene graph into pieces
-> GLB from the pieces -> MJCF visuals -> convex decomposition (CoACD, one bounded pass, with a hull fallback).

Nothing in a source file is executed: the loaders only parse geometry and images.
"""

from __future__ import annotations

import json
import math
import multiprocessing as mp
import queue
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import trimesh
from PIL import Image

TEX_PX = 256  # MuJoCo visual texture size
GLB_TEX_PX = 1024
MAX_COLLISION_FACES = 6000
DECOMP_TIMEOUT_S = 150
Y_TO_Z = trimesh.transformations.rotation_matrix(math.pi / 2, [1, 0, 0])
Z_TO_Y = np.linalg.inv(Y_TO_Z)

MIN_HULL_VOLUME_M3 = 2e-7
DEFAULT_DENSITY = 500.0  # kg/m3 of the hull volume for a model that gives none


@dataclass
class Settings:
    kind: str = "dynamic"  # "static" fixtures get no joint; "dynamic" ones a free joint
    density: float = DEFAULT_DENSITY
    max_hulls: int = 10
    target_longest_m: float | None = (
        None  # rescale so the longest side is about this (unit-ambiguous sources)
    )
    unit_scale: float | None = None  # explicit scale (user import "units" choice)
    y_up: bool | None = None  # None: glTF/glb are Y-up, everything else Z-up
    friction: float = 0.8

    def to_json(self) -> dict:
        return {k: v for k, v in self.__dict__.items() if v is not None}


# ---------------------------------------------------------------- loading

_MESH_EXT = (".gltf", ".glb", ".dae", ".obj", ".stl")


def _load_dae_raw(path: Path) -> trimesh.Scene:
    """Some Collada files hold geometry that trimesh's scene walk misses (no resolvable scene nodes). Read the
    geometry library directly: triangles only, the file's unit applied, no textures."""
    import collada

    c = collada.Collada(str(path), ignore=[collada.DaeError])
    unit = float(c.assetInfo.unitmeter or 1.0) if c.assetInfo else 1.0
    y_up = c.assetInfo is not None and c.assetInfo.upaxis == "Y_UP"
    sc = trimesh.Scene()
    for g in c.geometries:
        for prim in g.primitives:
            try:
                tris = prim.triangleset() if hasattr(prim, "triangleset") else prim
                v = np.asarray(tris.vertex, dtype=float) * unit
                idx = np.asarray(tris.vertex_index, dtype=int).reshape(-1, 3)
            except (ValueError, TypeError, AttributeError, IndexError):
                continue
            m = trimesh.Trimesh(v, idx, process=False)
            if y_up:
                m.apply_transform(Y_TO_Z)
            sc.add_geometry(m)
    return sc


def _load_one(path: Path) -> trimesh.Scene:
    from trimesh.resolvers import FilePathResolver

    kw = {}
    if path.suffix.lower() in (".dae", ".obj", ".gltf"):
        kw["resolver"] = FilePathResolver(path.parent, allow_anywhere=True)
    try:
        sc = trimesh.load(str(path), force="scene", **kw)
    except Exception:  # noqa: BLE001 - any parser failure means "try the raw reader / report no geometry"
        sc = trimesh.Scene()
    if not sc.geometry and path.suffix.lower() == ".dae":
        sc = _load_dae_raw(path)
    if not sc.geometry:
        raise ValueError(f"no geometry in {path.name}")
    return sc


def find_meshes(folder: Path) -> list[Path]:
    """The visual mesh files of a downloaded model folder, best format first, collision meshes skipped."""
    files = [
        f
        for f in folder.rglob("*")
        if f.suffix.lower() in _MESH_EXT
        and "collision" not in f.name.lower()
        and "thumbnail" not in str(f).lower()
    ]
    for exts in ((".gltf",), (".glb",), (".dae",), (".obj",), (".stl",)):
        pick = sorted(f for f in files if f.suffix.lower() in exts)
        if pick:
            return pick
    return []


def load_files(paths: list[Path], y_up: bool | None = None) -> trimesh.Scene:
    """Geometry of the given mesh files as one scene of baked pieces, Z-up."""
    if not paths:
        raise ValueError("no mesh file found")
    scene = trimesh.Scene()
    for p in paths:
        sc = _load_one(p)
        up = y_up if y_up is not None else p.suffix.lower() in (".gltf", ".glb")
        for m in sc.dump(concatenate=False):
            if up:
                m.apply_transform(Y_TO_Z)
            scene.add_geometry(m)
    return scene


def load_folder(folder: Path, y_up: bool | None = None) -> trimesh.Scene:
    return load_files(find_meshes(folder), y_up)


# ---------------------------------------------------------------- normalising


def _snap_power_of_ten(r: float) -> float:
    p = 10 ** round(math.log10(r))
    return p if abs(r / p - 1) < 0.3 else r


def normalise(
    scene: trimesh.Scene, s: Settings, known_dims_m: list[float] | None = None
) -> list[trimesh.Trimesh]:
    """Pieces in metres, footprint centred on the origin, lowest point on z = 0.

    Scale comes from, in order: an explicit unit scale; the source's own published dimensions (Poly Haven);
    a hinted longest side, snapped to a power of ten when close (cm / mm / inch-sized exports)."""
    pieces = [g.copy() for g in scene.geometry.values() if len(g.faces)]
    if not pieces:
        raise ValueError("model has no faces")
    lo = np.min([p.bounds[0] for p in pieces], axis=0)
    hi = np.max([p.bounds[1] for p in pieces], axis=0)
    ext = hi - lo
    scale = 1.0
    if s.unit_scale:
        scale = s.unit_scale
    elif known_dims_m:
        a, b = sorted(known_dims_m), sorted(float(x) for x in ext)
        r = float(np.median([x / y for x, y in zip(a, b) if y > 1e-9]))
        scale = _snap_power_of_ten(r) if abs(r - 1) > 0.05 else 1.0
    elif s.target_longest_m:
        r = s.target_longest_m / float(ext.max())
        scale = 1.0 if 0.75 < r < 1.35 else _snap_power_of_ten(r)
    if scale != 1.0:
        for p in pieces:
            p.apply_scale(scale)
        lo, hi = lo * scale, hi * scale
    shift = -np.array([(lo[0] + hi[0]) / 2, (lo[1] + hi[1]) / 2, lo[2]])
    for p in pieces:
        p.apply_translation(shift)
    return pieces


# ---------------------------------------------------------------- materials and images


def _pil(img) -> Image.Image | None:
    return img if isinstance(img, Image.Image) else None


def _shrink(img: Image.Image, px: int) -> Image.Image:
    if max(img.size) > px:
        img = img.copy()
        img.thumbnail((px, px), Image.LANCZOS)
    return img


def _material_info(piece: trimesh.Trimesh) -> dict:
    """Base colour (0..1 rgba), base-colour image (or None) and metal/rough of a piece."""
    vis = piece.visual
    rgba = np.array([0.6, 0.6, 0.6, 1.0])
    img, metal, rough = None, 0.0, 0.7
    mat = getattr(vis, "material", None)
    if mat is not None:
        bc = getattr(mat, "baseColorFactor", None)
        if bc is not None:
            bc = np.asarray(bc, dtype=float)[:4]
            rgba = bc / 255.0 if bc.max() > 1.0 else bc
        elif getattr(mat, "diffuse", None) is not None:
            rgba = np.asarray(mat.diffuse, dtype=float)[:4] / 255.0
        img = _pil(getattr(mat, "baseColorTexture", None)) or _pil(
            getattr(mat, "image", None)
        )
        metal = float(getattr(mat, "metallicFactor", None) or 0.0)
        r = getattr(mat, "roughnessFactor", None)
        rough = float(r) if r is not None else 0.7
    elif vis.kind == "vertex" and len(vis.vertex_colors):
        rgba = np.asarray(vis.vertex_colors[0], dtype=float) / 255.0
    return {"rgba": rgba, "image": img, "metal": metal, "rough": rough}


def _has_uv(piece: trimesh.Trimesh) -> bool:
    return (
        piece.visual.kind == "texture" and getattr(piece.visual, "uv", None) is not None
    )


def _write_obj(path: Path, piece: trimesh.Trimesh, uv: bool) -> None:
    v, f = piece.vertices, piece.faces
    lines = [f"v {a:.6f} {b:.6f} {c:.6f}" for a, b, c in v]
    if uv:
        lines += [f"vt {a:.5f} {b:.5f}" for a, b in piece.visual.uv]
        lines += [f"f {a + 1}/{a + 1} {b + 1}/{b + 1} {c + 1}/{c + 1}" for a, b, c in f]
    else:
        lines += [f"f {a + 1} {b + 1} {c + 1}" for a, b, c in f]
    path.write_text("\n".join(lines) + "\n")


def _prepare_glb_images(pieces: list[trimesh.Trimesh]) -> None:
    """Keep the web file lean: cap texture size."""
    for p in pieces:
        mat = getattr(p.visual, "material", None)
        if mat is None or not hasattr(mat, "baseColorTexture"):
            continue
        for attr in (
            "baseColorTexture",
            "normalTexture",
            "metallicRoughnessTexture",
            "occlusionTexture",
            "emissiveTexture",
        ):
            im = _pil(getattr(mat, attr, None))
            if im is not None:
                setattr(mat, attr, _shrink(im, GLB_TEX_PX))


# ---------------------------------------------------------------- collision


def _decimate(mesh: trimesh.Trimesh, faces: int) -> trimesh.Trimesh:
    if len(mesh.faces) <= faces:
        return mesh
    try:
        return mesh.simplify_quadric_decimation(face_count=faces)
    except (ValueError, RuntimeError, ImportError):
        return mesh


def _hull_fallback(
    parts: list[trimesh.Trimesh], max_hulls: int
) -> list[trimesh.Trimesh]:
    """One hull per connected piece, the smallest merged together until max_hulls remain."""
    comps: list[trimesh.Trimesh] = []
    for m in parts:
        comps.extend(m.split(only_watertight=False))
    comps.sort(key=lambda c: -float(np.prod(c.extents)))
    keep, rest = comps[: max_hulls - 1], comps[max_hulls - 1 :]
    hulls = [c.convex_hull for c in keep]
    if rest:
        hulls.append(trimesh.util.concatenate(rest).convex_hull)
    return hulls or [trimesh.util.concatenate(parts).convex_hull]


def _coacd_worker(q, verts, faces, max_hulls, threshold) -> None:
    import coacd

    coacd.set_log_level("error")
    out = coacd.run_coacd(
        coacd.Mesh(verts, faces),
        threshold=threshold,
        max_convex_hull=max_hulls,
        preprocess_mode="auto",
        resolution=50,
        mcts_nodes=20,
        mcts_iterations=80,
        mcts_max_depth=3,
        seed=0,
        merge=True,
    )
    q.put([(np.asarray(v), np.asarray(f)) for v, f in out])


def decompose(
    pieces: list[trimesh.Trimesh], max_hulls: int, timeout: float = DECOMP_TIMEOUT_S
) -> tuple[list[trimesh.Trimesh], str]:
    """Convex decomposition: one bounded CoACD pass in a child process (killed past `timeout`). Falls back to one
    hull per connected piece when CoACD fails or runs out of time. Returns (hulls, method)."""
    plain = []
    for p in pieces:
        c = trimesh.Trimesh(p.vertices.copy(), p.faces.copy(), process=False)
        plain.append(c)
    merged = _decimate(trimesh.util.concatenate(plain), MAX_COLLISION_FACES)
    ctx = mp.get_context("spawn")
    q = ctx.Queue()
    proc = ctx.Process(
        target=_coacd_worker,
        args=(
            q,
            np.asarray(merged.vertices, dtype=np.float64),
            np.asarray(merged.faces, dtype=np.int32),
            max_hulls,
            0.06,
        ),
    )
    proc.start()
    method = "coacd"
    hulls: list[trimesh.Trimesh] = []
    deadline = time.monotonic() + timeout
    try:
        while True:
            try:
                res = q.get(timeout=1.0)
                hulls = [trimesh.Trimesh(v, f) for v, f in res if len(f) >= 4]
                break
            except queue.Empty:
                if not proc.is_alive() or time.monotonic() > deadline:
                    method = "hull-fallback"
                    break
    finally:
        if proc.is_alive():
            proc.terminate()
        proc.join(5)
    if not hulls:
        return _hull_fallback([merged], max_hulls), "hull-fallback"
    # Wire and sliver pieces give hulls MuJoCo rejects as having no volume; they add nothing to contact anyway.
    solid = [h for h in hulls if h.volume > MIN_HULL_VOLUME_M3]
    return (solid or [merged.convex_hull])[:max_hulls], method


# ---------------------------------------------------------------- the package


def _mjcf(model_id: str, visuals: list[dict], hulls: int, s: Settings) -> str:
    assets, geoms = [], []
    for i, v in enumerate(visuals):
        mesh = f"{model_id}_v{i}"
        assets.append(
            f'    <mesh name="{mesh}" file="visual/v{i}.obj" inertia="shell"/>'
        )
        if v["texture"]:
            assets.append(
                f'    <texture name="{mesh}_t" type="2d" file="visual/{v["texture"]}"/>'
            )
            assets.append(
                f'    <material name="{mesh}_m" texture="{mesh}_t" rgba="1 1 1 1" specular="{v["spec"]:.2f}" shininess="{v["shin"]:.2f}"/>'
            )
        else:
            r, g, b, a = v["rgba"]
            assets.append(
                f'    <material name="{mesh}_m" rgba="{r:.3f} {g:.3f} {b:.3f} {a:.3f}" specular="{v["spec"]:.2f}" shininess="{v["shin"]:.2f}"/>'
            )
        geoms.append(
            f'      <geom type="mesh" mesh="{mesh}" material="{mesh}_m" group="2" contype="0" conaffinity="0" density="0"/>'
        )
    for i in range(hulls):
        assets.append(
            f'    <mesh name="{model_id}_c{i}" file="collision/hull_{i}.obj"/>'
        )
        geoms.append(
            f'      <geom type="mesh" mesh="{model_id}_c{i}" group="3" density="{s.density:g}" '
            f'friction="{s.friction:g} 0.005 0.0001" rgba="0.7 0.2 0.2 0.3"/>'
        )
    joint = '      <freejoint name="root"/>\n' if s.kind == "dynamic" else ""
    return (
        f'<mujoco model="{model_id}">\n  <asset>\n'
        + "\n".join(assets)
        + f'\n  </asset>\n  <worldbody>\n    <body name="{model_id}">\n{joint}'
        + "\n".join(geoms)
        + "\n    </body>\n  </worldbody>\n</mujoco>\n"
    )


def build_package(
    pieces: list[trimesh.Trimesh], out: Path, model_id: str, s: Settings
) -> dict:
    """Write model.glb, model.xml, visual/, collision/ and return the measured facts (meta)."""
    out.mkdir(parents=True, exist_ok=True)
    (out / "visual").mkdir(exist_ok=True)
    (out / "collision").mkdir(exist_ok=True)

    web = [p.copy() for p in pieces]
    _prepare_glb_images(web)
    sc = trimesh.Scene()
    for i, p in enumerate(web):
        p.apply_transform(Z_TO_Y)
        sc.add_geometry(p, node_name=f"part{i}", geom_name=f"part{i}")
    (out / "model.glb").write_bytes(sc.export(file_type="glb"))

    visuals = []
    for i, p in enumerate(pieces):
        info = _material_info(p)
        uv = _has_uv(p)
        _write_obj(out / "visual" / f"v{i}.obj", p, uv)
        tex = None
        if uv and info["image"] is not None:
            src = info["image"]
            im = _shrink(src, TEX_PX).convert(
                "RGBA" if src.mode in ("RGBA", "LA") else "RGB"
            )
            tex = f"t{i}.png"
            im.save(out / "visual" / tex, optimize=True)
        visuals.append(
            {
                "texture": tex,
                "rgba": [float(x) for x in info["rgba"]],
                "spec": 0.15 + 0.5 * info["metal"],
                "shin": float(np.clip(1.0 - info["rough"], 0.05, 0.9)),
            }
        )

    hulls, method = decompose(pieces, s.max_hulls)
    for i, h in enumerate(hulls):
        _write_obj(out / "collision" / f"hull_{i}.obj", h, False)
    (out / "model.xml").write_text(_mjcf(model_id, visuals, len(hulls), s))

    ext = trimesh.util.concatenate(
        [trimesh.Trimesh(p.vertices, p.faces, process=False) for p in pieces]
    ).extents
    hull_vol = float(sum(max(h.volume, 0.0) for h in hulls))
    meta = {
        "size_m": [round(float(x), 4) for x in ext],
        "faces": int(sum(len(p.faces) for p in pieces)),
        "hulls": len(hulls),
        "collision": method,
        "mass_kg": round(hull_vol * s.density, 3) if s.kind == "dynamic" else None,
        "settings": s.to_json(),
    }
    (out / "meta.json").write_text(json.dumps(meta, indent=1))
    return meta


def check_mjcf(folder: Path) -> dict:
    """Compile model.xml in MuJoCo and report the body's real mass: the proof the entry is usable."""
    import mujoco

    m = mujoco.MjModel.from_xml_path(str(folder / "model.xml"))
    return {
        "mass_kg": round(float(m.body_mass.sum()), 3),
        "ngeom": int(m.ngeom),
        "nmesh": int(m.nmesh),
    }
