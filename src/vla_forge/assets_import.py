"""User model import: validate an uploaded file at the boundary, then convert it to the library's form.

Accepted: .glb, .gltf (self-contained), .obj, .stl, and a .zip holding any of those plus textures, or an MJCF /
URDF project (its mesh files are used; the XML is only read for mesh references, never executed).

The checks, in order, all before anything is parsed as geometry:

* size: at most MAX_UPLOAD bytes (50 MB); a zip at most MAX_UNPACKED bytes (150 MB) and MAX_MEMBERS files unpacked
* extension is on the allow list, and the leading bytes match it (glTF magic, a JSON document with an "asset" key,
  binary or ASCII STL, OBJ text with vertex lines, "PK" for a zip)
* zip members: no absolute paths, no `..`, no backslashes, no links, no hidden or device names, and only
  extensions on the allow list (meshes, images, mtl, bin, xml, urdf, json, txt). A script, binary or library
  anywhere in the archive rejects the whole upload
* glTF resources: only embedded `data:` URIs, or files in the same zip by plain relative name
* XML (MJCF/URDF): at most 5 MB and no DOCTYPE / ENTITY declarations

Nothing uploaded is ever executed or imported as code.
"""

from __future__ import annotations

import io
import json
import re
import secrets
import shutil
import struct
import threading
import time
import zipfile
from dataclasses import dataclass, field
from pathlib import PurePosixPath

from . import assets_convert as conv
from .assets_sources import REPO, safe_extract

MAX_UPLOAD = 50 * 1024 * 1024
MAX_UNPACKED = 150 * 1024 * 1024
MAX_MEMBERS = 400
MAX_FACES = 1_000_000
MAX_XML = 5 * 1024 * 1024

MESH_EXT = {".glb", ".gltf", ".obj", ".stl", ".dae"}
UPLOAD_EXT = {".glb", ".gltf", ".obj", ".stl", ".zip"}
ZIP_EXT = MESH_EXT | {
    ".png",
    ".jpg",
    ".jpeg",
    ".mtl",
    ".bin",
    ".xml",
    ".urdf",
    ".json",
    ".txt",
    ".webp",
    ".tga",
    ".bmp",
}
UNITS = {"m": 1.0, "cm": 0.01, "mm": 0.001, "in": 0.0254}

USER_DIR = REPO / "assets" / "user"
USER_MANIFEST = USER_DIR / "manifest.json"
_LOCK = threading.Lock()  # one conversion at a time: bounded memory and CPU


class ImportRejected(ValueError):
    """The upload failed validation. The message is safe to show the user."""


# ---------------------------------------------------------------- validation


def _ext(name: str) -> str:
    return PurePosixPath(name).suffix.lower()


def check_member_name(name: str) -> None:
    """One archive member path: relative, forward slashes, no parent steps, no hidden or odd names."""
    if (
        not name
        or name.startswith(("/", "\\"))
        or "\\" in name
        or "\x00" in name
        or re.match(r"^[A-Za-z]:", name)
    ):
        raise ImportRejected(f"unsafe path in the zip: {name!r}")
    parts = name.split("/")
    if any(p in ("", "..", ".") or p.startswith(".") for p in parts):
        raise ImportRejected(f"unsafe path in the zip: {name!r}")


def sniff_mesh(ext: str, head: bytes, size: int) -> None:
    """The first bytes of a mesh file must look like the format its extension claims."""
    if ext == ".glb":
        if (
            head[:4] != b"glTF"
            or len(head) < 12
            or struct.unpack("<I", head[4:8])[0] not in (1, 2)
        ):
            raise ImportRejected("the file does not start like a binary glTF (.glb)")
    elif ext == ".gltf":
        text = head.lstrip()[:1]
        if text != b"{":
            raise ImportRejected("the file does not look like glTF JSON")
    elif ext == ".stl":
        if head[:5].lower() == b"solid" and b"facet" in head[:2000].lower():
            return
        if len(head) >= 84:
            n = struct.unpack("<I", head[80:84])[0]
            if size == 84 + 50 * n and n > 0:
                return
        raise ImportRejected("the file is neither a binary nor an ASCII STL")
    elif ext == ".obj":
        try:
            text = head.decode("utf-8")
        except UnicodeDecodeError:
            raise ImportRejected("the OBJ file is not text") from None
        if not re.search(
            r"^v\s+[-+0-9.eE]+\s+[-+0-9.eE]+\s+[-+0-9.eE]+", text, re.MULTILINE
        ):
            raise ImportRejected("the OBJ file has no vertex lines at its start")
    elif ext == ".dae" and b"<COLLADA" not in head and b"<collada" not in head:
        raise ImportRejected("the file does not look like Collada (.dae)")


def _gltf_refs_ok(data: bytes, allow_files: bool) -> None:
    try:
        doc = json.loads(data)
    except (ValueError, UnicodeDecodeError):
        raise ImportRejected("the glTF file is not valid JSON") from None
    if not isinstance(doc, dict) or "asset" not in doc:
        raise ImportRejected("the glTF file has no asset block")
    for key in ("buffers", "images"):
        for item in doc.get(key, []):
            uri = item.get("uri") if isinstance(item, dict) else None
            if uri is None or uri.startswith("data:"):
                continue
            if not allow_files:
                raise ImportRejected(
                    "the glTF refers to other files: upload a .glb, or a zip with the files in it"
                )
            if (
                re.match(r"^[A-Za-z][A-Za-z0-9+.-]*:", uri)
                or ".." in uri
                or uri.startswith("/")
                or "\\" in uri
            ):
                raise ImportRejected(f"unsafe resource path in the glTF: {uri[:80]!r}")


def check_xml(data: bytes) -> None:
    if len(data) > MAX_XML:
        raise ImportRejected("an XML file in the zip is over 5 MB")
    low = data[:200000].lower()
    if b"<!doctype" in low or b"<!entity" in low:
        raise ImportRejected("XML with a DOCTYPE or ENTITY declaration is refused")


def validate_zip(data: bytes) -> list[str]:
    """Check an archive without extracting it. Returns the member names."""
    try:
        zf = zipfile.ZipFile(io.BytesIO(data))
    except zipfile.BadZipFile:
        raise ImportRejected("the file is not a valid zip") from None
    infos = [i for i in zf.infolist() if not i.is_dir()]
    if not infos:
        raise ImportRejected("the zip is empty")
    if len(infos) > MAX_MEMBERS:
        raise ImportRejected(f"the zip has more than {MAX_MEMBERS} files")
    if sum(i.file_size for i in infos) > MAX_UNPACKED:
        raise ImportRejected("the zip unpacks to more than 150 MB")
    names = []
    for i in infos:
        check_member_name(i.filename)
        if (i.external_attr >> 16) & 0o170000 == 0o120000:
            raise ImportRejected(f"links are not allowed in the zip: {i.filename!r}")
        ext = _ext(i.filename)
        if ext not in ZIP_EXT:
            raise ImportRejected(
                f"file type {ext or '(none)'} is not allowed in the zip ({i.filename})"
            )
        if (
            i.compress_size
            and i.file_size / max(i.compress_size, 1) > 200
            and i.file_size > 5_000_000
        ):
            raise ImportRejected(f"suspicious compression ratio: {i.filename!r}")
        with zf.open(i) as fh:
            head = fh.read(4096)
        if ext in MESH_EXT:
            sniff_mesh(ext, head, i.file_size)
        if ext == ".gltf":
            with zf.open(i) as fh:
                _gltf_refs_ok(fh.read(MAX_XML), allow_files=True)
        if ext in (".xml", ".urdf"):
            with zf.open(i) as fh:
                check_xml(fh.read(MAX_XML + 1))
        names.append(i.filename)
    if not any(_ext(n) in MESH_EXT for n in names):
        raise ImportRejected(
            "the zip holds no mesh file (.glb, .gltf, .obj, .stl or .dae)"
        )
    return names


def validate_upload(filename: str, data: bytes) -> str:
    """Raise ImportRejected, or return the extension of an acceptable upload."""
    if len(data) == 0:
        raise ImportRejected("the file is empty")
    if len(data) > MAX_UPLOAD:
        raise ImportRejected(f"the file is over {MAX_UPLOAD // (1024 * 1024)} MB")
    name = PurePosixPath(filename.replace("\\", "/")).name
    ext = _ext(name)
    if ext not in UPLOAD_EXT:
        raise ImportRejected(
            "supported files: .glb, .gltf, .obj, .stl, or a .zip with a model (or an MJCF / URDF project)"
        )
    if ext == ".zip":
        if data[:4] != b"PK\x03\x04":
            raise ImportRejected("the file does not start like a zip")
        validate_zip(data)
    else:
        sniff_mesh(ext, data[:4096], len(data))
        if ext == ".gltf":
            _gltf_refs_ok(data, allow_files=False)
    return ext


# ---------------------------------------------------------------- conversion


def slug(name: str) -> str:
    s = re.sub(
        r"[^a-z0-9]+", "_", PurePosixPath(name.replace("\\", "/")).stem.lower()
    ).strip("_")
    return s[:40] or "model"


def auto_scale(longest: float) -> float:
    """Unitless meshes (OBJ, STL): guess metres, centimetres or millimetres from the longest side."""
    if longest <= 5:
        return 1.0
    if longest <= 200:
        return 0.01
    return 0.001


@dataclass
class Job:
    id: str
    filename: str
    state: str = "queued"  # queued | converting | done | error
    message: str = ""
    model_id: str = ""
    started: float = field(default_factory=time.time)

    def to_json(self) -> dict:
        return {
            "id": self.id,
            "filename": self.filename,
            "state": self.state,
            "message": self.message,
            "model_id": self.model_id,
        }


def _user_entries() -> list[dict]:
    if USER_MANIFEST.exists():
        return json.loads(USER_MANIFEST.read_text()).get("models", [])
    return []


def user_models() -> list[dict]:
    return _user_entries()


def _save_user(entries: list[dict]) -> None:
    USER_DIR.mkdir(parents=True, exist_ok=True)
    USER_MANIFEST.write_text(json.dumps({"version": 1, "models": entries}, indent=1))


def convert_upload(
    filename: str,
    data: bytes,
    name: str = "",
    units: str = "auto",
    kind: str = "dynamic",
    job: Job | None = None,
) -> dict:
    """Validate and convert one upload into assets/user/<id>/. Returns its manifest entry. Raises ImportRejected."""
    if units not in UNITS and units != "auto":
        raise ImportRejected("units must be auto, m, cm, mm or in")
    if kind not in ("static", "dynamic"):
        raise ImportRejected("kind must be static or dynamic")
    ext = validate_upload(filename, data)
    model_id = f"u_{slug(name or filename)}_{secrets.token_hex(3)}"
    work = USER_DIR / "_work" / model_id
    work.mkdir(parents=True)
    try:
        if job:
            job.state = "converting"
        if ext == ".zip":
            with zipfile.ZipFile(io.BytesIO(data)) as zf:
                safe_extract(zf, work)
            meshes = [p for p in conv.find_meshes(work)]
            if not meshes:
                raise ImportRejected("no usable mesh in the zip")
            y_up = None
        else:
            path = work / f"upload{ext}"
            path.write_bytes(data)
            meshes, y_up = [path], None
        with _LOCK:
            scene = conv.load_files(meshes, y_up)
            if sum(len(g.faces) for g in scene.geometry.values()) > MAX_FACES:
                raise ImportRejected(
                    "the model has more than a million faces; reduce it first"
                )
            s = conv.Settings(kind=kind, max_hulls=8 if kind == "dynamic" else 14)
            if units != "auto":
                s.unit_scale = UNITS[units]
            elif ext in (".glb", ".gltf"):
                s.unit_scale = 1.0
            else:
                pieces0 = [g for g in scene.geometry.values() if len(g.faces)]
                longest = (
                    max(float(g.extents.max()) for g in pieces0) if pieces0 else 1.0
                )
                lo = min(g.bounds[0] for g in pieces0)
                hi = max(g.bounds[1] for g in pieces0)
                longest = float((hi - lo).max())
                s.unit_scale = auto_scale(longest)
            pieces = conv.normalise(scene, s)
            out = USER_DIR / model_id
            meta = conv.build_package(pieces, out, model_id, s)
            meta["mujoco"] = conv.check_mjcf(out)
    except ImportRejected:
        shutil.rmtree(USER_DIR / model_id, ignore_errors=True)
        raise
    except Exception as e:
        shutil.rmtree(USER_DIR / model_id, ignore_errors=True)
        raise ImportRejected(f"could not convert this model: {type(e).__name__}") from e
    finally:
        shutil.rmtree(work, ignore_errors=True)

    entry = {
        "id": model_id,
        "name": (name or PurePosixPath(filename).stem)[:60],
        "category": "mine",
        "kind": kind,
        "description": f"Imported from {PurePosixPath(filename).name}",
        "tags": ["mine", "import", kind],
        "origin": "user",
        "source": {
            "name": "user import",
            "url": "",
            "upstream": PurePosixPath(filename.replace("\\", "/")).name[:120],
            "author": "uploaded by you",
            "license": "user supplied",
            "license_url": "",
        },
        "size_m": meta["size_m"],
        "mass_kg": meta["mujoco"]["mass_kg"] if kind == "dynamic" else None,
        "faces": meta["faces"],
        "hulls": meta["hulls"],
        "collision": meta["collision"],
        "units": units,
    }
    with _LOCK:
        entries = _user_entries()
        entries.append(entry)
        _save_user(entries)
    if job:
        job.state, job.model_id = "done", model_id
    return entry


def trash_user_model(model_id: str) -> bool:
    """Take a user model out of the library (moved to assets/user/_trash/, not erased)."""
    with _LOCK:
        entries = _user_entries()
        keep = [e for e in entries if e["id"] != model_id]
        if len(keep) == len(entries):
            return False
        src = USER_DIR / model_id
        if src.exists():
            (USER_DIR / "_trash").mkdir(exist_ok=True)
            shutil.move(str(src), str(USER_DIR / "_trash" / model_id))
        _save_user(keep)
    return True
