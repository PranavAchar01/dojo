"""Upload validation (the boundary) and one real conversion. Fixtures are generated here, not downloaded."""

import io
import json
import struct
import zipfile

import numpy as np
import pytest
import trimesh

from vla_forge import assets_import as imp


def stl_bytes() -> bytes:
    return trimesh.creation.box(extents=(0.1, 0.05, 0.03)).export(file_type="stl")


def glb_bytes() -> bytes:
    return trimesh.Scene(trimesh.creation.box(extents=(0.2, 0.1, 0.1))).export(
        file_type="glb"
    )


def obj_bytes() -> bytes:
    return trimesh.creation.box(extents=(2, 1, 1)).export(file_type="obj").encode()


def make_zip(files: dict[str, bytes], links: dict[str, str] | None = None) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for name, data in files.items():
            zf.writestr(name, data)
        for name, target in (links or {}).items():
            info = zipfile.ZipInfo(name)
            info.external_attr = 0o120777 << 16
            zf.writestr(info, target)
    return buf.getvalue()


def test_accepts_the_formats():
    assert imp.validate_upload("a.stl", stl_bytes()) == ".stl"
    assert imp.validate_upload("a.glb", glb_bytes()) == ".glb"
    assert imp.validate_upload("a.obj", obj_bytes()) == ".obj"
    ascii_stl = b"solid x\nfacet normal 0 0 1\nouter loop\nvertex 0 0 0\nvertex 1 0 0\nvertex 0 1 0\nendloop\nendfacet\nendsolid x\n"
    assert imp.validate_upload("a.stl", ascii_stl) == ".stl"
    assert imp.validate_upload("b.zip", make_zip({"m/part.stl": stl_bytes()})) == ".zip"


def test_size_cap():
    with pytest.raises(imp.ImportRejected, match="over 50 MB"):
        imp.validate_upload("big.stl", b"solid " + b"x" * (imp.MAX_UPLOAD + 1))
    with pytest.raises(imp.ImportRejected, match="empty"):
        imp.validate_upload("a.glb", b"")


@pytest.mark.parametrize(
    "name", ["a.exe", "a.py", "a.sh", "a.txt", "a", "a.glb.exe", "a.dylib"]
)
def test_extension_allow_list(name):
    with pytest.raises(imp.ImportRejected, match="supported files"):
        imp.validate_upload(name, glb_bytes())


def test_magic_bytes_must_match_extension():
    with pytest.raises(imp.ImportRejected):
        imp.validate_upload("a.glb", stl_bytes())
    with pytest.raises(imp.ImportRejected):
        imp.validate_upload(
            "a.stl", b"MZ\x90\x00" + b"\x00" * 200
        )  # a Windows executable named .stl
    with pytest.raises(imp.ImportRejected):
        imp.validate_upload("a.obj", bytes(range(256)) * 4)
    with pytest.raises(imp.ImportRejected):
        imp.validate_upload("a.zip", glb_bytes())
    with pytest.raises(imp.ImportRejected):
        imp.validate_upload("a.gltf", b"not json at all")


def test_gltf_external_references_refused():
    doc = {
        "asset": {"version": "2.0"},
        "buffers": [{"uri": "http://evil.example/x.bin", "byteLength": 4}],
    }
    with pytest.raises(imp.ImportRejected, match="other files"):
        imp.validate_upload("a.gltf", json.dumps(doc).encode())
    ok = {
        "asset": {"version": "2.0"},
        "buffers": [
            {"uri": "data:application/octet-stream;base64,AAAA", "byteLength": 3}
        ],
    }
    assert imp.validate_upload("a.gltf", json.dumps(ok).encode()) == ".gltf"


@pytest.mark.parametrize(
    "member",
    [
        "../evil.stl",
        "/etc/passwd.stl",
        "a/../../b.stl",
        "..\\x.stl",
        "C:/x.stl",
        ".hidden/x.stl",
        "a/./b.stl",
    ],
)
def test_zip_path_traversal_refused(member):
    with pytest.raises(imp.ImportRejected, match="unsafe path"):
        imp.validate_upload("m.zip", make_zip({member: stl_bytes()}))


def test_zip_links_refused():
    z = make_zip({"ok.stl": stl_bytes()}, links={"link.stl": "/etc/passwd"})
    with pytest.raises(imp.ImportRejected, match="links"):
        imp.validate_upload("m.zip", z)


@pytest.mark.parametrize(
    "name", ["run.py", "install.sh", "lib.so", "tool.exe", "x.js", "x.dylib", "noext"]
)
def test_zip_scripts_and_binaries_refused(name):
    z = make_zip({"part.stl": stl_bytes(), name: b"print('hi')"})
    with pytest.raises(imp.ImportRejected, match="not allowed in the zip"):
        imp.validate_upload("m.zip", z)


def test_zip_needs_a_mesh_and_a_valid_one():
    with pytest.raises(imp.ImportRejected, match="no mesh"):
        imp.validate_upload("m.zip", make_zip({"tex.png": b"\x89PNG\r\n\x1a\n"}))
    with pytest.raises(imp.ImportRejected):
        imp.validate_upload("m.zip", make_zip({"part.stl": b"definitely not an stl"}))
    with pytest.raises(imp.ImportRejected, match="valid zip"):
        imp.validate_upload("m.zip", b"PK\x03\x04" + b"garbage" * 20)


def test_zip_bombs_and_member_limits():
    many = {f"f{i}.txt": b"x" for i in range(imp.MAX_MEMBERS + 1)}
    many["m.stl"] = stl_bytes()
    with pytest.raises(imp.ImportRejected, match="more than"):
        imp.validate_upload("m.zip", make_zip(many))
    big = make_zip({"m.stl": stl_bytes(), "pad.txt": b"\0" * (imp.MAX_UNPACKED + 1)})
    with pytest.raises(imp.ImportRejected, match="unpacks|compression"):
        imp.validate_upload("m.zip", big)


def test_xml_entities_refused():
    xml = b'<?xml version="1.0"?><!DOCTYPE x [<!ENTITY a "aaaa">]><robot name="&a;"/>'
    with pytest.raises(imp.ImportRejected, match="DOCTYPE"):
        imp.validate_upload("m.zip", make_zip({"part.stl": stl_bytes(), "r.urdf": xml}))
    ok = b'<?xml version="1.0"?><robot name="r"><link name="l"/></robot>'
    assert (
        imp.validate_upload("m.zip", make_zip({"part.stl": stl_bytes(), "r.urdf": ok}))
        == ".zip"
    )


def test_slug_and_units():
    assert imp.slug("My Part (v2).STL") == "my_part_v2"
    assert imp.slug("../../etc/passwd") == "passwd"
    assert (
        imp.auto_scale(0.3) == 1.0
        and imp.auto_scale(30) == 0.01
        and imp.auto_scale(3000) == 0.001
    )


def test_convert_stl_end_to_end(tmp_path, monkeypatch):
    """A 40 mm bracket exported in millimetres becomes a metre-sized, MuJoCo-loadable entry."""
    monkeypatch.setattr(imp, "USER_DIR", tmp_path)
    monkeypatch.setattr(imp, "USER_MANIFEST", tmp_path / "manifest.json")
    part = trimesh.creation.box(extents=(40, 20, 10)).export(file_type="stl")  # mm
    e = imp.convert_upload("bracket.stl", part, name="Bracket", units="mm")
    assert (
        e["origin"] == "user"
        and e["category"] == "mine"
        and e["source"]["name"] == "user import"
    )
    assert np.allclose(sorted(e["size_m"]), [0.01, 0.02, 0.04], atol=1e-4)
    folder = tmp_path / e["id"]
    for f in (
        "model.glb",
        "model.xml",
        "meta.json",
        "collision/hull_0.obj",
        "visual/v0.obj",
    ):
        assert (folder / f).exists()
    assert e["mass_kg"] and e["mass_kg"] > 0
    assert [m["id"] for m in imp.user_models()] == [e["id"]]
    assert not list((tmp_path / "_work").iterdir())  # the work folder is cleaned up
    assert imp.trash_user_model(e["id"]) and imp.user_models() == []
    assert (tmp_path / "_trash" / e["id"] / "model.xml").exists()  # moved, not erased


def test_rejected_upload_leaves_nothing(tmp_path, monkeypatch):
    monkeypatch.setattr(imp, "USER_DIR", tmp_path)
    monkeypatch.setattr(imp, "USER_MANIFEST", tmp_path / "manifest.json")
    with pytest.raises(imp.ImportRejected):
        imp.convert_upload("x.exe", b"MZ", name="x")
    assert imp.user_models() == [] and not any(tmp_path.glob("u_*"))


def test_struct_header_sanity():
    # a binary STL whose triangle count does not match its size is refused
    bad = b"\0" * 80 + struct.pack("<I", 5) + b"\0" * 10
    with pytest.raises(imp.ImportRejected):
        imp.validate_upload("a.stl", bad)
