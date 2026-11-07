"""HTTP surface of the model library (web/assets.js is its client). Mounted by server.py with one line.

GET    /api/assets                 list / search (?q=, ?category=, ?origin=library|user, ?kind=static|dynamic)
GET    /api/assets/categories      category names with counts
GET    /api/assets/{id}            one entry with provenance
GET    /api/assets/{id}/glb        the web file
GET    /api/assets/{id}/thumb      the cached thumbnail (404 until scripts/assets_thumbs.mjs has run)
GET    /api/assets/{id}/mjcf       the MuJoCo MJCF text
POST   /api/assets/import-url      JSON {url, name, units, kind}: the server downloads a public link, then as below
POST   /api/assets/import          raw body = the file, X-Filename header, ?name= ?units= ?kind=
GET    /api/assets/import/{job}    conversion status
DELETE /api/assets/{id}            user models only: moved to assets/user/_trash/, not erased
"""

from __future__ import annotations

import asyncio
import ipaddress
import json
import socket
import threading
import urllib.request
from pathlib import Path
from urllib.parse import unquote, urlparse

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import FileResponse, PlainTextResponse

from . import assets_import as imp
from .assets_sources import REPO

router = APIRouter(prefix="/api/assets")
LIB = REPO / "assets" / "library"
JOBS: dict[str, imp.Job] = {}


def library() -> list[dict]:
    """Every entry: the sourced library first, then the user's imports."""
    out: list[dict] = []
    manifest = LIB / "manifest.json"
    if manifest.exists():
        out.extend(json.loads(manifest.read_text())["models"])
    products = LIB / "manifest_products.json"
    if products.exists():
        out.extend(json.loads(products.read_text())["models"])
    out.extend(imp.user_models())
    return out


def entry(model_id: str) -> dict:
    for e in library():
        if e["id"] == model_id:
            return e
    raise HTTPException(404, "no such model")


def folder(e: dict) -> Path:
    return (imp.USER_DIR if e["origin"] == "user" else LIB) / e["id"]


def search(
    items: list[dict], q: str = "", category: str = "", origin: str = "", kind: str = ""
) -> list[dict]:
    terms = [t for t in q.lower().split() if t]
    out = []
    for e in items:
        if category and e["category"] != category:
            continue
        if origin and e["origin"] != origin:
            continue
        if kind and e["kind"] != kind:
            continue
        hay = " ".join(
            [
                e["id"],
                e["name"],
                e["description"],
                e["category"],
                " ".join(e["tags"]),
                e["source"]["name"],
                e["source"]["author"],
            ]
        ).lower()
        if all(t in hay for t in terms):
            out.append(e)
    return out


def brief(e: dict) -> dict:
    f = folder(e)
    return {
        **e,
        "has_thumb": (f / "thumb.png").exists(),
        "glb": f"/api/assets/{e['id']}/glb",
        "thumb": f"/api/assets/{e['id']}/thumb",
    }


@router.get("")
def list_models(q: str = "", category: str = "", origin: str = "", kind: str = ""):
    return [brief(e) for e in search(library(), q, category, origin, kind)]


@router.get("/categories")
def categories():
    counts: dict[str, int] = {}
    for e in library():
        counts[e["category"]] = counts.get(e["category"], 0) + 1
    return [
        {"category": c, "count": n}
        for c, n in sorted(counts.items(), key=lambda kv: (kv[0] == "mine", kv[0]))
    ]


@router.post("/import")
async def import_model(
    request: Request, name: str = "", units: str = "auto", kind: str = "dynamic"
):
    filename = unquote(request.headers.get("x-filename", ""))
    if not filename:
        raise HTTPException(400, "send the file name in the X-Filename header")
    declared = request.headers.get("content-length")
    if declared and declared.isdigit() and int(declared) > imp.MAX_UPLOAD + 1024:
        raise HTTPException(
            413, f"the file is over {imp.MAX_UPLOAD // (1024 * 1024)} MB"
        )
    body = bytearray()
    async for chunk in request.stream():
        body.extend(chunk)
        if len(body) > imp.MAX_UPLOAD:
            raise HTTPException(
                413, f"the file is over {imp.MAX_UPLOAD // (1024 * 1024)} MB"
            )
    data = bytes(body)
    try:  # cheap checks answer at once; conversion runs in a worker thread
        imp.validate_upload(filename, data)
    except imp.ImportRejected as e:
        raise HTTPException(422, str(e)) from None
    return await _start(filename, data, name, units, kind)


async def _start(filename: str, data: bytes, name: str, units: str, kind: str) -> dict:
    job = imp.Job(id=f"j{len(JOBS) + 1}_{int(imp.time.time())}", filename=filename)
    JOBS[job.id] = job

    def run() -> None:
        try:
            imp.convert_upload(filename, data, name, units, kind, job)
        except imp.ImportRejected as e:
            job.state, job.message = "error", str(e)

    threading.Thread(target=run, daemon=True).start()
    await asyncio.sleep(0)
    return job.to_json()


def _public_host(host: str) -> bool:
    """True when every address the host resolves to is a public one (no loopback, LAN or link-local)."""
    try:
        infos = socket.getaddrinfo(host, None)
    except OSError:
        return False
    return bool(infos) and all(ipaddress.ip_address(i[4][0]).is_global for i in infos)


class _PublicRedirects(urllib.request.HTTPRedirectHandler):
    """Follow a redirect only when it stays on http(s) and a public host."""

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        parts = urlparse(newurl)
        if parts.scheme not in ("http", "https") or not _public_host(
            parts.hostname or ""
        ):
            raise imp.ImportRejected("that link redirects to a non-public address")
        return super().redirect_request(req, fp, code, msg, headers, newurl)


_OPENER = urllib.request.build_opener(_PublicRedirects)


def _fetch(url: str) -> tuple[str, bytes]:
    parts = urlparse(url)
    if parts.scheme not in ("http", "https") or not parts.hostname:
        raise imp.ImportRejected("use a web link that starts with http or https")
    if not _public_host(parts.hostname):
        raise imp.ImportRejected("that address is not a public web link")
    req = urllib.request.Request(url, headers={"User-Agent": "dojo-model-import"})
    try:
        with _OPENER.open(req, timeout=30) as r:
            data = r.read(imp.MAX_UPLOAD + 1)
    except OSError as e:
        raise imp.ImportRejected(f"could not download that link ({e})") from None
    if len(data) > imp.MAX_UPLOAD:
        raise imp.ImportRejected(
            f"the file is over {imp.MAX_UPLOAD // (1024 * 1024)} MB"
        )
    name = unquote(Path(urlparse(url).path).name) or "model"
    return name, data


@router.post("/import-url")
async def import_url(body: dict):
    url = str(body.get("url", "")).strip()
    try:
        filename, data = await asyncio.to_thread(_fetch, url)
        imp.validate_upload(filename, data)
    except imp.ImportRejected as e:
        raise HTTPException(422, str(e)) from None
    return await _start(
        filename,
        data,
        str(body.get("name", "")),
        str(body.get("units", "auto")),
        str(body.get("kind", "dynamic")),
    )


@router.get("/import/{job_id}")
def import_status(job_id: str):
    job = JOBS.get(job_id)
    if job is None:
        raise HTTPException(404, "no such import")
    out = job.to_json()
    if job.state == "done":
        out["model"] = brief(entry(job.model_id))
    return out


@router.get("/{model_id}")
def get_model(model_id: str):
    return brief(entry(model_id))


@router.get("/{model_id}/glb")
def get_glb(model_id: str):
    p = folder(entry(model_id)) / "model.glb"
    if not p.exists():
        raise HTTPException(404, "no web file")
    return FileResponse(
        p,
        media_type="model/gltf-binary",
        headers={"Cache-Control": "public, max-age=3600"},
    )


@router.get("/{model_id}/thumb")
def get_thumb(model_id: str):
    p = folder(entry(model_id)) / "thumb.png"
    if not p.exists():
        raise HTTPException(404, "no thumbnail yet")
    return FileResponse(
        p, media_type="image/png", headers={"Cache-Control": "public, max-age=86400"}
    )


@router.get("/{model_id}/mjcf")
def get_mjcf(model_id: str):
    p = folder(entry(model_id)) / "model.xml"
    return PlainTextResponse(p.read_text(), media_type="application/xml")


@router.delete("/{model_id}")
def delete_model(model_id: str):
    e = entry(model_id)
    if e["origin"] != "user":
        raise HTTPException(403, "library models are read-only")
    imp.trash_user_model(model_id)
    return {"ok": True}


factory_router = APIRouter()


@factory_router.get("/api/factory")
def factory(refresh: bool = False):
    """The factory line exactly as the engine produced it: the dressed two-station cell, the recorded episodes of the
    scripted expert and the success counts. Generated by assets_hero.run when no run is on disk yet."""
    from . import assets_hero

    f = assets_hero.OUT / "factory_hero.json"
    if refresh or not f.exists():
        assets_hero.run(3)
    return FileResponse(f, media_type="application/json")
