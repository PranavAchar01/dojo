"""HTTP surface of the build plan (inline in the prompt window) (web/plan.js is its only client). Mounted by server.py with one line."""

from __future__ import annotations

import asyncio
import base64
import io
import sys
from functools import lru_cache

import numpy as np
from fastapi import APIRouter, HTTPException
from fastapi.responses import Response
from PIL import Image
from pydantic import BaseModel, Field

from . import hero_bricks as hb
from . import hero_muse as hm
from . import hero_rules as hr

router = APIRouter(prefix="/api/build")


class RulesIn(BaseModel):
    gap: int = 4
    pillar: int = Field(2)
    deck: int = 6
    tier: bool = True

    def rules(self) -> hb.Rules:
        return hb.Rules(self.gap, self.pillar, self.deck, self.tier)


class Row(BaseModel):
    name: str = ""
    part: str
    colour: str
    col: int
    row: int
    lvl: int
    yaw: int = 0
    role: str = ""
    source: str = "user"


class Plan(BaseModel):
    rules: RulesIn = RulesIn()
    bricks: list[Row]


class Fix(Plan):
    instruction: str = Field(min_length=3, max_length=400)


class Read(BaseModel):
    rules: RulesIn = RulesIn()
    image: str | None = (
        None  # base64 png or jpeg; empty reads the rendered instruction photo
    )


def _rows(plan: Plan) -> list[dict]:
    return [r.model_dump() for r in plan.bricks]


@lru_cache(maxsize=8)
def _photo(gap: int, pillar: int, deck: int, tier: bool) -> bytes:
    from . import hero_render as dr

    keep = hb.DUCK
    try:
        hb.set_assembly(hb.generate(hb.Rules(gap, pillar, deck, tier)))
        buf = io.BytesIO()
        img = dr.render(
            dr.scene(dr.target_poses(), size=768),
            dr.camera(az=-35, el=24, dist=0.15),
            768,
            768,
        )
        Image.fromarray(img).save(buf, "PNG")
        return buf.getvalue()
    finally:
        hb.set_assembly(keep)


@router.get("/state")
def state():
    ok, why = hm.available()
    active = hb.load_active()
    rules, bricks = active or (hb.Rules(), hb.generate())
    return {
        "rules": rules.as_dict(),
        "bricks": hb.to_json(bricks),
        "in_use": active is not None,
        "parts": {
            p: {"name": hb.PART_NAME[p], "footprint": hb.PART_FOOTPRINT[p]}
            for p in hb.PART_FOOTPRINT
        },
        "colours": {k: list(v) for k, v in hb.COLOURS.items()},
        "muse": {"model": "Muse Spark", "live": ok, "why": why},
        "attribution": hb.manifest()["attribution"],
        "licence": hb.manifest()["library_licence"],
        "settle": {"seconds": hr.SETTLE_S, "max_mm": hr.SETTLE_MAX_MM},
    }


@router.get("/mesh/{part}")
def mesh(part: str):
    if part not in hb.PART_FOOTPRINT:
        raise HTTPException(404, "unknown part")
    v, f = hb.part_mesh_cached(part)
    return {"v": np.round(v, 6).ravel().tolist(), "f": f.ravel().tolist()}


@router.get("/photo.png")
def photo(gap: int = 4, pillar: int = 2, deck: int = 6, tier: bool = True):
    bad = hb.Rules(gap, pillar, deck, tier).problems()
    if bad:
        raise HTTPException(422, "; ".join(bad))
    return Response(_photo(gap, pillar, deck, tier), media_type="image/png")


@router.post("/generate")
def generate(r: RulesIn):
    bad = r.rules().problems()
    if bad:
        raise HTTPException(422, "; ".join(bad))
    return {"bricks": hb.to_json(hb.generate(r.rules()))}


@router.post("/check")
async def check(plan: Plan):
    bricks, problems = hm.validate(_rows(plan), "user")
    rules = plan.rules.rules()
    if not bricks:
        return {"results": [], "problems": problems}
    results = await asyncio.to_thread(hr.check, bricks, rules)
    return {"results": hr.to_json(results), "problems": [] if results else problems}


@router.post("/read")
async def read(req: Read):
    ok, why = hm.available()
    if not ok:
        raise HTTPException(503, why)
    r = req.rules.rules()
    if req.image:
        try:
            raw = base64.b64decode(req.image.split(",")[-1], validate=False)
            Image.open(io.BytesIO(raw)).verify()
        except Exception as e:
            raise HTTPException(422, "that is not an image") from e
        png = raw
    else:
        png = _photo(r.gap, r.pillar, r.deck, r.tier)
    out = await asyncio.to_thread(hm.read_image, png)
    for s in out["reply"]["steps"]:
        s["source"] = "muse"
    bricks, problems = hm.validate(out["reply"]["steps"], "muse")
    return {
        "model": out["model"],
        "usage": out["usage"],
        "summary": out["reply"]["summary"],
        "bricks": hb.to_json(bricks),
        "problems": problems,
        "score": hm.score(bricks, hb.generate(r), r) if not req.image else None,
    }


@router.post("/fix")
async def fix(req: Fix):
    ok, why = hm.available()
    if not ok:
        raise HTTPException(503, why)
    rows = _rows(req)
    out = await asyncio.to_thread(hm.revise, rows, req.rules.rules(), req.instruction)
    if out["ok"]:
        out["diff"] = hm.diff(rows, out["bricks"])
    return out


@router.post("/use")
async def use(plan: Plan):
    bricks, problems = hm.validate(_rows(plan), "user")
    if problems:
        raise HTTPException(422, "; ".join(problems))
    rules = plan.rules.rules()
    hb.save_active(rules, bricks)
    return {"ok": True, "bricks": len(bricks), "table": "/api/build/table.png"}


@router.get("/table.png")
async def table():
    """One frame of the MuJoCo table with the build in use (a short subprocess, so the server stays light)."""
    p = await asyncio.create_subprocess_exec(
        "nice",
        "-n",
        "19",
        sys.executable,
        "-m",
        "vla_forge.hero_sim",
        "-",
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.DEVNULL,
    )
    out, _ = await p.communicate()
    if p.returncode != 0 or not out:
        raise HTTPException(500, "the table render failed")
    return Response(out, media_type="image/png")
