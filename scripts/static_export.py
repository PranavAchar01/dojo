"""Export a recorded static snapshot of the Dojo app into dist/ for Vercel.

Usage: python scripts/static_export.py [BASE_URL]   (default http://pranavs-mac-mini:8100)
GET JSON and media endpoints are saved under dist/_s/api; a fetch shim serves them
and turns anything that needs the live backend into an honest notice.
"""

from __future__ import annotations

import json
import re
import shutil
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import urlopen

ROOT = Path(__file__).resolve().parents[1]
DIST = ROOT / "dist"
BASE = (sys.argv[1] if len(sys.argv) > 1 else "http://pranavs-mac-mini:8100").rstrip(
    "/"
)
MEDIA = (".png", ".jpg", ".jpeg", ".glb", ".webp")
SKIP_DIRS = {"__pycache__"}

SHIM = """(function(){
var NOTE="Live training runs on the local server; this is a recorded snapshot.";
function note(){var d=document.getElementById("snap-note");if(!d){d=document.createElement("div");d.id="snap-note";
d.style.cssText="position:fixed;left:50%;bottom:16px;transform:translateX(-50%);z-index:99999;background:#222;color:#fff;font:500 13px Figtree,system-ui,sans-serif;padding:10px 16px;border-radius:10px;box-shadow:0 4px 18px rgba(0,0,0,.3);max-width:90vw";
d.textContent=NOTE;document.body.appendChild(d);}d.style.display="block";clearTimeout(d._t);d._t=setTimeout(function(){d.style.display="none"},5000);}
function map(u){var p=new URL(u,location.origin);if(p.origin!==location.origin||p.pathname.indexOf("/api/")!==0)return null;
var path=p.pathname.replace(/\\/$/,"");return /\\.(png|jpe?g|glb|webp)$/i.test(path)?"/_s"+path:"/_s"+path+".json";}
var of=window.fetch.bind(window);
window.fetch=function(input,init){var url=typeof input==="string"?input:input.url;var m=(init&&init.method||(input.method)||"GET").toUpperCase();
if(url.indexOf("/api/")===-1||!map(url))return of(input,init);
if(m!=="GET"){note();return Promise.resolve(new Response(JSON.stringify({error:NOTE,detail:NOTE}),{status:503,headers:{"content-type":"application/json"}}));}
return of(map(url),{cache:"force-cache"}).then(function(r){return r.ok?r:new Response("{}",{status:404});});};
window.EventSource=function(){this.close=function(){};this.addEventListener=function(){};};
document.addEventListener("DOMContentLoaded",function(){var b=document.createElement("div");b.textContent="Recorded snapshot: live training runs on the local server.";
b.style.cssText="position:fixed;right:10px;top:8px;z-index:99998;font:500 11px Figtree,system-ui,sans-serif;color:#555;background:rgba(255,255,255,.85);padding:3px 8px;border-radius:8px;pointer-events:none";document.body.appendChild(b);});
})();
"""


def get(path: str) -> bytes | None:
    try:
        with urlopen(BASE + path, timeout=120) as r:
            return r.read()
    except (HTTPError, URLError, TimeoutError):
        return None


def save(path: str, data: bytes) -> None:
    clean = path.split("?")[0]
    rel = clean if clean.lower().endswith(MEDIA) else clean + ".json"
    out = DIST / "_s" / rel.lstrip("/")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(data)


def fetch_save(path: str) -> bytes | None:
    data = get(path)
    if data is not None:
        save(path, data)
    return data


def jget(path: str):
    data = fetch_save(path)
    try:
        return json.loads(data) if data else None
    except ValueError:
        return None


def main() -> None:
    if DIST.exists():
        shutil.rmtree(DIST)
    shutil.copytree(
        ROOT / "web",
        DIST / "static",
        ignore=shutil.ignore_patterns(*SKIP_DIRS, ".*"),
    )
    shim = DIST / "static" / "snapshot-shim.js"
    shim.write_text(SHIM)
    tag = '<script src="/static/snapshot-shim.js"></script>'
    for name, dest in (("index.html", "app.html"), ("landing.html", "landing.html")):
        html = (ROOT / "web" / name).read_text()
        html = re.sub(r"(<head[^>]*>)", r"\1" + tag, html, count=1)
        (DIST / dest).write_text(html)
    for p in (DIST / "static").glob("*.html"):
        h = p.read_text()
        if tag not in h and "<head" in h:
            p.write_text(re.sub(r"(<head[^>]*>)", r"\1" + tag, h, count=1))

    jobs: list[str] = []
    for ep in (
        "state",
        "library",
        "robots",
        "agents",
        "llm",
        "meshes",
        "lines",
        "assets",
        "assets/categories",
        "optimizer/runs",
        "optimizer/skills",
    ):
        jget("/api/" + ep)
    lib = jget("/api/library") or {"tiles": []}
    state = jget("/api/state") or {}
    robots = jget("/api/robots")
    rids = [
        x["id"]
        for v in (robots or {}).values()
        if isinstance(v, list)
        for x in v
        if isinstance(x, dict) and "id" in x
    ]
    for rid in filter(None, rids):
        jobs += [f"/api/robots/{rid}/view", f"/api/robots/{rid}/report"]
    for t in lib["tiles"]:
        sid = t["id"]
        jobs.append(f"/api/skills/{sid}/thumb.png")
        jobs.append(f"/api/skills/{sid}")
        if t.get("replay"):
            first = jget(f"/api/skills/{sid}/replay/0")
            n = len((first or {}).get("all", []))
            jobs += [f"/api/skills/{sid}/replay/{i}" for i in range(1, n)]
            jobs += [f"/api/skills/{sid}/replay-events", f"/api/replays/{sid}/scene"]
            jobs += [
                f"/api/replays/{sid}/rollout/{r}/{i}"
                for r in range(8)
                for i in range(6)
            ]
    for r in state.get("runs", []):
        jobs.append(f"/api/runs/{r.get('id') or r.get('run')}")
    runs = jget("/api/optimizer/runs") or []
    for run in [r for r in runs if r.get("default")][:1]:
        rid = run["id"]
        d = jget(f"/api/optimizer/runs/{rid}") or {}
        jobs += [f"/api/optimizer/runs/{rid}/scene", f"/api/optimizer/runs/{rid}/frame"]
        blob = json.dumps(d)
        for rel in set(re.findall(r'"([\w./-]+\.jpg)"', blob)):
            jobs.append(f"/api/optimizer/runs/{rid}/img/{rel}")
        jobs += [
            f"/api/optimizer/runs/{rid}/img/r{r}/m{k}.jpg"
            for r in range(8)
            for k in range(100)
        ]
        jobs += [
            f"/api/optimizer/runs/{rid}/img/{n}.jpg" for n in ("hero_r0", "hero_final")
        ]
    with ThreadPoolExecutor(12) as ex:
        list(ex.map(fetch_save, dict.fromkeys(jobs)))

    (DIST / "vercel.json").write_text(
        json.dumps(
            {
                "cleanUrls": False,
                "rewrites": [
                    {"source": "/app", "destination": "/app.html"},
                    {"source": "/", "destination": "/landing.html"},
                    {"source": "/api/:p*.png", "destination": "/_s/api/:p*.png"},
                    {"source": "/api/:p*.jpg", "destination": "/_s/api/:p*.jpg"},
                    {"source": "/api/:p*.glb", "destination": "/_s/api/:p*.glb"},
                ],
            },
            indent=1,
        )
    )
    n = sum(1 for f in DIST.rglob("*") if f.is_file())
    size = sum(f.stat().st_size for f in DIST.rglob("*") if f.is_file()) / 1e6
    print(f"exported {n} files, {size:.1f} MB")


if __name__ == "__main__":
    main()
