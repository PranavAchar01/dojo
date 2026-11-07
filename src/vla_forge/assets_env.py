"""The factory's floor texture and HDRI sky, both CC0 from Poly Haven, fetched into assets/library/env/.

    uv run --group assets python -m vla_forge.assets_env

The concrete floor is a PBR set (diffuse, roughness, normal; 1k JPEG for the web view, 512 px PNG for MuJoCo,
which reads no JPEG). The sky is a 1k Radiance HDR lit by a real machine shop. Provenance goes to
assets/library/env/provenance.json and is merged into manifest.json under "environment".
"""

from __future__ import annotations

import io
import json

from PIL import Image

from .assets_sources import REPO, _get

ENV = REPO / "assets" / "library" / "env"
FLOOR, SKY = "concrete_floor_02", "machine_shop_02"
API = "https://api.polyhaven.com"


def _info(asset: str) -> dict:
    i = json.loads(_get(f"{API}/info/{asset}"))
    return {
        "name": i["name"],
        "author": ", ".join(i.get("authors", {})) or "Poly Haven",
        "license": "CC0-1.0",
        "license_url": "https://creativecommons.org/publicdomain/zero/1.0/",
        "source_name": "Poly Haven",
        "source_url": f"https://polyhaven.com/a/{asset}",
    }


def main() -> None:
    ENV.mkdir(parents=True, exist_ok=True)
    tex = json.loads(_get(f"{API}/files/{FLOOR}"))
    out = {}
    for key, name in (
        ("Diffuse", "floor_diff"),
        ("Rough", "floor_rough"),
        ("nor_gl", "floor_nor"),
    ):
        data = _get(tex[key]["1k"]["jpg"]["url"])
        (ENV / f"{name}.jpg").write_bytes(data)
        Image.open(io.BytesIO(data)).convert("RGB").resize((512, 512)).save(
            ENV / f"{name}.png"
        )
    sky = json.loads(_get(f"{API}/files/{SKY}"))["hdri"]["1k"]["hdr"]["url"]
    (ENV / "sky.hdr").write_bytes(_get(sky))
    out["floor"] = {"id": FLOOR, "kind": "pbr texture", **_info(FLOOR)}
    out["sky"] = {"id": SKY, "kind": "hdri", **_info(SKY)}
    (ENV / "provenance.json").write_text(json.dumps(out, indent=1))
    print(
        {k: v["source_url"] for k, v in out.items()},
        sorted(p.name for p in ENV.iterdir()),
    )


if __name__ == "__main__":
    main()
