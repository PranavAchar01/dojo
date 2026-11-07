"""Screenshots 0, 1, 2 (rules vs render, photo, exploded) and 5 (Muse Spark's read, live, scored).

usage: hero_shots.py OUTDIR [--muse]
"""

import sys
import textwrap
from pathlib import Path

from PIL import Image, ImageDraw

from vla_forge import hero_bricks as hb
from vla_forge import hero_muse as hm
from vla_forge import hero_render as r
from vla_forge import hero_rules as hr

out = Path(sys.argv[1])
out.mkdir(parents=True, exist_ok=True)
STAMP = "2026-11-07-hero"
rules = hb.Rules()
bricks = hb.generate(rules)


def panel(
    lines: list[tuple[str, str]], photo: Image.Image, title: str, path: Path
) -> None:
    w, h = 2100, 1000
    img = Image.new("RGB", (w, h), "white")
    d = ImageDraw.Draw(img)
    d.text((40, 30), title, fill="#1c2b33", font=r._font(40))
    y = 100
    for mark, text in lines:
        col = {"ok": "#1a8f4e", "bad": "#d64545", "": "#465a69"}[mark]
        pre = "PASS  " if mark == "ok" else "FAIL  " if mark == "bad" else ""
        for i, ln in enumerate(textwrap.wrap(pre + text.strip(), 78) or [""]):
            d.text(
                (40 + (0 if i == 0 or not pre else 70), y),
                ln,
                fill=col,
                font=r._font(22),
            )
            y += 32
        y += 6
    img.paste(photo.resize((900, 900)), (w - 940, 70))
    img.save(path)


def check_lines(results) -> list[tuple[str, str]]:
    out = []
    for x in results:
        out.append(("ok" if x.ok else "bad", x.label))
        out.append(("", "     " + x.detail))
    return out


photo_path = out / f"{STAMP}-1-photo.png"
r.photo(photo_path)
r.exploded(out / f"{STAMP}-2-exploded.png")
head = [
    (
        "",
        f"Rules: gap {rules.gap} studs, pillars {rules.pillar} bricks, deck {rules.deck} studs, brick on deck",
    ),
    ("", "Parts: LDraw 3003 / 2456 (CC BY 4.0). Every line below is checked by code."),
]
panel(
    head + check_lines(hr.check(bricks, rules)),
    Image.open(photo_path),
    "Rules vs our build",
    out / f"{STAMP}-0-official-vs-ours.png",
)

if "--muse" in sys.argv:
    png = photo_path.read_bytes()
    png_small = out / "_read-input.png"
    Image.open(photo_path).convert("RGB").resize((768, 768)).save(png_small)
    res = hm.read_image(png_small.read_bytes())
    rows = res["reply"]["steps"]
    for s in rows:
        s["source"] = "muse"
    read, problems = hm.validate(rows, "muse")
    sc = hm.score(read, bricks, rules)
    keep = hb.DUCK
    if read and not problems:
        hb.set_assembly(read)
        read_photo = out / "_read-render.png"
        r.photo(read_photo)
        hb.set_assembly(keep)
        right = Image.open(read_photo)
    else:
        right = Image.new("RGB", (900, 900), "#f1f4f7")
    lines = [
        (
            "",
            f"Muse Spark ({res['model']}) read the photo; {res['usage'].get('prompt_tokens', 0)} in / {res['usage'].get('completion_tokens', 0)} out tokens.",
        ),
        (
            "",
            f"Matches {sc['exact_matches']} of {sc['bricks_expected']} bricks exactly; read {sc['bricks_read']} bricks.",
        ),
        (
            "",
            f"Rule checks on its read: {sc['rules_passed']} of {sc['rules_total']} pass.",
        ),
        ("", "Validator problems: " + ("; ".join(problems) or "none")),
    ] + [("ok" if x["ok"] else "bad", x["label"]) for x in sc["rules"]]
    panel(
        lines,
        right,
        "Muse Spark's read of the bridge photo",
        out / f"{STAMP}-5-muse-steps.png",
    )
    print({k: v for k, v in sc.items() if k != "rules"}, problems)
    png_small.unlink()
    (out / "_read-render.png").unlink(missing_ok=True)
