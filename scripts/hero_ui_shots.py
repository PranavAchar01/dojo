"""Build plan screenshots 6 (the view) and 7 (a fix in words with its diff). usage: hero_ui_shots.py PORT OUTDIR"""

import sys

from playwright.sync_api import sync_playwright

port, out = sys.argv[1], sys.argv[2]
with sync_playwright() as p:
    b = p.chromium.launch(args=["--use-gl=swiftshader", "--enable-unsafe-swiftshader"])
    pg = b.new_page(viewport={"width": 1500, "height": 1500})
    errs = []
    pg.on("pageerror", lambda e: errs.append(str(e)))
    pg.goto(f"http://127.0.0.1:{port}/app")
    pg.wait_for_timeout(1500)
    pg.evaluate("setView('plan')")
    pg.wait_for_selector("#bp-body tr")
    pg.wait_for_timeout(3000)
    pg.screenshot(path=f"{out}/2026-11-07-hero-6-buildplan.png")
    # fix in words: nudge by hand so the checklist reacts, then show the Muse revision if live
    pg.fill("#r-gap", "3")
    pg.dispatch_event("#r-gap", "change")
    pg.wait_for_timeout(2500)
    pg.click('[data-nudge="col"][data-d="1"][data-i="4"]')
    pg.wait_for_timeout(2500)
    if not pg.is_disabled("#bp-fix-go"):
        pg.fill("#bp-fix", "make the pillars one brick taller")
        pg.click("#bp-fix-go")
        pg.wait_for_selector(".bp-prop", timeout=120000)
    pg.wait_for_timeout(500)
    pg.screenshot(path=f"{out}/2026-11-07-hero-7-fix-in-words.png", full_page=True)
    print("errors:", errs)
    b.close()
