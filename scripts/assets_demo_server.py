"""Minimal server for the Models panel and the factory view: only the assets router and the static web folder."""

import sys
from pathlib import Path

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from vla_forge import assets_api

app = FastAPI()
app.include_router(assets_api.router)
app.mount("/static", StaticFiles(directory=Path(__file__).resolve().parents[1] / "web"), name="static")

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=int(sys.argv[1]) if len(sys.argv) > 1 else 8195)
