"""Show the instruction photo to Muse Spark and save its build steps. Needs VLA_FORGE_META_CREDITS=1.
Usage: duck_muse.py PHOTO OUT.json"""

import json
import sys
from pathlib import Path

from vla_forge import llama

SCHEMA = {
    "type": "object",
    "properties": {
        "summary": {"type": "string"},
        "steps": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "step": {"type": "integer"},
                    "brick": {"type": "string", "description": "e.g. 2x4 yellow brick"},
                    "placement": {"type": "string", "description": "where it goes relative to the bricks already placed"},
                    "layer": {"type": "integer", "description": "0 is the table layer"},
                    "orientation": {"type": "string", "enum": ["long side along the duck", "across the duck"]},
                    "why": {"type": "string"},
                },
                "required": ["step", "brick", "placement", "layer", "orientation"],
            },
        },
    },
    "required": ["summary", "steps"],
}
SYSTEM = (
    "You write assembly instructions for a robot that builds LEGO models from a photo. Be precise and use only "
    "what the photo shows. Studs are the round bumps; one stud is 8 mm; one brick layer is 9.6 mm."
)
USER = (
    "This photo shows a finished LEGO duck made of exactly 6 bricks. The duck faces the right of the picture "
    "(beak side). List the bricks and give the build order from the bottom up, one step per brick: brick type and "
    "colour, how many studs it spans, which layer, how it is oriented, and where it sits relative to the bricks "
    "already placed (for example how many studs it is shifted forward or back). Return JSON."
)

if llama.api_key() is None or not llama.meta_allowed():
    sys.exit("Meta calls are off (set VLA_FORGE_META_CREDITS=1) or no key")
client = llama.MetaClient(llama.api_key(), "muse-spark-1.3")
_req = client.request
client.request = lambda *a, **k: {**_req(*a, **k), "max_tokens": 16000}  # a photo needs more reasoning room
reply = client.chat_json(SYSTEM, USER, SCHEMA, [Path(sys.argv[1]).read_bytes()])
Path(sys.argv[2]).write_text(json.dumps({"model": "muse-spark-1.3", "usage": client.usage, "reply": reply}, indent=2))
print("ok", client.usage, "spent", llama.meta_spent())
