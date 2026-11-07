"""The Robot tab's cache key, the stored payloads and the thumbnail endpoint."""

import gzip
import json

from fastapi import Request

from vla_forge import robotcache, server, thumbs
from vla_forge.sim import robots as R


def test_config_key_follows_the_parts():
    a = robotcache.config_key("so100.native.native")
    assert a == robotcache.config_key("so100.native.native")
    assert a != robotcache.config_key("so100.locked.native")  # another wrist choice
    assert a != robotcache.config_key("koch.native.native")  # another arm
    assert robotcache.config_key("so101.native.native") != robotcache.config_key(
        "so101.native.franka_hand"
    )


def test_part_key_differs_per_part():
    assert robotcache.part_key("arm", "so101") != robotcache.part_key("arm", "koch")


def test_payload_is_built_once_then_read_from_disk(tmp_path, monkeypatch):
    monkeypatch.setattr(robotcache, "ROOT", tmp_path)
    robotcache._mem.clear()
    calls = []

    def make():
        calls.append(1)
        return {"ok": True}

    first = robotcache._once("k1", "view", make)
    robotcache._mem.clear()  # a new process: memory is empty, disk is not
    second = robotcache._once("k1", "view", make)
    assert first == second and len(calls) == 1
    assert json.loads(gzip.decompress(second)) == {"ok": True}


def _request(**headers) -> Request:
    return Request(
        {
            "type": "http",
            "headers": [
                (k.replace("_", "-").encode(), v.encode()) for k, v in headers.items()
            ],
        }
    )


def test_blob_is_gzip_with_an_etag_and_revalidates():
    data = gzip.compress(b'{"a":1}')
    r = server._blob(data, "abc", _request(accept_encoding="gzip"))
    assert r.headers["content-encoding"] == "gzip" and r.headers["etag"] == '"abc"'
    assert (
        server._blob(
            data, "abc", _request(accept_encoding="gzip", if_none_match='"abc"')
        ).status_code
        == 304
    )
    assert (
        server._blob(data, "abc", _request()).body == b'{"a":1}'
    )  # a client that cannot gzip gets plain JSON


def test_every_card_has_a_thumbnail_name():
    names = [thumbs.thumb_name(*w) for w in thumbs.wanted()]
    assert len(names) == len(set(names))
    assert thumbs.thumb_name("arm", "so101") == "arm-so101.webp"
    assert thumbs.thumb_name("wrist", "locked", "koch") == "wrist-locked-on-koch.webp"
    assert len([w for w in thumbs.wanted() if w[0] == "arm"]) == len(R.ARMS)


def test_thumbnail_endpoint_serves_long_cached_webp_and_404s_unknown():
    from fastapi import HTTPException

    r = server.part_thumb("arm", "so101")
    assert r.media_type == "image/webp" and "max-age" in r.headers["cache-control"]
    assert (
        server.part_thumb("wrist", "locked", arm="koch").path.name
        == "wrist-locked-on-koch.webp"
    )
    for slot, part in (("arm", "nope"), ("bogus", "so101")):
        try:
            server.part_thumb(slot, part)
        except HTTPException as e:
            assert e.status_code == 404
        else:
            raise AssertionError("expected 404")
