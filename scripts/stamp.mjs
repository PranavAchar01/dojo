// Stamp a PNG with the tag of the one stage that is allowed to make thumbnails (web/stage.js: STAGE_TAG).
// tests/test_thumbnail_pipeline.py fails for any thumbnail without it.
import { readFileSync, writeFileSync } from "node:fs";
import { crc32 } from "node:zlib";

export const TAG = "dojo-stage-v1";

export function stampPng(path) {
  const b = readFileSync(path);
  const data = Buffer.from(`dojo-stage\0${TAG}`, "latin1");
  const head = Buffer.alloc(8);
  head.writeUInt32BE(data.length, 0);
  head.write("tEXt", 4, "latin1");
  const crc = Buffer.alloc(4);
  crc.writeUInt32BE(crc32(Buffer.concat([head.subarray(4), data])) >>> 0, 0);
  const at = 8 + 8 + 13 + 4; // signature, IHDR chunk header, IHDR data, IHDR crc
  writeFileSync(path, Buffer.concat([b.subarray(0, at), head, data, crc, b.subarray(at)]));
}
