"""Generate native/exl3/map.bin for the exl3deq Vyb kernel from the verified
tile_slot_composition.json (oracle-probed ground truth, see README.md).

Layout: 16-byte header (cw: i64 LE = output columns, ntiles: i64 LE), then one
16-byte entry per slot (slots ordered by (row, col)):
  bytes 0..7   nibble index x4 (u16 LE)
  bytes 8..11  bit position x4 (u8)
  bytes 12..13 row within tile (u16 LE)
  bytes 14..15 col within tile (u16 LE)
"""

import json
import struct
import sys

COLUMNS = 4096          # down_proj fixture: n = out_features
TILES = 32768           # (2048/16) * (4096/16)

src = sys.argv[1] if len(sys.argv) > 1 else "tile_slot_composition.json"
dst = sys.argv[2] if len(sys.argv) > 2 else "map.bin"

raw = json.load(open(src))
entries = []
for key, comp in sorted(raw.items(), key=lambda kv: tuple(int(x) for x in kv[0].split(","))):
    r, c = (int(x) for x in key.split(","))
    assert len(comp) == 4, f"slot {key}: {len(comp)} members"
    entries.append((r, c, comp))

hdr = struct.pack("<qq", COLUMNS, TILES)
body = b""
for r, c, comp in entries:
    body += struct.pack("<4H", *(m for m, _ in comp))
    body += struct.pack("<4B", *(p for _, p in comp))
    body += struct.pack("<HH", r, c)
assert len(body) == 256 * 16
open(dst, "wb").write(hdr + body)
print(f"wrote {dst}: header 16B + {len(entries)} slots x 16B (cw={COLUMNS}, ntiles={TILES})")
