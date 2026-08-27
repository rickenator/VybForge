#!/usr/bin/env python3
# Reference tooling: writes a small deterministic GGUF v3 file for testing the
# Vyb-native GGUF reader. Tensor DATA is minimal (a couple of bytes) so the
# parser test focuses on header + metadata KV + tensor index correctness.
import struct, os

out = os.path.join(os.path.dirname(__file__), "test.gguf")

def s(x):  # length-prefixed string
    b = x.encode()
    return struct.pack("<Q", len(b)) + b

b = bytearray()
# magic 'GGUF', version 3
b += b"GGUF"
b += struct.pack("<I", 3)          # version
b += struct.pack("<Q", 2)          # n_tensors
b += struct.pack("<Q", 1)          # n_kv
# KV: general.name = "tiny"
b += s("general.name")
b += struct.pack("<I", 8)          # value type = string
b += s("tiny")
# tensor infos
b += s("t0")
b += struct.pack("<I", 2)          # n_dims
b += struct.pack("<QQ", 2, 3)      # dims [2,3]
b += struct.pack("<I", 0)          # type u8
b += struct.pack("<Q", 72)         # offset
b += s("t1")
b += struct.pack("<I", 1)          # n_dims
b += struct.pack("<Q", 4)          # dims [4]
b += struct.pack("<I", 6)          # type f32
b += struct.pack("<Q", 74)         # offset
# tensor data (minimal)
while len(b) < 72:
    b += b"\x00"
b += bytes([1, 2, 3, 4])           # t0: 4 bytes at 72..
assert len(b) >= 74
while len(b) < 74:
    b += b"\x00"
b += struct.pack("<4f", 1.0, 2.0, 3.0, 4.0)  # t1: 16 bytes at 74

with open(out, "wb") as f:
    f.write(b)
print(f"wrote {out} ({len(b)} bytes)")
