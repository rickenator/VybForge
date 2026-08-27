#!/usr/bin/env python3
# Reference tooling: writes a small deterministic GGUF v3 file for testing the
# Vyb-native GGUF reader AND the q4_0 tensor-data dequant slice.
# Tensors: t0 = u8 [2,3]; t1 = f32 [4] = {1,2,3,4}; t2 = Q4_0 [32] (1 block).
# Q4_0 block = [f16 scale][32x4-bit]; scale=1.0 and nibble_i = i =>
#   dequant_i = (i - 8) * 1.0.
# Tensor DATA is placed AFTER the entire info region (data_base) — the offsets
# are computed, not hand-guessed.
import struct, os

out = os.path.join(os.path.dirname(__file__), "test.gguf")

def s(x):
    b = x.encode()
    return struct.pack("<Q", len(b)) + b

# ----- header + kv -----
head = bytearray(b"GGUF")
head += struct.pack("<I", 3)            # version
head += struct.pack("<Q", 3)            # n_tensors
head += struct.pack("<Q", 1)            # n_kv
kv = s("general.name") + struct.pack("<I", 8) + s("tiny")

# ----- tensor infos (with placeholder offsets to patch later) -----
infos = []
data_bytes = []
def tensor(name, nd, dims, ty, data):
    infos.append([name, nd, dims, ty, 0])   # offset patched later
    data_bytes.append(data)

tensor("t0", 2, [2, 3], 0, bytes([1, 2, 3, 4]))
tensor("t1", 1, [4], 6, struct.pack("<4f", 1.0, 2.0, 3.0, 4.0))
q4 = bytes([0x00, 0x3C])
for e in range(0, 32, 2):
    q4 += bytes([(e & 0xF) | (((e + 1) & 0xF) << 4)])
tensor("t2", 1, [32], 2, q4)

# compute info sizes to find data_base
def info_size(namelen, nd):
    return 8 + namelen + 4 + nd * 8 + 4 + 8
base = 24 + len(kv)
for (name, nd, dims, ty, _), _ in zip(infos, data_bytes):
    pass
sizes = [info_size(len(x[0]), x[1]) for x in infos]
info_total = sum(sizes)
data_base = base + info_total           # 24 + kv + info_total

# fill offsets contractually
off = data_base
for k, (_, _, _, _, _), _ in zip(range(3), infos, data_bytes):
    pass
# assign offsets sequentially
cursor = data_base
for n in range(3):
    infos[n][4] = cursor
    cursor += len(data_bytes[n])
    assert cursor <= data_base + info_total + 18

# ----- assemble file -----
b = bytearray(head)
b += kv
for (name, nd, dims, ty, oset) in infos:
    b += s(name)
    b += struct.pack("<I", nd)
    for d in dims:
        b += struct.pack("<Q", d)
    b += struct.pack("<I", ty)
    b += struct.pack("<Q", oset)
assert len(b) == data_base, (len(b), data_base)
for data in data_bytes:
    b += data

with open(out, "wb") as f:
    f.write(b)
print(f"wrote {out} ({len(b)} bytes; data_base={data_base}; "
      f"t0@{infos[0][4]} t1@{infos[1][4]} t2@{infos[2][4]})")
