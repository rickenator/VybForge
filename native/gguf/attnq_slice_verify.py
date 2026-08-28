#!/usr/bin/env python3
# Verification: cross-check the streamed+dequantized real attn_q weight slice
# (native/host/q4k_load_driver.vyb -> native/out/attnq_f32_slice.txt) against a
# faithful llama.cpp block_q4_K dequant of the same packed bytes from the model.
import os, struct

repo = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MODEL = "/home/rick/Models/qwen3/Qwen3-4B-Q4_K_M.gguf"
ABSOFF = 332415936
NB = 2048
PBS = NB * 144

with open(MODEL, "rb") as fh:
    fh.seek(ABSOFF)
    raw = fh.read(PBS)
assert len(raw) == PBS

def get_scale_min_k4(j, s):
    if j < 4:
        return s[j] & 63, s[j + 4] & 63
    return (s[j+4] & 0xF) | ((s[j-4] >> 6) << 4), (s[j+4] >> 4) | ((s[j] >> 6) << 4)

def deq_one(b):
    d = struct.unpack("<e", b[0:2])[0]
    dmin = struct.unpack("<e", b[2:4])[0]
    scales, qs = b[4:16], b[16:144]
    out = []
    for j in range(0, 256, 64):
        t = j // 64
        for gl, is_ in ((0, 2 * t), (1, 2 * t + 1)):
            sc, m = get_scale_min_k4(is_, scales)
            d1, m1 = d * sc, dmin * m
            for l in range(32):
                bq = qs[t * 32 + l]
                nib = (bq & 0xF) if gl == 0 else (bq >> 4)
                out.append(d1 * nib - m1)
    return out

ref = []
for blk in range(NB):
    ref.extend(deq_one(raw[blk * 144:(blk + 1) * 144]))

vyb = [float(x) for x in open(os.path.join(repo, "native/out/attnq_f32_slice.txt")).read().split()]
assert len(vyb) == NB * 256, (len(vyb), NB * 256)

bad = 0
mx = 0.0
for i in range(NB * 256):
    err = abs(vyb[i] - ref[i])
    mx = max(mx, err)
    if err > 1e-6:
        bad += 1
print(f"attn_q slice: {NB} blocks / {NB * 256} values")
print("bad =", bad, " maxerr =", mx)
print("ATTNQ_SLICE_VERIFY:", "OK" if bad == 0 else "FAIL")
raise SystemExit(0 if bad == 0 else 1)
