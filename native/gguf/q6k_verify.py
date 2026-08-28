#!/usr/bin/env python3
# Verification: cross-check the streamed+dequantized real attn_v (Q6_K) slice
# against a faithful llama.cpp block_q6_K dequant (elements placed by position).
import os, struct

repo = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MODEL = "/home/rick/Models/qwen3/Qwen3-4B-Q4_K_M.gguf"
ABSOFF = 338314688
NB = 2048
PBS = NB * 210

with open(MODEL, "rb") as fh:
    fh.seek(ABSOFF)
    raw = fh.read(PBS)
assert len(raw) == PBS

def deq_one(b):
    d = struct.unpack("<e", b[208:210])[0]
    ql, qh = b[0:128], b[128:192]
    sc = [x if x < 128 else x - 256 for x in b[192:208]]  # int8 signed
    out = [0.0] * 256
    for o in (0, 128):
        g = o // 128
        ql2 = ql[g * 64:g * 64 + 64]
        qh2 = qh[g * 32:g * 32 + 32]
        scb = g * 8
        for l in range(32):
            isv = l // 16
            q1 = ((ql2[l] & 0xF) | (((qh2[l] >> 0) & 3) << 4)) - 32
            q2 = ((ql2[32 + l] & 0xF) | (((qh2[l] >> 2) & 3) << 4)) - 32
            q3 = ((ql2[l] >> 4) | (((qh2[l] >> 4) & 3) << 4)) - 32
            q4 = ((ql2[32 + l] >> 4) | (((qh2[l] >> 6) & 3) << 4)) - 32
            out[o + l] = d * sc[scb + isv + 0] * q1
            out[o + 32 + l] = d * sc[scb + isv + 2] * q2
            out[o + 64 + l] = d * sc[scb + isv + 4] * q3
            out[o + 96 + l] = d * sc[scb + isv + 6] * q4
    return out

ref = []
for blk in range(NB):
    ref.extend(deq_one(raw[blk * 210:(blk + 1) * 210]))

vyb = [float(x) for x in open(os.path.join(repo, "native/out/attnv_f32_slice.txt")).read().split()]
assert len(vyb) == NB * 256, (len(vyb), NB * 256)
bad = 0
mx = 0.0
for i in range(NB * 256):
    err = abs(vyb[i] - ref[i])
    mx = max(mx, err)
    if err > 1e-6:
        bad += 1
print(f"attn_v (Q6_K) slice: {NB} blocks / {NB*256} values")
print("bad =", bad, " maxerr =", mx)
print("Q6K_VERIFY:", "OK" if bad == 0 else "FAIL")
raise SystemExit(0 if bad == 0 else 1)
