#!/usr/bin/env python3
# Verification: cross-check the Vyb q4kdeq kernel (native/out/q4k_vyb.txt)
# against a faithful python port of llama.cpp dequantize_row_q4_K, on one real
# 144-byte block of a Q4_K tensor pulled from the live Qwen3-4B model.
# Run: python3 q4k_verify.py   (after native/host/q4k_driver.vyb has run)
import os, struct

repo = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MODEL = "/home/rick/Models/qwen3/Qwen3-4B-Q4_K_M.gguf"
TSV = os.path.join(repo, "native/out/qwen3_4b_tensors.tsv")
BLK = os.path.join(repo, "native/out/q4k_block.bin")
REF = os.path.join(repo, "native/out/q4k_ref.txt")
VYB = os.path.join(repo, "native/out/q4k_vyb.txt")

# abs data offset of the first Q4_K tensor's data (blk.0.attn_q.weight, type 12)
abs_off = None
for line in open(TSV):
    if "blk.0.attn_q.weight" in line and "\t12\t" in line:
        toks = line.rstrip("\n").split("\t")
        for t in toks:
            if t.startswith("off="):
                abs_off = int(t[4:])
                break
        break
assert abs_off is not None, "blk.0.attn_q.weight (Q4_K) not found in tsv"

with open(MODEL, "rb") as fh:
    fh.seek(abs_off)
    raw = fh.read(144)
assert len(raw) == 144, len(raw)
open(BLK, "wb").write(raw)

def get_scale_min_k4(j, s):
    if j < 4:
        return s[j] & 63, s[j + 4] & 63
    return (s[j+4] & 0xF) | ((s[j-4] >> 6) << 4), (s[j+4] >> 4) | ((s[j] >> 6) << 4)

d = struct.unpack("<e", raw[0:2])[0]
dmin = struct.unpack("<e", raw[2:4])[0]
scales, qs = raw[4:16], raw[16:144]
ref = []
for j in range(0, 256, 64):
    t = j // 64
    for gl, is_ in ((0, 2 * t), (1, 2 * t + 1)):
        sc, m = get_scale_min_k4(is_, scales)
        d1, m1 = d * sc, dmin * m
        for l in range(32):
            b = qs[t * 32 + l]
            nib = (b & 0xF) if gl == 0 else (b >> 4)
            ref.append(d1 * nib - m1)
open(REF, "w").write("\n".join(f"{x:.10g}" for x in ref) + "\n")

if not os.path.exists(VYB):
    print("q4_K block + reference written. Run native/host/q4k_driver.vyb, then re-run this.")
    raise SystemExit(0)

vyb = [float(x) for x in open(VYB).read().split()]
assert len(vyb) == 256, len(vyb)
bad = 0
mx = 0.0
for i in range(256):
    err = abs(vyb[i] - ref[i])
    mx = max(mx, err)
    if err > 1e-6:
        bad += 1
        if bad <= 5:
            print(f"  idx {i}: vyb={vyb[i]:.10g} ref={ref[i]:.10g}")
print("q4_K dequant: block abs_off =", abs_off)
print("bad =", bad, " maxerr =", mx)
print("Q4K_VERIFY:", "OK" if bad == 0 else "FAIL")
raise SystemExit(0 if bad == 0 else 1)
