"""P2 gate 2: MOE_VYB_MATCH — Vyb MoE kernels vs the numpy reference.

The Vyb driver dumps f32 buffers as the raw 4-byte words (decimal Ints), decoded
here; `float32(w & 0xFFFFFFFF)` in little-endian is the value.

Usage: python3 native/glm53/verify_moe_vyb.py
"""

import sys

import numpy as np

PREP = "/home/rick/Projects/VybForge/native/glm53/prep"
OUT = "/home/rick/Projects/VybForge/native/out"

HIDDEN, TOPK = 4096, 8


def words_to_f32(path):
    ws = [int(x) for x in open(path).read().split()]
    out = []
    for w in ws:
        b = np.array([w & 0xFFFFFFFF], dtype=np.uint64).tobytes()[:4]
        out.append(np.frombuffer(b, dtype="<f4")[0])
    return np.array(out, dtype=np.float32)


y = np.fromfile(f"{PREP}/y.f32", dtype="<f4")
S = y.size // HIDDEN
y = y.reshape(S, HIDDEN)

ref_out = np.fromfile(f"{PREP}/ref_out.f32", dtype="<f4").reshape(S, HIDDEN)
ref_idx = np.fromfile(f"{PREP}/ref_idx.f32", dtype="<f4").reshape(S, TOPK)
ref_w = np.fromfile(f"{PREP}/ref_w.f32", dtype="<f4").reshape(S, TOPK)

vyb_out = words_to_f32(f"{OUT}/moe_out.txt").reshape(S, HIDDEN)
rvals = words_to_f32(f"{OUT}/moe_route.txt")
vyb_idx = rvals[: S * TOPK].reshape(S, TOPK)
vyb_w = rvals[S * TOPK: S * TOPK * 2].reshape(S, TOPK)

set_ok = 0
worst_w = 0.0
for s in range(S):
    a = dict(zip(vyb_idx[s].astype(int).tolist(), vyb_w[s].tolist()))
    b = dict(zip(ref_idx[s].astype(int).tolist(), ref_w[s].tolist()))
    if set(a) == set(b):
        set_ok += 1
        for k in a:
            worst_w = max(worst_w, abs(a[k] - b[k]))
    else:
        print(f"  token {s}: vyb {sorted(a)} vs ref {sorted(b)}")
print(f"routing: {set_ok}/{S} tokens identical expert sets, max |dw| {worst_w:.3e}")

d = np.abs(vyb_out - ref_out)
scale = float(np.abs(ref_out).max())
rel = d.max() / max(scale, 1e-12)
print(f"output : max abs {d.max():.3e}  scale {scale:.3e}  rel {rel:.3e}")

ok = (set_ok == S) and worst_w < 1e-4 and rel < 1e-5
print("MOE_VYB_MATCH:", "OK" if ok else "FAIL")
sys.exit(0 if ok else 1)
