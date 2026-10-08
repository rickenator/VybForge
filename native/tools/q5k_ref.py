#!/usr/bin/env python3
"""Q5_K dequant reference (VybForge#10 phase 3).

Q5_K is the full-attention layers' weight type in Qwen3.8-27B Ridge (q/k/v, 51 tensors). Our
numpy port of `dequantize_row_q5_K` is checked against the compiled upstream function
(`native/tools/ggml_dequant_authority.py` spec "q5_K"), on real tensors, before either is used to
judge the GPU kernel.

Writes native/out/q5k_ref.txt:
    Q5_K <tensor>@<element> -> v0 v1 v2 v3 v4 v5
"""
import os
import sys

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
TSV = os.path.join(REPO, "native/gguf/ridge-3.7bpw-inventory.tsv")
GGUF = os.environ.get(
    "VYBFORGE_RIDGE_GGUF",
    os.path.expanduser("~/Models/qwen38-27b-ridge/Qwen3.8-27B-Ridge-3.7bpw.gguf"),
)
OUT = os.path.join(REPO, "native/out/q5k_ref.txt")

QK_K = 256
BLOCK = 176          # d f16, dmin f16, scales[12], qh[32], qs[128]
SLICE_BLOCKS = 16    # must match native/host/q5k_load_driver.vyb


def get_scale_min_k4(scales, j):
    """Vectorized llama.cpp get_scale_min_k4: 6-bit scale/min pairs packed in scales[12]."""
    q = scales
    sc = np.where(j < 4,
                  q[:, j] & 63,
                  (q[:, j + 4] & 0xF) | ((q[:, j - 4] >> 6) << 4))
    m = np.where(j < 4,
                 q[:, j + 4] & 63,
                 (q[:, j + 4] >> 4) | ((q[:, j] >> 6) << 4))
    return sc, m


def q5_K_ours(raw):
    """Faithful port of llama.cpp dequantize_row_q5_K (ggml-quants.c)."""
    nb = len(raw) // BLOCK
    b = np.frombuffer(raw, np.uint8).reshape(nb, BLOCK)
    d = b[:, 0:2].copy().view("<f2").astype(np.float64)[:, 0]
    dmin = b[:, 2:4].copy().view("<f2").astype(np.float64)[:, 0]
    scales = b[:, 4:16].astype(np.int64).copy()
    qh = b[:, 16:48].astype(np.int64)
    qs = b[:, 48:176].astype(np.int64)

    out = np.empty((nb, QK_K), np.float64)
    for t in range(4):
        sc0, m0 = get_scale_min_k4(scales, 2 * t)
        sc1, m1 = get_scale_min_k4(scales, 2 * t + 1)
        d1 = d * sc0.astype(np.float64)
        mi1 = dmin * m0.astype(np.float64)
        d2 = d * sc1.astype(np.float64)
        mi2 = dmin * m1.astype(np.float64)
        u1 = 1 << (2 * t)
        u2 = 2 << (2 * t)
        ql = qs[:, 32 * t:32 * t + 32]
        base = 64 * t
        lo = (ql & 0xF) + np.where((qh & u1) != 0, 16, 0)
        hi = (ql >> 4) + np.where((qh & u2) != 0, 16, 0)
        out[:, base:base + 32] = d1[:, None] * lo - mi1[:, None]
        out[:, base + 32:base + 64] = d2[:, None] * hi - mi2[:, None]
    return out.reshape(-1)


def parse_inventory():
    tens = {}
    with open(TSV) as fh:
        for line in fh:
            if line.startswith("#"):
                continue
            toks = line.rstrip("\n").split("\t")
            if len(toks) < 7:
                continue
            name, dims, tid, _tn, _role, off, nbytes = toks[:7]
            tens[name] = dict(shape=[int(x) for x in dims.split("x")], type=int(tid),
                              off=int(off), bytes=int(nbytes))
    return tens


def authority(raw):
    spec = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ggml_dequant_authority.py")
    try:
        import importlib.util
        s = importlib.util.spec_from_file_location("ggml_auth_q5k", spec)
        mod = importlib.util.module_from_spec(s)
        s.loader.exec_module(mod)
    except Exception as exc:
        print(f"Q5_K_AUTH_ERR {type(exc).__name__}: {exc}")
        return None
    return mod.dequantize(raw, "q5_K")


def main():
    if not os.path.exists(GGUF):
        print(f"Q5_K_REF_SKIP model not found: {GGUF}")
        return 0
    tens = parse_inventory()
    names = [n for n, t in tens.items() if t["type"] == 13]
    if not names:
        print("Q5_K_REF_FAIL inventory has no type-13 tensors")
        return 1
    # The full-attention layers (blk.3, 7, ... — every 4th) carry Q5_K q/k/v.
    chosen = []
    for want in ("blk.3.attn_q.weight", "blk.3.attn_k.weight", "blk.7.attn_v.weight"):
        if want in names:
            chosen.append(want)
    for n in names:
        if len(chosen) >= 3:
            break
        if n not in chosen:
            chosen.append(n)

    lines, agreed, differed = [], 0, 0
    with open(OUT, "w") as fh:
        for name in chosen:
            t = tens[name]
            nbytes = min(SLICE_BLOCKS * BLOCK, t["bytes"])
            with open(GGUF, "rb") as f:
                f.seek(t["off"])
                raw = f.read(nbytes)
            if len(raw) != nbytes:
                print(f"Q5_K_REF_FAIL short read {name}")
                return 1
            ours = q5_K_ours(raw)

            theirs = authority(raw)
            if theirs is None:
                indep = "llama.cpp-C=unavailable"
            elif theirs.size != ours.size:
                indep = f"llama.cpp-C=SIZE {theirs.size} vs {ours.size}"
                differed += 1
            else:
                md = float(np.max(np.abs(theirs - ours)))
                if md == 0.0:
                    indep = "llama.cpp-C=IDENTICAL"
                    agreed += 1
                else:
                    indep = f"llama.cpp-C=DIFFERS maxabs={md:.3e}"
                    differed += 1

            vals = " ".join(f"{v:.17g}" for v in ours[:6])
            fh.write(f"Q5_K {name}@{t['off']} -> {vals}\n")
            print(f"Q5_K_REF {name} blocks={nbytes // BLOCK} numel={ours.size} {indep}")

    print(f"Q5_K_REF_TENSORS={len(chosen)}")
    print(f"Q5_K_REF_INDEP identical={agreed} differing={differed}")
    if differed:
        print("Q5_K_REF_FAIL our port disagrees with llama.cpp's own code")
        return 1
    if agreed == 0:
        print("Q5_K_REF_WARN no compiled authority available")
    print(f"Q5_K_REF_WROTE {OUT}")
    print("Q5_K_REF_DONE")
    return 0


if __name__ == "__main__":
    sys.exit(main())
