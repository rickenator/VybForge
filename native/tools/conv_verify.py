#!/usr/bin/env python3
"""Causal depthwise short convolution, checked against ggml's own GGML_OP_SSM_CONV.

ggml stores ne0 contiguously, so numpy shapes here are (last axis = ne0):
    s : (d_inner, ncs)  with ne = (ncs, d_inner)   -- the concat of the conv state window and tokens
    w : (d_inner, d_conv) with ne = (d_conv, d_inner) -- the GGUF tensor is {d_conv, d_inner}
    out: (n_t, d_inner) with ne = (d_inner, n_t)

The maths (ops.cpp FIR, the spec's step 3): out[t, ch] = sum_{j<d_conv} s[t+j, ch] * w[j, ch],
i.e. causally left-padded by d_conv-1 frames, which is exactly why the layer carries the last
d_conv-1 frames as state.

Usage: conv_verify.py
"""
import os
import subprocess
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
LLAMA = os.environ.get("VYBFORGE_LLAMA", os.path.expanduser("~/Projects/llama.cpp"))
BUILD = os.path.join(REPO, "native/build")
SRC = os.path.join(HERE, "conv_authority.c")
BIN = os.path.join(BUILD, "conv_authority")
MAXREL = float(os.environ.get("VYBFORGE_CONV_MAXREL", "1e-6"))

D_CONV = 4        # qwen35 ssm.conv_kernel
D_INNER = 10240   # 2*16*128 (q||k) + 48*128 (v) for the model; smaller here keeps it quick
N_T = 5


def build():
    os.makedirs(BUILD, exist_ok=True)
    cmd = [os.environ.get("CC", "gcc"), "-O2", "-o", BIN, SRC,
           f"-I{LLAMA}/ggml/include", f"-L{LLAMA}/build/bin",
           "-lggml", "-lggml-base", "-lggml-cpu", f"-Wl,-rpath,{LLAMA}/build/bin"]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        print("CONV_VERIFY_FAIL build: " + (r.stderr.strip()[:300] or r.stdout.strip()[:300]))
        return False
    return True


def main():
    d_conv, d_inner, n_t = D_CONV, D_INNER, N_T
    ncs = d_conv - 1 + n_t
    rng = np.random.default_rng(20261008)
    s = rng.normal(0.0, 0.2, size=(d_inner, ncs)).astype(np.float32)   # ne (ncs, d_inner)
    w = rng.normal(0.0, 0.3, size=(d_inner, d_conv)).astype(np.float32)  # ne (d_conv, d_inner)

    print(f"CONV_VERIFY d_conv={d_conv} d_inner={d_inner} n_t={n_t} ncs={ncs}")
    if not build():
        return 1

    inp, outp = os.path.join(BUILD, "conv_in.bin"), os.path.join(BUILD, "conv_out.bin")
    with open(inp, "wb") as fh:
        for a in (s, w):
            fh.write(a.astype("<f4").tobytes())

    r = subprocess.run([BIN, inp, outp, str(d_conv), str(d_inner), str(n_t), "1"],
                       capture_output=True, text=True)
    if r.returncode != 0:
        print("CONV_VERIFY_FAIL run: " + (r.stdout + r.stderr)[:300])
        return 1
    print("  " + r.stdout.strip())

    ref = np.fromfile(outp, dtype="<f4").reshape(n_t, d_inner).astype(np.float64)

    mine = np.zeros((n_t, d_inner), dtype=np.float64)
    for t in range(n_t):
        for j in range(d_conv):
            mine[t] += s[:, t + j].astype(np.float64) * w[:, j].astype(np.float64)

    diff = np.abs(mine - ref)
    rel = float(np.max(diff) / max(float(np.max(np.abs(ref))), 1e-30))
    print(f"CONV_VERIFY n={mine.size} maxabs={float(np.max(diff)):.3e} maxrel={rel:.3e} "
          f"(authority scale {float(np.max(np.abs(ref))):.3e})")
    if rel <= MAXREL:
        print(f"CONV_VERIFY_SUMMARY maxrel={rel:.3e}")
        print(f"CONV_VERIFY_DONE within maxrel {MAXREL:g} of ggml's ssm_conv")
        return 0
    print(f"CONV_VERIFY_FAIL worse than maxrel {MAXREL:g}")
    return 1


if __name__ == "__main__":
    sys.exit(main())
