#!/usr/bin/env python3
"""Verify the GPU KV-cache context build (kvctx) per-layer roped-K/V vs the numpy reference
(kvctx_ref.py). The GPU dumps first-64 f64 of each layer's DKr and DV; numpy wrote full [9,NKV]
f64 .bin. Compare the first 64 elements at sample layers."""
import numpy as np, os, sys
out = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "native", "out"))
bad = []
for L in [0, 8, 17, 26, 35]:
    for which, suf in [("DKr", "DKr"), ("DV", "DV")]:
        g = np.loadtxt(os.path.join(out, f"kvctx_L{L}_{suf}_gpu.txt"))
        r = np.fromfile(os.path.join(out, f"kvctx_L{L}_{suf}_ref.bin"), "<f8").reshape(-1)[:64]
        n = min(len(g), len(r)); g = g[:n]; r = r[:n]
        m = np.abs(r) > 1e-8
        rel = np.max(np.abs(g - r)[m] / np.abs(r)[m]) if m.any() else 0.0
        corr = np.corrcoef(g, r)[0, 1] if n > 1 else 1.0
        f = "OK" if rel < 2e-2 and corr > 0.999 else "FAIL"
        if f != "OK": bad.append(f"L{L}{suf}(rel={rel:.2e})")
        print(f"  L{L} {suf}: maxrel={rel:.2e} corr={corr:.5f} {f}")
print("KVCTX_VERIFY:", "OK" if not bad else f"FAIL {bad}")
sys.exit(0 if not bad else 1)
