#!/usr/bin/env python3
"""Gate for the T2 training-forward activation cache: compares each cached activation
(native/out/train_fwd_<NAME>.txt, written by native/train/train_fwd.vyb) to the numpy
reference (native/out/train_fwd_ref_<NAME>.txt) and requires max-relative-on-hard-elems
< 2e-5 (the Vyb-vs-libm transcendental residual). Verification-only (runs after the Vyb driver).
"""
import numpy as np, os, sys
base = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
out = os.path.join(base, "native/out")
NAMES = ["XN", "DQr", "DKr", "DV", "DCtx", "X1N", "Hu"]
tol = 2e-5
bad = []
for n in NAMES:
    v = np.loadtxt(os.path.join(out, f"train_fwd_{n}.txt"))
    r = np.loadtxt(os.path.join(out, f"train_fwd_ref_{n}.txt"))
    m = np.abs(r) > 1e-3
    rel = np.max(np.abs(v - r)[m] / np.abs(r)[m]) if m.any() else 0.0
    corr = np.corrcoef(v, r)[0, 1]
    flag = "OK" if rel < tol else "FAIL"
    print(f"{n:5s} n={v.size:6d} maxrel={rel:.3e} corr={corr:.6f}  {flag}")
    if rel >= tol or corr < 0.999:
        bad.append(n)
if bad:
    print(f"TRAIN_FWD_CACHE_VERIFY: FAIL ({bad})")
    sys.exit(1)
print("TRAIN_FWD_CACHE_VERIFY: OK")
