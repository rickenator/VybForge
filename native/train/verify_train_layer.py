#!/usr/bin/env python3
"""Gate for M2b: the real blk.0 LoRA-augmented forward. Compares each cached activation
(Vyb tl_*.txt) to the numpy reference (tl_*_ref.txt); tol ~2e-5 rel (transcendental)."""
import numpy as np, os, sys
out = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "native", "out"))
tol = 5e-5
NAMES = ["xn","q","k","v","qn","kn","qr","kr","ctx","o","x1","x1n","gate","up","hu","m","xo",
         "s_q","s_k","s_v","s_o","s_g","s_u","s_d"]
bad = []
for n in NAMES:
    v = np.loadtxt(os.path.join(out, f"tl_{n}.txt"))
    r = np.loadtxt(os.path.join(out, f"tl_{n}_ref.txt"))
    m = np.abs(r) > 1e-3
    rel = np.max(np.abs(v - r)[m] / np.abs(r)[m]) if m.any() else 0.0
    corr = np.corrcoef(v, r)[0, 1]
    flag = "OK" if rel < tol and corr > 0.999 else "FAIL"
    print(f"{n:6s} n={v.size:7d} maxrel={rel:.3e} corr={corr:.6f} {flag}")
    if rel >= tol or corr <= 0.999:
        bad.append(n)
print("M2B_VERIFY:", "OK" if not bad else f"FAIL {bad}")
sys.exit(0 if not bad else 1)
