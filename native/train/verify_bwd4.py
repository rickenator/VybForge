#!/usr/bin/env python3
"""Gate for T3.4 dRoPE adjoint: bwd4_dxqr/dxkr vs analytic; tol 2e-5 rel on |ref|>1e-4."""
import numpy as np, os, sys
out = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "native", "out"))
tol = 2e-5
bad = []
for name in ["dxqr", "dxkr"]:
    v = np.loadtxt(os.path.join(out, f"bwd4_{name}.txt"))
    r = np.loadtxt(os.path.join(out, f"bwd4_{name}_ref.txt"))
    m = np.abs(r) > 1e-4
    rel = np.max(np.abs(v - r)[m] / np.abs(r)[m]) if m.any() else 0.0
    corr = np.corrcoef(v, r)[0, 1]
    print(f"{name} maxrel={rel:.3e} corr={corr:.6f} {'OK' if rel < tol else 'FAIL'}")
    if rel >= tol or corr < 0.999:
        bad.append(name)
print("BWD4_VERIFY:", "OK" if not bad else f"FAIL {bad}")
sys.exit(0 if not bad else 1)
