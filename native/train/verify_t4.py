#!/usr/bin/env python3
"""Gate for T4: LoRA-apply output (t4_h.txt) and AdamW-updated params (t4_pnew.txt) vs refs."""
import numpy as np, os, sys
out = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "native", "out"))
tol = 2e-5
bad = []
for name, t in [("h", 2e-5), ("pnew", 2e-5)]:
    v = np.loadtxt(os.path.join(out, f"t4_{name}.txt"))
    r = np.loadtxt(os.path.join(out, f"t4_{name}_ref.txt"))
    m = np.abs(r) > 1e-4
    rel = np.max(np.abs(v - r)[m] / np.abs(r)[m]) if m.any() else 0.0
    corr = np.corrcoef(v, r)[0, 1]
    print(f"{name} maxrel={rel:.3e} corr={corr:.6f} {'OK' if rel < t else 'FAIL'}")
    if rel >= t or corr < 0.999:
        bad.append(name)
print("T4_VERIFY:", "OK" if not bad else f"FAIL {bad}")
sys.exit(0 if not bad else 1)
