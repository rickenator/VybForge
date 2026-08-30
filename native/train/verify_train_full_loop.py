#!/usr/bin/env python3
"""Gate for M2e-3: the GPU full-36-layer LoRA TRAINING LOOP (m2e3_loss.txt) vs the numpy
oracle (m2e3_loss_ref.txt). PRIMARY gate = per-step loss match (loop correctness). SECONDARY
note = descent. NB: for a deep real-Qwen3 stack, backward gradients vanish to ~1e-13 by L0 and
saturate at the top layers, so an MSE-overfit-to-a-fixed-random-target is NOT expected to
descend much (top layers carry the signal). Loop correctness is what matters here; corpus
next-token CE with head-gradient is the production objective that will show real descent.
"""
import numpy as np, os, sys
out = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "native", "out"))
tol = 5e-3

g = np.loadtxt(os.path.join(out, "m2e3_loss.txt"))
r = np.loadtxt(os.path.join(out, "m2e3_loss_ref.txt"))
assert g.size == r.size, f"loss length mismatch {g.size} vs {r.size}"
base = max(abs(r[0]), 1.0)
rel = np.max(np.abs(g - r) / base)
corr = np.corrcoef(g, r)[0, 1]
bad = []
flag = "OK" if rel < tol and corr > 0.999 else "FAIL"
if rel >= tol or corr <= 0.999: bad.append(f"loss-match ({rel:.2e})")
print(f"steps={g.size} loss0={g[0]:.6g} lossN={g[-1]:.6g} maxrel={rel:.3e}(/l0) corr={corr:.6f} {flag}")
print("(note: deep-stack vanishing gradient -> MSE-overfit descent is weak; see docstring)")
print("M2E3_VERIFY:", "OK" if not bad else f"FAIL {bad}")
sys.exit(0 if not bad else 1)
