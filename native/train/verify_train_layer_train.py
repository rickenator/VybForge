#!/usr/bin/env python3
"""Gate for M2d: the Vyb real-blk.0 LoRA TRAINING LOOP (m2d_loss.txt, per-step loss)
vs the numpy oracle (m2d_loss_ref.txt). Requires: (1) per-step loss close to oracle
(rel tol), (2) loss DESCENDS / overfits the fixed target over N steps."""
import numpy as np, os, sys
out = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "native", "out"))
tol = 5e-3

g = np.loadtxt(os.path.join(out, "m2d_loss.txt"))
r = np.loadtxt(os.path.join(out, "m2d_loss_ref.txt"))
assert g.size == r.size, f"loss length mismatch {g.size} vs {r.size}"

# per-step relative agreement (oracle has floor 0 but loss decreases -> compare to step-0 scale)
base = max(abs(r[0]), 1e-6)
rel = np.max(np.abs(g - r) / base)
corr = np.corrcoef(g, r)[0, 1]

first, last = g[0], g[-1]
descend = last < first * 0.999
drop = 1.0 - last / max(first, 1e-12)

bad = []
flag = "OK" if rel < tol and corr > 0.999 else "FAIL"
if rel >= tol or corr <= 0.999: bad.append("loss-match")
print(f"steps={g.size} loss0={first:.6g} lossN={last:.6g} drop={drop*100:.1f}% "
      f"maxrel={rel:.3e}(/l0) corr={corr:.6f} {flag}")
if not descend:
    bad.append("descent")
    print(f"DESCENT: FAIL (loss {first:.6g} -> {last:.6g}, not descending)")
else:
    print(f"DESCENT: OK (loss {first:.6g} -> {last:.6g}, overfits target)")

print("M2D_TRAIN_VERIFY:", "OK" if not bad else f"FAIL {bad}")
sys.exit(0 if not bad else 1)
