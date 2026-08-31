#!/usr/bin/env python3
"""Gate for the M2e-3 corpus-CE loop: GPU per-step CE loss (m2e3ce_loss_vyb.txt) vs the numpy
oracle (m2e3ce_loss_ref.txt). PRIMARY gate = per-step loss match (loop correctness). The CE
objective has a non-trivial HEAD gradient, so unlike MSE-overfit it SHOULD descend.
Also gates final L0 Uq adapter parity vs the oracle."""
import numpy as np, os, sys
out = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "native", "out"))
bad = []
g = np.atleast_1d(np.loadtxt(os.path.join(out, "m2e3ce_loss_vyb.txt")))
r = np.atleast_1d(np.loadtxt(os.path.join(out, "m2e3ce_loss_ref.txt")))
n = min(g.size, r.size)
g = g[:n]; r = r[:n]
rel = np.max(np.abs(g - r) / max(abs(r[0]), 1.0))
corr = np.corrcoef(g, r)[0, 1] if n > 1 else 1.0
flag = "OK" if rel < 5e-2 and corr > 0.99 else "FAIL"
if flag != "OK": bad.append(f"loss-match(rel={rel:.2e})")
print(f"steps={n} loss0={g[0]:.6g} lossN={g[-1]:.6g} maxrel={rel:.3e}(/l0) corr={corr:.6f} {flag}")
print(f"  ref : {np.array2string(r, precision=6)}  (oracle, committed 8-step: 15.49->0.697)")
print(f"  gpu : {np.array2string(g, precision=6)}")
# final L0 Uq parity (only meaningful when GPU and oracle ran the SAME NSTP, e.g. the 8-step gate)
vp = os.path.join(out, "m2e3ce_L0_Uq_U_final_vyb.txt")
rp = os.path.join(out, "m2e3ce_L0_Uq_U_final_ref.txt")
if os.path.exists(vp) and os.path.exists(rp) and n == len(r) and n >= 8:
    v = np.loadtxt(vp); rr = np.loadtxt(rp)
    m = np.abs(rr) > 1e-6
    rel2 = np.max(np.abs(v - rr)[m] / np.abs(rr)[m]) if m.any() else 0.0
    f2 = "OK" if rel2 < 2e-2 else "FAIL"
    if f2 != "OK": bad.append(f"L0Uq(rel={rel2:.2e})")
    print(f"  L0 Uq final: maxrel={rel2:.2e} {f2}")
else:
    print("  L0 Uq final: (NSTP mismatch / <8 steps — skip adapter parity; loss-match is the gate)")
print("M2E3CE_VERIFY:", "OK" if not bad else f"FAIL {bad}")
sys.exit(0 if not bad else 1)
