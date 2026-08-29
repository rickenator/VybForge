#!/usr/bin/env python3
"""Gate for M2c: the Vyb real-blk.0 backward (b_dU{nm}.txt, b_dV{nm}.txt, b_dx.txt) vs the
FD-validated numpy oracle (b_*_ref.txt). tol 2e-4 rel. Checks all 7 LoRA projections."""
import numpy as np, os, sys
out = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "native", "out"))
tol = 2e-4
NAMES = ["U_q","V_q","U_k","V_k","U_v","V_v","U_o","V_o","U_g","V_g","U_u","V_u","U_d","V_d"]
bad = []
for nm in NAMES:
    v = np.loadtxt(os.path.join(out, f"b_d{nm}.txt"))
    r = np.loadtxt(os.path.join(out, f"b_d{nm.replace('_','')}_ref.txt"))  # ref: b_dUq_ref (no underscore)
    m = np.abs(r) > 1e-4
    rel = np.max(np.abs(v - r)[m] / np.abs(r)[m]) if m.any() else 0.0
    corr = np.corrcoef(v, r)[0, 1]
    flag = "OK" if rel < tol and corr > 0.999 else "FAIL"
    print(f"d{nm:3s} n={v.size:5d} maxrel={rel:.3e} corr={corr:.6f} {flag}")
    if rel >= tol or corr <= 0.999:
        bad.append("d"+nm)
# dx
v = np.loadtxt(os.path.join(out, "b_dx.txt")); r = np.loadtxt(os.path.join(out, "b_dx_ref.txt"))
m = np.abs(r) > 1e-4
rel = np.max(np.abs(v - r)[m] / np.abs(r)[m]) if m.any() else 0.0
corr = np.corrcoef(v, r)[0, 1]
print(f"dx n={v.size} maxrel={rel:.3e} corr={corr:.6f} {'OK' if rel<tol and corr>0.999 else 'FAIL'}")
if rel >= tol or corr <= 0.999: bad.append("dx")
print("M2C_BACK_VERIFY:", "OK" if not bad else f"FAIL {bad}")
sys.exit(0 if not bad else 1)
