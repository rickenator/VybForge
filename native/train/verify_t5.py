#!/usr/bin/env python3
"""Gate for T5 (training-step driver): the Vyb-trained LoRA params (t5_U_vyb, t5_V_vyb)
must match the numpy reference (t5_Ufinal_ref, t5_Vfinal_ref) after N steps; and the ref
loss trajectory must descend (overfit)."""
import numpy as np, os, sys
out = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "native", "out"))
tol = 2e-5
bad = []
for name in ["U", "V"]:
    v = np.loadtxt(os.path.join(out, f"t5_{name}_vyb.txt"))
    r = np.loadtxt(os.path.join(out, f"t5_{name}final_ref.txt"))
    m = np.abs(r) > 1e-4
    rel = np.max(np.abs(v - r)[m] / np.abs(r)[m]) if m.any() else 0.0
    corr = np.corrcoef(v, r)[0, 1]
    print(f"{name} maxrel={rel:.3e} corr={corr:.6f} {'OK' if rel < tol else 'FAIL'}")
    if rel >= tol or corr < 0.999:
        bad.append(name)
losses = np.loadtxt(os.path.join(out, "t5_losses_ref.txt"))
print("ref losses:", [round(float(x), 4) for x in losses],
      "-> descent" if losses[-1] < losses[0] else "-> !! no descent")
if not (losses[-1] < losses[0]):
    bad.append("loss_descent")
print("T5_VERIFY:", "OK" if not bad else f"FAIL {bad}")
sys.exit(0 if not bad else 1)
