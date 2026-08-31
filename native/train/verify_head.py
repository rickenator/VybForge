#!/usr/bin/env python3
"""Gate for the CE-head probe: GPU (head_vyb_*.txt / head_loss_vyb.txt) vs numpy (head_ref_*.txt).
Checks logits, dlog, dh, dxo, plab and the scalar CE loss. tol ~1e-4 rel."""
import numpy as np, os, sys
out = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "native", "out"))
bad = []
def cmp(name, tol=2e-3):
    g = np.loadtxt(os.path.join(out, f"head_vyb_{name}.txt"))
    r = np.loadtxt(os.path.join(out, f"head_ref_{name}.txt"))
    n = min(g.size, r.size); g = g[:n]; r = r[:n]
    m = np.abs(r) > 1e-4
    rel = np.max(np.abs(g - r)[m] / np.abs(r)[m]) if m.any() else 0.0
    corr = np.corrcoef(g, r)[0, 1] if n > 1 else 1.0
    f = "OK" if rel < tol and corr > 0.999 else "FAIL"
    if f != "OK": bad.append(f"{name}(rel={rel:.2e})")
    print(f"  {name}: n={n} maxrel={rel:.2e} corr={corr:.6f} {f}")
print("CE-head GPU vs numpy:")
cmp("embedded"); cmp("logits"); cmp("dlg"); cmp("dh"); cmp("dxo"); cmp("plab")
gl = float(open(os.path.join(out, "head_loss_vyb.txt")).read())
rl = float(open(os.path.join(out, "head_ref_loss.txt")).read())
rel = abs(gl - rl) / max(abs(rl), 1e-8)
f = "OK" if rel < 2e-3 else "FAIL"
if f != "OK": bad.append(f"loss({rel:.2e})")
print(f"  loss: gpu={gl:.9g} ref={rl:.9g} rel={rel:.2e} {f}")
print("HEAD_VERIFY:", "OK" if not bad else f"FAIL {bad}")
sys.exit(0 if not bad else 1)
