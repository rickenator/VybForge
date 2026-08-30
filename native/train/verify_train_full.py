#!/usr/bin/env python3
"""Gate for M2e-1: the GPU full-36-layer LoRA forward activation cache
(m2e_L{layer}_{t}_vyb.txt + m2e_hidden_vyb.txt) vs the numpy reference
(m2e_L{layer}_{t}_ref.txt, m2e_hidden_ref.txt). Checks the FINAL 36-layer hidden
(strict, tol 2e-4) plus the full per-layer cached activation set, so every layer
and every tensor feeds the gate."""
import numpy as np, os, sys
out = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "native", "out"))
tol = 2e-4

def check(name_v, name_r):
    v = np.loadtxt(os.path.join(out, name_v))
    r = np.loadtxt(os.path.join(out, name_r))
    if v.size != r.size:
        return "FAIL", 9.9, 0.0, v.size, r.size
    m = np.abs(r) > 1e-4
    rel = np.max(np.abs(v - r)[m] / np.abs(r)[m]) if m.any() else 0.0
    corr = np.corrcoef(v, r)[0, 1]
    flag = "OK" if rel < tol and corr > 0.999 and not np.isnan(rel) else "FAIL"
    return flag, rel, corr, v.size, r.size

bad = []
f, rel, corr, nv, nr = check("m2e_hidden_vyb.txt", "m2e_hidden_ref.txt")
print(f"hidden(36L) n={nv} maxrel={rel:.3e} corr={corr:.6f} {f}")
if f != "OK": bad.append("hidden")

TENS = ["x","xn","q","k","v","qn","kn","qr","kr","ctx","o","x1","x1n","gate","up","hu","m","xo",
        "s_q","s_k","s_v","s_o","s_g","s_u","s_d"]
tot = 0; ok = 0
for L in range(36):
    for t in TENS:
        tot += 1
        f, rel, corr, nv, nr = check(f"m2e_L{L}_{t}_vyb.txt", f"m2e_L{L}_{t}_ref.txt")
        if f != "OK":
            bad.append(f"L{L}_{t}")
        else:
            ok += 1
print(f"per-layer tensors checked {ok}/{tot}")

print("M2E1_VERIFY:", "OK" if not bad else f"FAIL {bad[:24]}")
sys.exit(0 if not bad else 1)
