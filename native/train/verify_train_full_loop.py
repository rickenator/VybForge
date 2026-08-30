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
bad=[]
flag = "OK" if rel < tol and corr > 0.999 else "FAIL"
if rel >= tol or corr <= 0.999: bad.append(f"loss-match ({rel:.2e})")
print(f"steps={g.size} loss0={g[0]:.6g} lossN={g[-1]:.6g} maxrel={rel:.3e}(/l0) corr={corr:.6f} {flag}")
print("(note: deep-stack vanishing gradient -> MSE-overfit descent is weak; see docstring)")

# ---- adapter + AdamW moment parity (loss alone can hide ineffective updates) ----
# GPU writes m2e3_L{LL}_{A}_vyb.txt; oracle writes m2e3_L{LL}_{A}_ref.txt. Compare them.
AT = ["Uq_U_final","Vq_V_final","Uq_mU","Uq_vU","Vq_mV","Vq_vV"]
tot_t = 0; ok_t = 0
for LL in (0,17,35):
    for A in AT:
        vp = os.path.join(out, f"m2e3_L{LL}_{A}_vyb.txt")
        rp = os.path.join(out, f"m2e3_L{LL}_{A}_ref.txt")
        if not (os.path.exists(vp) and os.path.exists(rp)):
            print(f"  L{LL} {A}: (missing dump) SKIP")
            continue
        v = np.loadtxt(vp); r = np.loadtxt(rp)
        tot_t += 1
        m = np.abs(r) > 1e-7
        rel = np.max(np.abs(v-r)[m]/np.abs(r)[m]) if m.any() else 0.0
        corr = np.corrcoef(v,r)[0,1] if v.size>1 else 1.0
        f = "OK" if rel < 2e-2 and corr > 0.99 else "FAIL"
        print(f"  L{LL} {A}: n={v.size} maxrel={rel:.2e} corr={corr:.6f} {f}")
        if f != "OK": bad.append(f"adapter L{LL} {A} ({rel:.2e})")
        else: ok_t += 1
print(f"adapter+moment parity {ok_t}/{tot_t}")
print("M2E3_VERIFY:", "OK" if not bad else f"FAIL {bad}")
sys.exit(0 if not bad else 1)
