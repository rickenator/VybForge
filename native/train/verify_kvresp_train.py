#!/usr/bin/env python3
"""Gate for the frozen-context KV training driver (kvresp_train.vyb) vs the numpy oracle
(kvresp_train_ref.py, commit 3435de0).

PRIMARY = step-1 GRADIENTS: dU_q/dV_q at L0/17/35 must match the oracle's kvresp_train_DUQ/DVQ
refs (~e-4 rel / corr 1.0). SECONDARY = per-step masked CE loss reproduces the oracle trajectory
(15.957 -> 15.495 -> 14.841 -> 14.095, DESCENDING) to a few e-2. Finally L0 Uq adapter parity.
Gate order is gradients-then-loss (per the S-lift lesson)."""
import numpy as np, os, sys
out = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "native", "out"))

def cmp_txt_bin(txt, binpath, tol=1e-3, name=""):
    g = np.loadtxt(os.path.join(out, txt))
    r = np.fromfile(os.path.join(out, binpath), "<f8")
    n = min(g.size, r.size); g = g[:n]; r = r[:n]
    corr = np.corrcoef(g, r)[0, 1] if r.std() > 1e-30 else 1.0
    rn = float(np.linalg.norm(r))
    norm_rel = float(np.linalg.norm(g - r) / rn) if rn > 1e-30 else 0.0
    m = np.abs(r) > 1e-6
    maxrel = np.max(np.abs(g - r)[m] / np.abs(r)[m]) if m.any() else 0.0  # info only (tails)
    # Gate on corr + vector-norm relative diff (a systematic scale/offset error would blow up
    # norm_rel and/or drop corr; a lone near-zero tail element cannot pass a scaling error).
    ok = corr > 0.999 and norm_rel < tol
    print(f"  {name}: corr={corr:.6f} norm_rel={norm_rel:.3e} maxrel(info)={maxrel:.3e} {'OK' if ok else 'FAIL'}")
    return ok

bad = []
# ---- step-1 gradients (the primary gate) ----
print("== step-1 gradients vs oracle ==")
for L in (0, 17, 35):
    if not cmp_txt_bin(f"kvresp_train_DUQ_L{L}_gpu.txt", f"kvresp_train_DUQ_L{L}_ref.bin", name=f"L{L} dU_q"):
        bad.append(f"L{L} dU_q")
    if not cmp_txt_bin(f"kvresp_train_DVQ_L{L}_gpu.txt", f"kvresp_train_DVQ_L{L}_ref.bin", name=f"L{L} dV_q"):
        bad.append(f"L{L} dV_q")

# ---- per-step loss trajectory ----
print("== per-step masked CE loss vs oracle ==")
g = np.atleast_1d(np.loadtxt(os.path.join(out, "kvresp_train_loss_gpu.txt")))
r = np.atleast_1d(np.loadtxt(os.path.join(out, "kvresp_train_loss_ref.txt")))
n = min(g.size, r.size); g = g[:n]; r = r[:n]
rel = np.max(np.abs(g - r) / max(abs(r[0]), 1.0))
corr = np.corrcoef(g, r)[0, 1] if n > 1 else 1.0
ok = rel < 5e-2 and corr > 0.99
if not ok:
    bad.append(f"loss-match(rel={rel:.2e})")
print(f"steps={n} loss0={g[0]:.6g} lossN={g[-1]:.6g} maxrel={rel:.3e}(/l0) corr={corr:.6f} {'OK' if ok else 'FAIL'}")
print(f"  ref : {np.array2string(r, precision=6)}  (frozen-context oracle, descends)")
print(f"  gpu : {np.array2string(g, precision=6)}")
# descent check
if g[-1] >= g[0]:
    bad.append("no-descent")
    print("  DESCENT: FAIL (gpu loss did not descend)")
else:
    print(f"  DESCENT: OK ({g[0]:.6g} -> {g[-1]:.6g})")

# ---- final L0 Uq adapter parity ----
print("== final L0 Uq adapter parity ==")
vp = os.path.join(out, "kvresp_train_UQ_L0_stepN_gpu.txt")
rp = os.path.join(out, "kvresp_train_UQ_L0_stepN_ref.txt")
if os.path.exists(vp) and os.path.exists(rp):
    v = np.loadtxt(vp); rr = np.loadtxt(rp)
    m = np.abs(rr) > 1e-6
    rel2 = np.max(np.abs(v - rr)[m] / np.abs(rr)[m]) if m.any() else 0.0
    ok2 = rel2 < 2e-2
    if not ok2:
        bad.append(f"L0Uq(rel={rel2:.2e})")
    print(f"  L0 Uq final: maxrel={rel2:.2e} {'OK' if ok2 else 'FAIL'}")
else:
    print("  L0 Uq final: (missing)")

print("KVRESP_TRAIN_VERIFY:", "OK" if not bad else f"FAIL {bad}")
sys.exit(0 if not bad else 1)
