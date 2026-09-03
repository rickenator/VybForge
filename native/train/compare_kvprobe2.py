#!/usr/bin/env python3
"""Compare PROBE2 per-token ASLB dumps (diagkv_L*_*.txt) vs numpy kvprobe_L*_*.bin refs, at response
token 1 (abs row 10), layers 0/17/35. Localizes which field the per-token fill gets wrong at depth."""
import numpy as np, os
out = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "native", "out"))
def cmp(name, gpath, ref):
    g = np.loadtxt(os.path.join(out, gpath))
    r = np.fromfile(os.path.join(out, ref), "<f8")
    n = min(g.size, r.size); g = g[:n]; r = r[:n]
    corr = np.corrcoef(g, r)[0, 1] if r.std() > 1e-30 else 1.0
    rn = float(np.linalg.norm(r)); nr = float(np.linalg.norm(g - r)) / rn if rn > 0 else 0.0
    m = np.abs(r) > 1e-6
    maxrel = np.max(np.abs(g - r)[m] / np.abs(r)[m]) if m.any() else 0.0
    ok = corr > 0.999 and nr < 1e-2
    print(f"  {name:22s}: corr={corr:.6f} normrel={nr:.3e} maxrel={maxrel:.3e} {'OK' if ok else 'FAIL'}")
    return nr

bad = 0.0
fails = []
print("== layer-input (offset 0 = residual/attn_norm input, MY manual fill) ==")
for L in (0, 17, 35):
    nr = cmp(f"L{L} xin", f"diagkv_L{L}_xin_row10.txt", f"kvprobe_L{L}_xin_ref.bin")
    if nr > 1e-2: fails.append(f"L{L} xin")
print("== L17 detail ==")
for nm, ref, key in [("xn","kvprobe_L17_xn_ref.bin","xn"),("dqr","kvprobe_L17_dqr_ref.bin","dqr"),
                     ("dkr","kvprobe_L17_dkr_ref.bin","dkr"),("ctx","kvprobe_L17_ctx_ref.bin","ctx"),
                     ("m2","kvprobe_L17_m2_ref.bin","m2"),("sq","kvprobe_L17_sq_ref.bin","sq")]:
    nr = cmp(f"L17 {nm}", f"diagkv_L17_{nm}_row10.txt", ref)
    if nr > 1e-2: fails.append(f"L17 {nm}")
print("\nPROBE2_RESULT:", "ALL OK (forward ASLB fully correct -> bug is backward-buffer/state)" if not fails else f"FAIL FIELDS: {fails}")
