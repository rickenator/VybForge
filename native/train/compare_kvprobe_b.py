#!/usr/bin/env python3
"""Compare the L35 step-1 BACKWARD chain-link dumps (diagb_L35_*.txt) vs numpy oracle refs
(kvprobe_b_*_ref.bin) at response row 10. Pinpoints WHERE the per-token backward diverges from the
(known-good) oracle at the TOP layer:
  dCtx2 (dattn input, from o-proj/ffn backward) ; dQr2 (dattn dQ output, = oracle gq) ; dq2 (norm-back)."""
import numpy as np, os
out = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "native", "out"))
def cmp(name, gpath, ref):
    g = np.loadtxt(os.path.join(out, gpath)); r = np.fromfile(os.path.join(out, ref), "<f8")
    n = min(g.size, r.size); g = g[:n]; r = r[:n]
    corr = np.corrcoef(g, r)[0, 1] if r.std() > 1e-30 else 1.0
    rn = float(np.linalg.norm(r)); nr = float(np.linalg.norm(g - r)) / rn if rn > 0 else 0.0
    m = np.abs(r) > 1e-6; maxrel = np.max(np.abs(g - r)[m] / np.abs(r)[m]) if m.any() else 0.0
    ok = corr > 0.999 and nr < 1e-2
    print(f"  {name:8s}: corr={corr:.6f} normrel={nr:.3e} maxrel={maxrel:.3e} {'OK' if ok else 'FAIL'}")
    return (name, nr) if not ok else None
fails = []
for nm, gp, rp in [("dCtx2","diagb_L35_dctx2_row10.txt","kvprobe_b_dctx_L35_row10_ref.bin"),
                   ("dQr2 ","diagb_L35_dqr2_row10.txt","kvprobe_b_gq_L35_row10_ref.bin"),
                   ("dQn2 ","diagb_L35_dqn2_row10.txt","kvprobe_b_dqn_L35_row10_ref.bin"),
                   ("DQ   ","diagb_L35_dq_row10.txt","kvprobe_b_dqval_L35_row10_ref.bin"),
                   ("dq2  ","diagb_L35_dq2_row10.txt","kvprobe_b_dqnorm_L35_row10_ref.bin")]:
    f = cmp(nm, gp, rp)
    if f: fails.append(f)
print("\nCHAIN_LINK_RESULT:", "ALL OK" if not fails else f"DIVERGES AT: {fails}")
