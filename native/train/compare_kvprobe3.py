#!/usr/bin/env python3
"""Compare PROBE3 unprobed-field dumps (diagkv_L0_*_row10.txt) vs numpy kvprobe_L0_*_ref.bin at L0 row 10."""
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
    return name if not ok else None
fails = []
for nm in ["dq","dk","dv","do","x1","x1n","gr","up","hu","sk","sv","so","sg","su","sd"]:
    f = cmp(nm, f"diagkv_L0_{nm}_row10.txt", f"kvprobe_L0_{nm}_ref.bin")
    if f: fails.append(f)
print("\nPROBE3_RESULT:", "ALL OK" if not fails else f"WRONG FIELDS: {fails}")
