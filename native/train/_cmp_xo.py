import numpy as np, os
out="native/out"
for s in (1,2):
    g=np.loadtxt(f"{out}/g_xo_s{s}.txt")
    o=np.loadtxt(f"{out}/o_xo_s{s}.txt")
    m=np.abs(o)>1e-4
    rel=np.max(np.abs(g-o)[m]/np.abs(o)[m]) if m.any() else 0
    corr=np.corrcoef(g,o)[0,1]
    print(f"step {s} xo: |g|={np.linalg.norm(g):.4e} |o|={np.linalg.norm(o):.4e} rel={rel:.3e} corr={corr:.6f}")
