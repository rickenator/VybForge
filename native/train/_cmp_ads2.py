import numpy as np, os
out="native/out"
for nm in ["Uq35_final","Vq35_final","Ug35","Uo35"]:
    g=np.loadtxt(f"{out}/g_{nm}.txt"); o=np.loadtxt(f"{out}/o_{nm}.txt")
    n=min(g.size,o.size); g=g[:n]; o=o[:n]
    m=np.abs(o)>1e-6
    rel=np.max(np.abs(g-o)[m]/np.abs(o)[m]) if m.any() else 0
    corr=np.corrcoef(g,o)[0,1] if n>1 else 1
    print(f"{nm}: n={n} |g_delta| adap: gpu_g0norm={np.linalg.norm(g):.4e} ord={np.linalg.norm(o):.4e} rel(big)={rel:.3e} corr={corr:.6f}")
