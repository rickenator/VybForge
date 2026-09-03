import numpy as np, os
out="native/out"
for nm in ["Uq35","Vq35","Uq0","Vq0"]:
    g=np.loadtxt(f"{out}/g_q1_{nm}.txt"); o=np.loadtxt(f"{out}/o_q1_{nm}.txt")
    m=np.abs(o)>1e-4
    rel=np.max(np.abs(g-o)[m]/np.abs(o)[m]) if m.any() else 0
    corr=np.corrcoef(g,o)[0,1] if g.size>1 else 1
    print(f"q1 {nm}: gnorm={np.linalg.norm(g):.4e} onorm={np.linalg.norm(o):.4e} rel={rel:.3e} corr={corr:.6f}")
