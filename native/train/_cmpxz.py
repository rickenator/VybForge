import numpy as np, os
out="native/out"
for step in (1,2):
    for nm in ["Uq35","Vq35","Uq0"]:
        g=np.loadtxt(f"{out}/z{step}g_{nm}.txt"); o=np.loadtxt(f"{out}/z{step}o_{nm}.txt")
        m=np.abs(o)>1e-4
        rel=np.max(np.abs(g-o)[m]/np.abs(o)[m]) if m.any() else 0
        corr=np.corrcoef(g,o)[0,1] if g.size>1 else 1
        print(f"post-step{step} {nm}: rel={rel:.3e} corr={corr:.6f} {'OK' if rel<1e-2 else 'DIFF'}")
