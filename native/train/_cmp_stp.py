import numpy as np, os
out="native/out"
for nm in ["Uq35","Vq35"]:
    def chk(a,b):
        g=np.loadtxt(f"{out}/{a}"); o=np.loadtxt(f"{out}/{b}")
        n=min(g.size,o.size); g=g[:n]; o=o[:n]
        m=np.abs(o)>1e-4
        rel=np.max(np.abs(g-o)[m]/np.abs(o)[m]) if m.any() else 0
        corr=np.corrcoef(g,o)[0,1] if n>1 else 1
        print(f"{a} vs {b}: gnorm={np.linalg.norm(g):.4e} onorm={np.linalg.norm(o):.4e} rel={rel:.3e} corr={corr:.6f}")
    chk(f"h_{nm}_s.txt", f"h_{nm}_s.txt")
