import numpy as np, os
out="native/out"
for nm in ["Uq35_final","Vq35_final","mUq35"]:
    vp=f"{out}/g_{nm}.txt"; rp=f"{out}/o_{nm}.txt"
    v=np.loadtxt(vp); r=np.loadtxt(rp)
    m=np.abs(r)>1e-12
    rel=np.max(np.abs(v-r)[m]/np.abs(r)[m]) if m.any() else 0.0
    corr=np.corrcoef(v,r)[0,1] if v.size>1 else 1.0
    print(f"{nm}: gpu|.|={np.linalg.norm(v):.4e} ord|.|={np.linalg.norm(r):.4e} rel={rel:.3e} corr={corr:.6f} {'OK' if rel<1e-2 else 'DIFF'}")
