import os, numpy as np, importlib.util as _u
spec=_u.spec_from_file_location("l0","native/gguf/layer0_ref.py")
l0=_u.module_from_spec(spec); spec.loader.exec_module(l0)
src=open("native/train/train_full_loop_ref.py").read()
ns={"__file__":"native/train/train_full_loop_ref.py"}
exec(src[src.index("import os"):src.index("# hyperparams")], ns)
out="native/out"; D=2560; S=2; H=32; KVH=8; HD=128; FF=9728; NQ=H*HD; NKV=KVH*HD
fwd=ns["forward_layers"]; bwd=ns["backward_layer"]
x0=np.fromfile(f"{out}/m2e_input.bin","<f8").reshape(S,D)
t=np.fromfile(f"{out}/tl_t.bin","<f8").reshape(S,D)
cache,xo=fwd(x0)
ns["XCACHE"]={}; cur=x0
for L in range(36): ns["XCACHE"][L]=cur; cur=cache[L]["xo"]
dLin=0.5*(xo-t)*2/(S*D)
# oracle: chained dx is computed inside backward; capture per-layer CHAINED input grad = the dx passed in
# We'll reproduce: the dG passed INTO backward_layer(L) is the chained upstream. Capture those.
dx=dLin.copy()
ords={}
for L in range(35,-1,-1):
    ords[L]=dx.copy()          # the chained upstream INTO layer L' backward = gradient of layer L input
    gu,dx=bwd(L,dx.copy(),cache[L])
for L in (35,17,0):
    gp=np.loadtxt(f"{out}/g_dg_L{L}.txt")  # GPU DG at this layer (the chained upstream used at L)
    o=ords[L]
    print(f"L{L}: gpu|DG|={np.linalg.norm(gp):.4e} ord|dx|={np.linalg.norm(o):.4e} corr={np.corrcoef(gp,o)[0,1]:.6f}")
