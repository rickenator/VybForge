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
dx=dLin.copy()
ords={}
for L in range(35,-1,-1):
    ords[L]=dx.reshape(-1).copy()
    gu,dx=bwd(L,dx.copy(),cache[L])
for L in (35,17,0):
    gp=np.loadtxt(f"{out}/g_dg_L{L}.txt").reshape(-1)
    o=ords[L].reshape(-1)
    print(f"L{L}: gp.size={gp.size} o.size={o.size}")
    n=min(gp.size,o.size)
    print(f"  corr={np.corrcoef(gp[:n],o[:n])[0,1]:.6f} |gp|={np.linalg.norm(gp):.4e} |o|={np.linalg.norm(o):.4e}")
