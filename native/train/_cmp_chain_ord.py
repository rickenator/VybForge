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
for L in range(35,-1,-1):
    gu,dx=bwd(L,dx.copy(),cache[L])
    if L in (35,17,0):
        print(f"oracle chained |dx| @ L{L}: {np.linalg.norm(dx):.4e}  max {np.abs(dx).max():.3e}")
