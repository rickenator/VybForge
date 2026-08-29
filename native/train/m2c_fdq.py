#!/usr/bin/env python3
"""Stage-by-stage FD check of the analytic Q-path backward intermediates vs the ref.
Perturb one element of the input to each stage and recompute the sub-forward to the loss
(whose dL/dxo = dG), compare FD to the analytic grad of that stage (b_*.txt)."""
import os, importlib.util, numpy as np
repo = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
out = os.path.join(repo, "native", "out")
spec = importlib.util.spec_from_file_location("l0", os.path.join(repo, "native/gguf/layer0_ref.py"))
l0 = importlib.util.module_from_spec(spec); spec.loader.exec_module(l0)
D,H,KVH,HD,FF=2560,32,8,128,9728; NQ,NKV,S,R,alpha_r=H*HD,KVH*HD,2,2,2.0; scale=1.0/np.sqrt(HD)
tens=l0.parse_tsv(); pre="blk.0."
rw=lambda n: l0.read_weight(tens[pre+n+".weight"], tuple(tens[pre+n+".weight"]["shape"]))
W={n:rw(n) for n in ["attn_q","attn_k","attn_v","attn_output","ffn_gate","ffn_up","ffn_down"]}
LN={n:rw(n) for n in ["attn_norm","attn_q_norm","attn_k_norm","ffn_norm"]}
x=np.fromfile(os.path.join(out,"layer0_input.bin"),"<f8").reshape(S,D)
dG=np.fromfile(os.path.join(out,"tl_dG.bin"),"<f8").reshape(S,D)
ps={"q":(D,NQ),"k":(D,NKV),"v":(D,NKV),"o":(NQ,D),"g":(D,FF),"u":(D,FF),"d":(FF,D)}
Lo={nm:(np.fromfile(os.path.join(out,f"tl_U{nm}.bin"),"<f8").reshape(ps[nm][0],R),
      np.fromfile(os.path.join(out,f"tl_V{nm}.bin"),"<f8").reshape(R,ps[nm][1])) for nm in ps}
WK={"q":"attn_q","k":"attn_k","v":"attn_v","o":"attn_output","g":"ffn_gate","u":"ffn_up","d":"ffn_down"}
lp=lambda a,nm: (a@W[WK[nm]]+alpha_r*((a@Lo[nm][0])@Lo[nm][1]))
xn=l0.rmsnorm(x,LN["attn_norm"]); q0=lp(xn,"q"); k0=lp(xn,"k"); v=lp(xn,"v")
qn0=l0.rmsnorm(q0.reshape(S,H,HD),LN["attn_q_norm"]); kn0=l0.rmsnorm(k0.reshape(S,KVH,HD),LN["attn_k_norm"])
qr0,kr=l0.rope(qn0,kn0,np.arange(S)); ctx0=l0.gqa(qr0,kr,v.reshape(S,KVH,HD))
def tail(ctx_):
    o=lp(ctx_.reshape(S,NQ),"o"); x1=x+o
    x1n=l0.rmsnorm(x1,LN["ffn_norm"]); g=lp(x1n,"g"); u=lp(x1n,"u")
    hu=(g/(1+np.exp(-g)))*u; m=lp(hu,"d"); return x1+m
def f_ctx(c): return tail(c)
def f_qr(q_): return tail(l0.gqa(q_,kr,v.reshape(S,KVH,HD)))
def f_qn(n_):
    qr_,_=l0.rope(n_,kn0,np.arange(S)); return tail(l0.gqa(qr_,kr,v.reshape(S,KVH,HD)))
def f_q(q_):
    n_=l0.rmsnorm(q_.reshape(S,H,HD),LN["attn_q_norm"])
    qr_,_=l0.rope(n_,kn0,np.arange(S)); return tail(l0.gqa(qr_,kr,v.reshape(S,KVH,HD)))
xo0=tail(ctx0); target=xo0-dG
loss=lambda xo: 0.5*np.sum((xo-target)**2); baseline=loss(xo0)
def fd(f, arr, idx, eps=1e-6):
    a=arr.copy(); a[idx]+=eps; return (loss(f(a))-baseline)/eps
def cmp(name, f, arr, ref, arr_nm, idx):
    v=fd(f,arr,idx); a=ref.reshape(-1)[np.ravel_multi_index(idx,arr.shape)]
    print(f"{name}[{idx}]: FD={v:.6g} an={a:.6g}  {'OK' if abs(v-a)/(abs(a)+1e-9)<2e-3 else 'X'}")
cmp("dCtx", f_ctx, ctx0, np.loadtxt(os.path.join(out, "b_dCtx_ref.txt")), "dCtx", (0,0))
cmp("dQr",  f_qr,  qr0,  np.loadtxt(os.path.join(out, "b_dQr_ref.txt")),  "dQr",  (1,0,0))
cmp("dQn",  f_qn,  qn0,  np.loadtxt(os.path.join(out, "b_dQn_ref.txt")),  "dQn",  (1,0,0))
cmp("dq",   f_q,   q0,   np.loadtxt(os.path.join(out, "b_dq_ref.txt")),    "dq",   (1,7))
