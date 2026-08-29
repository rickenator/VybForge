#!/usr/bin/env python3
"""Finite-difference check of the M2c analytic backward (train_layer_back_ref.py).
Perturbs one LoRA U (and V), recomputes the FULL forward, compares the loss-gradient
(0.5*||xo - dG_src||^2 style, whose dL/dxo equals the upstream grad dG) to the analytic
dU/dV. Verifies the backward-chain ground truth before programming the Vyb driver."""
import os, numpy as np, importlib.util
repo = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
out = os.path.join(repo, "native", "out")
spec = importlib.util.spec_from_file_location("l0", os.path.join(repo, "native/gguf/layer0_ref.py"))
l0 = importlib.util.module_from_spec(spec); spec.loader.exec_module(l0)
D,H,KVH,HD,FF = 2560,32,8,128,9728
NQ,NKV,S,R,alpha_r = H*HD,KVH*HD,2,2,2.0
tens = l0.parse_tsv(); pre="blk.0."
rw=lambda n: l0.read_weight(tens[pre+n+".weight"], tuple(tens[pre+n+".weight"]["shape"]))
W={n:rw(n) for n in ["attn_q","attn_k","attn_v","attn_output","ffn_gate","ffn_up","ffn_down"]}
LN={n:rw(n) for n in ["attn_norm","attn_q_norm","attn_k_norm","ffn_norm"]}
x=np.fromfile(os.path.join(out,"layer0_input.bin"),"<f8").reshape(S,D)
dG=np.fromfile(os.path.join(out,"tl_dG.bin"),"<f8").reshape(S,D)
ps={"q":(D,NQ),"k":(D,NKV),"v":(D,NKV),"o":(NQ,D),"g":(D,FF),"u":(D,FF),"d":(FF,D)}
Lo={nm:(np.fromfile(os.path.join(out,f"tl_U{nm}.bin"),"<f8").reshape(ps[nm][0],R),
       np.fromfile(os.path.join(out,f"tl_V{nm}.bin"),"<f8").reshape(R,ps[nm][1])) for nm in ps}
WK={"q":"attn_q","k":"attn_k","v":"attn_v","o":"attn_output","g":"ffn_gate","u":"ffn_up","d":"ffn_down"}
def fwd(Lo_):
    xn=l0.rmsnorm(x,LN["attn_norm"])
    def lp(a,nm):
        U,V=Lo_[nm]; s=a@U
        return a@W[WK[nm]]+alpha_r*(s@V), s
    q,_=lp(xn,"q"); k,_=lp(xn,"k"); v,_=lp(xn,"v")
    qn=l0.rmsnorm(q.reshape(S,H,HD),LN["attn_q_norm"]); kn=l0.rmsnorm(k.reshape(S,KVH,HD),LN["attn_k_norm"])
    qr,kr=l0.rope(qn,kn,np.arange(S)); ctx=l0.gqa(qr,kr,v.reshape(S,KVH,HD))
    o,_=lp(ctx.reshape(S,NQ),"o"); x1=x+o
    x1n=l0.rmsnorm(x1,LN["ffn_norm"]); g,_=lp(x1n,"g"); u,_=lp(x1n,"u"); hu=(g/(1+np.exp(-g)))*u
    m,_=lp(hu,"d"); return x1+m
xo0=fwd(Lo)
target = xo0 - dG   # so dL/dxo|_{xo0} = dG, matching the analytic backward's upstream grad
def loss(xo): return 0.5*np.sum((xo-target)**2)
# analytic grads (re-read from ref output)
def an(nm): return (np.loadtxt(os.path.join(out,f"b_dU{nm}_ref.txt")).reshape(ps[nm][0],R),
                    np.loadtxt(os.path.join(out,f"b_dV{nm}_ref.txt")).reshape(R,ps[nm][1]))
baseline=loss(xo0)
for nm in ["q","g","d"]:
    U0,V0=Lo[nm]; eps=1e-6
    # U fd
    aU=np.zeros_like(U0); aV=np.zeros_like(V0)
    idx=np.unravel_index(np.argmax(np.abs(U0)), U0.shape)
    Lo2={k:v for k,v in Lo.items()}; Lo2[nm]=(U0.copy(),V0)
    Lu=Lo2[nm][0].copy(); Lu[idx]+=eps; Lo2[nm]=(Lu,Lo2[nm][1])
    fdU=(loss(fwd(Lo2))-baseline)/eps
    aU[idx]=an(nm)[0][idx]
    # V fd
    Lo2={k:v for k,v in Lo.items()}; Lo2[nm]=(U0.copy(),V0.copy()); Lv=Lo2[nm][1].copy()
    jdx=np.unravel_index(np.argmax(np.abs(V0)), V0.shape); Lv[jdx]+=eps; Lo2[nm]=(U0,Lv)
    fdV=(loss(fwd(Lo2))-baseline)/eps
    aV[jdx]=an(nm)[1][jdx]
    print(f"{nm}: FD_U={fdU:.6g} an_U={aU[idx]:.6g}; FD_V={fdV:.6g} an_V={aV[jdx]:.6g}")
