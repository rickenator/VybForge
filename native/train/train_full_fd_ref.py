#!/usr/bin/env python3
"""FD gradient validation for the Vyb GPU full-stack LoRA backward.
Independent check that does NOT trust the loop: for a FIXED forward point (original LoRA,
no training), pick one adapter element at bottom/middle/top layer, perturb by +h along a unit
direction, recompute full-36 forward + loss, get d(Loss)/d(adapter element) by central finite
difference. Compare to the GPU's analytic dU/dV gradient (same element) via corr + scale.

This isolates backward correctness from AdamW/loop state. Outputs a reference of (layer, rank,
in_idx, out_idx, fd_grad) that the GPU probe (train_full_fd.vyb) reproduces via its own
single-forward+backward, then verify_fd compares.

We do the FD in numpy over the SAME real model + LoRA (m2e_l{L}_{UV}.bin), MSE loss to tl_t.bin.
"""
import os, importlib.util, numpy as np

repo = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MODEL = "/home/rick/Models/qwen3/Qwen3-4B-Q4_K_M.gguf"
out = os.path.join(repo, "native", "out")
spec = importlib.util.spec_from_file_location("l0", os.path.join(repo, "native/gguf/layer0_ref.py"))
l0 = importlib.util.module_from_spec(spec); spec.loader.exec_module(l0)

D, H, KVH, HD, FF = 2560, 32, 8, 128, 9728
NQ, NKV, S, R, alpha_r = H*HD, KVH*HD, 2, 2, 2.0
tens = l0.parse_tsv()
proj_shapes = {"q":(D,NQ),"k":(D,NKV),"v":(D,NKV),"o":(NQ,D),"g":(D,FF),"u":(D,FF),"d":(FF,D)}
WK = {"q":"attn_q","k":"attn_k","v":"attn_v","o":"attn_output","g":"ffn_gate","u":"ffn_up","d":"ffn_down"}
seq = np.arange(S)
INVF = np.fromfile(os.path.join(out, "layer0_invfreq.bin"), "<f8")
freqs = lambda s_: s_ * INVF[:HD//2]

def rw(L, n):
    name = f"blk.{L}.{n}.weight"; t = tens[name]
    return l0.read_weight(t, tuple(t["shape"]))
def LN(L): return {n: rw(L,n) for n in ["attn_norm","attn_q_norm","attn_k_norm","ffn_norm"]}
def load_Lo(L):
    return {nm:(np.fromfile(os.path.join(out,f"m2e_l{L}_U{nm}.bin"),"<f8").reshape(*proj_shapes[nm][:1]+(R,)),
             np.fromfile(os.path.join(out,f"m2e_l{L}_V{nm}.bin"),"<f8").reshape(R,proj_shapes[nm][1])) for nm in proj_shapes}
def rms(v,w): return l0.rmsnorm(v,w)

# Preload weights once
ALL_LN = [LN(L) for L in range(36)]
ALL_W = [{n: rw(L,n) for n in ["attn_q","attn_k","attn_v","attn_output","ffn_gate","ffn_up","ffn_down"]} for L in range(36)]
Lo0 = load_Lo  # live?

def lproj(a, nm, L, Lo):
    U,V=Lo[L][nm]; s=a@U; return a@ALL_W[L][WK[nm]]+alpha_r*(s@V), s

def forward(x0, Lo):
    cur = x0
    for L in range(36):
        W=ALL_W[L]; lnn=ALL_LN[L]
        xn = rms(cur, lnn["attn_norm"])
        q,_=lproj(xn,"q",L,Lo); k,_=lproj(xn,"k",L,Lo); v,_=lproj(xn,"v",L,Lo)
        qn = rms(q.reshape(S,H,HD), lnn["attn_q_norm"]); kn = rms(k.reshape(S,KVH,HD), lnn["attn_k_norm"])
        qr,kr = l0.rope(qn,kn,seq)
        ctx = l0.gqa(qr,kr, v.reshape(S,KVH,HD))
        o,_=lproj(ctx.reshape(S,NQ),"o",L,Lo); x1=cur+o
        x1n = rms(x1, lnn["ffn_norm"])
        g,_=lproj(x1n,"g",L,Lo); up,_=lproj(x1n,"u",L,Lo); hu=(g/(1+np.exp(-g)))*up
        m,_=lproj(hu,"d",L,Lo); cur=x1+m
    return cur

x0 = np.fromfile(os.path.join(out,"m2e_input.bin"),"<f8").reshape(S,D)
t  = np.fromfile(os.path.join(out,"tl_t.bin"),"<f8").reshape(S,D)
def loss(Lo):
    xo = forward(x0, Lo)
    return 0.5*np.mean((xo-t)**2)

# build live Lo (dict-of-dict) from load_Lo
Lo = {L: load_Lo(L) for L in range(36)}

# Choose representative adapters (bottom/middle/top) x projection: q & ffn_down give good spread.
cases = [
  (0,  "q",0,0),      # bottom, Uq[0,0]
  (0,  "q",5,1),      # bottom, Uq[5,1]
  (17, "q",100,1),
  (17, "g",200,0),
  (35, "q",50,1),
  (35, "q",100,0),
]
# For V final we also test a V element. We'll pick element indices within U shape (In,R).
# deep-copy helpers for the full Lo dict (36 layers, 7 projs each)
def clone_Lo(Lo):
    return {L: {pr:(U.copy(),V.copy()) for pr,(U,V) in Lo[L].items()} for L in range(36)}

results=[]
for (LL, proj, i, r) in cases:
    U=Lo[LL][proj][0]
    h=0.01*max(abs(U[i,r]),1e-4)
    LoP=clone_Lo(Lo); LoP[LL][proj][0][i,r]+=h; lp=loss(LoP)
    LoM=clone_Lo(Lo); LoM[LL][proj][0][i,r]-=h; lm=loss(LoM)
    fd=(lp-lm)/(2*h)
    results.append((LL,proj,"U",i,r,float(fd)))
    print(f"L{LL} {proj} U[{i},{r}] h={h:.2e} fd={fd:.6e}")
casesV=[(0,"q",0,0),(17,"g",1,5),(35,"d",0,3)]
for (LL,proj,rr,oi) in casesV:
    V=Lo[LL][proj][1]
    h=0.01*max(abs(V[rr,oi]),1e-4)
    LoP=clone_Lo(Lo); LoP[LL][proj][1][rr,oi]+=h; lp=loss(LoP)
    LoM=clone_Lo(Lo); LoM[LL][proj][1][rr,oi]-=h; lm=loss(LoM)
    fd=(lp-lm)/(2*h)
    results.append((LL,proj,"V",rr,oi,float(fd)))
    print(f"L{LL} {proj} V[{rr},{oi}] h={h:.2e} fd={fd:.6e}")

np.savetxt(os.path.join(out,"m2e3_fd_cases.txt"), np.array(results), fmt="%s")
print("M2E3_FD_REF_DONE cases", len(results))
