#!/usr/bin/env python3
# Reference verification for Vyb GPU greedy+stochastic autoregressive decode.
# Re-runs the full decode (embed -> L-layer stack -> lm_head -> logits) so we
# also verify the per-step logits, then replicates the temperature+top_k+top_p
# sampler and the seeded LCG draw, comparing the generated token sequence.
import numpy as np, os, math
base="native/out"
def L(n,s): return np.loadtxt(os.path.join(base,n)).reshape(s)
D,V,H,HD,KVH,HM,LAYERS,G=16,16,4,4,2,32,4,4
NCOL,HK=H*HD,KVH*HD; eps=1e-6; TEMP,K,P=0.7,8,0.9
wte=L("wte.txt",(V,D)); lmh=L("lmh.txt",(D,V))
w1=L("w1.txt",D); w2=L("w2.txt",D)
Wq=L("wq.txt",(D,NCOL)); Wk=L("wk.txt",(D,HK)); Wv=L("wv.txt",(D,HK)); Wo=L("wo.txt",(D,D))
Wg=L("wg.txt",(D,HM)); Wu=L("wu.txt",(D,HM)); Wd=L("wd.txt",(HM,D)); freq=L("freq.txt",2)
def rms(x,w): return x/np.sqrt((x**2).mean(1,keepdims=True)+eps)*w
def ropemat(m,nh,hd):
    S=m.shape[0]; o=m.reshape(S,nh,hd).copy()
    for s in range(S):
        for j in range(hd//2):
            c,sn=np.cos(s*freq[j]),np.sin(s*freq[j])
            o[s,:,2*j],o[s,:,2*j+1]=o[s,:,2*j]*c-o[s,:,2*j+1]*sn, o[s,:,2*j]*sn+o[s,:,2*j+1]*c
    return o.reshape(S,nh*hd)
scale=1/np.sqrt(HD)
def layers(rx):
    for _ in range(LAYERS):
        RN1=rms(rx,w1); Qr=RN1@Wq; Kr=RN1@Wk; Vr=RN1@Wv
        Q=ropemat(Qr,H,HD); K=ropemat(Kr,KVH,HD); S=rx.shape[0]
        Ctx=np.zeros((S,NCOL))
        for s in range(S):
            for h in range(H):
                kvh=h//(H//KVH); q=Q[s,h*HD:(h+1)*HD]; Ks=K[:s+1,kvh*HD:(kvh+1)*HD]; Vv=Vr[:s+1,kvh*HD:(kvh+1)*HD]
                log=(q@Ks.T)*scale; a=np.exp(log-log.max()); a/=a.sum(); Ctx[s,h*HD:(h+1)*HD]=a@Vv
        O=Ctx@Wo; X1=rx+O; RN2=rms(X1,w2); GG=RN2@Wg; U=RN2@Wu; Hu=(GG/(1+np.exp(-GG)))*U; M=Hu@Wd
        rx=X1+M
    return rx
def filter_probs(lv):
    tv=lv/TEMP
    e=np.exp(tv-tv.max()); pr=e/e.sum()
    order=np.argsort(-pr)
    kept=[]
    cum=0.0
    for i in range(V):
        if i>=K: break
        if cum < P:
            kept.append(order[i]); cum+=pr[order[i]]
    kp=pr[kept]; kp=kp/kp.sum()
    return kept, np.array(kp)
toks=[1,5,3,2]; prng=12345; all_log=[]; ref_tokens=[]
for g in range(G):
    HS=layers(wte[np.array(toks)])
    lv=HS[-1]@lmh
    all_log.append(lv)
    kept,kp=filter_probs(lv)
    prng=(prng*1664525+1013904223)&0xFFFFFFFF
    u=prng/4294967296.0
    acc=0.0; nxt=kept[-1]
    for i in range(len(kp)):
        acc+=kp[i]
        if u<=acc: nxt=kept[i]; break
    ref_tokens.append(nxt)
    toks.append(nxt)
gpu_log=np.loadtxt(os.path.join(base,"sample_logits.txt")).reshape(G,V)
gpu_tokens=np.loadtxt(os.path.join(base,"gen_tokens.txt")).astype(int).tolist()
lgdiff=np.abs(gpu_log-np.array(all_log)).max()
print("max |gpu logits - ref logits| =", lgdiff)
print("ref  tokens:", ref_tokens, "  gpu tokens:", gpu_tokens[4:])
print("per-step top1 prob:", [round(float(filter_probs(l)[1].max()),4) for l in all_log])
print("DECODE_VERIFY:", "OK" if (lgdiff<1e-2 and ref_tokens==gpu_tokens[4:]) else "FAIL")
