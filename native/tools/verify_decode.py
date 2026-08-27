#!/usr/bin/env python3
# Reference verification for the Vyb GPU greedy autoregressive decode
# (decode_driver.vyb): replicates embed -> L-layer stack -> lm_head -> argmax
# on the dumped weights, compares the generated token ids to gen_tokens.txt.
import numpy as np, os
base="native/out"
def L(n,s): return np.loadtxt(os.path.join(base,n)).reshape(s)
D,V,H,HD,KVH,HM,LAYERS,G=16,16,4,4,2,32,4,4
NCOL,HK=H*HD,KVH*HD; eps=1e-6
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
        O=Ctx@Wo; X1=rx+O; RN2=rms(X1,w2); G=RN2@Wg; U=RN2@Wu; Hu=(G/(1+np.exp(-G)))*U; M=Hu@Wd
        rx=X1+M
    return rx
toks=[1,5,3,2]
for g in range(G):
    X=wte[np.array(toks)]
    HSt=layers(X)
    logits=HSt[-1]@lmh
    nxt=int(np.argmax(logits))
    toks.append(nxt)
actual=np.loadtxt(os.path.join(base,"gen_tokens.txt")).astype(int).tolist()
print("ref  tokens:", toks)
print("gpu  tokens:", actual)
print("DECODE_VERIFY:", "OK" if toks==actual else "FAIL  (ref!=gpu)")
