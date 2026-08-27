#!/usr/bin/env python3
# Reference verification for the Vyb GPU multi-layer stack (stack_driver.vyb):
# loops the single-layer forward LAYERS times over the weight-tied residual
# stream and compares the final output to native/out/stack_out.txt.
import numpy as np, os
base = "native/out"
def L(n,s): return np.loadtxt(os.path.join(base,n)).reshape(s)
S,D,H,HD,KVH,HM,LAYERS=4,16,4,4,2,32,4
NCOL,HK=H*HD,KVH*HD; eps=1e-6
X=L("weights.txt",(S,D)); w1=L("w1.txt",D); w2=L("w2.txt",D)
Wq=L("wq.txt",(D,NCOL)); Wk=L("wk.txt",(D,HK)); Wv=L("wv.txt",(D,HK)); Wo=L("wo.txt",(D,D))
Wg=L("wg.txt",(D,HM)); Wu=L("wu.txt",(D,HM)); Wd=L("wd.txt",(HM,D)); freq=L("freq.txt",2)
def rms(x,w): return x/np.sqrt((x**2).mean(1,keepdims=True)+eps)*w
def ropemat(m,nh,hd):
    o=m.reshape(S,nh,hd).copy()
    for s in range(S):
        for j in range(hd//2):
            c,sn=np.cos(s*freq[j]),np.sin(s*freq[j])
            o[s,:,2*j],o[s,:,2*j+1]=o[s,:,2*j]*c-o[s,:,2*j+1]*sn, o[s,:,2*j]*sn+o[s,:,2*j+1]*c
    return o.reshape(S,nh*hd)
scale=1/np.sqrt(HD)
def layer(rx):
    RN1=rms(rx,w1); Qr=RN1@Wq; Kr=RN1@Wk; Vr=RN1@Wv
    Q=ropemat(Qr,H,HD); K=ropemat(Kr,KVH,HD)
    Ctx=np.zeros((S,NCOL))
    for s in range(S):
        for h in range(H):
            kvh=h//(H//KVH); q=Q[s,h*HD:(h+1)*HD]; Ks=K[:s+1,kvh*HD:(kvh+1)*HD]; Vv=Vr[:s+1,kvh*HD:(kvh+1)*HD]
            log=(q@Ks.T)*scale; a=np.exp(log-log.max()); a/=a.sum(); Ctx[s,h*HD:(h+1)*HD]=a@Vv
    O=Ctx@Wo; X1=rx+O; RN2=rms(X1,w2); G=RN2@Wg; U=RN2@Wu; Hu=(G/(1+np.exp(-G)))*U; M=Hu@Wd
    return X1+M
rx=X.copy()
for _ in range(LAYERS):
    rx=layer(rx)
g=np.loadtxt(os.path.join(base,"stack_out.txt")).reshape(S,D)
d=np.abs(rx-g)
print("GPU stack_out[0,:3] =", g[0,:3])
print("ref [0,:3] =", rx[0,:3])
print("max abs diff =", d.max())
print("LAYERS =", LAYERS, "   STACK_VERIFY:", "OK" if d.max()<1e-3 else "FAIL")
