#!/usr/bin/env python3
# Reference verification for the stochastic sampler (sampler.vyb): replicates
# temperature+top_k+top_p+LCG on the same crafted logits and compares kept set
# and all draws exactly against native/out/sampler_out.txt.
import numpy as np, os
tv=np.array([i*1.5-10.0 for i in range(16)])
V=16; TEMP,K,P=0.7,8,0.9
e=np.exp(tv/TEMP - (tv/TEMP).max()); pr=e/e.sum()
order=np.argsort(-pr); kept=[]; cum=0.0
for i in range(V):
    if i>=K: break
    if cum<P:
        kept.append(int(order[i])); cum+=pr[order[i]]
kp=pr[kept]; kp=kp/kp.sum()
prng=12345; draws=[]
for _ in range(12):
    prng=(prng*1664525+1013904223)&0xFFFFFFFF
    u=prng/4294967296.0
    acc=0.0; nxt=kept[-1]
    for i in range(len(kp)):
        acc+=kp[i]
        if u<=acc: nxt=kept[i]; break
    draws.append(nxt)
lines=open(os.path.join("native/out","sampler_out.txt")).read().strip().splitlines()
g_kept=[int(x) for x in lines[0].split()]
g_probs=[float(x) for x in lines[1].split()]
g_draws=[int(x) for x in lines[2].split()]
ok = (g_kept==kept) and (g_draws==draws) and max(abs(np.array(g_probs)-np.array(kp)))<1e-6
print("ref  kept ids:", kept, " probs:", [round(float(p),6) for p in kp])
print("gpu  kept ids:", g_kept, " probs:", [round(float(p),6) for p in g_probs])
print("ref  draws:", draws)
print("gpu  draws:", g_draws)
print("SAMPLER_VERIFY:", "OK" if ok else "FAIL")
