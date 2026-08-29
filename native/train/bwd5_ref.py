#!/usr/bin/env python3
"""Reference for T3.5 dAttention (causal softmax backward). Synthetic roped q,k,v and
upstream dCtx; analytic dQ,dK,dV. Verification-only."""
import numpy as np, os
out = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "native", "out"))
S, H, KVH, HD = 2, 2, 1, 4
scale = 1.0 / np.sqrt(HD)
rng = np.random.default_rng(23)
q = rng.normal(size=(S, H, HD))
k = rng.normal(size=(S, KVH, HD))
v = rng.normal(size=(S, KVH, HD))
dC = rng.normal(size=(S, H, HD))
q.astype("<f8").tofile(os.path.join(out, "bwd5_q.bin"))
k.astype("<f8").tofile(os.path.join(out, "bwd5_k.bin"))
v.astype("<f8").tofile(os.path.join(out, "bwd5_v.bin"))
dC.astype("<f8").tofile(os.path.join(out, "bwd5_dc.bin"))

dQ = np.zeros_like(q); dK = np.zeros_like(k); dV = np.zeros_like(v)
for s in range(S):
    for h in range(H):
        qs = q[s, h]
        # causal weights
        sc = np.array([scale * np.dot(qs, k[t, 0]) for t in range(s + 1)])
        mx = sc.max(); a = np.exp(sc - mx); ssum = a.sum(); w = a / ssum
        l = np.array([np.dot(dC[s, h], v[t, 0]) for t in range(s + 1)])
        mean = (w * l).sum()
        dsc = a * (l - mean) / ssum
        dQ[s, h] = scale * sum(dsc[t] * k[t, 0] for t in range(s + 1))
        for t in range(s + 1):
            dK[t, 0] += scale * dsc[t] * qs
            dV[t, 0] += w[t] * dC[s, h]
np.savetxt(os.path.join(out, "bwd5_dq_ref.txt"), dQ.reshape(-1), fmt="%.12g")
np.savetxt(os.path.join(out, "bwd5_dk_ref.txt"), dK.reshape(-1), fmt="%.12g")
np.savetxt(os.path.join(out, "bwd5_dv_ref.txt"), dV.reshape(-1), fmt="%.12g")
print("BWD5_REF_DONE", dQ.size, dK.size, dV.size)
