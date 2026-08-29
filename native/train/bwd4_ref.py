#!/usr/bin/env python3
"""Reference for T3.4 dRoPE adjoint. Synthetic roped-q/k upstream grads; analytic
pre-rope grads via the inverse (transpose) half-pair rotation, sharing the same
inv-freq table the Vyb kernel gets. Verification-only."""
import numpy as np, os
out = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "native", "out"))
S, H, KVH, HD = 2, 2, 1, 4
half = HD // 2
POS = 0
fr = np.array([0.7, 0.3])
rng = np.random.default_rng(17)
# grads of the ROPED q/k, laid out [S,H,HD] and [S,KVH,HD]
dqr = rng.normal(size=(S, H, HD))
dkr = rng.normal(size=(S, KVH, HD))
dxqr = np.zeros_like(dqr); dxkr = np.zeros_like(dkr)
for s in range(S):
    for hh in range(max(H, KVH)):
        # q pairs
        for i in range(half):
            ang = (POS + s) * fr[i]; c, sn = np.cos(ang), np.sin(ang)
            dqi, dqj = dqr[s, hh, i], dqr[s, hh, i + half]
            dxqr[s, hh, i] = c * dqi + sn * dqj
            dxqr[s, hh, i + half] = -sn * dqi + c * dqj
        # k pairs (KVH=1)
        if hh < KVH:
            for i in range(half):
                ang = (POS + s) * fr[i]; c, sn = np.cos(ang), np.sin(ang)
                dki, dkj = dkr[s, hh, i], dkr[s, hh, i + half]
                dxkr[s, hh, i] = c * dki + sn * dkj
                dxkr[s, hh, i + half] = -sn * dki + c * dkj
dqr.astype("<f8").tofile(os.path.join(out, "bwd4_dqr.bin"))
dkr.astype("<f8").tofile(os.path.join(out, "bwd4_dkr.bin"))
fr.astype("<f8").tofile(os.path.join(out, "bwd4_fr.bin"))
np.savetxt(os.path.join(out, "bwd4_dxqr_ref.txt"), dxqr.reshape(-1), fmt="%.12g")
np.savetxt(os.path.join(out, "bwd4_dxkr_ref.txt"), dxkr.reshape(-1), fmt="%.12g")
print("BWD4_REF_DONE", dxqr.size, dxkr.size)
