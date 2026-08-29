#!/usr/bin/env python3
"""Reference for T3.3 dRMSNorm. Synthetic x[S,D], w[D], dy[S,D]; analytic dx and dw:
  inv = (mean(x^2)+eps)^-1/2 ; T_s = sum_j dy_sj*w_j*x_sj ;
  dx_sd = dy_sd*w_d*inv - x_sd*(inv^3/D)*T_s ;  dw_d = sum_s dy_sd*x_sd*inv
"""
import numpy as np, os
out = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "native", "out"))
S, D = 4, 16
eps = 1e-6
rng = np.random.default_rng(13)
x = rng.normal(size=(S, D)) * 1.2
w = rng.uniform(0.5, 1.5, D)
dy = rng.normal(size=(S, D))
x.astype("<f8").tofile(os.path.join(out, "bwd3_x.bin"))
w.astype("<f8").tofile(os.path.join(out, "bwd3_w.bin"))
dy.astype("<f8").tofile(os.path.join(out, "bwd3_dy.bin"))
dx = np.zeros_like(x); dw = np.zeros(D)
for s in range(S):
    m = (x[s] ** 2).mean()
    inv = 1.0 / np.sqrt(m + eps)
    T = (dy[s] * w * x[s]).sum()
    dx[s] = dy[s] * w * inv - x[s] * (inv ** 3 / D) * T
    dw += dy[s] * x[s] * inv
np.savetxt(os.path.join(out, "bwd3_dx_ref.txt"), dx.reshape(-1), fmt="%.12g")
np.savetxt(os.path.join(out, "bwd3_dw_ref.txt"), dw.reshape(-1), fmt="%.12g")
print("BWD3_REF_DONE")
