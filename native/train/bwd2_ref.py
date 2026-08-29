#!/usr/bin/env python3
"""Reference for T3.2 dGEMM + LoRA grads. Determinstic synthetic LoRA layer inputs;
computes ANALYTIC grads (dU, dV, ds, dx_lora, dx_base) and writes .bin inputs + refs.
Model: h = x@W + alpha_r*(s@V), s = x@U (alpha_r = alpha/r). Backward to U,V and to x.
"""
import numpy as np, os
out = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "native", "out"))
S, IN, R, O = 4, 16, 8, 32
alpha_r = 2.0
rng = np.random.default_rng(11)
x = rng.normal(size=(S, IN))
U = rng.normal(size=(IN, R))
V = rng.normal(size=(R, O))
W = rng.normal(size=(IN, O))
dY = rng.normal(size=(S, O))
s = x @ U
gv = alpha_r * dY                      # LoRA-scaled upstream grad
ds = gv @ V.T                          # [S,R]
dU = x.T @ ds                          # [IN,R]
dV = s.T @ gv                          # [R,O]
dx_lora = ds @ U.T                     # [S,IN]
dx_base = dY @ W.T                     # [S,IN]
bins = {"x": x, "U": U, "V": V, "W": W, "dY": dY, "s": s, "gv": gv}
for name, a in bins.items():
    a.astype("<f8").tofile(os.path.join(out, f"bwd2_{name}.bin"))
refs = {"ds": ds, "dU": dU, "dV": dV, "dx_lora": dx_lora, "dx_base": dx_base}
for name, a in refs.items():
    np.savetxt(os.path.join(out, f"bwd2_{name}_ref.txt"), a.reshape(-1), fmt="%.12g")
print("BWD2_REF_DONE", {k: v.shape for k, v in bins.items()}, "gv (alpha_r)=2.0")
