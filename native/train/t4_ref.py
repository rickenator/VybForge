#!/usr/bin/env python3
"""Reference for T4: LoRA apply h = x@W + alpha_r*((x@U)@V) and one AdamW step.
Writes inputs .bin + refs. Verification-only."""
import numpy as np, os
out = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "native", "out"))
rng = np.random.default_rng(31)
S, IN, R, OUT = 3, 8, 4, 16
alpha_r = 2.0
x = rng.normal(size=(S, IN))
W = rng.normal(size=(IN, OUT))
U = rng.normal(size=(IN, R))
V = rng.normal(size=(R, OUT))
for nm, a in [("x", x), ("W", W), ("U", U), ("V", V)]:
    a.astype("<f8").tofile(os.path.join(out, f"t4_{nm}.bin"))
h = x @ W + alpha_r * ((x @ U) @ V)
np.savetxt(os.path.join(out, "t4_h_ref.txt"), h.reshape(-1), fmt="%.12g")

# ---- one AdamW step ----
n = 20
lr, b1, b2 = 0.001, 0.9, 0.999
eps_, wd = 1e-8, 0.01
t = 5
b1t, b2t = b1 ** t, b2 ** t
p = rng.normal(size=n); g = rng.normal(size=n)
m = np.zeros(n); v = np.zeros(n)
p.astype("<f8").tofile(os.path.join(out, "t4_p.bin")); g.astype("<f8").tofile(os.path.join(out, "t4_g.bin"))
m.astype("<f8").tofile(os.path.join(out, "t4_m.bin")); v.astype("<f8").tofile(os.path.join(out, "t4_v.bin"))
m2 = b1 * m + (1 - b1) * g
v2 = b2 * v + (1 - b2) * g ** 2
mhat = m2 / (1 - b1t); vhat = v2 / (1 - b2t)
p_new = p - lr * (mhat / (np.sqrt(vhat) + eps_) + wd * p)
np.savetxt(os.path.join(out, "t4_pnew_ref.txt"), p_new, fmt="%.12g")
print("T4_REF_DONE h", h.shape, "adamw n", n, "b1t", round(b1t,6), "b2t", round(b2t,8))
