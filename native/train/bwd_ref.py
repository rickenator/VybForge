#!/usr/bin/env python3
"""Reference for the T3.1 backward kernels (dSiLU, dresid): generates deterministic
synthetic activations, computes the ANALYTIC gradients, and writes input .bin files +
grad references. Verification-only (kernels are all Vyb)."""
import numpy as np, os
out = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "native", "out"))
n = 1024
rng = np.random.default_rng(7)
g = rng.uniform(-3.0, 3.0, n)
u = rng.uniform(-2.0, 2.0, n)
dhu = rng.uniform(-1.0, 1.0, n)
dy = rng.uniform(-1.0, 1.0, n)
for name, a in [("g", g), ("u", u), ("dhu", dhu), ("dy", dy)]:
    (a.astype("<f8")).tofile(os.path.join(out, f"bwd_{name}.bin"))
# analytic grads
sig = 1.0 / (1.0 + np.exp(-g))
silu = g * sig
silup = sig + g * sig * (1.0 - sig)
dg = dhu * u * silup
du = dhu * silu
np.savetxt(os.path.join(out, "bwd_dg_ref.txt"), dg, fmt="%.12g")
np.savetxt(os.path.join(out, "bwd_du_ref.txt"), du, fmt="%.12g")
np.savetxt(os.path.join(out, "bwd_da_ref.txt"), dy, fmt="%.12g")
np.savetxt(os.path.join(out, "bwd_db_ref.txt"), dy, fmt="%.12g")
print("BWD_REF_DONE n=", n)
