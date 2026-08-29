#!/usr/bin/env python3
"""Reference for T5 (training-step driver): a LoRA OUTPUT-PROJECTION training loop in numpy.
  y = x@W + alpha_r*((x@U)@V),  MSE loss = 0.5*sum((y-t)^2)  (no /N)
Backward: dy=y-t; gv=alpha_r*dy; s=x@U; ds=gv@V^T; dU=x^T@ds; dV=s^T@gv.
AdamW updates U and V only (W frozen), bias-corrected m/v, lr/b1/b2/eps/wd.
Writes initial weights/inputs and the FINAL U,V + per-step losses as the reference.
The Vyb driver (t5_probe.vyb) must reproduce the same final U,V (and hence the loss descent).
"""
import numpy as np, os
out = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "native", "out"))
S, IN, R, OUT = 3, 4, 2, 3
alpha_r, lr, b1, b2, eps, wd, NSTEPS = 2.0, 0.05, 0.9, 0.999, 1e-8, 0.01, 5
rng = np.random.default_rng(41)
x = rng.normal(size=(S, IN)); t = rng.normal(size=(S, OUT))
W = rng.normal(size=(IN, OUT)); W *= 0.2
U = rng.normal(size=(IN, R)) * 0.1; V = rng.normal(size=(R, OUT)) * 0.1
mU = np.zeros_like(U); vU = np.zeros_like(U)
mV = np.zeros_like(V); vV = np.zeros_like(V)
for nm, a in [("x", x), ("t", t), ("W", W), ("U", U), ("V", V)]:
    a.astype("<f8").tofile(os.path.join(out, f"t5_{nm}.bin"))
mU.astype("<f8").tofile(os.path.join(out, "t5_mU.bin")); vU.astype("<f8").tofile(os.path.join(out, "t5_vU.bin"))
mV.astype("<f8").tofile(os.path.join(out, "t5_mV.bin")); vV.astype("<f8").tofile(os.path.join(out, "t5_vV.bin"))

b1t, b2t = 1.0, 1.0
losses = []
for step in range(1, NSTEPS + 1):
    y = x @ W + alpha_r * ((x @ U) @ V)
    e = y - t
    losses.append(0.5 * (e ** 2).sum())
    gv = alpha_r * e
    s = x @ U
    ds = gv @ V.T
    dU = x.T @ ds
    dV = s.T @ gv
    b1t *= b1; b2t *= b2
    # IN-PLACE moment/param updates (mU/vU/mV/vV must persist across steps; a plain
    # `m = b1*m + ...` rebinds the local and never writes the persistent array).
    mU[:] = b1 * mU + (1 - b1) * dU
    vU[:] = b2 * vU + (1 - b2) * dU ** 2
    mUhat = mU / (1 - b1t); vUhat = vU / (1 - b2t)
    U[:] = U - lr * (mUhat / (np.sqrt(vUhat) + eps) + wd * U)
    mV[:] = b1 * mV + (1 - b1) * dV
    vV[:] = b2 * vV + (1 - b2) * dV ** 2
    mVhat = mV / (1 - b1t); vVhat = vV / (1 - b2t)
    V[:] = V - lr * (mVhat / (np.sqrt(vVhat) + eps) + wd * V)
    np.savetxt(os.path.join(out, f"t5_U_ref_s{step}.txt"), U.reshape(-1), fmt="%.12g")
    np.savetxt(os.path.join(out, f"t5_V_ref_s{step}.txt"), V.reshape(-1), fmt="%.12g")
np.savetxt(os.path.join(out, "t5_Ufinal_ref.txt"), U.reshape(-1), fmt="%.12g")
np.savetxt(os.path.join(out, "t5_Vfinal_ref.txt"), V.reshape(-1), fmt="%.12g")
np.savetxt(os.path.join(out, "t5_losses_ref.txt"), np.array(losses), fmt="%.8g")
print("T5_REF_DONE losses:", [round(l, 4) for l in losses])
