#!/usr/bin/env python3
"""Standalone repro reference: two adamw launches (n=8 U-like, n=6 V-like) with KNOWN
gradients. Expected v-moments = 0.001 * g^2 (init m=v=0, b2=0.999). Verification-only."""
import numpy as np, os
out = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "native", "out"))
gU = (np.arange(8) * 0.5 + 0.3).astype("<f8")
gV = (np.arange(6) * 0.3 + 0.1).astype("<f8")
gU.tofile(os.path.join(out, "rp_gU.bin")); gV.tofile(os.path.join(out, "rp_gV.bin"))
np.savetxt(os.path.join(out, "rp_vU_ref.txt"), 0.001 * gU ** 2, fmt="%.12g")
np.savetxt(os.path.join(out, "rp_vV_ref.txt"), 0.001 * gV ** 2, fmt="%.12g")
print("ADAMW_REPRO_REF_DONE")
