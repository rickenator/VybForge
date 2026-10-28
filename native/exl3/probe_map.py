"""Empirically extract the EXL3 4bpw window->slot mapping from the oracle.

All-zero trellis must decode to a constant (validates decode_mcg exactly).
Then flip single nibbles and watch which output slots move.
"""

import sys

import numpy as np
import torch
from exllamav3.ext import exllamav3_ext as ext

dev = "cuda:0"
K = 4.0

def recon(trellis_np):
    tr = torch.from_numpy(trellis_np.copy()).to(dev)
    k, n = tr.shape[0] * 16, tr.shape[1] * 16
    w = torch.empty((k, n), dtype=torch.float16, device=dev)
    ext.reconstruct(w, tr, K, True, False)
    torch.cuda.synchronize()
    return w.float().cpu().numpy()

# --- 0. decode validation: all zeros -> constant? ---
base = recon(np.zeros((1, 8, 64), dtype=np.int16))
vals = np.unique(base)
print("all-zero oracle: unique values:", vals[:4], "count:", len(vals))

# my decode of window 0:
import numpy as np
prod = np.uint32(0) * np.uint32(0xCBAC1FED)
y = (prod & np.uint32(0x8FFF8FFF)) ^ np.uint32(0x3B603B60)
lo = (y & np.uint32(0xFFFF)).astype(np.uint16).view(np.float16)
hi = (y >> np.uint32(16)).astype(np.uint16).view(np.float16)
print("my decode(0):", float(lo) + float(hi))

# --- 1. single-nibble probes ---
t0 = np.zeros((1, 8, 64), dtype=np.int16)
d0 = recon(t0)
t16 = t0.reshape(8, 64).astype(np.uint32)  # 8 subtiles, 64 uint16 each

for m in (0, 1, 2, 3, 4, 7, 8, 255):
    t = t16.copy()
    word, off = m // 4, 4 * (m % 4)
    t[0, word] |= np.uint32(0xF << off)
    d = recon(t.reshape(1, 8, 64).astype(np.int16))
    delta = np.abs(d[0:16, 0:128] - d0[0:16, 0:128]) > 1e-4
    pos = np.argwhere(delta)
    print(f"nibble {m:3d} (word {word}, off {off:2d}): {len(pos)} slots moved -> {pos[:10].tolist()}")
