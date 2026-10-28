"""Bisect the EXL3 reference: compare the ROTATED-basis tile decode only
(ext.reconstruct, no Hadamard/scales) vs the reference's `what`.

Usage: env -u PYTHONPATH ~/venvs/exl3/bin/python bisect_ref.py [fixture_dir]
"""

import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import exl3_ref  # noqa: E402
from exl3_ref import load_fixture  # noqa: E402

fixture = sys.argv[1] if len(sys.argv) > 1 else "fixture"
d = load_fixture(fixture)

trellis = d["trellis"]
kb, nb, _ = trellis.shape
k, n = kb * 16, nb * 16

# --- reference What (rotated basis, pre-Hadamard) ---
t = trellis.reshape(kb * nb, 64).astype(np.uint32)
sh = np.uint32([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15])
bits = ((t[:, :, None] >> sh) & np.uint32(1)).reshape(kb * nb, 1024).astype(np.uint8)
nib = (bits[:, 0::4] << 0) | (bits[:, 1::4] << 1) | (bits[:, 2::4] << 2) | (bits[:, 3::4] << 3)
vmap = exl3_ref.tile_value_map()
idx_j = np.array([vmap[j][0] for j in range(256)])
idx_c = np.array([vmap[j][1] for j in range(256)])
tiles = exl3_ref.decode_mcg(nib)
what = np.zeros((kb * nb, 16, 16), dtype=np.float32)
what[:, idx_j, idx_c] = tiles
what = what.reshape(kb, nb, 16, 16).transpose(0, 2, 1, 3).reshape(k, n)

# --- oracle rotated basis ---
from exllamav3.ext import exllamav3_ext as ext  # noqa: E402

dev = "cuda:0"
tr = torch.from_numpy(trellis.copy()).to(dev)
w_or = torch.empty((k, n), dtype=torch.float16, device=dev)
ext.reconstruct(w_or, tr, float(64 // 16), True, False)
torch.cuda.synchronize()
w_or = w_or.float().cpu().numpy()

diff = np.abs(what - w_or)
print(f"what std {what.std():.6f}   oracle std {w_or.std():.6f}")
print(f"max abs diff {diff.max():.3e}   frac>3e-3 {(diff > 3e-3).mean():.4f}")

# how many tiles agree fully?
tdiff = diff.reshape(kb, 16, nb, 16).transpose(0, 2, 1, 3).reshape(kb * nb, 16, 16)
tile_ok = (tdiff.max(axis=(1, 2)) <= 3e-3).sum()
print(f"tiles fully OK: {tile_ok} / {kb * nb}")
