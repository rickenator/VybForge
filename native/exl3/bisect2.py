"""Bisect v2: rotated-basis (pre-Hadamard) compare on the real fixture."""
import sys

import numpy as np
import torch

sys.path.insert(0, "/home/rick/Projects/VybForge/native/exl3")
from exl3_ref import load_fixture, load_slot_composition, decode_mcg  # noqa: E402
from exllamav3.ext import exllamav3_ext as ext  # noqa: E402

d = load_fixture("fixture")
trellis = d["trellis"]
kb, nb, _ = trellis.shape
k, n = kb * 16, nb * 16

t = trellis.reshape(kb * nb, 64).astype(np.uint32)
bits = ((t[:, :, None] >> np.uint32(np.arange(16))) & np.uint32(1)).reshape(kb * nb, 1024).astype(np.uint8)
nib = (bits[:, 0::4] | (bits[:, 1::4] << 1) | (bits[:, 2::4] << 2) | (bits[:, 3::4] << 3))

slots, cidx, cpos = load_slot_composition()
w16 = np.zeros((kb * nb, 256), dtype=np.uint32)
for i in range(4):
    w16 |= nib[:, cidx[:, i]].astype(np.uint32) << np.uint32(4 * cpos[:, i])
vals = decode_mcg(w16)

what = np.zeros((k, n), dtype=np.float32)
rows = np.array([s[0] for s in slots])
cols = np.array([s[1] for s in slots])
rr = (np.arange(kb)[:, None, None] * 16 + rows[None, None, :])
cc = (np.arange(nb)[None, :, None] * 16 + cols[None, None, :])
what[rr, cc] = vals.reshape(kb, nb, 256)

tr = torch.from_numpy(trellis.copy()).to("cuda:0")
w_or = torch.empty((k, n), dtype=torch.float16, device="cuda:0")
ext.reconstruct(w_or, tr, 4.0, True, False)
torch.cuda.synchronize()
w_or = w_or.float().cpu().numpy()

diff = np.abs(what - w_or)
print(f"what std {what.std():.6f}  oracle std {w_or.std():.6f}")
print(f"max abs diff {diff.max():.3e}  frac>3e-3 {(diff > 3e-3).mean():.4f}")
tdiff = diff.reshape(kb, 16, nb, 16).transpose(0, 2, 1, 3).reshape(kb * nb, 16, 16)
ok_tiles = int((tdiff.max(axis=(1, 2)) <= 3e-3).sum())
print(f"tiles fully OK: {ok_tiles} / {kb * nb}")
