"""Identify the kernel's effective Hadamard ordering from the oracle output."""
import sys

import numpy as np
import torch

sys.path.insert(0, "/home/rick/Projects/VybForge/native/exl3")
from exl3_ref import load_fixture, load_slot_composition, decode_mcg, hadamard128, R_SCALE  # noqa: E402
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
ext.reconstruct_had_slice(w_or, tr,
                          torch.from_numpy(d["suh"].copy()).to("cuda:0"),
                          torch.from_numpy(d["svh"].copy()).to("cuda:0"),
                          4.0, True, False, 0)
torch.cuda.synchronize()
w_or = w_or.float().cpu().numpy()

H = hadamard128()
n_bits = 7
def bitrev(p):
    r = 0
    for b in range(n_bits):
        r = (r << 1) | ((p >> b) & 1)
    return r
P = np.eye(128)[[bitrev(i) for i in range(128)]]
Hbr = P @ H @ P

suh = d["suh"].astype(np.float32)
svh = d["svh"].astype(np.float32)
i0, j0 = 0, 0
Wt = what[i0 * 128:(i0 + 1) * 128, j0 * 128:(j0 + 1) * 128]
M = w_or[i0 * 128:(i0 + 1) * 128, j0 * 128:(j0 + 1) * 128] / (suh[i0 * 128:(i0 + 1) * 128, None] * svh[None, j0 * 128:(j0 + 1) * 128])

cands = {
    "H@W@H (natural)": H @ Wt @ H,
    "Hbr@W@Hbr (bitrev both)": Hbr @ Wt @ Hbr,
    "H@W@Hbr (col nat, row br)": H @ Wt @ Hbr,
    "Hbr@W@H (col br, row nat)": Hbr @ Wt @ H,
}
for name, C in cands.items():
    C = C * R_SCALE * R_SCALE
    diff = np.abs(C - M)
    print(f"{name:28s}: max abs diff {diff.max():.4e}")
