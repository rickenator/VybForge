"""Solve the full EXL3 nibble->slot map with ONE random-filled probe."""

import sys

import numpy as np
import torch
from exllamav3.ext import exllamav3_ext as ext

dev = "cuda:0"
MAGIC = np.uint32(0xCBAC1FED)
AMASK = np.uint32(0x8FFF8FFF)
XOR = np.uint32(0x3B603B60)

def decode_all(nib):
    """nib: uint32[256] -> fp16[256] window values (window j = nibbles j..j+3)."""
    w = (nib[np.arange(256)] |
         (nib[(np.arange(256) + 1) % 256] << np.uint32(4)) |
         (nib[(np.arange(256) + 2) % 256] << np.uint32(8)) |
         (nib[(np.arange(256) + 3) % 256] << np.uint32(12))).astype(np.uint32)
    with np.errstate(over="ignore"):
        prod = (w * MAGIC) & np.uint32(0xFFFFFFFF)
    y = (prod & AMASK) ^ XOR
    lo = (y & np.uint32(0xFFFF)).astype(np.uint16).view(np.float16).astype(np.float64)
    hi = (y >> np.uint32(16)).astype(np.uint16).view(np.float16).astype(np.float64)
    return lo + hi  # keep fp64 for uniqueness; oracle rounds to fp16

def recon(trellis_np):
    tr = torch.from_numpy(trellis_np.copy()).to(dev)
    w = torch.empty((tr.shape[0] * 16, tr.shape[1] * 16), dtype=torch.float16, device=dev)
    ext.reconstruct(w, tr, 4.0, True, False)
    torch.cuda.synchronize()
    return w.float().cpu().numpy()

for seed in range(50):
    rs = np.random.RandomState(seed)
    nib = rs.randint(1, 16, 256).astype(np.uint32)
    sub = np.zeros((8, 64), dtype=np.uint32)
    for m in range(256):
        sub[0, m // 4] |= nib[m] << np.uint32(4 * (m % 4))
    expected = decode_all(nib)
    out = recon(sub.reshape(1, 8, 64).astype(np.int16))[0:16, 0:16].astype(np.float64)
    # window value must equal fp16(expected)
    exp16 = expected.astype(np.float16).astype(np.float64)
    if len(np.unique(exp16)) != 256:
        continue
    slot = {}
    ok = True
    for j in range(256):
        hits = np.argwhere(out == exp16[j])
        if len(hits) != 1:
            ok = False
            break
        slot[j] = (int(hits[0][0]), int(hits[0][1]))
    if not ok or len(set(slot.values())) != 256:
        continue
    print(f"seed {seed}: FULL BIJECTIVE MAP solved")
    m = np.zeros((256, 2), dtype=np.int32)
    for j in range(256):
        m[j] = slot[j]
    np.save("nibble_slot_map.npy", m)
    for rem in range(8):
        js = [j for j in range(256) if j % 8 == rem]
        rows = sorted({slot[j][0] for j in js})
        cols = sorted({slot[j][1] for j in js})
        print(f"j%8={rem}: rows={rows} cols={cols}")
    break
else:
    print("no seed gave a fully unique map")
    sys.exit(1)
