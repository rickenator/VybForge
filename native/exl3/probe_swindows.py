"""Pin the EXL3 s-window (j%8 in 5,6,7) contents and slots empirically.

For m in 0..15: set nibble m = 0xF (others 0), record moved slots + values.
A window starting at j contains nibble m iff j <= m <= j+3; its value is then
dec(w16 with nibble m at bit position 4*(m-j)). Cross-probe matching pins
(window start -> slot) for j in -3..15, which covers the full 8-periodic rule.
"""

import numpy as np
import torch
from exllamav3.ext import exllamav3_ext as ext

dev = "cuda:0"
MAGIC = np.uint32(0xCBAC1FED)
AMASK = np.uint32(0x8FFF8FFF)
XOR = np.uint32(0x3B603B60)

def decw(nibs):
    """nibs: tuple of 4 nibble values (LSB first) -> fp16 value as float."""
    w = np.uint32(nibs[0] | (nibs[1] << 4) | (nibs[2] << 8) | (nibs[3] << 12))
    with np.errstate(over="ignore"):
        prod = (w * MAGIC) & np.uint32(0xFFFFFFFF)
    y = (prod & AMASK) ^ XOR
    lo = np.uint16(y & np.uint32(0xFFFF)).view(np.float16)
    hi = np.uint16(y >> np.uint32(16)).view(np.float16)
    return float(np.float16(float(lo) + float(hi)))

def recon(trellis_np):
    tr = torch.from_numpy(trellis_np.copy()).to(dev)
    w = torch.empty((tr.shape[0] * 16, tr.shape[1] * 16), dtype=torch.float16, device=dev)
    ext.reconstruct(w, tr, 4.0, True, False)
    torch.cuda.synchronize()
    return w.float().cpu().numpy()

base = recon(np.zeros((1, 8, 64), dtype=np.int16))[0:16, 0:16]

# window start -> slot (to fill)
wslot = {}
for m in range(16):
    t = np.zeros((8, 64), dtype=np.uint32)
    t[0, m // 4] |= np.uint32(0xF) << np.uint32(4 * (m % 4))
    out = recon(t.reshape(1, 8, 64).astype(np.int16))[0:16, 0:16]
    moved = np.argwhere(np.abs(out - base) > 1e-4)
    # candidate windows containing nibble m
    cands = {}
    for j in range(m - 3, m + 1):
        nibs = [0, 0, 0, 0]
        nibs[m - j] = 0xF
        cands[decw(nibs)] = j % 256
    line = []
    for r, c in moved:
        v = float(np.float16(out[r, c]))
        if v in cands:
            j = cands[v]
            wslot[j] = (int(r), int(c))
            line.append(f"win{j}->({r},{c})")
        else:
            line.append(f"({r},{c})={v:.5g}?")
    print(f"nibble {m:2d}: {'  '.join(line)}")

print()
for j in sorted(wslot):
    print(f"window {j:3d} (j%8={j%8}, lane {j//8}): slot {wslot[j]}")
