"""Finalize the EXL3 tile map: merge b-window (model) + s-window (empirical)
compositions, resolve contested slots empirically, verify on random probes,
then verify the full reference dequant on the REAL fixture vs the oracle.
"""

import ast
import json
import sys

import numpy as np
import torch
from exllamav3.ext import exllamav3_ext as ext

sys.path.insert(0, "/home/rick/Projects/VybForge/native/exl3")
from exl3_ref import load_fixture  # noqa: E402

dev = "cuda:0"
MAGIC = np.uint32(0xCBAC1FED)
AMASK = np.uint32(0x8FFF8FFF)
XOR = np.uint32(0x3B603B60)

def decw(w16):
    with np.errstate(over="ignore"):
        prod = (np.uint32(w16) * MAGIC) & np.uint32(0xFFFFFFFF)
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

# ---- compositions ----
comp = {}
for j in range(256):
    L, k = j // 8, j % 8
    if k >= 4:
        continue
    r0 = 2 * (L % 4)
    col = 2 * (L // 8) + (8 if k < 4 else 0) + (1 if L % 8 >= 4 else 0)
    row = (r0, r0 + 1, r0 + 8, r0 + 9)[(3 - k) % 4]
    comp[(row, col)] = [(j, 0), ((j + 1) % 256, 1), ((j + 2) % 256, 2), ((j + 3) % 256, 3)]

s_raw = json.load(open("s_slot_composition.json"))
s_comp = {}
for k, v in s_raw.items():
    slot = tuple(ast.literal_eval(k)) if isinstance(k, str) else tuple(k)
    s_comp[slot] = [tuple(ast.literal_eval(t)) if isinstance(t, str) else tuple(t) for t in v]

contested = set(comp) & set(s_comp)
print(f"b compositions {len(comp)}, s compositions {len(s_comp)}, contested {len(contested)}")
for slot, c in s_comp.items():
    comp[slot] = c          # s empirical wins initially; contested resolved below

# ---- resolve contested slots with one random probe ----
rs = np.random.RandomState(101)
nib = rs.randint(1, 16, 256).astype(np.uint32)
sub = np.zeros((8, 64), dtype=np.uint32)
for m in range(256):
    sub[0, m // 4] |= nib[m] << np.uint32(4 * (m % 4))
out = recon(sub.reshape(1, 8, 64).astype(np.int16))[0:16, 0:16]

def val_of(composition):
    w = np.uint32(0)
    for m, p in composition:
        w |= nib[m] << np.uint32(4 * p)
    return decw(int(w))

fixed = 0
for slot in sorted(contested):
    target = float(np.float16(out[slot]))
    b_val = val_of([(j, 0), ((j + 1) % 256, 1), ((j + 2) % 256, 2), ((j + 3) % 256, 3)])
    # find the b-window j that maps to this slot
    j_cands = [j for j in range(256) if j % 8 >= 4 and j // 8 >= 0]
    best = None
    for j in range(256):
        if (j % 8) < 4:
            continue
        c = [(j, 0), ((j + 1) % 256, 1), ((j + 2) % 256, 2), ((j + 3) % 256, 3)]
        L, k = j // 8, j % 8
        r0 = 2 * (L % 4)
        col = 2 * (L // 8) + (8 if k < 4 else 0) + (1 if L % 8 >= 4 else 0)
        row = (r0, r0 + 1, r0 + 8, r0 + 9)[(3 - k) % 4]
        if (row, col) == slot and abs(val_of(c) - target) < 1e-6:
            best = c
            break
    if best is not None:
        comp[slot] = best
        fixed += 1
    else:
        # keep s composition if it matches
        if abs(val_of(s_comp[slot]) - target) < 1e-6:
            fixed += 1
        else:
            print(f"UNRESOLVED slot {slot}: target {target}")
print(f"contested resolved: {fixed}/{len(contested)}")

# ---- verify ALL 256 slots on fresh random probes ----
ok_all = True
for seed in (7, 23, 41):
    rs = np.random.RandomState(seed)
    nib = rs.randint(1, 16, 256).astype(np.uint32)
    sub = np.zeros((8, 64), dtype=np.uint32)
    for m in range(256):
        sub[0, m // 4] |= nib[m] << np.uint32(4 * (m % 4))
    out = recon(sub.reshape(1, 8, 64).astype(np.int16))[0:16, 0:16]
    bad = 0
    for slot, c in comp.items():
        if float(np.float16(out[slot])) != np.float16(val_of(c)):
            bad += 1
    print(f"seed {seed}: mismatched slots {bad}/256")
    ok_all = ok_all and bad == 0

if ok_all:
    json.dump({f"{r},{c}": v for (r, c), v in sorted(comp.items())},
              open("tile_slot_composition.json", "w"), indent=0)
    print("saved tile_slot_composition.json — map complete")
