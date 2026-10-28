"""Build the COMPLETE empirical nibble->slot map for one 16x16 EXL3 tile.

b-windows (j%8 in 0..4, 160 slots): sliding-window model, verified.
s-windows (j%8 in 5..7, 96 slots): content model resisted hand-derivation, so
derive each slot's 16-bit window composition directly from single-nibble probes:

  for every nibble m: set nibble m = 0xF (others 0), record which s-slots move
  and the value each shows. A slot moved by nibbles {m...} with value
  dec(nib at bit position p) pins (m -> p) for that slot's 16-bit window.

Output: nibble_slot_map.npy = for each slot, the list of (nibble, bitpos) that
compose it — the ground truth the reference (and later the Vyb port) uses.
"""

import collections
import json

import numpy as np
import torch
from exllamav3.ext import exllamav3_ext as ext

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

# ---- known b-window map (j%8 in 0..4): sliding window, lane L=j//8 ----
bmap = {}
for j in range(256):
    L, k = j // 8, j % 8
    if k >= 4:
        continue
    r0 = 2 * (L % 4)
    col = 2 * (L // 8) + (8 if k < 4 else 0) + (1 if L % 8 >= 4 else 0)
    row = (r0, r0 + 1, r0 + 8, r0 + 9)[(3 - k) % 4]
    bmap[j] = (row, col)
bslots = set(bmap.values())

# ---- response matrix over all 256 nibbles ----
base = recon(np.zeros((1, 8, 64), dtype=np.int16))[0:16, 0:16]
assert float(np.abs(base - decw(0)).max()) == 0.0, "baseline != dec(0)"

resp = collections.defaultdict(dict)   # slot -> {nibble: value}
for m in range(256):
    t = np.zeros((8, 64), dtype=np.uint32)
    t[0, m // 4] |= np.uint32(0xF) << np.uint32(4 * (m % 4))
    out = recon(t.reshape(1, 8, 64).astype(np.int16))[0:16, 0:16]
    moved = np.argwhere(np.abs(out - base) > 1e-4)
    for r, c in moved:
        resp[(int(r), int(c))][m] = float(np.float16(out[r, c]))

s_slots = sorted(set(resp) - bslots)
print(f"slots responding: {len(resp)} (b {len(bslots)}, s {len(s_slots)})")

# ---- per s-slot: window composition from the responding nibbles ----
# each slot should respond to exactly 4 nibbles; the value dec(0xF << 4p) for the
# probe nibble identifies that nibble's bit position p in the slot's window.
POS_OF_VAL = {}
for p in range(4):
    POS_OF_VAL[round(decw(0xF << np.uint32(4 * p)), 5)] = p

slot_comp = {}     # slot -> list of (nibble, bitpos)
weird = []
for slot in s_slots:
    comp = []
    for m, v in resp[slot].items():
        key = round(v, 5)
        if key in POS_OF_VAL:
            comp.append((m, POS_OF_VAL[key]))
        else:
            weird.append((slot, m, v))
    comp.sort()
    slot_comp[slot] = comp
print("unexpected probe values:", weird[:6], f"(total {len(weird)})")

sizes = collections.Counter(len(c) for c in slot_comp.values())
print("s-slot composition sizes:", dict(sizes))

# ---- assemble the full nibble -> slot map ----
# b-windows: window j -> slot (sliding content, nibbles j..j+3 at positions 0..3)
nibble_to_slot = {}
for j, slot in bmap.items():
    for pos in range(4):
        nibble_to_slot[(j + pos) % 256] = slot
# s-windows: slot -> its nibbles (each slot owns its 4 nibbles exclusively)
for slot, comp in slot_comp.items():
    for m, p in comp:
        if m in nibble_to_slot and nibble_to_slot[m] != slot:
            print(f"CONFLICT nibble {m}: {nibble_to_slot[m]} vs {slot}")
        nibble_to_slot[m] = slot
print("nibbles mapped:", len(nibble_to_slot), "of 256")

out = np.zeros((256, 2), dtype=np.int32)
missing = [m for m in range(256) if m not in nibble_to_slot]
print("missing nibbles:", missing)
if not missing:
    for m in range(256):
        out[m] = nibble_to_slot[m]
    np.save("nibble_slot_map.npy", out)
    print("saved nibble_slot_map.npy")
    json.dump({str(k): v for k, v in slot_comp.items()}, open("s_slot_composition.json", "w"), indent=0)
    print("saved s_slot_composition.json")
