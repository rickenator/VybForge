"""Arbitrate the cw=4096 deq disagreement: compute the oracle's raw What (pre-Hadamard)
for down_proj expert 0 and print the f64-pair reinterpretations of What[32,0:2] and
What[0,1024:1026] in the same format the probe's ODCHK prints (dtoh reads 8 bytes
= two adjacent f32 reinterpreted as one f64).

Oracle decode replicated from loader.dequant_exl3_fast lines 130-152 (tile/slot
placement only; no Hadamard, no scales).
"""
import sys
import numpy as np
from safetensors import safe_open

sys.path.insert(0, "/home/rick/Projects/VybForge/native/glm53")
import verify_layer3_moe as B
from exl3 import exl3_ref
import os

CKPT = B.V.CKPT
idx = B.index()
pre = "model.language_model.layers.3.mlp.experts.0.down_proj."
with safe_open(f"{CKPT}/{idx[pre + 'trellis']}", framework="pt", device="cpu") as f:
    tre = f.get_tensor(pre + "trellis").numpy()

kb, nb, words = trellis_shape = tre.shape
print(f"down trellis shape: {tre.shape}  (kb*16={kb*16} rows, nb*16={nb*16} cols)")
tr = tre.view(np.uint16) if tre.dtype == np.int16 else tre
t = tr.reshape(kb * nb, 64).astype(np.uint32)
shifts = np.uint32(np.arange(16))
bits = ((t[:, :, None] >> shifts) & np.uint32(1)).reshape(kb * nb, 1024).astype(np.uint8)
nib = (bits[:, 0::4] << 0) | (bits[:, 1::4] << 1) | (bits[:, 2::4] << 2) | (bits[:, 3::4] << 3)
slots, cidx, cpos = exl3_ref.load_slot_composition(
    os.path.join("/home/rick/Projects/VybForge/native/exl3", "tile_slot_composition.json"))
w16 = np.zeros((kb * nb, 256), dtype=np.uint32)
for i in range(4):
    w16 |= nib[:, cidx[:, i]].astype(np.uint32) << np.uint32(4 * cpos[:, i])
vals = exl3_ref.decode_mcg(w16)
rows = np.array([s[0] for s in slots])
cols = np.array([s[1] for s in slots])
what = np.zeros((kb * 16, nb * 16), dtype=np.float32)
rr = (np.arange(kb)[:, None, None] * 16 + rows[None, None, :])
cc = (np.arange(nb)[None, :, None] * 16 + cols[None, None, :])
what[rr, cc] = vals.reshape(kb, nb, 256)

def pair64(a, b):
    raw = np.empty(2, dtype=np.float32)
    raw[0], raw[1] = a, b
    return raw.view(np.float64)[0]

print(f"ORACLE What[32,0:2] as f64-pair = {pair64(what[32,0], what[32,1])!r}")
print(f"ORACLE What[0,1024:1026] as f64-pair = {pair64(what[0,1024], what[0,1025])!r}")
print(f"ORACLE What[33,0:2] = {pair64(what[33,0], what[33,1])!r}")
print(f"ORACLE What[1,1024:1026] = {pair64(what[1,1024], what[1,1025])!r}")
print("probe46 ODCHK said: r32c01=0.000169754 r0c1024=-0.000459194 | r33c01=-5.31078e-05 r1c1024=0.148926")
