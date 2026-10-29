"""EXL3_VYB_MATCH gate: Vyb exl3deq kernel output vs the oracle-verified numpy
reference (exl3_ref), on the real GLM-5.3 expert fixture.

  A. rotated basis: Vyb f64 output vs reference What (tolerance absorbs the
     reference's fp16 intermediate rounding of the codebook sum: |diff| bound
     1.5e-3 relative).
  B. full pipeline: per-128-block Hadamard + suh/svh applied to the Vyb output
     in numpy, fp16-cast, vs exl3_ref.dequant_exl3 (which is itself
     oracle-verified, EXL3_REF_VERIFY: OK). Gate 3e-3 absolute.

Usage: python3 native/exl3/verify_exl3_vyb.py [fixture_dir] [vyb_txt]
"""

import json
import sys

import numpy as np

sys.path.insert(0, "/home/rick/Projects/VybForge/native/exl3")
from exl3_ref import decode_mcg, dequant_exl3, hadamard128, load_fixture, R_SCALE  # noqa: E402

fixture = sys.argv[1] if len(sys.argv) > 1 else "/home/rick/Projects/VybForge/native/exl3/fixture"
vyb_txt = sys.argv[2] if len(sys.argv) > 2 else "/home/rick/Projects/VybForge/native/out/exl3_rotated.txt"

d = load_fixture(fixture)
trellis = d["trellis"]
kb, nb, _ = trellis.shape
k, n = kb * 16, nb * 16

# --- reference rotated basis (bit-exact vs oracle, bisect2.py) ---
t = trellis.reshape(kb * nb, 64).astype(np.uint32)
bits = ((t[:, :, None] >> np.uint32(np.arange(16))) & np.uint32(1)).reshape(kb * nb, 1024).astype(np.uint8)
nib = (bits[:, 0::4] | (bits[:, 1::4] << 1) | (bits[:, 2::4] << 2) | (bits[:, 3::4] << 3)).astype(np.uint32)
slots, cidx, cpos = None, None, None
raw = json.load(open("/home/rick/Projects/VybForge/native/exl3/tile_slot_composition.json"))
skeys = sorted(raw, key=lambda kk: tuple(int(x) for x in kk.split(",")))
rows = np.array([int(x) for kk in skeys for x in kk.split(",")])[0::2]
cols = np.array([int(x) for kk in skeys for x in kk.split(",")])[1::2]
cidx = np.array([[int(m) for m, _ in raw[kk]] for kk in skeys], dtype=np.int64)
cpos = np.array([[int(p) for _, p in raw[kk]] for kk in skeys], dtype=np.int64)
w16 = np.zeros((kb * nb, 256), dtype=np.uint32)
for i in range(4):
    w16 |= nib[:, cidx[:, i]] << np.uint32(4 * cpos[:, i])
vals = decode_mcg(w16)
what_ref = np.zeros((k, n), dtype=np.float32)
what_ref[(np.arange(kb)[:, None, None] * 16 + rows[None, None, :]),
         (np.arange(nb)[None, :, None] * 16 + cols[None, None, :])] = vals.reshape(kb, nb, 256)

# --- Vyb kernel output ---
txt = open(vyb_txt).read()
vyb = np.array(txt.split(), dtype=np.float64).reshape(k, n)
del txt

diff = np.abs(vyb - what_ref)
rel = np.abs(vyb - what_ref) / np.maximum(np.abs(what_ref), 1.0)
print(f"A. rotated basis: max abs {diff.max():.3e}  max rel {rel.max():.3e}")
gate_a = rel.max() <= 1.5e-3

# --- B. full pipeline, entirely on the GPU (Hadamard + suh/svh in Vyb) ---
w_txt = open(vyb_txt.replace("exl3_rotated", "exl3_w")).read()
w_vyb = np.array(w_txt.split(), dtype=np.float64).reshape(k, n)
del w_txt
w_ref_full = dequant_exl3(trellis, d["suh"], d["svh"]).astype(np.float32)
diff_b = np.abs(w_vyb - w_ref_full)
print(f"B. full W (GPU):  max abs {diff_b.max():.3e}  |diff|>3e-3: {int((diff_b > 3e-3).sum())}")
gate_b = bool((diff_b > 3e-3).sum() == 0)

ok = gate_a and gate_b
print("EXL3_VYB_MATCH:", "OK" if ok else "FAIL")
sys.exit(0 if ok else 1)
