"""EXL3_REF_VERIFY: numpy reference vs exllamav3 oracle on the real expert fixture.

Oracle: exllamav3 ext.reconstruct_had_slice (the exact kernel TensorFold serves
GLM-5.3-Flash with), run on godzilla's 3090. Reference: exl3_ref.dequant_exl3.

Usage: env -u PYTHONPATH ~/venvs/exl3/bin/python verify_ref.py [fixture_dir]
"""

import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from exl3_ref import dequant_exl3, load_fixture  # noqa: E402

fixture = sys.argv[1] if len(sys.argv) > 1 else "fixture"
d = load_fixture(fixture)

# --- reference ---
w_ref = dequant_exl3(d["trellis"], d["suh"], d["svh"]).astype(np.float32)

# --- oracle ---
from exllamav3.ext import exllamav3_ext as ext  # noqa: E402

dev = "cuda:0"
trellis = torch.from_numpy(d["trellis"]).to(dev)          # int16 [k/16, n/16, 64]
suh = torch.from_numpy(d["suh"]).to(dev)                  # fp16 [k]
svh = torch.from_numpy(d["svh"]).to(dev)                  # fp16 [n]
K = float(trellis.shape[-1] // 16)
w_or = torch.empty((trellis.shape[0] * 16, trellis.shape[1] * 16),
                   dtype=torch.float16, device=dev)
ext.reconstruct_had_slice(w_or, trellis, suh, svh, K, True, False, 0)
torch.cuda.synchronize()
w_or = w_or.float().cpu().numpy()

diff = np.abs(w_ref - w_or)
denom = np.maximum(np.abs(w_or), 1e-8)
maxabs = float(diff.max())
maxrel = float((diff / denom).max())
bad3 = int((diff > 3e-3).sum())
scale = float(np.abs(w_or).std())
print(f"mcg id          : {d['mcg'].item()}")
print(f"shape           : {w_or.shape}")
print(f"oracle std      : {scale:.6f}  min {w_or.min():.6f} max {w_or.max():.6f}")
print(f"ref    std      : {np.abs(w_ref).std():.6f}")
print(f"max abs diff    : {maxabs:.3e}")
print(f"max rel diff    : {maxrel:.3e}")
print(f"|diff| > 3e-3   : {bad3} of {diff.size}")
ok = bad3 == 0
print("EXL3_REF_VERIFY:", "OK" if ok else "FAIL")
sys.exit(0 if ok else 1)
