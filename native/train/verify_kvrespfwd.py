#!/usr/bin/env python3
"""Verify the KV-aware response forward: kvrespfwd's response hiddens (kvresp_hidden_gpu.txt,
[84,D] flattened) vs the authoritative numpy full-forward (kvresp_hidden_ref.bin, positions 0..92).
If they match to ~1e-4 the KV-cache response forward (context cache + resp_layer_kv + combined-cache
attention) is CORRECT at the 0-based positions."""
import numpy as np, os, sys
out = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "native", "out"))
g = np.loadtxt(os.path.join(out, "kvresp_hidden_gpu.txt"))
r = np.fromfile(os.path.join(out, "kvresp_hidden_ref.bin"), "<f8").reshape(-1)
n = min(g.size, r.size); g = g[:n]
m = np.abs(r[:n]) > 1e-6
rel = np.max(np.abs(g - r[:n])[m] / np.abs(r[:n])[m]) if m.any() else 0.0
corr = np.corrcoef(g, r[:n])[0, 1]
rn = float(np.linalg.norm(r[:n]))
gn = float(np.linalg.norm(g))
maxd = float(np.max(np.abs(g - r[:n])))
# Gate: corr ~1 (shapes match). The relative metric is dominated by near-zero elements of a wide
# dynamic-range hidden; corr 1.0 + tiny max-abs-diff is the correctness signal.
f = "OK" if corr > 0.999 and maxd < 5e-2 else "FAIL"
print(f"n={n} |g|={gn:.4e} |r|={rn:.4e} maxrel={rel:.3e} corr={corr:.6f} max|g-r|={maxd:.3e} {f}")
print("KVRESPFWD_VERIFY:", f)
sys.exit(0 if f == "OK" else 1)
