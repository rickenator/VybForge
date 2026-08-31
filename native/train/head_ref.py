#!/usr/bin/env python3
"""Reference for the GPU CE-head probe: output_norm -> tied lm_head -> softmax CE -> head-back.
Uses REAL D=2560/S=4 but a SMALL synthetic EMB vocab (V=100) so the numpy side is cheap while
still exercising the exact head math the corpus-CE loop needs (embed, hgemm, softmax-CE + dlog,
head-back, rmsnorm back). Writes input .bin files (read by head_probe.vyb) + reference outputs."""
import os, numpy as np
repo = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
out = os.path.join(repo, "native", "out")
D, S, V, EPS = 2560, 4, 100, 1e-6
rng = np.random.default_rng(0)
EMB = rng.normal(0, 0.02, size=(V, D)).astype("<f8")   # [VOCAB, D]
ON  = rng.uniform(0.9, 1.1, size=(D,)).astype("<f8")
xo  = rng.normal(0, 0.1, size=(S, D)).astype("<f8")
ids = np.array([13, 7, 42, 99], dtype=np.int64)
labels = np.array([7, 42, 99, 5], dtype=np.int64)

def rmsnorm(v, w):
    return v * (1.0 / np.sqrt((v ** 2).mean(-1, keepdims=True) + EPS)) * w

def rmsnorm_layer(v_, gy, wn):
    inv = 1.0 / np.sqrt((v_ ** 2).mean() + EPS)
    T = (gy * wn * v_).sum()
    return gy * wn * inv - v_ * (inv ** 3 / D) * T

x0 = EMB[ids]                 # embed
h = rmsnorm(xo, ON)           # output_norm
logits = h @ EMB.T            # [S,V]
lse = logits.max(-1, keepdims=True)
e = np.exp(logits - lse); p = e / e.sum(-1, keepdims=True)
loss = float(-np.log(p[np.arange(S), labels] + 1e-30).mean())
dlog = p.copy(); dlog[np.arange(S), labels] -= 1.0; dlog /= S
dh = dlog @ EMB
dxo = np.stack([rmsnorm_layer(xo[s], dh[s], ON) for s in range(S)])

for name, arr in [("head_emb", EMB), ("head_on", ON), ("head_xo", xo)]:
    arr.tofile(os.path.join(out, name + ".bin"))
ids.tofile(os.path.join(out, "head_ids.bin")); labels.tofile(os.path.join(out, "head_labels.bin"))
(numpy_embedded := EMB[ids]).tofile(os.path.join(out, "head_embedded.bin"))
np.savetxt(os.path.join(out, "head_ref_embedded.txt"), numpy_embedded.reshape(-1), fmt="%.9g")
# refs
np.savetxt(os.path.join(out, "head_ref_logits.txt"), logits.reshape(-1), fmt="%.9g")
np.savetxt(os.path.join(out, "head_ref_dlg.txt"), dlog.reshape(-1), fmt="%.9g")
np.savetxt(os.path.join(out, "head_ref_dh.txt"), dh.reshape(-1), fmt="%.9g")
np.savetxt(os.path.join(out, "head_ref_dxo.txt"), dxo.reshape(-1), fmt="%.9g")
np.savetxt(os.path.join(out, "head_ref_plab.txt"), p[np.arange(S), labels], fmt="%.9g")
open(os.path.join(out, "head_ref_loss.txt"), "w").write(f"{loss:.10g}\n")
print("HEAD_REF_DONE loss", loss, "descend-ref")
