#!/usr/bin/env python3
# Python REFERENCE VERIFICATION for the Vyb GPU one-layer forward.
# Reads the inputs/weights + GPU output dumped by layer_driver.vyb (Vyb-native
# GPU path), recomputes the forward with numpy, and compares. Verification only.
import numpy as np, os

rd = {}
base = os.path.join(os.path.dirname(__file__), "..", "out")
def load(n):
    return np.loadtxt(os.path.join(base, n))

S, D, H, HD, KVH, HM = 4, 16, 4, 4, 2, 32
NCOL, NKCOL = H*HD, KVH*HD
eps = 1e-6

X   = load("weights.txt").reshape(S, D)
w1  = load("w1.txt")
w2  = load("w2.txt")
Wq  = load("wq.txt").reshape(D, NCOL)
Wk  = load("wk.txt").reshape(D, NKCOL)
Wv  = load("wv.txt").reshape(D, NKCOL)
Wo  = load("wo.txt").reshape(D, D)
Wg  = load("wg.txt").reshape(D, HM)
Wu  = load("wu.txt").reshape(D, HM)
Wd  = load("wd.txt").reshape(HM, D)
freq = load("freq.txt")
gpu_out = load("out.txt").reshape(S, D)

def rmsnorm(x, w):
    r = 1.0 / np.sqrt((x**2).mean(axis=1, keepdims=True) + eps)
    return x * r * w

def rope(mat, cols, nheads, hd):
    # mat[S, cols]; rotate per (s, head, i-half)
    out = mat.copy().reshape(S, nheads, hd)
    for s in range(S):
        for j in range(hd//2):
            ang = s * freq[j]
            c, sn = np.cos(ang), np.sin(ang)
            q0 = out[s, :, 2*j]; q1 = out[s, :, 2*j+1]
            out[s, :, 2*j]   = q0*c - q1*sn
            out[s, :, 2*j+1] = q0*sn + q1*c
    return out.reshape(S, cols)

RN1 = rmsnorm(X, w1)
Q = RN1 @ Wq; K = RN1 @ Wk; V = RN1 @ Wv
Q = rope(Q, NCOL, H, HD); K = rope(K, NKCOL, KVH, HD)
scale = 1.0/np.sqrt(HD)
Ctx = np.zeros((S, NCOL))
for s in range(S):
    for h in range(H):
        kvh = h // (H//KVH)
        q = Q[s, h*HD:(h+1)*HD]
        Ks = K[:s+1, kvh*HD:(kvh+1)*HD]      # causal t<=s
        logits = (q @ Ks.T) * scale
        a = np.exp(logits - logits.max())
        a = a / a.sum()
        Ctx[s, h*HD:(h+1)*HD] = a @ V[:s+1, kvh*HD:(kvh+1)*HD]
O = Ctx @ Wo
X1 = X + O
RN2 = rmsnorm(X1, w2)
G = RN2 @ Wg; U = RN2 @ Wu
Hu = (G / (1 + np.exp(-G))) * U      # SiLU
M = Hu @ Wd
ref_out = X1 + M

diff = np.abs(ref_out - gpu_out)
print("reference forward computed")
print("GPU out[:3] =", gpu_out[0,:3])
print("ref  out[:3] =", ref_out[0,:3])
print("max abs diff =", diff.max())
print("mean abs diff =", diff.mean())
# GPU uses Vyb-native polynomial exp/sin/cos (~1e-10), integrated over the whole
# forward on O(1..100) random weights: max ~2e-4 (rel ~1e-5) is a correct layer.
good = diff.max() < 1e-3
print("LAYER_VERIFY:", "OK" if good else "FAIL")
