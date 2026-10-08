#!/usr/bin/env python3
"""Which memory does the op actually read per head?

Three arithmetic setups, each chosen so the op's output directly REVEALS the per-head slice:

  A. state per head = diag(D_h), gate = 0 (decay 1), beta = 0 (no update), q = 1
     => out[j, h] = D_h * q[j] / sqrt(S). Reading out[:, h] shows which D_h that head used.
  B. state = 0, gate = 0, beta = 1, k = e_0, q = e_0, v[:, h] = V_h
     => out[j, h] = V_h exactly. Reading out[:, h] shows which v head that head used.
  C. state per head = diag(1), beta = 0, q = e_0, gate = G_h
     => out[0, h] = exp(G_h). Shows which gate entry that head used.
"""
import os
import subprocess

import numpy as np

BUILD = "/home/rick/Projects/VybForge/native/build"
BIN = os.path.join(BUILD, "gdn_authority")


def run(S, H_K, H_V, q, k, v, g, be, st, tag):
    T, B = 1, 1
    inp, outp, stp = f"{BUILD}/p_in.bin", f"{BUILD}/p_out.bin", f"{BUILD}/p_st.bin"
    with open(inp, "wb") as fh:
        for a in (q, k, v, g, be, st):
            fh.write(a.astype("<f4").tobytes())
    r = subprocess.run([BIN, inp, outp, stp, str(S), str(H_K), str(H_V), str(T), str(B), "1"],
                       capture_output=True, text=True)
    if r.returncode != 0:
        print(f"  {tag}: ERR {r.stdout}{r.stderr}")
        return None
    return np.fromfile(outp, dtype="<f4").reshape(S, H_V, T, B).astype(np.float64)[:, :, 0, 0]


S, H_K, H_V = 4, 1, 3

# ---- A: state mapping (D_h per head, no update) ----
D = np.array([1.0, 2.0, 4.0])
st = np.zeros((S, S, H_V, 1), dtype=np.float32)
for h in range(H_V):
    st[:, :, h, 0] = np.eye(S) * D[h]
q = np.ones((S, H_K, 1, 1), dtype=np.float32)
k = np.ones((S, H_K, 1, 1), dtype=np.float32)
v = np.zeros((S, H_V, 1, 1), dtype=np.float32)
g = np.zeros((1, H_V, 1, 1), dtype=np.float32)
be = np.zeros((1, H_V, 1, 1), dtype=np.float32)
outA = run(S, H_K, H_V, q, k, v, g, be, st, "A")
print("A. state per head = diag(D), decay 1, beta 0, q = 1")
if outA is not None:
    for h in range(H_V):
        print(f"   head {h}: op out = {outA[:, h]}   (D[{h}]={D[h]} would give {D[h]/np.sqrt(S):.4f})")

# ---- B: v mapping (update only) ----
V = np.array([10.0, 20.0, 40.0])
st0 = np.zeros((S, S, H_V, 1), dtype=np.float32)
k = np.zeros((S, H_K, 1, 1), dtype=np.float32)
k[0, 0, 0, 0] = 1.0
q = np.zeros((S, H_K, 1, 1), dtype=np.float32)
q[0, 0, 0, 0] = 1.0
v = np.zeros((S, H_V, 1, 1), dtype=np.float32)
for h in range(H_V):
    v[:, h, 0, 0] = V[h]
g = np.zeros((1, H_V, 1, 1), dtype=np.float32)
be = np.ones((1, H_V, 1, 1), dtype=np.float32)
outB = run(S, H_K, H_V, q, k, v, g, be, st0, "B")
print("\nB. state 0, beta 1, k=e0, q=e0, v per head = V")
if outB is not None:
    for h in range(H_V):
        print(f"   head {h}: op out = {outB[:, h]}   (V[{h}]={V[h]})")

# ---- C: gate mapping ----
G = np.array([0.0, -1.0, -2.0])
st1 = np.zeros((S, S, H_V, 1), dtype=np.float32)
for h in range(H_V):
    st1[:, :, h, 0] = np.eye(S)
q = np.zeros((S, H_K, 1, 1), dtype=np.float32)
q[0, 0, 0, 0] = 1.0
k = np.ones((S, H_K, 1, 1), dtype=np.float32)
v = np.zeros((S, H_V, 1, 1), dtype=np.float32)
g = np.zeros((1, H_V, 1, 1), dtype=np.float32)
for h in range(H_V):
    g[0, h, 0, 0] = G[h]
be = np.zeros((1, H_V, 1, 1), dtype=np.float32)
outC = run(S, H_K, H_V, q, k, v, g, be, st1, "C")
print("\nC. state diag(1), beta 0, q=e0, gate per head = G")
if outC is not None:
    for h in range(H_V):
        print(f"   head {h}: op out[0] = {outC[0, h]:.6f}   (exp(G[{h}])={np.exp(G[h]):.6f})")
