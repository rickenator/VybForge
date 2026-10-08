#!/usr/bin/env python3
"""Delta-function scan: put a single 1.0 at flat offset k of the state and see which (out index,
head) picks it up. That yields the op's actual addressing as a lookup table, with no guessing.

Setup makes the read trivial: gate = 0 (decay 1), beta = 0 (no update), q = ones, so
out[j, h] = scale * sum_i S[i, j] for the op's S, i.e. whichever slot k contributes, its column j
and head h are readable directly.
"""
import os
import subprocess

import numpy as np

BUILD = "/home/rick/Projects/VybForge/native/build"
BIN = os.path.join(BUILD, "gdn_authority")
S, H_K, H_V = 4, 1, 2
scale = 1.0 / np.sqrt(S)

q = np.ones((S, H_K, 1, 1), dtype=np.float32)
k = np.ones((S, H_K, 1, 1), dtype=np.float32)
v = np.zeros((S, H_V, 1, 1), dtype=np.float32)
g = np.zeros((1, H_V, 1, 1), dtype=np.float32)
be = np.zeros((1, H_V, 1, 1), dtype=np.float32)

rows = []
for flat in range(S * S * H_V):
    st = np.zeros((S, S, H_V, 1), dtype=np.float32)
    st.ravel()[flat] = 1.0
    inp, outp, stp = f"{BUILD}/k_in.bin", f"{BUILD}/k_out.bin", f"{BUILD}/k_st.bin"
    with open(inp, "wb") as fh:
        for a in (q, k, v, g, be, st):
            fh.write(a.astype("<f4").tobytes())
    r = subprocess.run([BIN, inp, outp, stp, str(S), str(H_K), str(H_V), "1", "1", "1"],
                       capture_output=True, text=True)
    if r.returncode != 0:
        print("ERR", r.stdout, r.stderr)
        break
    out = np.fromfile(outp, dtype="<f4").reshape(S, H_V, 1, 1).astype(np.float64)[:, :, 0, 0]
    hit = np.argwhere(np.abs(out - scale) < 1e-6)
    # my assumption: flat index within a head is (j*S + i) for head h = flat // (S*S)
    mine_h = flat // (S * S)
    mine_j = (flat % (S * S)) // S
    for (j, h) in hit:
        rows.append((flat, mine_h, mine_j, h, j))
    if not len(hit):
        rows.append((flat, mine_h, mine_j, None, None))

print(f"scan over {S*S*H_V} state slots, H_v={H_V}, S={S}")
print(f"{'flat':>4} {'my h':>4} {'my j':>4} | {'op h':>4} {'op j':>4} | {'agrees'}")
for flat, mh, mj, oh, oj in rows:
    ok = "yes" if (oh == mh and oj == mj) else "NO"
    print(f"{flat:>4} {mh:>4} {mj:>4} | {str(oh):>4} {str(oj):>4} | {ok}")

bad = [r for r in rows if r[3] != r[1] or r[4] != r[2]]
print(f"\nslots where the op disagrees with my mapping: {len(bad)} of {S*S*H_V}")
