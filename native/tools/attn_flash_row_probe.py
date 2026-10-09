#!/usr/bin/env python3
"""Which keys did the flash op actually read? A row-identity probe (phase 4, unit 10 step 4f).

Five layout hypotheses for the flash path's mismatch have been tested and refuted; this replaces the
sixth guess with a measurement. The fixture is built so the answer is legible in the output:

  * q = 0 and k = 0 (both projections zeroed), so every score is 0 and the softmax is UNIFORM over
    whichever keys the op believes are allowed;
  * V is set so that token t carries the constant value t+1 in every dim of every kv head
    (hid[t, 0] = t+1, W_v[:, 0] = 1, everything else zero).

Then attention(query q, head h, dim d) = mean of (t+1) over the keys q attends to — a NUMBER THAT NAMES
THE KEY SET. Under a causal mask and query q that is mean(1..q+1) = (q+2)/2, so:

    query 0 -> 1.0     query 1 -> 1.5     query 2 -> 2.0     query 3 -> 2.5   ...

Any other value names a different key set: a reversed or transposed key axis, an off-by-one, an
unwritten tile, or a mask that allows the wrong half. v is identical across kv heads on purpose, so the
grouping question (already refuted separately) cannot contaminate this one.

Usage: attn_flash_row_probe.py    (env: VYBFORGE_LLAMA, CC; VYBFORGE_ATTN_S, default 6)
"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
# attn_verify reads S from this env var at import time, and the dump offsets below are computed from it:
# set it BEFORE the import or the two disagree (an empty slice, which is how this was found)
os.environ.setdefault("VYBFORGE_ATTN_S", "6")
import attn_verify as av

S = av.S
NH, HD, NKV, D = av.NH, av.HD, av.NKV, av.D


def fixture_favor0():
    """Score every key by -(query+1)*(key+1)*scale, so the weights concentrate on KEY 0 — which the causal
    mask allows for every query. A correct key axis therefore returns V(key 0) = 1.0 for EVERY query,
    which is the discriminating reading: the flat-score probe cannot tell "one key (the diagonal)"
    apart from "the key axis is read from the wrong place"."""
    Wqg = np.zeros((NH * 2 * HD, D), dtype=np.float32)
    Wk = np.zeros((NKV * HD, D), dtype=np.float32)
    Wv = np.zeros((NKV * HD, D), dtype=np.float32)
    Wv[:, 0] = 1.0
    Wo = np.zeros((D, NH * HD), dtype=np.float32)
    nmq, nmk = np.ones((HD,), dtype=np.float32), np.ones((HD,), dtype=np.float32)
    hid = np.zeros((S, D), dtype=np.float32)
    hid[:, 0] = np.arange(1, S + 1)
    for h in range(NH):
        Wqg[2 * h * HD + 0, 0] = -1.0      # q[t,h,0] = -(t+1)
    for j in range(NKV):
        Wk[j * HD + 0, 0] = 1.0            # k[k,j,0] =  (k+1)
    return Wqg, Wk, Wv, Wo, nmq, nmk, hid


def fixture():
    """The legible fixture: flat scores, V carrying the token index."""
    Wqg = np.zeros((NH * 2 * HD, D), dtype=np.float32)
    Wk = np.zeros((NKV * HD, D), dtype=np.float32)
    Wv = np.zeros((NKV * HD, D), dtype=np.float32)
    Wv[:, 0] = 1.0                      # v[t, j, d] = hid[t, 0]
    Wo = np.zeros((D, NH * HD), dtype=np.float32)
    nmq = np.ones((HD,), dtype=np.float32)
    nmk = np.ones((HD,), dtype=np.float32)
    hid = np.zeros((S, D), dtype=np.float32)
    hid[:, 0] = np.arange(1, S + 1)     # token t carries t+1
    return Wqg, Wk, Wv, Wo, nmq, nmk, hid


def attn_stage(dump):
    """The `attn` slot of the positional dump, as (S, NH, HD)."""
    off = (S * NH * 2 * HD                      # qg
           + 4 * S * NH * HD                    # q_pre, gate_pre, q_norm, q_rope
           + 2 * S * NKV * HD                   # k_norm, k_rope
           + 2 * S * S * NH)                    # scores, probs (zero placeholders in GQA mode)
    n = S * NH * HD
    return np.asarray(dump[off:off + n], dtype=np.float64).reshape(S, NH, HD)


def main():
    print(f"ROW PROBE geometry n_head={NH} n_kv={NKV} S={S} — V carries t+1, scores flat")
    which = os.environ.get("VYBFORGE_PROBE", "flat")
    Wqg, Wk, Wv, Wo, nmq, nmk, hid = (fixture_favor0() if which == "favor0" else fixture())
    scale = 4.0 if which == "favor0" else av.KQS
    if which == "favor0":
        print("ROW PROBE scores favour key 0 -> a correct key axis gives V(key 0) = 1.0 for EVERY query")
    for mode in ("explicit", "flash"):
        av.run_authority(Wqg, Wk, Wv, Wo, nmq, nmk, hid, mode=mode, kq_scale=scale)
        a = attn_stage(np.fromfile(av.OUT, dtype="<f4"))
        print(f"ROW PROBE mode={mode}: the value the op produced per query (head 0, dim 0) "
              f"and the key set that value names")
        for q in range(S):
            vals = a[q, 0, :]
            const = float(np.nanmax(vals) - np.nanmin(vals)) < 1e-6
            v = float(vals[0])
            # invert: v = mean over keys of (t+1)  =>  if contiguous from 0, keys = 0..(2v-2)
            print(f"ROW PROBE   q={q:2d} value={v:9.5f} (constant over dims: {const}) "
                  f"expected for keys 0..{q}: {(q + 2) / 2:9.5f}  "
                  f"{'MATCH' if abs(v - (q + 2) / 2) < 1e-3 else 'DIFFERENT KEY SET'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
