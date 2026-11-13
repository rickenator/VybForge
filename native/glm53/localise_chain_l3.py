#!/usr/bin/env python3
"""Localise the DECODE_CHAIN layer-3 divergence.

Reads native/out/decchain_dbg.txt (a 4-layer / 1-step debug run of
glm_decode_chain.vyb) and the gold layer-3 decode oracle dumps, and reports where
the decode walk's layer-3 values stop matching:
  1. DSA cache rows 0..19 vs gold_l3dec_{ckv,ik,igsc}.bin rows 0..19
  2. the appended row 20 vs gold row 20
  3. attn / streams1 / mlp / streams2 vs gold_l3dec_step0_*.npy (gold used the
     numpy chain input, so small ~1e-8 diffs are the known Vyb-vs-numpy residue;
     a ~1e-1 diff is the bug)
  4. streams2 vs the prefill's own chain3 token-20 half (the gate quantity)
"""
import re
import numpy as np

SCALE = 2.0 ** 40
OUT = 'native/out'

def dumps_from(path, name):
    rx = re.compile(r'^' + re.escape(name) + r' (.*)$')
    with open(path) as f:
        for line in f:
            m = rx.match(line.strip())
            if m:
                return np.array([int(x) for x in m.group(1).split()], dtype=np.int64) / SCALE
    return None

def gold_bin(path):
    raw = open(path, 'rb').read()
    return np.frombuffer(raw, dtype='<f8')

dbg = OUT + '/decchain_dbg.txt'
ckv = dumps_from(dbg, 'd0l3ckv')
ik = dumps_from(dbg, 'd0l3ik')
gsc = dumps_from(dbg, 'd0l3gsc')
print('cache shapes:', None if ckv is None else ckv.shape,
      None if ik is None else ik.shape, None if gsc is None else gsc.shape)

gckv = gold_bin(OUT + '/gold_l3dec_ckv.bin').reshape(22, 512)
gik = gold_bin(OUT + '/gold_l3dec_ik.bin').reshape(22, 128)
gigsc = gold_bin(OUT + '/gold_l3dec_igsc.bin').reshape(22, 128)
for nm, mine, g in [('ckv', ckv, gckv), ('ik', ik, gik), ('gsc', gsc, gigsc)]:
    if mine is None:
        continue
    w = mine.shape[0]
    m2 = mine.reshape(w, -1)
    print(f'{nm}: rows0-19 max_abs {np.abs(m2[:20] - g[:20]).max():.3e}   '
          f'row20 vs gold20 max_abs {np.abs(m2[20] - g[20]).max():.3e}')

for nm, f in [('attn', 'gold_l3dec_step0_attn.npy'),
              ('mlp', 'gold_l3dec_step0_mlp.npy'),
              ('s2', 'gold_l3dec_step0_streams2.npy')]:
    mine = dumps_from(dbg, 'd0l3' + nm)
    g = np.load(f'{OUT}/{f}').ravel().astype(np.float64)
    if mine is None:
        print(nm, 'missing')
        continue
    n = min(mine.shape[0], g.shape[0])
    print(f'{nm} vs gold: max_abs {np.abs(mine[:n] - g[:n]).max():.3e}  (n={n})')

# the gate quantity: token 20 half of chain3
rx = re.compile(r'^chain3 (.*)$')
ref = None
for line in open(OUT + '/prefill_vyb.txt'):
    m = rx.match(line.strip())
    if m:
        ref = np.array([int(x) for x in m.group(1).split()], dtype=np.int64) / SCALE
dec = dumps_from(dbg, 'd0l3s2')
if ref is not None and dec is not None:
    print(f's2 (chain step0 L3) vs prefill chain3 token20: max_abs {np.abs(dec - ref[:16384]).max():.3e}')
    # where: first mismatching element
    d = np.abs(dec - ref[:16384])
    idx = int(d.argmax())
    print(f'  worst idx {idx} (hc={idx//4096} elem={idx%4096})  dec={dec[idx]:.6g} ref={ref[idx]:.6g}')
