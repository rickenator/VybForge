#!/usr/bin/env python3
"""DECODE_CHAIN_MATCH: the chained 45-layer decode walk vs the batched prefill.

The modified glm_prefill.vyb dumps, per layer, the LAST TWO prompt tokens' streams
(2^40-scaled ints, tokens 20 then 21, 32768 values) as `chain<N>` dumps() lines into
native/out/prefill_vyb.txt.  glm_decode_chain.vyb walks tokens 20 and 21 stepwise
through all 45 layers and dumps `d<t>l<n>s2` lines (16384 values) into
native/out/decchain_vyb.txt.  Under causal attention the stepwise walk is exact, so
the comparison is element-wise, tolerance ~1e-9 (fp64 noise; the single-layer gates
sit at ~2e-12).

Run:  env -u PYTHONPATH ~/venvs/exl3/bin/python native/glm53/verify_decode_chain.py
"""
import re
import sys

SCALE = 2.0 ** 40
TOL = 1e-9
NLAY = 45

def main():
    pre_ref = {}
    rx_chain = re.compile(r'^chain(\d+) (.*)$')
    for line in open('native/out/prefill_vyb.txt'):
        m = rx_chain.match(line.strip())
        if m:
            pre_ref[int(m.group(1))] = [int(x) for x in m.group(2).split()]
    dec = {}
    rx_d = re.compile(r'^d(\d+)l(\d+)s2 (.*)$')
    for line in open('native/out/decchain_vyb.txt'):
        m = rx_d.match(line.strip())
        if m:
            dec[(int(m.group(1)), int(m.group(2)))] = [int(x) for x in m.group(3).split()]

    missing = [n for n in range(NLAY) if n not in pre_ref]
    if missing:
        print(f'DECODE_CHAIN_MATCH: FAIL (prefill ref missing layers {missing[:5]}...)')
        return 1
    missing = [(t, n) for t in (0, 1) for n in range(NLAY) if (t, n) not in dec]
    if missing:
        print(f'DECODE_CHAIN_MATCH: FAIL (decode dumps missing {missing[:5]}...)')
        return 1

    worst = 0.0
    worst_where = None
    per_layer = {}
    for n in range(NLAY):
        ref = pre_ref[n]
        if len(ref) != 32768:
            print(f'DECODE_CHAIN_MATCH: FAIL (chain{n} has {len(ref)} values, want 32768)')
            return 1
        w = 0.0
        for t in (0, 1):
            d = dec[(t, n)]
            if len(d) != 16384:
                print(f'DECODE_CHAIN_MATCH: FAIL (d{t}l{n}s2 has {len(d)} values, want 16384)')
                return 1
            base = t * 16384
            for i in range(16384):
                e = abs(ref[base + i] - d[i]) / SCALE
                if e > w:
                    w = e
        per_layer[n] = w
        if w > worst:
            worst = w
            worst_where = n
    for n in range(NLAY):
        print(f'  layer {n:2d} max_abs {per_layer[n]:.3e}')
    print(f'worst elementwise {worst:.3e} (layer {worst_where if worst_where is not None else 0})')
    if worst <= TOL:
        print('DECODE_CHAIN_MATCH: OK')
        return 0
    print('DECODE_CHAIN_MATCH: FAIL')
    return 1

if __name__ == '__main__':
    sys.exit(main())
