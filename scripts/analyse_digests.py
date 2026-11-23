#!/usr/bin/env python3
"""analyse_digests.py — per-layer drift between the deq and fused digest runs.

Reads the two "phase1 layer N <digest>" streams produced by scripts/digest_ab.sh
(/tmp/d_deq.txt and /tmp/d_fused.txt) and reports, per layer, the relative
difference of each digest field plus the trend across layers.

The question this answers: does the ~1e-7 per-layer difference COMPOUND over 45
layers (drift, disqualifying) or stay bounded (accumulation-order noise, fine)?
"""
import re, sys, os

def parse(path):
    out = {}
    if not os.path.exists(path):
        return out
    for line in open(path):
        m = re.match(r"phase1 layer (\d+) streams n=(\d+) (.*)", line.strip())
        if not m:
            continue
        layer = int(m.group(1))
        fields = {}
        for tok in m.group(3).split():
            if "=" in tok:
                k, v = tok.split("=", 1)
                try:
                    fields[k] = int(v)
                except ValueError:
                    fields[k] = float(v)
        out[layer] = fields
    return out

deq = parse("/tmp/d_deq.txt")
fus = parse("/tmp/d_fused.txt")
if not deq or not fus:
    print("missing digests: deq=%d fused=%d" % (len(deq), len(fus)))
    sys.exit(1)

common = sorted(set(deq) & set(fus))
print("layers compared: %d (deq has %d, fused has %d)" % (len(common), len(deq), len(fus)))
print()
print("%5s %12s %14s %14s %14s" % ("layer", "max_rel", "field", "deq_val", "fused_val"))
worst = (0.0, None, None)
identical = 0
diverge_from = None
for L in common:
    a, b = deq[L], fus[L]
    if a == b:
        identical += 1
        continue
    if diverge_from is None:
        diverge_from = L
    mr, mf = 0.0, ""
    for k in a:
        if k not in b or a[k] == 0:
            continue
        rel = abs(b[k] - a[k]) / max(abs(a[k]), 1)
        if rel > mr:
            mr, mf = rel, k
    if mr > worst[0]:
        worst = (mr, L, mf)
    print("%5d %12.2e %14s %14d %14d" % (L, mr, mf, a.get(mf, 0), b.get(mf, 0)))

print()
print("layers byte-identical : %d / %d" % (identical, len(common)))
print("first differing layer : %s" % (diverge_from if diverge_from is not None else "none"))
if worst[1] is not None:
    print("worst relative delta  : %.2e at layer %d (field %s)" % worst)
print("float32 epsilon       : 1.19e-07")
print()
if identical == len(common):
    print("VERDICT: PASS — all digests bit-identical")
elif worst[0] < 1e-6:
    print("VERDICT: NUMERICALLY EQUIVALENT but not bit-identical (max %.1e, ~%.0fx eps)"
          % (worst[0], worst[0] / 1.19e-7))
else:
    print("VERDICT: DRIFT — max relative delta %.2e exceeds float32 noise; investigate" % worst[0])
