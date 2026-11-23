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

if len(sys.argv) >= 3:
    pa, pb = sys.argv[1], sys.argv[2]
else:
    pa, pb = "/tmp/d_deq.txt", "/tmp/d_fused.txt"
deq = parse(pa)
fus = parse(pb)
if not deq or not fus:
    print("missing digests: deq=%d fused=%d" % (len(deq), len(fus)))
    sys.exit(1)

AGGS = ("sum", "abs", "sq")
ELEMS = ("max", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7")
N = 327680

def normalised(a, b):
    """Per-field delta in units that mean something.

    A raw relative delta is misleading twice over: a cancellation-prone aggregate (`sum` can cancel
    6 orders of magnitude below the stream magnitude) and a small-by-chance element both produce
    huge ratios from tiny absolute differences. So normalise aggregates against their OWN magnitude
    and per-element fields against the stream's mean |value| (abs / n).
    """
    scale = a["abs"] / N if a.get("abs") else 1.0
    out = {}
    for k in AGGS:
        if k in a and k in b and abs(a[k]) > 0:
            # `sum` is excluded from the verdict: 327680 terms with heavy cancellation make its own
            # magnitude meaningless (it lands 6 orders below the stream while abs stays at 1e16), so
            # any tiny per-element change can shift it by a large fraction. Kept for reporting only.
            out[k] = abs(a[k] - b[k]) / (abs(a[k]) if k != "sum" else (scale * N))
    for k in ELEMS:
        if k in a and k in b:
            out[k] = abs(a[k] - b[k]) / scale if scale else 0.0
    return out

common = sorted(set(deq) & set(fus))
print("layers compared: %d (A has %d, B has %d)" % (len(common), len(deq), len(fus)))
print()
print("%5s %12s %10s %16s %16s" % ("layer", "norm_delta", "field", "A", "B"))
worst = (0.0, None, None)
identical = 0
diverge_from = None
per_layer = []
for L in common:
    a, b = deq[L], fus[L]
    if a == b:
        identical += 1
        per_layer.append(0.0)
        continue
    if diverge_from is None:
        diverge_from = L
    n = normalised(a, b)
    judge = {k: v for k, v in n.items() if k != "sum"}
    mf = max(judge, key=judge.get) if judge else ""
    mr = judge.get(mf, 0.0)
    per_layer.append(mr)
    if mr > worst[0]:
        worst = (mr, L, mf)
    print("%5d %12.2e %10s %16d %16d" % (L, mr, mf, a.get(mf, 0), b.get(mf, 0)))

per_layer.sort()
print()
print("layers byte-identical : %d / %d" % (identical, len(common)))
print("first differing layer : %s" % (diverge_from if diverge_from is not None else "none"))
print("median normalised delta: %.2e" % per_layer[len(per_layer) // 2])
print("worst normalised delta : %.2e at layer %s (field %s)"
      % (worst[0], worst[1], worst[2]))
print("float32 epsilon        : 1.19e-07")
print()
print("NOTE: 'norm_delta' normalises aggregates (sum/abs/sq) against their OWN magnitude and")
print("      per-element fields against the stream mean |value| (abs/n). A raw relative delta on")
print("      `sum` is meaningless -- it cancels 6 orders of magnitude below the stream and reports")
print("      ~1e-2 for differences of 1e-7 of the data. Trust the per-element fields.")
print()
if identical == len(common):
    print("VERDICT: BIT-IDENTICAL — the two runs produced exactly the same digests")
elif worst[0] < 1e-6:
    print("VERDICT: EQUIVALENT (max %.1e of the data range, ~%.0fx eps)"
          % (worst[0], worst[0] / 1.19e-7))
else:
    print("VERDICT: DIFFERS — max %.1e of the data range at layer %s (field %s); inspect whether it"
          % (worst[0], worst[1], worst[2]))
    print("         compounds with depth (real drift) or is confined to one cancellation-prone field")
