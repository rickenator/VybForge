#!/usr/bin/env python3
"""DECODE_MTP_MATCH: the chained decode driver's MTP head (layer 45) vs the fp64 oracle.

gold_mtp.py reads the driver's `mtptrunk{r}` dumps from native/out/decchain_vyb.txt,
recomputes the MTP input composition + layer 45 + final head with real checkpoint
weights, and writes gold_mtp_{hin,x,hidden,logits}.npy. This verifier compares the
driver's own dumps element-wise against the gold arrays:

    gate A  mtphin{r}   vs gold_mtp_hin.npy   (eh_proj input composition; fp64, ~1e-12)
    gate B  mtpx{r}     vs gold_mtp_x.npy     (post-layer-45 hidden; the routed experts
                                               carry EXL3 fp16 dequant noise -> 2e-4 bar,
                                               the same convention as the L3 gates)
    gate C  mtplogits{t} vs gold_mtp_logits.npy + top-1 agreement

Run:  env -u PYTHONPATH ~/venvs/exl3/bin/python native/glm53/verify_decode_mtp.py
"""
import json
import re
import sys

import numpy as np

SCALE = 2.0 ** 40
TOL_HIN = 1e-8
TOL_X_REL = 2e-4      # gate B: per-row max_abs graded against the row's own magnitude
                      # (the L3 gate convention -- the EXL3 fp16 dequant residue scales
                      # with the activation, and layer 45's hidden runs ~10-30x larger
                      # than the trunk's, so an absolute 2e-4 bar would measure noise)
TOL_LOGITS = 1e-2
OUT = "native/out"


def read_dumps(path, prefix):
    rx = re.compile(r"^" + re.escape(prefix) + r"(\d+) (.*)$")
    out = {}
    for line in open(path):
        m = rx.match(line.strip())
        if m:
            out[int(m.group(1))] = np.array([int(x) for x in m.group(2).split()],
                                            dtype=np.float64) / SCALE
    return out


def main():
    meta = json.load(open(f"{OUT}/gold_mtp.json"))
    nrows = meta["nrows"]
    nsteps = meta["nsteps"]

    drv_hin = read_dumps(f"{OUT}/decchain_vyb.txt", "mtphin")
    drv_x = read_dumps(f"{OUT}/decchain_vyb.txt", "mtpx")
    drv_log = read_dumps(f"{OUT}/decchain_vyb.txt", "mtplogits")

    gold_hin = np.load(f"{OUT}/gold_mtp_hin.npy")
    gold_x = np.load(f"{OUT}/gold_mtp_x.npy")
    gold_logits = np.load(f"{OUT}/gold_mtp_logits.npy")

    worst_a = 0.0
    worst_b = 0.0
    worst_b_rel = 0.0
    worst_c = 0.0
    missing = []
    for r in range(nrows):
        if r not in drv_hin or r not in drv_x:
            missing.append(r)
            continue
        if len(drv_hin[r]) != 4096 or len(drv_x[r]) != 4096:
            print(f"DECODE_MTP_MATCH: FAIL (row {r} dumps have {len(drv_hin[r])}/"
                  f"{len(drv_x[r])} values, want 4096)")
            return 1
        worst_a = max(worst_a, float(np.abs(drv_hin[r] - gold_hin[r]).max()))
        d = float(np.abs(drv_x[r] - gold_x[r]).max())
        rel = d / max(float(np.abs(gold_x[r]).max()), 1.0)
        worst_b = max(worst_b, d)
        worst_b_rel = max(worst_b_rel, rel)
    if missing:
        print(f"DECODE_MTP_MATCH: FAIL (driver dumps missing rows {missing})")
        return 1

    t = nsteps - 1
    if t not in drv_log:
        print("DECODE_MTP_MATCH: FAIL (no mtplogits dump for the final step)")
        return 1
    if len(drv_log[t]) != 154880:
        print(f"DECODE_MTP_MATCH: FAIL (mtplogits has {len(drv_log[t])} values, want 154880)")
        return 1
    worst_c = float(np.abs(drv_log[t] - gold_logits).max())
    drv_top1 = int(np.argmax(drv_log[t]))
    gold_top1 = int(meta["top1"])

    print(f"  gate A  eh_proj input  worst max_abs {worst_a:.3e}  (tol {TOL_HIN:.0e})")
    print(f"  gate B  layer-45 hid   worst max_abs {worst_b:.3e}, "
          f"worst vs row scale {worst_b_rel:.3e}  (tol {TOL_X_REL:.0e})")
    print(f"  gate C  logits         worst max_abs {worst_c:.3e}  (tol {TOL_LOGITS:.0e})")
    print(f"  draft top-1: driver {drv_top1}  oracle {gold_top1}")
    ok = (worst_a <= TOL_HIN and worst_b_rel <= TOL_X_REL
          and worst_c <= TOL_LOGITS and drv_top1 == gold_top1)
    print(f"worst elementwise {max(worst_a, worst_b, worst_c):.3e}")
    print("DECODE_MTP_MATCH: OK" if ok else "DECODE_MTP_MATCH: FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
