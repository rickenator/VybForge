"""Gate for the Vyb step-decode driver (native/host/glm_decode_step.vyb) -> DEC_STEP_VYB.

Reads the driver's dumped stages (v*2^40 integers, same convention as the layer-0 gate)
from native/out/dec_step_vyb.txt and compares them against the gold step-decode arrays
native/out/gold_dec_step{0,1}_{attn,mlp,streams2}.npy.

Usage: env -u PYTHONPATH ~/venvs/exl3/bin/python native/glm53/verify_decode_step_vyb.py
"""

import sys

import numpy as np

OUT = "native/out"
SCALE = 1099511627776.0  # 2^40
TOL = {"attn": 2e-11, "mlp": 2e-4, "streams2": 2e-4}


def main():
    stages = {}
    with open(f"{OUT}/dec_step_vyb.txt") as f:
        for line in f:
            parts = line.split()
            if parts:
                stages[parts[0]] = np.array([int(x) for x in parts[1:]]) / SCALE

    worst = 0.0
    ok = True
    for t in (0, 1):
        for name, gold_name in (("attn", "attn"), ("mlp", "mlp"), ("s2", "streams2")):
            key = f"s{t}{name}"
            gold = np.load(f"{OUT}/gold_dec_step{t}_{gold_name}.npy").ravel().astype(np.float64)
            got = stages.get(key)
            if got is None or got.size != gold.size:
                print(f"DEC_STEP_VYB: FAIL {key} missing or wrong size "
                      f"({None if got is None else got.size} vs {gold.size})")
                ok = False
                continue
            err = float(np.abs(got - gold).max())
            tol = TOL[gold_name]
            stat = "OK " if err <= tol else "FAIL"
            if err > tol:
                ok = False
            worst = max(worst, err)
            print(f"  {stat} step{t} {gold_name:9s} max_abs={err:.3e}  (tol {tol:.0e})")

    if ok:
        print(f"DEC_STEP_VYB: OK  worst={worst:.3e}")
        return 0
    print("DEC_STEP_VYB: FAIL")
    return 1


if __name__ == "__main__":
    sys.exit(main())
