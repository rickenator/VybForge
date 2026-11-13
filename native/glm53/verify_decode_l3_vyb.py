"""Gate for the Vyb layer-3 step-decode driver (native/host/glm_decode_l3.vyb) -> DEC_L3_STEP_VYB.

Reads the driver's dumped stages (v*2^40 integers, same convention as the layer-0 gate)
from native/out/l3dec_vyb.txt and compares them against the gold step-decode arrays
native/out/gold_l3dec_step{0,1}_{attn,mlp,streams2}.npy written by gold_decode_l3.py.

Tolerances follow the established layer-3 gate convention (verify_l3_full.py /
verify_l3_moe_all.py): mlp and streams2 carry the EXL3 routed-expert dequant noise, which
is a digest-relative 2e-4 bar there — per-element, the routed path sits at ~5e-4 max
(EXL3_LOAD_VYB measured 7.8e-5 per weight element against the numpy oracle), so the mlp /
streams2 comparison is the same sum/abs/sq/max digest-relative form, not elementwise.
attn is pure bf16-weight math (no EXL3), so it keeps a tight elementwise check.

Usage: env -u PYTHONPATH ~/venvs/exl3/bin/python native/glm53/verify_decode_l3_vyb.py
"""

import sys

import numpy as np

OUT = "native/out"
SCALE = 1099511627776.0  # 2^40
TOL = {"attn": 1e-9, "mlp": 2e-4, "streams2": 2e-4}


def dig(v):
    f = np.ascontiguousarray(v, dtype=np.float64).ravel()
    return {"n": int(f.size), "sum": float(f.sum()), "abs": float(np.abs(f).sum()),
            "sq": float((f * f).sum()), "max": float(np.abs(f).max())}


def main():
    stages = {}
    with open(f"{OUT}/l3dec_vyb.txt") as f:
        for line in f:
            parts = line.split()
            if parts:
                stages[parts[0]] = np.array([int(x) for x in parts[1:]]) / SCALE

    worst = 0.0
    ok = True
    for t in (0, 1):
        for name, gold_name in (("attn", "attn"), ("mlp", "mlp"), ("s2", "streams2")):
            key = f"s{t}{name}"
            gold = np.load(f"{OUT}/gold_l3dec_step{t}_{gold_name}.npy").ravel().astype(np.float64)
            got = stages.get(key)
            if got is None or got.size != gold.size:
                print(f"DEC_L3_STEP_VYB: FAIL {key} missing or wrong size "
                      f"({None if got is None else got.size} vs {gold.size})")
                ok = False
                continue
            if gold_name == "attn":
                # pure bf16 path: elementwise, tight
                err = float(np.abs(got - gold).max())
                stat = "OK " if err <= TOL["attn"] else "FAIL"
                if err > TOL["attn"]:
                    ok = False
                worst = max(worst, err)
                print(f"  {stat} step{t} {gold_name:9s} max_abs={err:.3e}  (tol {TOL['attn']:.0e})")
            else:
                # EXL3-carrying stages: the same digest-relative bar as L3_FULL_VYB_MATCH
                g, a = dig(gold), dig(got)
                rels = {}
                for k in ("sum", "abs", "sq", "max"):
                    den = abs(g["abs"]) if k == "sum" else max(1.0, abs(g[k]))
                    rels[k] = abs(a[k] - g[k]) / den
                el = float(np.abs(got - gold).max())
                good = all(r <= TOL[gold_name] for r in rels.values())
                if not good:
                    ok = False
                worst = max(worst, max(rels.values()))
                print(f"  {'OK ' if good else 'FAIL'} step{t} {gold_name:9s} "
                      f"sum {rels['sum']:.2e}  abs {rels['abs']:.2e}  "
                      f"sq {rels['sq']:.2e}  max {rels['max']:.2e}  (tol {TOL[gold_name]:.0e})"
                      f"  [elementwise {el:.2e}]")

    if ok:
        print(f"DEC_L3_STEP_VYB: OK  worst={worst:.3e}")
        return 0
    print("DEC_L3_STEP_VYB: FAIL")
    return 1


if __name__ == "__main__":
    sys.exit(main())
