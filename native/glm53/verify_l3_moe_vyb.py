"""Gate: L3_MOE_ROUTE_VYB + L3_MOE_SHARED_VYB — layer 3's router and shared expert in Vyb.

Compares, against the numpy gold (native/glm53/gold_l3_moe.py):
  * the per-token top-8 expert SETS and their renormed weights,
  * the shared expert's output digest.

Usage: env -u PYTHONPATH ~/venvs/exl3/bin/python native/glm53/verify_l3_moe_vyb.py
"""

import json
import sys

import numpy as np

OUT = "native/out"
S40 = 1099511627776.0


def dig(v):
    f = np.ascontiguousarray(v, dtype=np.float64).ravel()
    return {"n": len(f), "sum": int(f.sum() * S40), "abs": int(np.abs(f).sum() * S40),
            "sq": int((f * f).sum() * S40), "max": int(np.abs(f).max() * S40),
            **{f"v{k}": int(f[k] * S40) if k < len(f) else 0 for k in range(8)}}


def parse(path):
    got, route, weights = {}, {}, {}
    for line in open(path):
        line = line.strip()
        if line.startswith("route "):
            for part in line.split()[1:]:
                k, rest = part.split("=", 1)
                route[k] = [int(x) for x in rest.strip("[]").split(",") if x]
            continue
        if line.startswith("weights "):
            for part in line.split()[1:]:
                k, rest = part.split("=", 1)
                weights[k] = [int(x) / S40 for x in rest.strip("[]").split(",") if x]
            continue
        parts = line.split()
        if not parts:
            continue
        fields = {}
        for p in parts[1:]:
            if "=" in p:
                k, v = p.split("=", 1)
                fields[k] = int(v)
        if "n" in fields:
            got[parts[0]] = fields
    return got, route, weights


def main():
    try:
        vyb, vroute, vw = parse(f"{OUT}/l3moe_vyb.txt")
    except FileNotFoundError:
        print("L3_MOE_ROUTE_VYB: FAIL (no native/out/l3moe_vyb.txt)")
        return 1
    gold = json.load(open(f"{OUT}/gold_l3_moe.json"))

    ok = True

    # ---- routing: sets and weights ----
    groute = gold["route"]
    grw = gold["rw"]
    set_bad = 0
    w_worst = 0.0
    for t in range(len(groute)):
        a = sorted(int(x) for x in groute[t])
        b = sorted(vroute.get(f"q{t}", []))
        if a != b:
            set_bad += 1
            if set_bad <= 3:
                print(f"  route q{t}: vyb {b} vs gold {a}")
            continue
        # weights are lane-ordered by the gold's own topk order; compare as multisets
        gw = sorted(float(x) for x in grw[t])
        vv = sorted(vw.get(f"q{t}", []))
        if len(vv) == len(gw):
            for x, y in zip(gw, vv):
                w_worst = max(w_worst, abs(x - y) / max(1e-9, abs(x)))
    print(f"routing sets: {len(groute) - set_bad}/{len(groute)} tokens agree; "
          f"worst weight rel {w_worst:.2e}")
    ok = ok and set_bad == 0 and w_worst <= 1e-4

    # ---- shared expert ----
    garr = "gold_l3_shared.npy"
    if "shared" in vyb:
        g = dig(np.load(f"{OUT}/{garr}"))
        f = vyb["shared"]
        worst = 0.0
        for k in ("sum", "abs", "sq", "max"):
            worst = max(worst, abs(f.get(k, 0) - g[k]) / max(1.0, abs(float(g[k]))))
        print(f"shared expert: rel (sum/abs/sq/max) worst {worst:.2e}  "
              f"{'OK' if worst <= 5e-6 else 'FAIL'}")
        ok = ok and worst <= 5e-6
    else:
        print("shared expert: missing from the Vyb output")
        ok = False

    print("L3_MOE_ROUTE_VYB:", "OK" if set_bad == 0 and w_worst <= 1e-4 else "FAIL")
    print("L3_MOE_SHARED_VYB:", "OK" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
