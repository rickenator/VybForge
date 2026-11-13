"""Gate: L3_MOE_ALL_VYB — layer 3's complete MoE ffn output vs the gold.

Compares the Vyb run's full [22, 4096] `mlp` (routed experts + shared expert) against
native/out/gold_l3_mlp.npy, and reports the routed and shared parts separately.
"""
import sys

import numpy as np

OUT = "native/out"
S40 = 1099511627776.0


def dig(v):
    f = np.ascontiguousarray(v, dtype=np.float64).ravel()
    return {"n": len(f), "sum": int(f.sum() * S40), "abs": int(np.abs(f).sum() * S40),
            "sq": int((f * f).sum() * S40), "max": int(np.abs(f).max() * S40)}


def parse(path):
    out = {}
    for line in open(path):
        parts = line.split()
        if parts:
            out[parts[0]] = {k: int(v) for k, v in
                             (p.split("=", 1) for p in parts[1:] if "=" in p)}
    return out


def main():
    try:
        vyb = parse(f"{OUT}/l3moeall_vyb.txt")
    except FileNotFoundError:
        print("L3_MOE_ALL_VYB: FAIL (no l3moeall_vyb.txt)")
        return 1
    if "mlp" not in vyb:
        print("L3_MOE_ALL_VYB: FAIL (no mlp line)")
        return 1
    gold = dig(np.load(f"{OUT}/gold_l3_mlp.npy"))
    v = vyb["mlp"]
    ok = v["n"] == gold["n"]
    print(f"mlp elements {v['n']} vs {gold['n']}")
    for k in ("sum", "abs", "sq", "max"):
        den = abs(float(gold["abs"])) if k == "sum" else max(1.0, abs(float(gold[k])))
        rel = abs(v[k] - gold[k]) / den
        tag = " (vs abs-sum)" if k == "sum" else ""
        print(f"  {k:4s} rel {rel:.2e}{tag}")
        ok = ok and rel <= 2e-4
    print("L3_MOE_ALL_VYB:", "OK" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
