"""Gate: L3_EXPERT_VYB — one token's routed-expert contribution vs the gold."""
import json
import sys

import numpy as np

OUT = "native/out"
S40 = 1099511627776.0


def dig(v):
    f = np.ascontiguousarray(v, dtype=np.float64).ravel()
    return {k: (int(v2 * S40) if isinstance(v2, float) else v2)
            for k, v2 in (("n", len(f)), ("sum", float(f.sum())), ("abs", float(np.abs(f).sum())),
                          ("sq", float((f * f).sum())), ("max", float(np.abs(f).max())))}


def parse(path):
    for line in open(path):
        parts = line.split()
        if parts and parts[0] == "routed":
            return {k: int(v) for k, v in (p.split("=", 1) for p in parts[1:] if "=" in p)}
    return None


def main():
    try:
        v = parse(f"{OUT}/l3exp_vyb.txt")
    except FileNotFoundError:
        print("L3_EXPERT_VYB: FAIL (no l3exp_vyb.txt)")
        return 1
    if v is None:
        print("L3_EXPERT_VYB: FAIL (no routed line)")
        return 1
    g = dig(np.load(f"{OUT}/gold_l3_routed.npy")[0])
    ok = v["n"] == g["n"]
    print(f"elements {v['n']} vs {g['n']}")
    for k in ("sum", "abs", "sq", "max"):
        # the SUM of 4096 signed values nearly cancels (|sum| is ~1/700 of abs-sum), so its
        # honest scale is the abs-sum, not its own magnitude.
        den = abs(float(g["abs"])) if k == "sum" else max(1.0, abs(float(g[k])))
        rel = abs(v[k] - g[k]) / den
        tag = " (vs abs-sum)" if k == "sum" else ""
        print(f"  {k:4s} rel {rel:.2e}{tag}")
        ok = ok and rel <= 1e-4
    print("L3_EXPERT_VYB:", "OK" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
