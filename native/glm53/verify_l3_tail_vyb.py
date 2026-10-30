"""Gate: L3_TAIL_VYB — hc_ffn mix -> post-norm -> MoE -> hc_apply, chained in Vyb.

Compares h2 (the post-attention norm, recomputed here from streams1 rather than read from
the gold), mlp (the MoE output) and streams2 (the layer's output residual) against the
gold's stage digests.
"""
import json
import sys

OUT = "native/out"


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
        v = parse(f"{OUT}/l3tail_vyb.txt")
    except FileNotFoundError:
        print("L3_TAIL_VYB: FAIL (no l3tail_vyb.txt)")
        return 1
    g = json.load(open(f"{OUT}/gold_l3_moe.json"))["stages"]
    ok = True
    for stage in ("h2", "mlp", "streams2"):
        if stage not in v or stage not in g:
            print(f"L3_TAIL_VYB: FAIL (missing {stage})")
            return 1
        a, b = v[stage], g[stage]
        if a["n"] != b["n"]:
            print(f"L3_TAIL_VYB: FAIL ({stage} n {a['n']} vs {b['n']})")
            return 1
        print(f"{stage}:")
        for k in ("sum", "abs", "sq", "max"):
            den = abs(float(b["abs"])) if k == "sum" else max(1.0, abs(float(b[k])))
            rel = abs(a[k] - b[k]) / den
            tag = " (vs abs-sum)" if k == "sum" else ""
            print(f"  {k:4s} rel {rel:.2e}{tag}")
            ok = ok and rel <= 2e-4
    print("L3_TAIL_VYB:", "OK" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
