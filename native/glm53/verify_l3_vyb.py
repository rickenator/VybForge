"""Gate: L3_ATTN_VYB_MATCH — layer 3's DSA attention in Vyb vs the numpy/HF-gated gold.

Compares the scaled-integer digests (same format both sides) and the indexer's per-query
token selection sets.

Usage: env -u PYTHONPATH ~/venvs/exl3/bin/python native/glm53/verify_l3_vyb.py
"""

import json
import sys

import numpy as np

OUT = "native/out"
S40 = 1099511627776.0


def dig(v):
    f = np.ascontiguousarray(v, dtype=np.float64).ravel()
    out = {"n": len(f), "sum": int(f.sum() * S40), "abs": int(np.abs(f).sum() * S40),
           "sq": int((f * f).sum() * S40), "max": int(np.abs(f).max() * S40)}
    for k in range(8):
        out[f"v{k}"] = int(f[k] * S40) if k < len(f) else 0
    return out


def parse(path):
    got, sel = {}, {}
    for line in open(path):
        line = line.strip()
        if line.startswith("sel "):
            for part in line.split()[1:]:
                key, rest = part.split("=", 1)
                toks = [int(x) for x in rest.strip("[]").split(",") if x]
                sel[key] = sorted(toks)
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
    return got, sel


def main():
    try:
        vyb, vsel = parse(f"{OUT}/l3_vyb.txt")
    except FileNotFoundError:
        print("L3_ATTN_VYB_MATCH: FAIL (no native/out/l3_vyb.txt)")
        return 1
    gold = json.load(open(f"{OUT}/gold_l3.json"))

    ok = True
    print("stage            n     sum rel     sq rel    max rel  verdict")
    for name, garr in (("ha", "gold_l3_ha.npy"), ("qres", "gold_l3_qres.npy"),
                       ("qb", "gold_l3_qb.npy"), ("ckv", "gold_l3_ckv.npy"),
                       ("ao", "gold_l3_o_pre.npy"), ("attn", "gold_l3_attn.npy")):
        if name not in vyb:
            print(f"{name:12s} missing from the Vyb output")
            ok = False
            continue
        g = dig(np.load(f"{OUT}/{garr}"))
        f = vyb[name]
        worst = 0.0
        for k in ("sum", "abs", "sq", "max"):
            worst = max(worst, abs(f.get(k, 0) - g[k]) / max(1.0, abs(float(g[k]))))
        row = f"{name:12s} {f['n']:7d} " + "  ".join(
            f"{abs(f.get(k, 0) - g[k]) / max(1.0, abs(float(g[k]))):9.2e}"
            for k in ("sum", "sq", "max"))
        verdict = worst <= 5e-6
        ok = ok and verdict
        print(row + f"  {'OK' if verdict else 'FAIL'}")

    # indexer selections (token sets, order-insensitive: pool order varies under bf16)
    gsel = gold["topk"]
    bad = 0
    for q in range(len(gsel)):
        key = f"q{q}"
        a = sorted(set(gsel[q]))
        b = vsel.get(key, [])
        if sorted(set(b)) != a:
            bad += 1
            if bad <= 3:
                print(f"  sel {key}: vyb {b} vs gold {a}")
    print(f"indexer selections: {len(gsel) - bad}/{len(gsel)} queries agree")
    ok = ok and bad == 0

    print("L3_ATTN_VYB_MATCH:", "OK" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
