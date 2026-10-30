"""Gate: L3_FULL_VYB_MATCH — layer 3 end to end, one Vyb process.

The driver chains: streams0 -> hc_attn mix -> input_layernorm -> h_a -> DSA attention ->
hc_apply -> streams1 -> hc_ffn mix -> post_attention_layernorm -> h2 -> MoE ->
hc_apply -> streams2.

Compared against the gold arrays: h_a, attn, streams1 (attention site + its hc apply) and
h2, mlp, streams2 (ffn site). The per-stage tolerance follows what each stage's own gate
established: attention ~1e-5, mlp ~1e-4 (EXL3 fp16 butterfly noise), streams ~1e-4.
"""
import json
import sys

import numpy as np

OUT = "native/out"
S40 = 1099511627776.0

GOLD = {
    "h_a": (f"{OUT}/gold_l3_ha.npy", 2e-5),
    "attn": (f"{OUT}/gold_l3_attn.npy", 2e-5),
    "streams1": (f"{OUT}/gold_l3_streams1.npy", 2e-4),
    "h2": (f"{OUT}/gold_l3_moe_h2.npy", 2e-4),
    "mlp": (f"{OUT}/gold_l3_mlp.npy", 2e-4),
    "streams2": (f"{OUT}/gold_l3_streams2.npy", 2e-4),
}


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
        v = parse(f"{OUT}/l3full_vyb.txt")
    except FileNotFoundError:
        print("L3_FULL_VYB_MATCH: FAIL (no l3full_vyb.txt)")
        return 1
    ok = True
    for stage, (gold_path, tol) in GOLD.items():
        if stage not in v:
            print(f"L3_FULL_VYB_MATCH: FAIL (no {stage} in the Vyb output)")
            return 1
        try:
            g = dig(np.load(gold_path))
        except FileNotFoundError:
            print(f"  {stage}: SKIP (no {gold_path})")
            continue
        a = v[stage]
        if a["n"] != g["n"]:
            print(f"  {stage}: FAIL n {a['n']} vs {g['n']}")
            ok = False
            continue
        rels = {}
        for k in ("sum", "abs", "sq", "max"):
            den = abs(float(g["abs"])) if k == "sum" else max(1.0, abs(float(g[k])))
            rels[k] = abs(a[k] - g[k]) / den
        good = all(r <= tol for r in rels.values())
        ok = ok and good
        print(f"  {stage:9s} sum {rels['sum']:.2e}  abs {rels['abs']:.2e}  "
              f"sq {rels['sq']:.2e}  max {rels['max']:.2e}  (tol {tol:.0e}) "
              f"{'ok' if good else 'FAIL'}")
    print("L3_FULL_VYB_MATCH:", "OK" if ok else "FAIL")
    # also fold in the gold's own stage digests for streams2 if the .npy is missing
    try:
        st = json.load(open(f"{OUT}/gold_l3_moe.json"))["stages"]
        if "streams2" in st and "streams2" in v:
            g = st["streams2"]
            rel = abs(v["streams2"]["abs"] - g["abs"]) / max(1.0, abs(float(g["abs"])))
            print(f"  (json streams2 abs rel {rel:.2e})")
    except Exception:
        pass
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
