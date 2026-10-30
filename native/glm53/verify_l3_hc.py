"""Gate: L3_HC_VYB — the attention-site hyper-connection mix/apply in isolation.

The full-layer run had streams1 1.3e-3 off the gold while every attention stage matched at
1e-11, so the hc mix/apply was the suspect. It was: the comb softmax used a max over the
whole 4x4 instead of a per-row max, so a row whose logits sit far below the block's peak
underflowed to zero on its small entries before the hc_eps floor went on -- 6x wrong on
those entries, 2e-3 on streams1. With a per-row max: comb 2e-8, streams1 2e-9.
"""
import sys

OUT = "native/out"


def parse(path):
    out = {}
    for line in open(path):
        p = line.split()
        if p:
            out[p[0]] = {k: int(v) for k, v in (x.split("=", 1) for x in p[1:] if "=" in x)}
    return out


def main():
    try:
        v = parse(f"{OUT}/l3hc_vyb.txt")
        g = parse(f"{OUT}/gold_l3hc.txt")
    except FileNotFoundError as e:
        print(f"L3_HC_VYB: FAIL ({e.filename})")
        return 1
    tol = {"mv": 1e-9, "collapsed": 1e-9, "postA": 1e-6, "combA": 1e-6, "h_a": 1e-9,
           "streams1": 1e-6}
    ok = True
    for stage, t in tol.items():
        if stage not in v or stage not in g:
            print(f"  {stage}: MISSING")
            ok = False
            continue
        worst = 0.0
        for k in ("sum", "abs", "sq", "max"):
            den = abs(float(g[stage]["abs"])) if k == "sum" else max(1.0, abs(float(g[stage][k])))
            worst = max(worst, abs(v[stage][k] - g[stage][k]) / den)
        good = worst <= t
        ok = ok and good
        print(f"  {stage:10s} worst {worst:.2e} (tol {t:.0e}) {'ok' if good else 'FAIL'}")
    print("L3_HC_VYB:", "OK" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
