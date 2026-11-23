#!/usr/bin/env python3
"""compare_logits.py — the decision-boundary gate for two GLM configurations.

Reads the `LOGITS tok=<i> argmax=<a> top1=<x> top2=<y> margin=<m>` lines emitted by
glm_serve under `P6_LOGITS=1` from two runs and answers the only question that decides
whether a faster path is safe:

    can any token's argmax flip?

A token flips only if the perturbation of its logits exceeds its top1-top2 margin, so the
single number that answers it is the SAFETY FACTOR = (smallest margin) / (largest top1
perturbation). >1 means no observed token can flip; the margin of safety is how many times
over.

Per-layer digests cannot answer this: they measure intermediate state, and their drift is
dominated by cancellation-prone aggregates. The head is where behaviour is actually decided.

Usage:
    compare_logits.py <logA> <logB> [--label-a NAME] [--label-b NAME]

Exit code 0 if every token agrees and the safety factor is >1, 1 otherwise.
"""
import re
import sys

LINE = re.compile(
    r"^LOGITS tok=(\d+) argmax=(\d+) top1=(-?[\d.eE+-]+) top2=(-?[\d.eE+-]+) margin=(-?[\d.eE+-]+)"
)


def parse(path):
    toks = {}
    for line in open(path, errors="replace"):
        m = LINE.match(line.strip())
        if not m:
            continue
        t = int(m.group(1))
        toks[t] = {
            "argmax": int(m.group(2)),
            "top1": float(m.group(3)),
            "top2": float(m.group(4)),
            "margin": float(m.group(5)),
        }
    return toks


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    labels = {"a": "A", "b": "B"}
    for i, flag in enumerate(("--label-a", "--label-b")):
        if flag in sys.argv:
            labels["ab"[i]] = sys.argv[sys.argv.index(flag) + 1]
    if len(args) < 2:
        print(__doc__)
        return 2

    A, B = parse(args[0]), parse(args[1])
    if not A or not B:
        print("missing LOGITS lines: A=%d B=%d" % (len(A), len(B)))
        print("Did both runs have P6_LOGITS=1? The probe only runs on the main path,")
        print("so VYBFORGE_SKIP_MTP=1 must also be set or the MTP stage may abort first.")
        return 1

    common = sorted(set(A) & set(B))
    if not common:
        print("no common tokens: the two runs share no LOGITS lines -- not comparable")
        return 1
    print("tokens compared: %d (A has %d, B has %d)" % (len(common), len(A), len(B)))
    if len(common) != len(A) or len(common) != len(B):
        print("WARNING: token sets differ -- the two runs are not comparable")

    disagree = []
    max_dtop1 = 0.0
    worst_tok = None
    min_margin = None
    min_margin_tok = None
    for t in common:
        a, b = A[t], B[t]
        if a["argmax"] != b["argmax"]:
            disagree.append(t)
        d = abs(b["top1"] - a["top1"])
        if d > max_dtop1:
            max_dtop1, worst_tok = d, t
        m = min(a["margin"], b["margin"])
        if min_margin is None or m < min_margin:
            min_margin, min_margin_tok = m, t

    print()
    print("%6s %8s %8s %14s %14s %12s" % ("tok", "argmaxA", "argmaxB", "top1_A", "top1_B", "margin_min"))
    for t in common:
        a, b = A[t], B[t]
        flag = "  <-- ARGMAX FLIP" if a["argmax"] != b["argmax"] else ""
        print("%6d %8d %8d %14.7f %14.7f %12.7f%s"
              % (t, a["argmax"], b["argmax"], a["top1"], b["top1"],
                 min(a["margin"], b["margin"]), flag))

    print()
    print("argmax agreement      : %d/%d" % (len(common) - len(disagree), len(common)))
    if disagree:
        print("  flipping tokens     : %s" % disagree)
    print("max |delta top1|      : %.3e (tok %s)" % (max_dtop1, worst_tok))
    print("smallest margin       : %.3e (tok %s)" % (min_margin, min_margin_tok))
    if max_dtop1 > 0:
        print("safety factor         : %.0fx  (smallest margin / largest perturbation)"
              % (min_margin / max_dtop1))
    else:
        print("safety factor         : infinite (logits identical)")

    print()
    if not disagree and (max_dtop1 == 0 or min_margin > max_dtop1):
        print("VERDICT: PASS — no token can flip; %d/%d argmax agree, perturbation is %.0fx "
              "inside the tightest margin" % (len(common), len(common),
                                              (min_margin / max_dtop1) if max_dtop1 else 0))
        return 0
    print("VERDICT: FAIL — %d token(s) disagree or the perturbation exceeds a margin" % len(disagree))
    return 1


if __name__ == "__main__":
    sys.exit(main())
