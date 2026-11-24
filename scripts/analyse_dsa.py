#!/usr/bin/env python3
"""analyse_dsa.py -- attribute the DSA layer cost from SEC_DSA lines.

The engine prints, once per DSA layer per step:

    SEC_DSA slot=<k> A=<ms> B=<ms> C=<ms> tot=<ms>

Three DISJOINT windows (round 168), so nothing is double-counted:
  A = resolve + weight read/dequant + hc mix gemm + hc_glue + CPU splits + input rms
  B = attn_site_c (its own weight dequant + attention + output projection)
  C = digests + hc_apply (pure CPU work, previously unmeasured)

Usage: analyse_dsa.py [logfile ...]        (default /tmp/serve_fgate.log)

Reports, per phase and per slot: n, min, p10, median, p90, max, and share of tot.
Step 0 is cold (checkpoint still streaming), so it is reported SEPARATELY and never
folded into the medians -- a cold step can be an order of magnitude slower and would
poison every conclusion, which is exactly the error this session kept making.
"""
import re
import statistics
import sys

LINE = re.compile(r"SEC_DSA slot=(\d+) A=(\d+) B=(\d+) C=(\d+) tot=(\d+)")


def load(paths):
    rows = []
    for path in paths:
        try:
            with open(path, "r", errors="replace") as fh:
                for line in fh:
                    m = LINE.search(line)
                    if m:
                        rows.append(tuple(int(g) for g in m.groups()))
        except FileNotFoundError:
            print(f"WARN: no such file: {path}")
    return rows


def show(name, values):
    if not values:
        return "n=0"
    values = sorted(values)
    p10 = values[max(0, int(0.10 * (len(values) - 1)))]
    p90 = values[min(len(values) - 1, int(0.90 * (len(values) - 1)))]
    return (
        f"n={len(values):3d} min={values[0]:6d} p10={p10:6d} "
        f"med={int(statistics.median(values)):6d} p90={p90:6d} max={values[-1]:6d}"
    )


def report(label, rows):
    if not rows:
        print(f"{label}: no SEC_DSA lines")
        return
    print(f"\n{label}  ({len(rows)} samples)")
    tot_med = statistics.median([r[4] for r in rows])
    for idx, phase in ((1, "A read+deq+mix"), (2, "B attn_site_c"), (3, "C digest+hc  ")):
        vals = [r[idx] for r in rows]
        med = statistics.median(vals)
        share = (100.0 * med / tot_med) if tot_med else 0.0
        print(f"  {phase}  {show(phase, vals)}   share={share:5.1f}%")
    print(f"  tot              {show('tot', [r[4] for r in rows])}")


def main(argv):
    paths = argv[1:] or ["/tmp/serve_fgate.log"]
    rows = load(paths)
    if not rows:
        print("No SEC_DSA lines. Is the engine the round-168 build, and NLAY>3 so a DSA layer runs?")
        return 1

    # step 0 is the first slot 0 occurrence; everything else is warm
    cold = [r for i, r in enumerate(rows) if i < max(r[0] for r in rows) and r[0] == 0]
    warm = [r for i, r in enumerate(rows) if r not in cold]
    report("COLD (step 0 -- reported, never folded into the medians)", cold)
    report("WARM (all later steps)", warm)

    # per-slot medians make layer-to-layer variation visible without mixing phases
    print("\nper-slot warm medians (slot 0 = first DSA layer):")
    for slot in sorted({r[0] for r in warm}):
        srows = [r for r in warm if r[0] == slot]
        print(
            f"  slot {slot}: A={int(statistics.median([r[1] for r in srows])):6d} "
            f"B={int(statistics.median([r[2] for r in srows])):6d} "
            f"C={int(statistics.median([r[3] for r in srows])):6d} "
            f"tot={int(statistics.median([r[4] for r in srows])):6d}"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
