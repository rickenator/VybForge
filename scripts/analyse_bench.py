#!/usr/bin/env python3
"""analyse_bench.py — medians from an in-process decode bench.

WHY THIS EXISTS: cold serve requests vary +-17% run to run (175 GB checkpoint, per-run page-cache
state), so a single run cannot decide a per-layer comparison below ~100 ms. Running N steps inside
ONE process removes that: step 0 pays the cold read, later steps are warm, and the spread between
them is the real measurement noise.

Reads the per-layer T_MOE / T_SPLIT lines emitted once per layer per step:
    T_MOE layer resolve=<ms> deq=<ms> mm=<ms> host=<ms> | mmG=.. mmU=.. mmD=.. set=.. xh=.. yh+rb=..
    T_SPLIT router=.. experts=.. shared=.. combine=.. hc_apply=.. after=..

Usage:
    analyse_bench.py <log> [--field mm] [--warm-only]

Reports, per field: n, min, p10, median, p90, max — and the median of step 0 separately from the
warm steps, so the cold-read effect is visible instead of silently inflating the average.
"""
import re
import statistics as st
import sys

MOE = re.compile(r"^T_MOE layer (.*)$")
SPLIT = re.compile(r"^T_SPLIT (.*)$")
STEPMARK = re.compile(r"^step(\d+) layer (\d+) done")
SEC = re.compile(r"^SEC ([A-Za-z0-9_]+) (\d+)$")


def parse(path):
    """Return {(step, layer): {field: ms}} for T_MOE and T_SPLIT lines."""
    rows, step, layer, order = {}, 0, -1, []
    for line in open(path, errors="replace"):
        line = line.strip()
        m = STEPMARK.match(line)
        if m:
            step, layer = int(m.group(1)), int(m.group(2))
            continue
        m = MOE.match(line)
        if m:
            fields = {}
            for tok in m.group(1).replace("|", " ").split():
                if "=" in tok:
                    k, v = tok.split("=", 1)
                    try:
                        fields[k] = int(v)
                    except ValueError:
                        pass
            rows[(step, layer)] = fields
            order.append((step, layer))
            continue
        # The DECODE path uses its own timers (`SEC mlpmoe`, `SEC moexpert`, ...) emitted once per
        # layer per step -- not T_MOE/T_SPLIT, which belong to the prefill (R=20) path. Parsing only
        # those gave 3 samples from the prefill and measured the wrong function entirely.
        m = SEC.match(line)
        if m:
            key = (step, layer)
            if key in rows:
                rows[key].setdefault(m.group(1), int(m.group(2)))
            else:
                rows[key] = {m.group(1): int(m.group(2))}
                order.append(key)
            continue
        m = SPLIT.match(line)
        if m:
            fields = {}
            for tok in m.group(1).split():
                if "=" in tok:
                    k, v = tok.split("=", 1)
                    try:
                        fields[k] = int(v)
                    except ValueError:
                        pass
            key = (step, layer)
            if key in rows:
                rows[key].update(fields)
            else:
                rows[key] = fields
                order.append(key)
    return rows, order


def stats(v):
    v = sorted(v)
    n = len(v)
    pick = lambda q: v[min(n - 1, max(0, int(round(q * (n - 1)))))]
    return dict(n=n, min=v[0], p10=pick(0.10), med=st.median(v), p90=pick(0.90), max=v[-1])


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    if not args:
        print(__doc__)
        return 2
    field = "mm"
    if "--field" in sys.argv:
        field = sys.argv[sys.argv.index("--field") + 1]
    warm_only = "--warm-only" in sys.argv

    rows, order = parse(args[0])
    if not rows:
        print("no T_MOE/T_SPLIT lines found in %s" % args[0])
        print("(the bench needs NSTEPS > 1 so the same layer is sampled on several steps)")
        return 1

    steps = sorted({s for s, _ in order})
    layers = sorted({l for _, l in order})
    print("log: %s" % args[0])
    print("steps=%d  layers=%d  samples=%d  field=%s" % (len(steps), len(layers), len(rows), field))

    # per-step medians show the cold->warm transition
    print()
    print("%6s %10s %10s %10s" % ("step", "n", "median", "max"))
    for s in steps:
        v = [rows[(ss, l)][field] for ss, l in order if ss == s and field in rows[(ss, l)]]
        if v:
            print("%6d %10d %10.0f %10.0f" % (s, len(v), st.median(v), max(v)))

    use = [s for s in steps if s != 0] if warm_only else steps
    vals = [rows[k][field] for k in order if k[0] in use and field in rows[k]]
    if not vals:
        print("\nno samples for %s" % field)
        return 1
    d = stats(vals)
    print()
    print("field %-8s (steps %s): n=%d  min=%.0f  p10=%.0f  median=%.0f  p90=%.0f  max=%.0f"
          % (field, "warm" if warm_only else "all", d["n"], d["min"], d["p10"], d["med"], d["p90"], d["max"]))
    spread = (d["max"] - d["min"]) / d["med"] * 100 if d["med"] else 0
    print("spread: %.0f%% of the median" % spread)
    if spread < 10:
        print("VERDICT: tight enough to compare variants (differences >%.0f ms are real)" % (d["med"] * 0.10))
    else:
        print("VERDICT: still noisy (+-%.0f%%); use --warm-only, more steps, or a larger effect" % (spread / 2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
