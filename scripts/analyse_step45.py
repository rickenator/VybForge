#!/usr/bin/env python3
"""analyse_step45.py -- attribute a full-depth glm_serve step and name the next hogs.

Reads a glm_serve log and produces, in descending order of cost:
  * per-phase totals and medians across the whole run (SEC P1attn / P1ffn / P1tail)
  * the same grouped into STEPS (blocks of NLAY layer-sites), so a step budget is visible
  * the intermittent ~900 ms events (SEC_DEQ, and SEC_ATTN phases over 200 ms), summed
  * the resolved-vs-unattributed split: everything the per-layer timers account for,
    against the request wall time, so the part nobody has instrumented yet is explicit

Usage:
  analyse_step45.py LOG [--nlay 45] [--wall 61.2[,58.4,...]]

`--wall` takes the per-request wall seconds printed by run_fgate.sh ("req1 61.2s"),
which is what makes the unattributed remainder computable rather than guessed.

Every number here is a measurement from the log. Nothing is inferred from geometry.
"""
import argparse
import re
import statistics
import sys
from collections import defaultdict

PHASES = {
    "P1attn": "attention site (DSA or KDA)",
    "P1ffn": "FFN / MoE branch",
    "P1tail": "layer tail (router, experts, combine)",
}
HITCH = 200  # ms: a single phase this slow is the ~900 ms event, not real work


def parse(path):
    phases = defaultdict(list)
    attn_inner = []      # SEC_ATTN lines
    deq_slow = []        # SEC_DEQ SLOW lines
    order = []           # sequence of (phase, ms) for step grouping
    with open(path, errors="replace") as fh:
        for line in fh:
            m = re.search(r"SEC (P1attn|P1ffn|P1tail)\s+(\d+)", line)
            if m:
                v = int(m.group(2))
                phases[m.group(1)].append(v)
                order.append((m.group(1), v))
                continue
            m = re.search(r"SEC_ATTN capslot=(\d+) S=(\d+) W=(\d+) I=(\d+) H=(\d+) P=(\d+) M=(\d+) T=(\d+) tot=(\d+)", line)
            if m:
                g = [int(x) for x in m.groups()]
                attn_inner.append({"slot": g[0], "S": g[1], "W": g[2], "I": g[3],
                                   "H": g[4], "P": g[5], "M": g[6], "T": g[7], "tot": g[8]})
                continue
            m = re.search(r"SEC_DEQ SLOW tot=(\d+)", line)
            if m:
                deq_slow.append(int(m.group(1)))
    return phases, attn_inner, deq_slow, order


def stat(vals):
    if not vals:
        return "n=0"
    v = sorted(vals)
    return (f"n={len(v):4d} med={int(statistics.median(v)):6d} "
            f"p90={v[min(len(v)-1, int(0.9*(len(v)-1)))]:6d} max={v[-1]:6d} "
            f"sum={sum(v):8d}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("log")
    ap.add_argument("--nlay", type=int, default=45)
    ap.add_argument("--wall", default="")
    args = ap.parse_args()

    phases, attn_inner, deq_slow, order = parse(args.log)
    if not order:
        print("No SEC lines found. Wrong log, or the driver did not emit timers.")
        return 1

    print("=" * 78)
    print("PER-PHASE, WHOLE RUN (ms)")
    print("=" * 78)
    for k, desc in PHASES.items():
        print(f"{k:8s} {desc:32s} {stat(phases.get(k, []))}")

    # ---- step budget ----
    # Each LAYER emits THREE timer lines (P1attn, P1ffn, P1tail), so the raw sequence must be grouped
    # into layers first. Treating each line as a layer silently divided the step into thirds and made the
    # per-step totals meaningless.
    per_layer = []
    for i in range(0, len(order) - 2, 3):
        per_layer.append(order[i][1] + order[i + 1][1] + order[i + 2][1])
    n = args.nlay
    steps = [sum(per_layer[i:i + n]) for i in range(0, len(per_layer), n)]
    steps = [s for s in steps if s > 0]
    print()
    print("=" * 78)
    print(f"PER-STEP BUDGET (blocks of {n} layer-sites, ms)")
    print("=" * 78)
    for i, s in enumerate(steps):
        print(f"  step {i}: {s:7d} ms   ({s/n:6.1f} ms/layer)")
    if len(steps) > 1:
        warm = steps[1:]
        print(f"  warm steps (excluding step 0): med={int(statistics.median(warm))} ms")

    # ---- hitches ----
    print()
    print("=" * 78)
    print(f"INTERMITTENT EVENTS (single phase >= {HITCH} ms) -- the ~900 ms signature")
    print("=" * 78)
    big = [v for _, v in order if v >= HITCH]
    tot_big = sum(big)
    print(f"  layer-level phases >= {HITCH} ms : {len(big)} events, {tot_big} ms total")
    if big:
        print(f"                                   med={int(statistics.median(big))} ms")
    inner_big = [a for a in attn_inner if a["tot"] >= HITCH]
    print(f"  SEC_ATTN totals >= {HITCH} ms    : {len(inner_big)} events, "
          f"{sum(a['tot'] for a in inner_big)} ms")
    for a in inner_big[:6]:
        worst = max(("S", a["S"]), ("W", a["W"]), ("I", a["I"]), ("H", a["H"]),
                    ("P", a["P"]), ("M", a["M"]), ("T", a["T"]), key=lambda kv: kv[1])
        print(f"      slot={a['slot']} tot={a['tot']:4d} worst={worst[0]}={worst[1]}")
    print(f"  SEC_DEQ slow calls            : {len(deq_slow)} events, {sum(deq_slow)} ms")

    # ---- account for the wall ----
    walls = [float(x) for x in args.wall.split(",") if x.strip()]
    if walls:
        print()
        print("=" * 78)
        print("WALL ACCOUNTING (per request)")
        print("=" * 78)
        accounted_per_step = statistics.median([s for s in steps]) if steps else 0
        for i, w in enumerate(walls):
            ms = w * 1000.0
            # a request runs one prompt pass then nsteps; compare against the median step
            print(f"  req{i+1}: wall={ms:8.0f} ms   timed-layer budget/step={accounted_per_step:7.0f} ms"
                  f"   unattributed={ms - accounted_per_step:8.0f} ms")
        print("  NOTE: unattributed includes prompt processing, HTTP, the checkpoint stream for a")
        print("  cold request, and every region with no timer. Do not read it as a single hog.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
