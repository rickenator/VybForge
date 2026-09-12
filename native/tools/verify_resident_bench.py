#!/usr/bin/env python3
"""Gate for the resident throughput bench.

Checks that the resident_driver run (a) produced the correct gold generated
ids for each interaction (no correctness regression from wrapping the gen loop),
and (b) reported a finite tokens/sec average. Prints the running per-run timing
and the resident average. Exits nonzero if the gold stream is wrong or t/s is
non-finite.

Inputs:
  native/out/chatgen_ids_vyb.txt  (last interaction's full stream, as decode)
  native/out/resident_bench.log   (the driver's stdout: RESIDENT_RUN=.. ms=..,
                                   RESIDENT_AVG_TOKENS_PER_SEC=..)
"""
import re, sys

log = open("native/out/resident_bench.log").read()
ids_file = "native/out/chatgen_ids_vyb.txt"
try:
    stream = [int(x) for x in open(ids_file).read().split()]
except FileNotFoundError:
    stream = []

# Autoregressive gold for prompt_The capital of France is (prompt_ids.txt) with GEN=3.
# decode_driver/chatgen_ref gold: continuation [19151,87054,83376].
GOLD = [785, 6722, 315, 9625, 374, 19151, 87054, 83376]

runs = re.findall(r"RESIDENT_RUN=(\d+) ms=(\d+) tokens=(\d+)", log)
secs = re.findall(r"RESIDENT_AVG_TOKENS_PER_SEC=([0-9.eE+-]+)", log)

fail = False
if not runs:
    print("RESIDENT-BENCH: FAIL — no RESIDENT_RUN lines in log")
    fail = True
else:
    for r, ms, toks in runs:
        print(f"  run {r}: {toks} tokens in {ms} ms -> {int(toks)/(int(ms)/1000.0):.2f} t/s")

# correctness on the (last) interaction
if stream and stream != GOLD:
    print(f"RESIDENT-BENCH: FAIL — generated stream {stream} != gold {GOLD}")
    fail = True
elif not stream:
    print("RESIDENT-BENCH: FAIL — no generated ids file")
    fail = True
else:
    print("RESIDENT-BENCH: gold stream MATCH — resident gen correct")

if not secs:
    print("RESIDENT-BENCH: FAIL — no RESIDENT_AVG_TOKENS_PER_SEC")
    fail = True
else:
    tps = float(secs[-1])
    if tps <= 0:
        print(f"RESIDENT-BENCH: FAIL — non-positive t/s {tps}")
        fail = True
    else:
        print(f"RESIDENT-BENCH: resident avg = {tps:.3f} tokens/sec")

print("RESIDENT-BENCH: " + ("FAIL" if fail else "PASS"))
sys.exit(1 if fail else 0)
