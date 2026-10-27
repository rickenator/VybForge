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
# Orientation-fixed forward (see VybForge#11): continuation [12095,11,323] = " Paris,".
# (Old gold [19151,87054,83376] was the pre-fix token salad anieQN-cond; corrected.)
GOLD = [785, 6722, 315, 9625, 374, 12095, 11, 323]

runs = re.findall(r"RESIDENT_RUN=(\d+) ms=(\d+) tokens=(\d+)", log)
secs = re.findall(r"RESIDENT_AVG_TOKENS_PER_SEC=([0-9.eE+-]+)", log)

fail = False
if not runs:
    print("RESIDENT-BENCH: FAIL — no RESIDENT_RUN lines in log")
    fail = True
else:
    for r, ms, toks in runs:
        print(f"  run {r}: {toks} tokens in {ms} ms -> {int(toks)/(int(ms)/1000.0):.2f} t/s")

# The first len(PROMPT) ids of the stream are the seed the driver was given: if they are not the gold prompt,
# the input (native/out/prompt_ids.txt) is wrong, not the forward. Say so instead of a bare stream mismatch.
GOLD_PROMPT = GOLD[:5]
if stream and stream[:len(GOLD_PROMPT)] != GOLD_PROMPT:
    print(f"RESIDENT-BENCH: FAIL — seed ids {stream[:len(GOLD_PROMPT)]}... are not the gold prompt {GOLD_PROMPT} "
          "(\"The capital of France is\"): native/out/prompt_ids.txt is stale or for another prompt; "
          "`make resident-bench` re-encodes it")
    fail = True
elif stream and stream != GOLD:
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
