#!/bin/bash
# bench_decode.sh — in-process, steady-state timing for the decode path.
#
# THE PROBLEM THIS SOLVES: measuring via separate cold serve requests varies +-17% run to run (a
# 175 GB checkpoint with per-run page-cache state), which makes any per-layer comparison below ~100 ms
# undecidable. Two runs of the SAME binary differed 505 vs 432 ms per layer. Running NSTEPS steps
# inside one process fixes it: step 0 pays the cold read and every later step is warm, so the spread
# across steps is the real noise and the median is a number a change can be judged against.
#
# Usage (ON lefty, from /srv/models/vybforge):
#   NLAY=6 STEPS=8 scripts/bench_decode.sh
#   NLAY=6 STEPS=8 FIELD=mmG scripts/bench_decode.sh
#   NLAY=45 STEPS=4 scripts/bench_decode.sh
#
# Keep NLAY small: cost is roughly prefill + STEPS x (per-layer x NLAY).
set -u
cd /srv/models/vybforge || { echo "FAIL: /srv/models/vybforge not found"; exit 1; }

NLAY=${NLAY:-6}
STEPS=${STEPS:-8}
FIELD=${FIELD:-mm}
PORT=${PORT:-18240}
run_fgate=${run_fgate:-/tmp/run_fgate.sh}
OUT=${OUT:-/tmp/bench_decode.txt}
# run_fgate.sh ALWAYS writes its server log to /tmp/serve_fgate.log (rm'd at start of each run);
# copying a differently-named path here silently produced "no log" instead of an error.
LOG=/tmp/serve_fgate.log

[ -f "$run_fgate" ] || { echo "FAIL: harness not found at $run_fgate"; exit 1; }

echo "bench: NLAY=$NLAY STEPS=$STEPS field=$FIELD port=$PORT"
rm -f "$OUT"
# NSTEPS>1 is the whole point; CTIMEOUT must exceed prefill + STEPS x step cost or the request is
# truncated at exactly the timeout with an empty body (which looks like a hang).
OUT="$OUT" NLAY="$NLAY" FG=1 REQS=1 NSTEPS="$STEPS" CTIMEOUT=${CTIMEOUT:-5400} PORT="$PORT" \
  bash "$run_fgate" > /tmp/bench_harness_out.txt 2>&1
if [ ! -f "$LOG" ]; then
  echo "FAIL: no log at $LOG — harness output was:"
  tail -5 /tmp/bench_harness_out.txt
  exit 1
fi
cp "$LOG" /tmp/bench_decode_saved.log

echo
grep -c "^T_MOE" /tmp/bench_decode_saved.log 2>/dev/null | sed 's/^/T_MOE samples: /'
echo
python3 scripts/analyse_bench.py /tmp/bench_decode_saved.log --field "$FIELD"
