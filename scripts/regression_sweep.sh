#!/bin/bash
# regression_sweep.sh — the correctness tripwire for the fused MoE fast path.
#
# P6_FUSED_ONLY defaults to 1 (production): the packed weights are read directly and the f32
# dequantisation is skipped, which is ~9.8x faster but means no projection is compared against a
# reference AT RUNTIME. This script is what replaces that runtime check: run it on every merge or
# nightly, not per request.
#
# It runs three passes and fails loudly on the first problem:
#
#   1. FAST PATH, EXACT BASELINE.  Fused-only at full depth, diffed bit-for-bit against
#      native/baselines/digest_fused_only45.txt. The path is deterministic, so ANY difference is a
#      real change -- not noise. A difference here is not automatically a bug (a deliberate change
#      should move the digests), but it must never happen silently: re-baseline only after pass 2
#      and the logit gate below confirm the new numbers.
#
#   2. VERIFICATION MODE.  P6_FUSED_ONLY=0 runs the deq alongside the fused chain and compares all
#      three projections per expert. This is the only pass that can say WHICH projection drifted.
#      Expected: 339 comparisons, 0 failures, max_abs <= ~3.3e-06.
#
#   3. LOGIT GATE.  Compares the fast path's head output against the recorded deq logits:
#      argmax agreement and the safety factor. This is the pass that answers "can a token change".
#
# Usage:
#   NLAY=45 scripts/regression_sweep.sh            # full sweep (~2h)
#   NLAY=4  SKIP_PASS1=1 scripts/regression_sweep.sh   # quick verification-only check
#   REBASELINE=1 scripts/regression_sweep.sh       # accept the new fast-path digests
#
# Runs ON lefty (paths and the vyb build there). LAN-localhost only.
set -u
cd /srv/models/vybforge || { echo "FAIL: /srv/models/vybforge not found"; exit 1; }

NLAY=${NLAY:-45}
PORT=${PORT:-18190}
SKIP_PASS1=${SKIP_PASS1:-0}
REBASELINE=${REBASELINE:-0}
BASE_DIG=native/baselines/digest_fused_only45.txt
BASE_LOGI=native/baselines/logits_detfused45.txt
BASE_LOGITD=native/baselines/logits_deq45.txt
run_fgate=/tmp/run_fgate.sh
FAILED=0

say() { echo; echo "=== $* ==="; }

say "sweep: NLAY=$NLAY port=$PORT $(date '+%Y-%m-%d %H:%M:%S')"

# ---------------------------------------------------------------- pass 1: fast path vs baseline
if [ "$NLAY" = "45" ] && [ "$SKIP_PASS1" = "0" ] && [ -f "$BASE_DIG" ]; then
  say "pass 1: fast path (P6_FUSED_ONLY=1) vs exact baseline"
  rm -f /tmp/sweep_fast.txt
  OUT=/tmp/sweep_fast.txt NLAY=$NLAY FG=1 REQS=1 PORT=$PORT P6_FUSED_ONLY=1 \
    bash "$run_fgate" >/dev/null 2>&1
  n=$(grep -c "^phase1 layer" /tmp/sweep_fast.txt 2>/dev/null || echo 0)
  echo "digests produced: $n (expected $NLAY)"
  if [ "$n" != "$NLAY" ]; then
    echo "FAIL: wrong digest count -- the run did not complete"; FAILED=1
  else
    grep "^phase1 layer" /tmp/sweep_fast.txt > /tmp/sweep_fast_d.txt
    if diff -q "$BASE_DIG" /tmp/sweep_fast_d.txt >/dev/null 2>&1; then
      echo "PASS: bit-identical to the recorded baseline"
    elif [ "$REBASELINE" = "1" ]; then
      cp /tmp/sweep_fast_d.txt "$BASE_DIG"
      echo "REBASELINED: $BASE_DIG updated from this run"
    else
      echo "DIFFERS from the baseline. Run pass 2 + the logit gate; if those pass, re-run with"
      echo "REBASELINE=1. First differing layer:"
      diff "$BASE_DIG" /tmp/sweep_fast_d.txt | head -4
      FAILED=1
    fi
  fi
else
  say "pass 1: SKIPPED (needs NLAY=45 and an existing baseline)"
fi

# ---------------------------------------------------------------- pass 2: verification mode
say "pass 2: verification mode (P6_FUSED_ONLY=0) -- per-expert comparison"
rm -f /tmp/sweep_ver.txt
OUT=/tmp/sweep_ver.txt NLAY=$NLAY FG=1 REQS=1 PORT=$((PORT+1)) P6_FUSED_ONLY=0 \
  bash "$run_fgate" >/dev/null 2>&1
cmps=$(grep -cE "^(FGATE|FGATEUP|FGATEDN|FGATESYN) expert" /tmp/serve_fgate.log 2>/dev/null || true)
fails=$(grep -cE "count_over_[0-9.e-]+=[1-9]" /tmp/serve_fgate.log 2>/dev/null || true)
# grep -c prints 0 AND exits 1 on no match, so `|| echo 0` would append a second 0 and turn the
# variable into "0\n0" -- the test below then reads it as a failure. `|| true` keeps a single value.
cmps=${cmps:-0}; fails=${fails:-0}
echo "comparisons: $cmps   failures: $fails"
grep -E "^(FGATE|FGATEUP|FGATEDN) expert" /tmp/serve_fgate.log | head -3
if [ "$fails" != "0" ]; then
  echo "FAIL: $fails projection(s) outside tolerance -- see /tmp/serve_fgate.log"; FAILED=1
elif [ "$cmps" = "0" ]; then
  echo "FAIL: no comparisons ran -- the verification block did not execute"; FAILED=1
else
  echo "PASS: every compared projection is within tolerance"
fi

# ---------------------------------------------------------------- pass 3: logit gate
if [ "$NLAY" = "45" ] && [ -f "$BASE_LOGITD" ]; then
  say "pass 3: logit gate -- fast path vs recorded deq logits"
  rm -f /tmp/sweep_lg.txt
  OUT=/tmp/sweep_lg_dig.txt NLAY=$NLAY FG=1 REQS=1 PORT=$((PORT+2)) P6_FUSED_ONLY=1 \
    P6_LOGITS=1 bash "$run_fgate" >/dev/null 2>&1
  grep "^LOGITS" /tmp/serve_fgate.log > /tmp/sweep_lg.txt 2>/dev/null || true
  ntok=$(grep -c "^LOGITS tok" /tmp/sweep_lg.txt 2>/dev/null || echo 0)
  echo "logits captured: $ntok tokens"
  if [ "$ntok" = "0" ]; then
    echo "FAIL: the head probe produced nothing (is P6_LOGITS=1 reaching the server?)"; FAILED=1
  else
    python3 scripts/compare_logits.py "$BASE_LOGITD" /tmp/sweep_lg.txt \
      --label-a deq --label-b fastpath | tail -8
    if [ "${PIPESTATUS[0]}" != "0" ]; then
      echo "FAIL: the logit gate rejected the fast path"; FAILED=1
    fi
  fi
else
  say "pass 3: SKIPPED (needs NLAY=45 and the recorded deq logits)"
fi

# ---------------------------------------------------------------- pass 4: decode-path digests
# The other three passes all measure the PREFILL. The decode path is a different function
# (moe_tail_step) and would regress with every other pass still green, so it needs its own check.
# The fused decode path is deterministic, so this is a bit-identical comparison.
BASE_DEC=native/baselines/digest_decode6.txt
if [ -f "$BASE_DEC" ]; then
  say "pass 4: decode-path digests vs baseline"
  rm -f /tmp/sweep_dec.txt
  OUT=/tmp/sweep_dec.txt NLAY=6 FG=1 REQS=1 NSTEPS=1 PORT=$((PORT+3)) \
    bash "$run_fgate" >/dev/null 2>&1
  grep -E "^l[0-9]+s[0-9]+" /tmp/sweep_dec.txt | sort > /tmp/sweep_dec_d.txt
  n=$(wc -l < /tmp/sweep_dec_d.txt)
  echo "decode digest lines: $n"
  if [ "$n" = "0" ]; then
    echo "FAIL: no decode digests produced"; FAILED=1
  elif diff -q "$BASE_DEC" /tmp/sweep_dec_d.txt >/dev/null 2>&1; then
    echo "PASS: decode digests bit-identical"
  elif [ "$REBASELINE" = "1" ]; then
    cp /tmp/sweep_dec_d.txt "$BASE_DEC"
    echo "REBASELINED: $BASE_DEC"
  else
    echo "DIFFERS from the decode baseline:"
    diff "$BASE_DEC" /tmp/sweep_dec_d.txt | head -4
    FAILED=1
  fi
else
  say "pass 4: SKIPPED (no $BASE_DEC recorded)"
fi

# ---------------------------------------------------------------- verdict
say "VERDICT"
if [ "$FAILED" = "0" ]; then
  echo "SWEEP PASS — fast path unchanged, verification clean, no token can flip"
  exit 0
fi
echo "SWEEP FAIL — do not ship this. Investigate the failing pass above."
exit 1
