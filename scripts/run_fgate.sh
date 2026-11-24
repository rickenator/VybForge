#!/bin/bash
# run_fgate.sh — start glm_serve with the fused-gate verification on, drive one
# generation request, print the FGATE lines. LAN-localhost only.
set -u
cd /srv/models/vybforge || exit 1
export VYB_STDLIB="$HOME/Projects/Vyb/stdlib"
export VYBFORGE_GX_DIR=/srv/models/hf/hub/models--Mia-AiLab--GLM-5.3-Flash-EXL3-4bpw-TensorFold/snapshots/078455ffe6472f9a52fbc1139f58b9db2881b25c
PORT=${PORT:-18102}
FG=${FG:-1}          # 1 = fused MoE projections, 0 = deq path
NLAY=${NLAY:-4}
# NSTEPS > 1 turns this into the DIFFERENTIAL decode measurement: wall(NSTEPS) - wall(1) is
# NSTEPS-1 decode steps, which cancels the one-time cold checkpoint read that dominates a single
# request (~175 GB at disk speed). That difference is the only reliable way to get a warm ms/token
# out of this harness without timestamps in the log.
NSTEPS=${NSTEPS:-1}
# A multi-step request costs ~NSTEPS x (one full streaming pass of the checkpoint), so the default
# 900 s curl timeout silently truncates it at exactly 900.0 s with an EMPTY body -- which reads like
# "the request hung" rather than "the timeout fired". Raise CTIMEOUT for any NSTEPS run.
CTIMEOUT=${CTIMEOUT:-900}
LOG=/tmp/serve_fgate.log
# IDS_OVERRIDE lets the caller run a different prompt set (>=22 ids required).
IDS="${IDS_OVERRIDE:-785,1234,4321,111,222,333,444,555,666,777,888,999,1010,1111,1212,1313,1414,1515,1616,1717,1818,1919,2020,2121,2222,2323,2424,2525,2626,2727,2828,2929,3030,3131,3232,3333,3434,3535,3636,3737,3838}"

rm -f "$LOG"
P6_FUSED_GATE="$FG" VYBFORGE_SKIP_MTP=1 VYBFORGE_SERVE_PORT="$PORT" VYBFORGE_NLAYERS="$NLAY" VYBFORGE_CHAIN_OUT="${OUT:-native/out/decchain_vyb.txt}" \
  nohup "$HOME/Projects/Vyb/build2/vyb" \
    --module-path native/glm53 \
    --module-path native/tensor \
    --module-path native/json \
    --module-path native/torchload \
    --module-path native/kernels \
    --module-path "$HOME/Projects/Vyb/bindings/cuda" \
    --module-path "$HOME/Projects/Vyb/stdlib/network" \
    native/host/glm_serve.vyb > "$LOG" 2>&1 &
SRV=$!
echo "started pid=$SRV port=$PORT nlayers=$NLAY"

for i in $(seq 1 60); do
  if grep -q "listening on" "$LOG" 2>/dev/null; then echo "listening after ${i}0s"; break; fi
  if ! kill -0 "$SRV" 2>/dev/null; then echo "SERVER DIED"; break; fi
  sleep 10
done

if grep -q "listening on" "$LOG"; then
  # WARM REPEATED MEASUREMENT: the first request pays the cold read of a 175 GB
  # checkpoint; later ones are warm. Report every request's wall time so the
  # steady state is visible instead of being hidden inside one cold number.
  REQS=${REQS:-5}
  # NOTE: the id list MUST vary per request -- an identical repeat is answered from
  # cache in ~0.1 ms and measures nothing. Appending the request index does that.
  echo "--- timed requests (n=${REQS}), seconds:"
  i=1
  while [ "$i" -le "$REQS" ]; do
    T=$(curl -s -o /tmp/resp.$$ -w '%{time_total}' -m "$CTIMEOUT" -X POST \
        "http://127.0.0.1:${PORT}/v1/generate?ids=${IDS},${i}&nsteps=${NSTEPS}")
    echo "req${i} ${T}s  $(head -c 80 /tmp/resp.$$)"
    i=$((i+1))
  done
  rm -f /tmp/resp.$$
fi

echo "--- SEC (MoE section timers):"
grep -E "SEC P1ffn|SEC P1tail|SEC P1attn" "$LOG" | head -12
echo "--- LOGITS lines:"
grep -E "^LOGITS" "$LOG" | head -8
echo "--- FGATE lines:"
grep FGATE "$LOG" | head -20
echo "--- failures:"
grep -iE "CHAIN_FAIL|Error|panic" "$LOG" | head -5
echo "--- tail:"
tail -4 "$LOG"
kill "$SRV" 2>/dev/null

# VALIDITY GUARD (round 195). Identical harness settings sometimes answer WITHOUT ever reaching
# moe_tail_step -- the log then has digests but zero "SEC moekern" lines, and any per-layer MoE number
# read from it is meaningless (it is prefill only, or a cached answer). A run that does not exercise the
# decode MoE must be treated as INVALID, never as a measurement.
# NOTE: `grep -c` exits 1 when it finds nothing, so `$(grep -c ... || echo 0)` yields "0\n0" and a
# numeric test on it silently falls into the wrong branch -- use -q to decide, then count.
if grep -q "SEC moekern" "$LOG" 2>/dev/null; then
  MOEKERN=$(grep -c "SEC moekern" "$LOG")
  rm -f "$LOG.invalid"
  echo "VALID_RUN: $MOEKERN moekern layer-lines."
else
  echo "INVALID_RUN: no SEC moekern lines -- the decode MoE stage never ran (or the process died in it);"
  echo "             do not use these numbers."
  echo "INVALID_RUN" > "$LOG.invalid"
fi
echo "DONE"
