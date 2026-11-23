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
LOG=/tmp/serve_fgate.log
IDS="785,1234,4321,111,222,333,444,555,666,777,888,999,1010,1111,1212,1313,1414,1515,1616,1717,1818,1919"

rm -f "$LOG"
P6_FUSED_GATE="$FG" VYBFORGE_SKIP_MTP=1 VYBFORGE_SERVE_PORT="$PORT" VYBFORGE_NLAYERS="$NLAY" \
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
    T=$(curl -s -o /tmp/resp.$$ -w '%{time_total}' -m 900 -X POST \
        "http://127.0.0.1:${PORT}/v1/generate?ids=${IDS},${i}&nsteps=1")
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
echo "DONE"
