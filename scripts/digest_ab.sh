#!/bin/bash
# digest_ab.sh — 45-layer gold-digest A/B: deq path vs fused MoE.
# Runs each configuration in a fresh process with a fresh output file (glm_serve
# APPENDS to VYBFORGE_CHAIN_OUT, so the default file is a 1 GB accumulation and
# useless for a diff), then compares the per-layer digests.
set -u
cd /srv/models/vybforge || exit 1
export VYB_STDLIB="$HOME/Projects/Vyb/stdlib"
export VYBFORGE_GX_DIR=/srv/models/hf/hub/models--Mia-AiLab--GLM-5.3-Flash-EXL3-4bpw-TensorFold/snapshots/078455ffe6472f9a52fbc1139f58b9db2881b25c
NL=${NL:-45}
IDS="785,1234,4321,111,222,333,444,555,666,777,888,999,1010,1111,1212,1313,1414,1515,1616,1717,1818,1919"

run_one () {   # $1=label  $2=port  $3=fused(0/1)  $4=outfile
  local label=$1 port=$2 fg=$3 out=$4
  local log=/tmp/digest_${label}.log
  rm -f "$out" "$log"
  echo "=== $label (FG=$fg) starting on port $port, out=$out"
  if [ "$fg" = "1" ]; then
    P6_FUSED_GATE=1 P6_FUSED_ONLY=1 VYBFORGE_SKIP_MTP=1 \
      VYBFORGE_SERVE_PORT="$port" VYBFORGE_NLAYERS="$NL" VYBFORGE_CHAIN_OUT="$out" \
      nohup "$HOME/Projects/Vyb/build2/vyb" \
        --module-path native/glm53 --module-path native/tensor --module-path native/json \
        --module-path native/torchload --module-path native/kernels \
        --module-path "$HOME/Projects/Vyb/bindings/cuda" \
        --module-path "$HOME/Projects/Vyb/stdlib/network" \
        native/host/glm_serve.vyb > "$log" 2>&1 &
  else
    P6_FUSED_GATE=0 VYBFORGE_SKIP_MTP=1 \
      VYBFORGE_SERVE_PORT="$port" VYBFORGE_NLAYERS="$NL" VYBFORGE_CHAIN_OUT="$out" \
      nohup "$HOME/Projects/Vyb/build2/vyb" \
        --module-path native/glm53 --module-path native/tensor --module-path native/json \
        --module-path native/torchload --module-path native/kernels \
        --module-path "$HOME/Projects/Vyb/bindings/cuda" \
        --module-path "$HOME/Projects/Vyb/stdlib/network" \
        native/host/glm_serve.vyb > "$log" 2>&1 &
  fi
  local srv=$!

  local i=0
  while [ $i -lt 90 ]; do
    if grep -q "listening on" "$log" 2>/dev/null; then break; fi
    if ! kill -0 "$srv" 2>/dev/null; then echo "!! $label server died before listening"; break; fi
    sleep 10; i=$((i+1))
  done
  if ! grep -q "listening on" "$log" 2>/dev/null; then
    echo "!! $label never listened"; tail -5 "$log"; return 1
  fi
  echo "--- $label listening; posting request (this runs all 45 layers)"
  curl -s -m 3600 -X POST "http://127.0.0.1:${port}/v1/generate?ids=${IDS}&nsteps=1" | head -c 120
  echo
  sleep 3
  kill "$srv" 2>/dev/null
  sleep 2
  kill -9 "$srv" 2>/dev/null
  local n
  n=$(grep -c "^phase1 layer" "$out" 2>/dev/null || echo 0)
  echo "--- $label digests captured: $n"
  grep -E "CHAIN_FAIL|ALL_FAIL|Error" "$log" | head -3
  return 0
}

run_one deq   18130 0 /tmp/digest_deq.txt
run_one fused 18131 1 /tmp/digest_fused.txt

echo "==================== COMPARISON ===================="
grep "^phase1 layer" /tmp/digest_deq.txt   2>/dev/null | sed 's/[[:space:]]*$//' > /tmp/d_deq.txt
grep "^phase1 layer" /tmp/digest_fused.txt 2>/dev/null | sed 's/[[:space:]]*$//' > /tmp/d_fused.txt
echo "deq layers:   $(wc -l < /tmp/d_deq.txt)"
echo "fused layers: $(wc -l < /tmp/d_fused.txt)"
if [ ! -s /tmp/d_deq.txt ] || [ ! -s /tmp/d_fused.txt ]; then
  echo "RESULT: INCOMPLETE — one side produced no digests"
else
  if diff -q /tmp/d_deq.txt /tmp/d_fused.txt >/dev/null; then
    echo "RESULT: PASS — all $(wc -l < /tmp/d_deq.txt) layer digests identical"
  else
    echo "RESULT: FAIL — differing layers:"
    diff /tmp/d_deq.txt /tmp/d_fused.txt | head -20
  fi
fi
pgrep -f glm_serve >/dev/null && { echo "(cleaning stray vyb)"; pkill -f glm_serve; }
echo "DIGEST_AB_DONE"
