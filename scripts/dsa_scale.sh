#!/bin/bash
# dsa_scale.sh — does the DSA layer cost scale with the NUMBER of DSA layers (accumulation), or with
# layer POSITION only (fixed per-layer cost)?
# If the per-layer cost is fixed, the max DSA layer cost is the same at NLAY=12/24/45.
# If state accumulates, the max grows with the number of DSA layers.
set -u
cd /srv/models/vybforge || exit 1
export VYB_STDLIB="$HOME/Projects/Vyb/stdlib"
export VYBFORGE_GX_DIR=/srv/models/hf/hub/models--Mia-AiLab--GLM-5.3-Flash-EXL3-4bpw-TensorFold/snapshots/078455ffe6472f9a52fbc1139f58b9db2881b25c
MP="--module-path native/glm53 --module-path native/tensor --module-path native/json --module-path native/torchload --module-path native/kernels --module-path $HOME/Projects/Vyb/bindings/cuda --module-path $HOME/Projects/Vyb/stdlib/network"
IDS="785,1234,4321,111,222,333,444,555,666,777,888,999,1010,1111,1212,1313,1414,1515,1616,1717,1818,1919,2020,2121"

for NL in 12 24 45; do
  PORT=$((18440 + NL))
  rm -f "/tmp/sc$NL.log"
  P6_FUSED_GATE=1 P6_FUSED_DEC=1 VYBFORGE_SKIP_MTP=1 VYBFORGE_SERVE=1 \
    VYBFORGE_SERVE_PORT=$PORT VYBFORGE_NLAYERS=$NL VYBFORGE_CHAIN_OUT="/tmp/sc$NL.txt" \
    nohup "$HOME/Projects/Vyb/build2/vyb" $MP native/host/glm_serve.vyb > "/tmp/sc$NL.log" 2>&1 &
  P=$!
  for i in $(seq 1 60); do grep -q "listening on" "/tmp/sc$NL.log" 2>/dev/null && break; sleep 5; done
  sleep 1
  curl -s -o /dev/null -m 4000 "http://127.0.0.1:${PORT}/v1/generate?ids=$IDS&nsteps=1"
  kill $P 2>/dev/null
  sleep 2
  NL=$NL python3 /srv/models/vybforge/scripts/dsa_scale_report.py "/tmp/sc$NL.log"
done
echo DONE
