#!/usr/bin/env bash
# tp_pair.sh -- run the TP=2 expert-parallel SERVE on lefty+righty, both ranks processing one request.
#
# Why both must be curled at once: each rank enters the per-layer all-reduce only while running its OWN
# forward pass. Curl one and leave the other idle and the first rank blocks forever in the collective.
#
# Addresses that have bitten before:
#   * orchestration ssh -> LAN (lefty 192.168.1.249, righty 192.168.1.250)
#   * NCCL rendezvous -> lefty's RoCE address 192.168.192.1, reachable from righty but NOT from godzilla
#   * righty has no /mnt/apex, so its checkpoint is a local copy at /srv/models/glm53-checkpoint
#
# Usage: scripts/tp_pair.sh [NLAY] [NSTEPS]
set -u

LEFTY=rick@192.168.1.249
RIGHTY=rick@192.168.1.250
LEFTY_ROCE=192.168.192.1
NLAY="${1:-45}"
NSTEPS="${2:-2}"
HTTP0=18660
HTTP1=18661
TPPORT=18460
MODEL_L="/mnt/apex/rick/vybforge/glm53-checkpoint"
MODEL_R="/srv/models/glm53-checkpoint"
HCA="NCCL_IB_HCA=rocep1s0f0,roceP2p1s0f0"
L_VYB='$HOME/Projects/Vyb/build2/vyb'
L_MODS='--module-path native/glm53 --module-path native/tensor --module-path native/json --module-path native/kernels --module-path native/torchload --module-path $HOME/Projects/Vyb/stdlib/network --module-path $HOME/Projects/Vyb/bindings/cuda'
R_MODS='--module-path native/glm53 --module-path native/tensor --module-path native/json --module-path native/kernels --module-path native/torchload --module-path /tmp/nccltest/stdlib/network --module-path /tmp/nccltest/bindings/cuda'

if [ "${SKIP_STAGE:-0}" = "0" ]; then
echo "== staging engine tree on righty (modules + ptx; the model copy is separate)"
ssh -o ConnectTimeout=8 "$RIGHTY" 'mkdir -p /srv/models/vybforge/native'
for d in native/glm53 native/tensor native/json native/kernels native/torchload native/host native/build native/exl3 scripts; do
  rsync -a "$d" "rick@$RIGHTY:/srv/models/vybforge/$(dirname $d)/" || echo "   STAGE FAILED: $d"
done
ssh -o ConnectTimeout=8 "$RIGHTY" 'ls /srv/models/vybforge/native/glm53/glm_engine.vyb >/dev/null 2>&1 && echo "   engine staged OK" || echo "   ENGINE STILL MISSING"'
# righty runs ./vyb from its working dir
ssh -o ConnectTimeout=8 "$RIGHTY" 'mkdir -p /srv/models/vybforge && cp /tmp/nccltest/vyb /srv/models/vybforge/vyb && chmod +x /srv/models/vybforge/vyb' && echo "   binary staged (from righty's own aarch64 build)" || echo "   BINARY STAGE FAILED"
echo "   staged"
else
  echo "== staging skipped (SKIP_STAGE=1); verifying engine present"
  ssh -o ConnectTimeout=8 "$RIGHTY" 'ls /srv/models/vybforge/native/glm53/glm_engine.vyb >/dev/null 2>&1 && echo "   engine present" || echo "   ENGINE MISSING"'
fi

echo "== starting rank 0 on lefty (http $HTTP0, tp $TPPORT, experts [0,144))"
ssh -o ConnectTimeout=8 "$LEFTY" "cd /srv/models/vybforge && VYB_STDLIB=\$HOME/Projects/Vyb/stdlib VYBFORGE_GX_DIR=$MODEL_L VYBFORGE_SKIP_MTP=1 VYBFORGE_RANK=0 VYBFORGE_WORLD=2 VYBFORGE_TP_PORT=$TPPORT VYBFORGE_SERVE=1 VYBFORGE_SERVE_PORT=$HTTP0 VYBFORGE_NLAYERS=$NLAY VYBFORGE_CHAIN_OUT=native/out/pair0.txt $HCA nohup $L_VYB $L_MODS native/host/glm_serve.vyb > /tmp/pair0.log 2>&1 & echo started"

echo "== starting rank 1 on righty (http $HTTP1, tp $TPPORT, experts [144,288))"
ssh -o ConnectTimeout=8 "$RIGHTY" "cd /srv/models/vybforge && VYB_STDLIB=/tmp/nccltest/stdlib VYBFORGE_GX_DIR=$MODEL_R VYBFORGE_SKIP_MTP=1 VYBFORGE_RANK=1 VYBFORGE_WORLD=2 VYBFORGE_TP_PORT=$TPPORT VYBFORGE_TP_PEER=$LEFTY_ROCE VYBFORGE_SERVE=1 VYBFORGE_SERVE_PORT=$HTTP1 VYBFORGE_NLAYERS=$NLAY VYBFORGE_CHAIN_OUT=native/out/pair1.txt $HCA nohup ./vyb $R_MODS native/host/glm_serve.vyb > /tmp/pair1.log 2>&1 & echo started"

# wait for BOTH to be listening (grep -q, never grep -c, so the count never leaks a newline into a test)
for i in $(seq 1 120); do
  a=0; b=0
  ssh -o ConnectTimeout=5 "$LEFTY" 'grep -q "glm_serve listening" /tmp/pair0.log 2>/dev/null' && a=1
  ssh -o ConnectTimeout=5 "$RIGHTY" 'grep -q "glm_serve listening" /tmp/pair1.log 2>/dev/null' && b=1
  if [ "$a" = "1" ] && [ "$b" = "1" ]; then break; fi
  sleep 2
done
echo "== listeners: $(ssh -o ConnectTimeout=5 "$LEFTY" 'grep -m1 -E "TP_COMM|TP_LISTEN|listening" /tmp/pair0.log' 2>/dev/null)"
echo "   rank1: $(ssh -o ConnectTimeout=5 "$RIGHTY" 'grep -m1 -E "TP_COMM|TP_LISTEN|listening" /tmp/pair1.log' 2>/dev/null)"

echo "== curling BOTH ranks at once (each enters the collective during its own forward pass)"
ssh -o ConnectTimeout=8 "$LEFTY" "curl -s -m 900 -X POST http://127.0.0.1:$HTTP0/v1/chat/completions -H 'Content-Type: application/json' -d '{\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}],\"max_tokens\":$NSTEPS}' > /tmp/pair0_resp.json 2>&1; echo rank0_rc=\$?" &
sleep 1
ssh -o ConnectTimeout=8 "$RIGHTY" "curl -s -m 900 -X POST http://127.0.0.1:$HTTP1/v1/chat/completions -H 'Content-Type: application/json' -d '{\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}],\"max_tokens\":$NSTEPS}' > /tmp/pair1_resp.json 2>&1; echo rank1_rc=\$?" &
wait

echo "== results"
for H in "$LEFTY" "$RIGHTY"; do
  echo "--- $H"
  ssh -o ConnectTimeout=8 "$H" 'grep -E "TP_RANK|TP_COMM|TP_LISTEN|SEC STEP|Segmentation|TP_FAIL" /tmp/pair0.log /tmp/pair1.log 2>/dev/null | tail -6'
done
echo "== chain digests"
ssh -o ConnectTimeout=8 "$LEFTY" 'md5sum /srv/models/vybforge/native/out/pair0.txt 2>/dev/null'
ssh -o ConnectTimeout=8 "$RIGHTY" 'md5sum /srv/models/vybforge/native/out/pair1.txt 2>/dev/null'

echo "== cleanup"
for H in "$LEFTY" "$RIGHTY"; do
  ssh -o ConnectTimeout=5 "$H" 'for p in $(ps -eo pid,cmd | grep "[g]lm_serve.vyb" | awk "{print \$1}"); do kill -9 $p 2>/dev/null; done' || true
done
echo done
