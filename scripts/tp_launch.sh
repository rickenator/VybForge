#!/usr/bin/env bash
# tp_launch.sh -- start the two-rank TP=2 launcher across lefty + righty and report both ranks.
#
# Addresses matter here and have bitten before:
#   * orchestration ssh goes to the LAN addresses (lefty 192.168.1.249, righty 192.168.1.250)
#   * righty's RoCE address 192.168.192.2 is reachable FROM LEFTY but NOT from godzilla, so the NCCL
#     rendezvous peer is lefty's RoCE address (192.168.192.1) and the control socket rides that rail.
#   * NCCL must be told which HCA to use or it picks the wrong pair and hangs.
#
# Usage:  scripts/tp_launch.sh [ITERS] [COUNT]
set -u

LEFTY=rick@192.168.1.249
RIGHTY=rick@192.168.1.250
LEFTY_ROCE=192.168.192.1
PORT="${PORT:-18420}"
ITERS="${1:-200}"
COUNT="${2:-4096}"
VYB_LEFTY='$HOME/Projects/Vyb/build2/vyb'
MODS_LEFTY='--module-path native/glm53 --module-path native/tensor --module-path native/json --module-path native/kernels --module-path native/torchload --module-path $HOME/Projects/Vyb/stdlib/network --module-path $HOME/Projects/Vyb/bindings/cuda'
HCA='NCCL_IB_HCA=rocep1s0f0,roceP2p1s0f0'

echo "== deploying"
scp -q native/nccl/tp_launch.vyb "$LEFTY:/srv/models/vybforge/native/nccl/" || exit 1
ssh -o ConnectTimeout=8 "$RIGHTY" 'mkdir -p /tmp/nccltest/native/nccl' || exit 1
scp -q native/nccl/tp_launch.vyb "$RIGHTY:/tmp/nccltest/native/nccl/" || exit 1

# clear any previous run by PID (never pkill -f: the pattern matches this script's own ssh command line)
for H in "$LEFTY" "$RIGHTY"; do
  ssh -o ConnectTimeout=8 "$H" 'for p in $(ps -eo pid,cmd | grep "tp_launch.vyb" | grep -v grep | awk "{print \$1}"); do kill -9 $p 2>/dev/null; done' || true
done

echo "== starting rank 0 on lefty (port $PORT, iters $ITERS, count $COUNT)"
ssh -o ConnectTimeout=8 "$LEFTY" "cd /srv/models/vybforge && VYB_STDLIB=\$HOME/Projects/Vyb/stdlib RANK=0 WORLD=2 PORT=$PORT COUNT=$COUNT ITERS=$ITERS $HCA nohup $VYB_LEFTY $MODS_LEFTY native/nccl/tp_launch.vyb > /tmp/tp0.log 2>&1 & echo started"

# wait for the listener rather than sleeping a fixed amount
for i in $(seq 1 40); do
  if ssh -o ConnectTimeout=5 "$LEFTY" 'grep -q "TPLAUNCH listening" /tmp/tp0.log 2>/dev/null'; then break; fi
  sleep 1
done
echo "== rank 0: $(ssh -o ConnectTimeout=5 "$LEFTY" 'grep -m1 "TPLAUNCH listening" /tmp/tp0.log 2>/dev/null || echo "no listener"')"

echo "== starting rank 1 on righty (peer $LEFTY_ROCE)"
ssh -o ConnectTimeout=8 "$RIGHTY" "cd /tmp/nccltest && VYB_STDLIB=/tmp/nccltest/stdlib RANK=1 WORLD=2 PEER_IP=$LEFTY_ROCE PORT=$PORT COUNT=$COUNT ITERS=$ITERS $HCA nohup ./vyb --module-path /tmp/nccltest/stdlib --module-path /tmp/nccltest/tensor --module-path /tmp/nccltest/bindings/cuda native/nccl/tp_launch.vyb > /tmp/tp1.log 2>&1 & echo started"

# both ranks run the same iteration count, so wait for the verdict lines
for i in $(seq 1 240); do
  a=$(ssh -o ConnectTimeout=5 "$LEFTY" 'grep -c "TPLAUNCH_VERDICT" /tmp/tp0.log 2>/dev/null || echo 0')
  b=$(ssh -o ConnectTimeout=5 "$RIGHTY" 'grep -c "TPLAUNCH_VERDICT" /tmp/tp1.log 2>/dev/null || echo 0')
  if [ "$a" -ge 1 ] && [ "$b" -ge 1 ]; then break; fi
  sleep 2
done

echo "================ RANK 0 (lefty) ================"
ssh -o ConnectTimeout=8 "$LEFTY" 'grep -E "TPLAUNCH|Segmentation|FAIL" /tmp/tp0.log | tail -8'
echo "================ RANK 1 (righty) ==============="
ssh -o ConnectTimeout=8 "$RIGHTY" 'grep -E "TPLAUNCH|Segmentation|FAIL" /tmp/tp1.log | tail -8'

echo "== cleanup"
for H in "$LEFTY" "$RIGHTY"; do
  ssh -o ConnectTimeout=5 "$H" 'for p in $(ps -eo pid,cmd | grep "tp_launch.vyb" | grep -v grep | awk "{print \$1}"); do kill -9 $p 2>/dev/null; done' || true
done
echo "done"
