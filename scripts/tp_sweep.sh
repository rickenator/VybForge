#!/usr/bin/env bash
# tp_sweep.sh -- run the two-rank launcher on lefty+righty over a range of payload sizes.
#
# Purpose: decide whether the per-layer all-reduce cost is LATENCY-bound (flat vs payload) or
# BANDWIDTH-bound (scales with payload). That decides how TP=2 can ever meet a ~20 ms/token bar:
# a latency floor must be OVERLAPPED with compute, while a bandwidth term can be shrunk.
#
# Designed to be run ON LEFTY (it can reach righty on both the LAN and the RoCE rail). Run it there:
#   ssh rick@192.168.1.249 'bash /srv/models/vybforge/scripts/tp_sweep.sh'
#
# Left alone: it kills only its own tp_launch processes, by PID.
set -u

RIGHTY=192.168.1.250
PEER=192.168.192.1
HCA="NCCL_IB_HCA=rocep1s0f0,roceP2p1s0f0"
ITERS="${ITERS:-100}"
COUNTS="${COUNTS:-512 1024 2048 4096 8192 16384}"
LEFTY_VYB="$HOME/Projects/Vyb/build2/vyb"
LEFTY_MODS="--module-path native/glm53 --module-path native/tensor --module-path native/json --module-path native/kernels --module-path native/torchload --module-path $HOME/Projects/Vyb/stdlib/network --module-path $HOME/Projects/Vyb/bindings/cuda"
RIGHTY_MODS="--module-path /tmp/nccltest/stdlib --module-path /tmp/nccltest/tensor --module-path /tmp/nccltest/bindings/cuda"

kill_mine() {
  for p in $(ps -eo pid,cmd | grep "[t]p_launch.vyb" | awk '{print $1}'); do kill -9 "$p" 2>/dev/null; done
}

port=18440
for C in $COUNTS; do
  port=$((port + 1))
  kill_mine
  cd /srv/models/vybforge || exit 1
  VYB_STDLIB="$HOME/Projects/Vyb/stdlib" RANK=0 WORLD=2 PORT=$port COUNT=$C ITERS=$ITERS \
    NCCL_IB_HCA=rocep1s0f0,roceP2p1s0f0 \
    nohup "$LEFTY_VYB" $LEFTY_MODS native/nccl/tp_launch.vyb > "/tmp/tp0_$C.log" 2>&1 &
  # wait for the listener instead of guessing
  for w in $(seq 1 30); do
    grep -q "TPLAUNCH listening" "/tmp/tp0_$C.log" 2>/dev/null && break
    sleep 1
  done

  ssh -o ConnectTimeout=8 "rick@$RIGHTY" \
    "cd /tmp/nccltest && VYB_STDLIB=/tmp/nccltest/stdlib RANK=1 WORLD=2 PEER_IP=$PEER PORT=$port COUNT=$C ITERS=$ITERS $HCA ./vyb $RIGHTY_MODS native/nccl/tp_launch.vyb 2>&1 | grep -E 'TPLAUNCH_LAT|TPLAUNCH_VERDICT' | tr '\n' ' '" \
    > "/tmp/tp1_$C.txt" 2>&1

  sleep 1
  r1=$(cat "/tmp/tp1_$C.txt")
  r0=$(grep -E 'TPLAUNCH_LAT|TPLAUNCH_VERDICT' "/tmp/tp0_$C.log" 2>/dev/null | tr '\n' ' ')
  echo "count=$C"
  echo "  r0: $r0"
  echo "  r1: $r1"
done
kill_mine
echo "sweep done"
