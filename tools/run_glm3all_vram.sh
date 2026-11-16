#!/bin/bash
# Run the glm3all MoE gate while sampling per-process GPU memory (pid + NAME),
# so a device jump can be attributed to OUR process or to a co-tenant.
cd "$(dirname "$0")/.." || exit 1
LOG=${1:-native/out/deqmem_run2.log}
SAM=${2:-native/out/deqmem_vram_samples.log}
: > "$SAM"
(
  while true; do
    ts=$(date +%s.%N)
    line=$(nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader | tr '\n' ';')
    echo "$ts $line" >> "$SAM"
    sleep 1
  done
) &
SAMPLER=$!
systemd-run --user --scope -q -p MemoryMax=80G timeout 3000 \
  env VYB_STDLIB=$HOME/Projects/Vyb/stdlib \
  $HOME/Projects/Vyb/build-release/vyb \
  --module-path native/json --module-path native/torchload --module-path native/glm53 \
  --module-path native/kernels --module-path native/tensor \
  --module-path $HOME/Projects/Vyb/bindings/cuda \
  --module-path $HOME/Projects/Vyb/stdlib/collections \
  native/host/glm_l3_moe_all.vyb > "$LOG" 2>&1 &
GATE=$!
sleep 1
echo "gate_wrapper_pid=$GATE vyb_pid=$(pgrep -f 'glm_l3_moe_all.vyb' | tr '\n' ',')" | tee -a "$SAM"
wait $GATE
RC=$?
kill $SAMPLER 2>/dev/null
wait $SAMPLER 2>/dev/null
echo "gate_exit=$RC"
tail -3 "$LOG"
