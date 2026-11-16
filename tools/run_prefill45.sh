#!/bin/bash
# Full 45-layer GLM-5.3-Flash prefill — the acceptance run.
#
# This file is the INNER runner (it just execs vyb and redirects to the log).
# Launch it as a transient systemd USER unit so it survives this shell/session:
#
#   systemd-run --user --unit=prefill45 --collect -q -p MemoryMax=80G \
#       --working-directory=/home/rick/Projects/VybForge \
#       /home/rick/Projects/VybForge/tools/run_prefill45.sh
#
# Watch:   tail -f native/out/prefill45.log     (ends with "PREFILL_OK layers=45")
# Status:  systemctl --user status prefill45     /  journalctl --user -u prefill45
# Stop:    systemctl --user stop prefill45
#
# Timing (measured 2026-11-16): ~3.9 s per expert -> ~18.6 min per MoE layer ->
# 43 MoE layers (3..45) ~= 13.5 h, so `timeout` is 86400 s (24 h). A 43200 s cap
# was measured to truncate it at ~layer 40.
#
# Checkpoint: the LOCAL ext4 copy (/usr/export) rather than the NFS export
# (/mnt/apex is the same data over NFS from apex).
# Needs a quiet GPU: `docker stop hyperqwen-abliterated-single-1` first, or the
# crash-looping vLLM engine (15-22 GB) will OOM this run. Verify with
# `nvidia-smi --query-compute-apps=pid,process_name,used_memory` before starting.
cd "$(dirname "$0")/.." || exit 1
exec timeout 86400 \
  env VYBFORGE_GX_DIR=/usr/export/rick/vybforge/glm53-checkpoint \
      VYBFORGE_PREFILL_OUT=native/out/prefill_vyb_45.txt \
      VYB_STDLIB="$HOME/Projects/Vyb/stdlib" \
  "$HOME/Projects/Vyb/build-release/vyb" \
  --module-path native/json --module-path native/torchload --module-path native/glm53 \
  --module-path native/kernels --module-path native/tensor \
  --module-path "$HOME/Projects/Vyb/bindings/cuda" \
  --module-path "$HOME/Projects/Vyb/stdlib/collections" \
  native/host/glm_prefill.vyb > native/out/prefill45.log 2>&1
