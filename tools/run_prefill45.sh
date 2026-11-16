#!/bin/bash
# Full 45-layer GLM-5.3-Flash prefill (the acceptance run), long-timeout variant.
#
# Measured 2026-11-16: one MoE layer (288 experts, 327 exl3 deqs at ~0.9 s each)
# takes ~14 min, so all 43 MoE layers (3..45) is a ~10-hour job -- the 3000 s
# `timeout` in the old handoff command kills it mid-layer. Run this detached:
#
#   setsid nohup tools/run_prefill45.sh >/dev/null 2>&1 &
#
# Then watch: tail -f native/out/prefill45.log   (PREFILL_OK layers=45 at the end)
#
# Checkpoint: use the LOCAL ext4 copy (/usr/export) rather than the NFS export
# (/mnt/apex is the same data over NFS from apex).
# Needs a quiet GPU: `docker stop hyperqwen-abliterated-single-1` first, or the
# crash-looping vLLM engine (15-22 GB) will OOM this run.
cd "$(dirname "$0")/.." || exit 1
exec systemd-run --user --scope -q -p MemoryMax=80G timeout 43200 \
  env VYBFORGE_GX_DIR=/usr/export/rick/vybforge/glm53-checkpoint \
      VYBFORGE_PREFILL_OUT=native/out/prefill_vyb_45.txt \
      VYB_STDLIB="$HOME/Projects/Vyb/stdlib" \
  "$HOME/Projects/Vyb/build-release/vyb" \
  --module-path native/json --module-path native/torchload --module-path native/glm53 \
  --module-path native/kernels --module-path native/tensor \
  --module-path "$HOME/Projects/Vyb/bindings/cuda" \
  --module-path "$HOME/Projects/Vyb/stdlib/collections" \
  native/host/glm_prefill.vyb > native/out/prefill45.log 2>&1
