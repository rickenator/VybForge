#!/usr/bin/env bash
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
. "$root/vybenv.sh" || exit 1   # VYBHOME / VYB / VYB_STDLIB (VybForge#15, rickenator/Vyb#424)

mkdir -p "$root/bin" "$root/data"
"$VYB" "$root/training/generate_dataset.vyb" --build "$root/bin/vybos-training-data" -O2
# Native Vyb prints main()'s integer return after stdout. Keep only JSONL rows.
"$root/bin/vybos-training-data" | sed -n '/^{/p' >"$root/data/vybos-configurator-all.jsonl"
# Split into train/eval — Vyb-native since Phase 2 (P2.5a); the inline interpreter
# heredoc it replaces is gone. Byte-parity with the committed splits is asserted by
# native/legit/run_split_gate.sh.
VYBFORGE_SPLIT_SRC="$root/data/vybos-configurator-all.jsonl" \
VYBFORGE_SPLIT_TRAIN="$root/data/vybos-configurator-train.jsonl" \
VYBFORGE_SPLIT_EVAL="$root/data/vybos-configurator-eval.jsonl" \
    "$VYB" "$root/training/split_dataset.vyb" --module-path "$root/native/json"
