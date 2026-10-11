#!/usr/bin/env bash
# build.sh — gates for the Whisper slice (voice loop, native/whisper).
#
#   ./build.sh weights    # checkpoint inventory: Vyb vs an independent parse
#   ./build.sh gold       # (re)generate the reference inventory + staged tensors
#
# The checkpoint is DISCOVERED, never hardcoded: VYBFORGE_ST_FILE wins, else the
# HF cache snapshot of openai/whisper-tiny.en. An absent file is a SKIP, not a pass.
set -euo pipefail
here="$(cd "$(dirname "$0")" && pwd)"
cd "$here"

VYBHOME="${VYBHOME:-$HOME/Projects/Vyb}"
VYB="${VYB:-$VYBHOME/build/vyb}"
export VYB_STDLIB="${VYB_STDLIB:-$VYBHOME/stdlib}"
PY="${PY:-$HOME/miniconda/bin/python}"

snapshot="/usr/export/LLM/hf/hub/models--openai--whisper-tiny.en/snapshots/87c7102498dcde7456f24cfd30239ca606ed9063/model.safetensors"
ST="${VYBFORGE_ST_FILE:-$snapshot}"
gold="../../fixtures/whisper/gold_stt"   # native/whisper -> repo/fixtures/whisper

if [ ! -f "$ST" ]; then
    echo "SKIP: checkpoint not found at $ST (set VYBFORGE_ST_FILE)"
    exit 0
fi
export VYBFORGE_ST_FILE="$ST"
export VYBFORGE_ST_OUT="$here/out/weights_listing_vyb.txt"
mkdir -p out

case "${1:-weights}" in
    weights)
        echo "== checkpoint: $ST"
        # --module-path is repeatable; it does NOT take a colon-separated list
        "$VYB" weights_probe.vyb \
            --module-path . \
            --module-path "$VYBHOME/../VybForge/native/torchload" \
            --module-path "$VYBHOME/../VybForge/native/json" \
            --build weights_probe
        ./weights_probe
        if [ ! -f "$gold/weights_listing_ref.txt" ]; then
            echo "== generating reference inventory"
            ( cd ../../fixtures/whisper && "$PY" oracle_weights.py "$ST" )
        fi
        "$PY" check_weights.py "$VYBFORGE_ST_OUT" "$gold/weights_listing_ref.txt"
        ;;
    gold)
        ( cd ../../fixtures/whisper && "$PY" oracle_weights.py "$ST" )
        ;;
    *) echo "usage: $0 [weights|gold]" >&2; exit 2 ;;
esac
