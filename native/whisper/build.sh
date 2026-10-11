#!/usr/bin/env bash
# build.sh — gates for the Whisper slice (voice loop, native/whisper).
#
#   ./build.sh weights    # checkpoint inventory: Vyb vs an independent parse
#   ./build.sh stages     # encoder front half on the GPU: conv1, conv2, +positional
#   ./build.sh kernels    # recompile the Whisper PTX kernels only
#   ./build.sh all        # weights + stages + kernels
#   ./build.sh gold       # (re)generate the reference artifacts
#
# The checkpoint is DISCOVERED, never hardcoded: VYBFORGE_ST_FILE wins, else the
# HF cache snapshot of openai/whisper-tiny.en. An absent file is a SKIP, not a pass.
#
# WHISPER_STAGES=<a,b> restricts the stage driver (default: every stage).
set -euo pipefail
here="$(cd "$(dirname "$0")" && pwd)"
cd "$here"

VYBHOME="${VYBHOME:-$HOME/Projects/Vyb}"
VYB="${VYB:-$VYBHOME/build/vyb}"
export VYB_STDLIB="${VYB_STDLIB:-$VYBHOME/stdlib}"
PY="${PY:-$HOME/miniconda/bin/python}"
ROOT="$here/../.."

snapshot="/usr/export/LLM/hf/hub/models--openai--whisper-tiny.en/snapshots/87c7102498dcde7456f24cfd30239ca606ed9063/model.safetensors"
ST="${VYBFORGE_ST_FILE:-$snapshot}"
gold="../../fixtures/whisper/gold_stt"   # native/whisper -> repo/fixtures/whisper

mkdir -p out build

kernels() {
    echo "== whisper_enc.ptx"
    "$VYB" --kernel kernels/whisper_enc.vyb --ptx build/whisper_enc.ptx \
        --module-path "$ROOT/native/kernels"
}

stages() {
    kernels
    if [ ! -f "$ROOT/native/build/gemm.ptx" ]; then
        echo "== building the shared gemm kernel (needed by the stage driver)"
        ( cd "$ROOT/native/build" && "$VYB" --kernel ../kernels/gemm.vyb --ptx gemm.ptx \
              --module-path "$ROOT/native/kernels" )
    fi
    echo "== encoder stages (from the repo root; module paths and PTX are repo-relative)"
    ( cd "$ROOT" && "$VYB" native/whisper/host/encoder_driver.vyb \
          --module-path native/mel \
          --module-path native/torchload \
          --module-path native/json \
          --module-path native/tensor \
          --module-path "$VYBHOME/bindings/cuda" \
          > native/whisper/out/vyb_stages.txt 2>&1 ) || true
    tail -1 "$here/out/vyb_stages.txt"
    "$PY" check_stages.py out/vyb_stages.txt
}

weights() {
    echo "== checkpoint: $ST"
    # --module-path is repeatable; it does NOT take a colon-separated list
    "$VYB" weights_probe.vyb \
        --module-path . \
        --module-path "$ROOT/native/torchload" \
        --module-path "$ROOT/native/json" \
        --build weights_probe
    ./weights_probe
    if [ ! -f "$gold/weights_listing_ref.txt" ]; then
        echo "== generating reference inventory"
        ( cd "$ROOT/fixtures/whisper" && "$PY" oracle_weights.py "$ST" )
    fi
    "$PY" check_weights.py "$VYBFORGE_ST_OUT" "$gold/weights_listing_ref.txt"
}

case "${1:-all}" in
    weights)
        [ -f "$ST" ] || { echo "SKIP: checkpoint not found at $ST (set VYBFORGE_ST_FILE)"; exit 0; }
        export VYBFORGE_ST_FILE="$ST" VYBFORGE_ST_OUT="$here/out/weights_listing_vyb.txt"
        weights ;;
    stages)
        [ -f "$ST" ] || { echo "SKIP: checkpoint not found at $ST (set VYBFORGE_ST_FILE)"; exit 0; }
        export VYBFORGE_ST_FILE="$ST"
        stages ;;
    kernels) kernels ;;
    all)
        [ -f "$ST" ] || { echo "SKIP: checkpoint not found at $ST (set VYBFORGE_ST_FILE)"; exit 0; }
        export VYBFORGE_ST_FILE="$ST" VYBFORGE_ST_OUT="$here/out/weights_listing_vyb.txt"
        weights; stages ;;
    gold)
        ( cd "$ROOT/fixtures/whisper" && "$PY" oracle_weights.py "$ST" && "$PY" oracle_stt.py ) ;;
    *) echo "usage: $0 [weights|stages|kernels|all|gold]" >&2; exit 2 ;;
esac
