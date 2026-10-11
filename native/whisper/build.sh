#!/usr/bin/env bash
# build.sh — gates for the Whisper slice (voice loop, native/whisper).
#
#   ./build.sh weights    # checkpoint inventory: Vyb vs an independent parse
#   ./build.sh stages     # the whole encoder on the GPU, stage-gated
#   ./build.sh decoder    # the decoder stages (same driver: one forward pass)
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
    ( cd "$ROOT" && "$VYB" native/whisper/host/forward_driver.vyb \
          --module-path native/mel \
          --module-path native/torchload \
          --module-path native/json \
          --module-path native/tensor \
          --module-path "$VYBHOME/bindings/cuda" \
          > native/whisper/out/vyb_stages.txt 2>&1 ) || true
    tail -1 "$here/out/vyb_stages.txt"
    # The encoder stage list is explicit: the checker's no-argument default is
    # "everything it knows about", which now includes decoder stages this run
    # does not emit.
    "$PY" check_stages.py out/vyb_stages.txt \
        conv1 conv2 e_pos e_in e_l0_ln1 e_l0_attn e_l0_res1 e_l0_ln2 e_l0_mlp \
        e_layer0 e_layer1 e_layer2 e_layer3 e_final
}

decoder() {
    kernels
    # One driver runs the whole forward pass (the decoder needs the encoder's
    # output and Vyb tensors cannot travel through a file), so the decoder gate
    # is the same program with a different stage list.
    echo "== decoder stages (from the repo root, same driver as the encoder)"
    ( cd "$ROOT" && "$VYB" native/whisper/host/forward_driver.vyb \
          --module-path native/mel \
          --module-path native/torchload \
          --module-path native/json \
          --module-path native/tensor \
          --module-path "$VYBHOME/bindings/cuda" \
          > native/whisper/out/vyb_dec.txt 2>&1 ) || true
    tail -1 "$here/out/vyb_dec.txt"
    # GROW THIS LIST as decoder stages land. A stage the driver does not emit
    # yet is not a failure — a gate that is permanently red for work that has not
    # been done teaches nothing (and hides the failure that matters).
    "$PY" check_stages.py out/vyb_dec.txt \
        d_embed d_l0_ln1 d_l0_selfattn d_l0_res1 d_l0_crossattn d_l0_mlp d_step0_logits
    # ACCEPTANCE: the first generated token. The driver prints "ARGMAX_ID" and the
    # id on the next line; the reference text starts " And so my fellow Americans",
    # and 843 IS " And" in this tokenizer.
    ID=$(grep -A1 "^ARGMAX_ID$" "$here/out/vyb_dec.txt" | tail -1 | tr -d ' ')
    if [ "$ID" = "843" ]; then
        echo "first-token gate: OK (843 = ' And')"
    else
        echo "first-token gate: FAIL (got '$ID', want 843)"; exit 1
    fi
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
    decoder)
        [ -f "$ST" ] || { echo "SKIP: checkpoint not found at $ST (set VYBFORGE_ST_FILE)"; exit 0; }
        export VYBFORGE_ST_FILE="$ST"
        decoder ;;
    kernels) kernels ;;
    all)
        [ -f "$ST" ] || { echo "SKIP: checkpoint not found at $ST (set VYBFORGE_ST_FILE)"; exit 0; }
        export VYBFORGE_ST_FILE="$ST" VYBFORGE_ST_OUT="$here/out/weights_listing_vyb.txt"
        weights; stages ;;
    gold)
        ( cd "$ROOT/fixtures/whisper" && "$PY" oracle_weights.py "$ST" && "$PY" oracle_stt.py ) ;;
    *) echo "usage: $0 [weights|stages|kernels|all|gold]" >&2; exit 2 ;;
esac
