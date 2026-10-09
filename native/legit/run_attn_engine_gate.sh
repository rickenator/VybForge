#!/usr/bin/env bash
# P4.9 Ridge's ATTENTION block on the ENGINE's own weight path — VybForge#10 phase 4, unit 10 step 5.
#
# What it proves, and why P4.8 is not enough. P4.8 (native/legit/run_attn_block_gate.sh) proved the
# block's ten stages against ggml on a SYNTHETIC fixture: it pins the READING of llama.cpp's
# attention (the joint q+gate split, the per-head RMS norms, the partial rope, the output gate). It
# says nothing about whether the engine can RUN that block — staging the model's own quantized
# weights (attn_q is a JOINT q+gate matrix, Q5_K, which no engine path had ever dequantised), the
# split kernel, the causal attention with 24 query heads over 4 kv heads, and `wo`.
#
# This runs THE ENGINE'S OWN DRIVER — native/host/model_driver.vyb in probe mode
# (VYB_ATTN_PROBE=<layer>) — which stages every operand from the live GGUF by name through its own
# dequant kernels and multiplies with layer.ptx's `gemm`, and compares its ten dumped stages against
# native/tools/attn_authority.c fed the SAME weights (dequantised by the independent `gguf` package).
#
# Teeth (all required to be distinguishable, or the gate fails):
#   * the staged attn_q/attn_output operands at the exact addresses gemm reads, against the model's
#     own dequantised tensors — a good block fed a wrong operand is invisible to stage comparison;
#   * the attention recomputed from the ENGINE's own q_rope/k_rope under the contiguous grouping
#     (h // (H/KVH)) must MATCH, and under round-robin (h % KVH) must MISS;
#   * `gated` with the sigmoid dropped, and with the gate dropped entirely — both must MISS.
# NaN fails (`not (r <= bar)`).
#
# SKIPs (never PASSes) without a Vyb toolchain, libggml, CUDA, the Ridge model, its inventory/tensor
# index, the `gguf` package, or the engine's rope table. A gate that proved nothing is a FAIL.
set -u
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$root"

step() { printf '%-46s %s\n' "$1" "$2"; }
echo "ATTN engine-path gate — $(date '+%F %T')"
echo

if ! . "$root/vybenv.sh" >/dev/null 2>&1 || [ ! -x "$VYB" ]; then
  step "Vyb toolchain" "SKIP (no compiler)"
  echo; echo "ATTN ENGINE GATE: SKIP (no toolchain — nothing to run)"; exit 0
fi
LLAMA="${VYBFORGE_LLAMA:-$HOME/Projects/llama.cpp}"
if [ ! -f "$LLAMA/ggml/include/ggml.h" ] || [ ! -e "$LLAMA/build/bin/libggml.so" ]; then
  step "libggml authority" "SKIP (no llama.cpp checkout + build at $LLAMA)"
  echo; echo "ATTN ENGINE GATE: SKIP (nothing to compare against)"; exit 0
fi
MODEL="${VYBFORGE_RIDGE_GGUF:-$HOME/Models/qwen38-27b-ridge/Qwen3.8-27B-Ridge-3.7bpw.gguf}"
if [ ! -f "$MODEL" ]; then
  step "model present" "SKIP (not on disk: $MODEL)"
  echo; echo "ATTN ENGINE GATE: SKIP (no model — nothing to stage)"; exit 0
fi
# The engine driver loads both of these unconditionally, and they are in the Makefile's KERNELS list
# (so `make -f native/Makefile verify` builds them). A missing one is a BUILD problem, not a skip.
for k in attn35 q5k; do
  if [ ! -f "$root/native/build/$k.ptx" ]; then
    step "kernel $k.ptx" "FAIL (missing — run: make -f native/Makefile attn-engine)"
    echo; echo "ATTN ENGINE GATE: FAIL"; exit 1
  fi
done
echo "toolchain: $VYB"
echo

py=""
for cand in "${ATTNE_PY:-}" "$root/.venv/bin/python" python3; do
  if [ -n "$cand" ] && env -u PYTHONPATH "$cand" -c 'import numpy' >/dev/null 2>&1; then py="$cand"; break; fi
done
if [ -z "$py" ]; then
  step "reference interpreter" "FAIL (no python with numpy)"
  echo; echo "ATTN ENGINE GATE: FAIL"; exit 1
fi

work="${VYBFORGE_ATTNE_OUTDIR:-$root/native/out/attn_engine}"
mkdir -p "$work"
log="$work/verify.log"

env -u PYTHONPATH "$py" native/tools/attn_engine_verify.py >"$log" 2>&1
rc=$?

grep -E '^ATTN_ENGINE_VERIFY' "$log" | sed 's/^/      /'

if grep -q "^ATTN_ENGINE_VERIFY_SKIP" "$log"; then
  step "engine attention block" "SKIP ($(grep -m1 '^ATTN_ENGINE_VERIFY_SKIP' "$log" | sed 's/^ATTN_ENGINE_VERIFY_SKIP //'))"
  echo; echo "ATTN ENGINE GATE: SKIP (nothing ran)"; exit 0
fi

if grep -q "^ATTN_ENGINE_VERIFY_DONE" "$log" && [ "$rc" = "0" ]; then
  summ="$(grep -m1 '^ATTN_ENGINE_VERIFY_SUMMARY' "$log" | sed 's/^ATTN_ENGINE_VERIFY_SUMMARY //')"
  nst="$(grep -cE '^ATTN_ENGINE_VERIFY [a-z_]+ +n= *[0-9]+ maxrel=.* ok' "$log")"
  ntooth="$(grep -cE 'reproduces the engine.s attn|gated == attn|must MISS' "$log")"
  step "engine attention block ($nst stages)" "PASS (worst $(grep -oE 'worst=[0-9.e+-]+' "$log" | tail -1 | sed 's/worst=//'), 3 teeth)"
  echo "      $summ"
  echo; echo "ATTN ENGINE GATE: PASS"
  exit 0
fi

step "engine attention block (10 stages)" "FAIL (see $log)"
echo; echo "ATTN ENGINE GATE: FAIL"
exit 1
