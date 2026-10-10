#!/usr/bin/env bash
# P4.10 the FFN half of a Ridge block on the ENGINE's own weight path — VybForge#10 phase 4, W1.
#
# What it proves, and why the existing gates are not enough. `make prefill` already exercises the
# FFN arithmetic (pre-norm -> parallel gate/up -> SiLU(gate)*up -> down -> residual) on Qwen3-4B.
# What no engine path could do before W1 is STAGE Ridge's FFN weights: they are IQ2_S (160 tensors,
# the mid-stack, 4.25 GiB — the largest type in the file) and IQ3_S (32, the edge layers), and both
# carry their codebook as an ordinary FIFTH kernel argument, so they launch through `cuda_launch_n`
# (Vyb#476) rather than the four-slot `cuda_launch4i` every other dequant uses. Before W1
# `stage_one` had no case for types 21/22 and fell through to the F32 reader, so the weights were
# silently mis-staged. This runs THE ENGINE'S OWN DRIVER — native/host/model_driver.vyb in FFN-probe
# mode (VYB_FFN_PROBE=<layer>) — which stages ffn_gate/ffn_up/ffn_down + post_attention_norm by name
# from the live GGUF through its own kernels and multiplies with layer.ptx's `gemm`, and compares its
# seven dumped stages against a reference assembled from OUR numpy ports of the same two
# dequantizers (each verified element-wise against llama.cpp's own compiled dequantizer in S0.5/S0.8).
#
# Three type boundaries, because the FFN type is NOT a function of the block kind: blk.3 is all
# IQ3_S, blk.7 has an IQ3_S ffn_down with IQ2_S gate/up, blk.19 is all IQ2_S.
#
# Teeth (all required to be distinguishable, or the gate fails):
#   * the STAGED OPERANDS at the exact addresses gemm reads (B[k*N+n]) against the model's own
#     dequantised tensors — this is what catches the wrong in-dim (gate/up D, down FF);
#   * the residual mis-wirings must MISS: no residual, and the residual on the NORMED input;
#   * SiLU dropped (gate*up) must MISS.
# NaN fails (`not (r <= bar)`).
#
# SKIPs (never PASSes) without a Vyb toolchain, CUDA, the Ridge model, its inventory / engine tensor
# index / rope table, the llama.cpp source the iq3s table extraction reads, or a python with numpy.
# A gate that proved nothing is a FAIL.
set -u
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$root"

step() { printf '%-46s %s\n' "$1" "$2"; }
echo "FFN engine-path gate — $(date '+%F %T')"
echo

if ! . "$root/vybenv.sh" >/dev/null 2>&1 || [ ! -x "$VYB" ]; then
  step "Vyb toolchain" "SKIP (no compiler)"
  echo; echo "FFN ENGINE GATE: SKIP (no toolchain — nothing to run)"; exit 0
fi
MODEL="${VYBFORGE_RIDGE_GGUF:-$HOME/Models/qwen38-27b-ridge/Qwen3.8-27B-Ridge-3.7bpw.gguf}"
if [ ! -f "$MODEL" ]; then
  step "model present" "SKIP (not on disk: $MODEL)"
  echo; echo "FFN ENGINE GATE: SKIP (no model — nothing to stage)"; exit 0
fi
# The engine driver loads these unconditionally, and they are in the Makefile's KERNELS list (so
# `make -f native/Makefile verify` builds them). A missing one is a BUILD problem, not a skip.
for k in layer iq2s iq3s q4k q6k q8_0 qwen3 rope gdn q5k attn35; do
  if [ ! -f "$root/native/build/$k.ptx" ]; then
    step "kernel $k.ptx" "FAIL (missing — run: make -f native/Makefile ffn-engine)"
    echo; echo "FFN ENGINE GATE: FAIL"; exit 1
  fi
done
echo "toolchain: $VYB"
echo

py=""
for cand in "${FFNE_PY:-}" "$root/.venv/bin/python" python3; do
  if [ -n "$cand" ] && env -u PYTHONPATH "$cand" -c 'import numpy' >/dev/null 2>&1; then py="$cand"; break; fi
done
if [ -z "$py" ]; then
  step "reference interpreter" "FAIL (no python with numpy)"
  echo; echo "FFN ENGINE GATE: FAIL"; exit 1
fi

work="${VYBFORGE_FFNE_OUTDIR:-$root/native/out/ffn_engine}"
mkdir -p "$work"
log="$work/verify.log"

env -u PYTHONPATH "$py" native/tools/ffn_engine_verify.py >"$log" 2>&1
rc=$?

grep -E '^FFN_ENGINE_VERIFY' "$log" | sed 's/^/      /'

if grep -q "^FFN_ENGINE_VERIFY_SKIP" "$log"; then
  step "engine FFN block" "SKIP ($(grep -m1 '^FFN_ENGINE_VERIFY_SKIP' "$log" | sed 's/^FFN_ENGINE_VERIFY_SKIP //'))"
  echo; echo "FFN ENGINE GATE: SKIP (nothing ran)"; exit 0
fi

if grep -q "^FFN_ENGINE_VERIFY_DONE" "$log" && [ "$rc" = "0" ]; then
  nst="$(grep -cE '^FFN_ENGINE_VERIFY [a-z_]+ +n= *[0-9]+ maxrel=.* ok' "$log")"
  nlay="$(grep -cE '^FFN_ENGINE_VERIFY blk\.[0-9]+ D=' "$log")"
  worst="$(grep -oE 'worst=[0-9.e+-]+' "$log" | sed 's/worst=//' | sort -g | tail -1)"
  step "engine FFN block ($nlay layers, $nst stages)" "PASS (worst $worst, 3 teeth)"
  grep -oE '^FFN_ENGINE_VERIFY_SUMMARY.*' "$log" | sed 's/^/      /'
  echo; echo "FFN ENGINE GATE: PASS"
  exit 0
fi

step "engine FFN block" "FAIL (see $log)"
echo; echo "FFN ENGINE GATE: FAIL"
exit 1
