#!/usr/bin/env bash
# run_phase2_battery.sh — every Phase-2 acceptance gate in one command.
#
# Phase 1 (Python cleanup) is closed; its battery stays the regression gate:
#   native/legit/run_phase1_battery.sh
# This one covers the torch pipeline -> Vyb work, one line per step, and grows as
# steps land (doc/P2-TRAINER-PLAN.md has the plan and the gate for each step).
#
# Same toolchain caveat as the Phase-1 battery: build/vyb is rebuilt mid-session in
# the Vyb checkout, and a half-finished rebuild makes Vyb programs fail in ways that
# read as repo bugs. This prints the toolchain HEAD and binary mtime up front.
#
# Usage: ./native/legit/run_phase2_battery.sh
set -uo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
. "$root/vybenv.sh" || exit 1

fail=0
step() { printf '%-46s %s\n' "$1" "$2"; }

cd "$root"
echo "Phase 2 battery — $(date '+%F %T')"
echo "toolchain: $VYB  ($(git -C "$(dirname "$(dirname "$VYB")")" log --oneline -1 2>/dev/null || echo 'not a git checkout'))"
echo "binary mtime: $(date -r "$VYB" '+%F %T')"
echo

# P2.5a — dataset split in Vyb (replaces the inline heredoc in generate-data.sh)
out="$(./native/legit/run_split_gate.sh 2>&1)"
if echo "$out" | tail -1 | grep -q "PASS"; then
  step "P2.5a split_dataset.vyb" "$(echo "$out" | grep 'port parity' | sed 's/  */ /g')"
else
  step "P2.5a split_dataset.vyb" "FAIL"; echo "$out" | tail -8 | sed 's/^/      /'; fail=1
fi

# P2.1a — gen_kv_train.py -> gen_kv_train.vyb (driver parameterization)
out="$(./native/legit/run_kvgen_gate.sh 2>&1)"
if echo "$out" | tail -1 | grep -q "PASS"; then
  step "P2.1a gen_kv_train.vyb" "identity (93 9) + frozen baseline (513 429)"
else
  step "P2.1a gen_kv_train.vyb" "FAIL"; echo "$out" | tail -8 | sed 's/^/      /'; fail=1
fi

# P2.1b — build_fullmanifest.py -> build_fullmanifest.vyb (+ pre-tokenizer boundaries)
out="$(./native/legit/run_fullmanifest_gate.sh 2>&1)"
if echo "$out" | tail -1 | grep -q "PASS"; then
  step "P2.1b build_fullmanifest.vyb" "manifest parity (425 tokens) + pretok boundaries"
else
  step "P2.1b build_fullmanifest.vyb" "FAIL"; echo "$out" | tail -8 | sed 's/^/      /'; fail=1
fi

# P2.1c — _build_kv.py -> _build_kv.vyb (driver assembler)
out="$(./native/legit/run_kvbuild_gate.sh 2>&1)"
if echo "$out" | tail -1 | grep -q "PASS"; then
  step "P2.1c _build_kv.vyb" "driver parity (1249 lines) + front-end"
else
  step "P2.1c _build_kv.vyb" "FAIL"; echo "$out" | tail -8 | sed 's/^/      /'; fail=1
fi

# P2.2a — render_chat.vyb (corpus -> chat-templated text, Qwen3 template from GGUF metadata)
out="$(./native/legit/run_chat_render_gate.sh 2>&1)"
if echo "$out" | tail -1 | grep -q "PASS"; then
  step "P2.2a render_chat.vyb" "720-record render byte-identical to the GGUF template"
else
  step "P2.2a render_chat.vyb" "FAIL"; echo "$out" | tail -8 | sed 's/^/      /'; fail=1
fi

# ── S0.3 — native loaders (safetensors + torch .bin) ────────────────────────────────
# Gates: native/legit/run_safetensors_gate.sh, native/legit/run_torchbin_gate.sh
# (doc/SPIKINGBRAIN.md S0.3). Each gate checks the Vyb reader against an independent
# parse, a third-party library and structural invariants, on real checkpoint files.
#
# These need a checkpoint on disk, and its path differs per machine — so discover it
# rather than hardcode one (VybForge#15), and report an absent file as "skipped", never
# as PASS: a green battery has to mean the evidence actually ran.
#   VYBFORGE_ST_GATE_FILE=/path/to/model.safetensors   extra safetensors file to check
#   VYBFORGE_SB_DIR=/path/to/torch/checkpoint/dir      torch .bin shard directory
#   VYBFORGE_TB_SHARDS=all|<n>                         shard sweep width (default all)
#
# Why the torch side sweeps every shard rather than one: shards differ in what they
# exercise. A single-tensor shard uses BINPUT/SETITEM; a 41-tensor shard pushes the
# pickle memo past 255 and switches to LONG_BINPUT/LONG_BINGET and SETITEMS. Running one
# shard would have shown a green tick over both of those bugs.
st_cands=""
for f in "${VYBFORGE_ST_GATE_FILE:-}" "$root/artifacts/vybos-configurator-lora/adapter_model.safetensors"; do
  if [ -n "$f" ] && [ -f "$f" ]; then st_cands="$st_cands $f"; fi
done
if [ -z "$st_cands" ]; then
  step "S0.3 safetensors loader" "skipped (no .safetensors checkpoint found)"
else
  st_bad=0; st_n=0; st_detail=""
  for f in $st_cands; do
    out="$(./native/legit/run_safetensors_gate.sh "$f" 2>&1)"; st_n=$((st_n + 1))
    if echo "$out" | grep -q "SAFETENSORS GATE: PASS"; then
      st_detail="$st_detail $(basename "$f"):$(echo "$out" | grep -o 'tensors=[0-9]*' | head -1 | cut -d= -f2)"
    else
      st_bad=$((st_bad + 1)); echo "      $(basename "$f"):"; echo "$out" | tail -8 | sed 's/^/      /'
    fi
  done
  if [ $st_bad -eq 0 ]; then step "S0.3 safetensors loader" "PASS ($st_n file(s))$st_detail"
  else step "S0.3 safetensors loader" "FAIL ($st_bad/$st_n files)"; fail=1; fi
fi

sbdir=""
for d in "${VYBFORGE_SB_DIR:-}" "$HOME/Models/spikingbrain-v1-7b-base"; do
  if [ -n "$d" ] && [ -d "$d" ]; then sbdir="$d"; break; fi
done
shards=""
if [ -n "$sbdir" ]; then
  for s in "$sbdir"/pytorch_model-*.bin; do [ -f "$s" ] && shards="$shards $s"; done
fi
if [ -z "$shards" ]; then
  step "S0.3 torch .bin loader" "skipped (no torch .bin checkpoint found)"
else
  if [ "${VYBFORGE_TB_SHARDS:-all}" != "all" ]; then
    shards="$(echo $shards | tr ' ' '\n' | head -n "${VYBFORGE_TB_SHARDS}" | tr '\n' ' ')"
  fi
  tb_bad=0; tb_n=0; tb_t=0; tb_torch=0
  for f in $shards; do
    out="$(./native/legit/run_torchbin_gate.sh "$f" 2>&1)"; tb_n=$((tb_n + 1))
    if echo "$out" | grep -q "TORCHBIN GATE: PASS"; then
      tb_t=$((tb_t + $(echo "$out" | grep -o 'tensors=[0-9]*' | tail -1 | cut -d= -f2 | head -1)))
      echo "$out" | grep -q "TB_TORCHCHECK_OK" && tb_torch=$((tb_torch + 1))
    else
      tb_bad=$((tb_bad + 1)); echo "      $(basename "$f"):"
      echo "$out" | tail -6 | sed 's/^/      /'
    fi
  done
  case "$tb_torch" in
    "$tb_n") authority="torch authority ok" ;;
    0)       authority="torch authority SKIPPED (no interpreter with torch)" ;;
    *)       authority="torch authority partial ($tb_torch/$tb_n)" ;;
  esac
  if [ $tb_bad -eq 0 ]; then step "S0.3 torch .bin loader" "PASS ($tb_n/$tb_n shards, $tb_t tensors, $authority)"
  else step "S0.3 torch .bin loader" "FAIL ($tb_bad/$tb_n shards)"; fail=1; fi
fi

# ── S0.1 — dtypes (f16/bf16 storage, fp32 accumulation) ─────────────────────────────
# Gate: native/legit/run_dtype_gate.sh (doc/SPIKINGBRAIN.md S0.1). Eight conversions, both
# directions, compared on the GPU against numpy/torch: widening byte-exact, and both f16
# and bf16 narrowing at 0 value differences (bf16 needed the Vyb#441 rounding fix).
out="$(./native/legit/run_dtype_gate.sh 2>&1)"
if echo "$out" | grep -q "S0\.1 DTYPE GATE: PASS"; then
  step "S0.1 dtype gate" "PASS ($(echo "$out" | grep -cE '^S0\.1 [a-z_0-9]+ +PASS') conversions vs numpy/torch)"
else
  step "S0.1 dtype gate" "FAIL"; echo "$out" | tail -10 | sed 's/^/      /'; fail=1
fi

# ── S0.2a — config contract (the model's own dimensions, two sources) ───────────────
# Gate: native/legit/run_config_gate.sh (doc/SPIKINGBRAIN.md S0.2a). One ModelConfig read from
# a GGUF's metadata OR an HF config.json, every field checked against llama.cpp's gguf-dump and
# the python gguf package, plus transformers.AutoConfig on a SECOND real model. Refuses a
# truncated GGUF instead of inventing dimensions.
out="$(./native/legit/run_config_gate.sh 2>&1)"
if echo "$out" | grep -q "S0\.2a CONFIG GATE: PASS"; then
  step "S0.2a config contract" "PASS ($(echo "$out" | grep -o 'MCGATE_FIELDS_OK [0-9]*' | tr '\n' ' ' | sed 's/  */ /g'))"
else
  step "S0.2a config contract" "FAIL"; echo "$out" | tail -10 | sed 's/^/      /'; fail=1
fi

# ── S0.2e — capability / layer descriptor (VybForge#10 phase 2) ──────────────────────
# Gate: native/legit/run_caps_gate.sh. model_caps.vyb decides whether this build can RUN a model
# and refuses with a NAMED reason when it cannot; the gate checks that decision against the dense
# model (SUPPORTED, type counts cross-checked against an independent Python parser), the Ridge
# target (UNSUPPORTED with exactly five named reasons), the vision tower, and a truncated file.
# The Ridge/mmproj cases SKIP when those files are not on disk, and the gate FAILS if it proved
# nothing at all — a gate that skips everything has verified nothing.
out="$(./native/legit/run_caps_gate.sh 2>&1)"
if echo "$out" | grep -q "S0\.2e CAPABILITY GATE: PASS"; then
  step "S0.2e capability descriptor" "PASS ($(echo "$out" | grep -o 'PASS ([0-9]* cases[^)]*)' | tail -1 | sed 's/^PASS (//; s/)$//'))"
else
  step "S0.2e capability descriptor" "FAIL"; echo "$out" | tail -10 | sed 's/^/      /'; fail=1
fi

# ── S0.5 — Q8_0 dequant on real Ridge tensors (VybForge#10 phase 3) ─────────────────
# Gate: native/legit/run_q8_0_gate.sh. The first of the quant types the descriptor's refusal
# list named, now implemented and checked on real tensors from the 12 GiB GGUF — with the
# numpy reference cross-checked against the INDEPENDENT python gguf dequantizer, so the
# comparison is not our code agreeing with our code. SKIPs without the model, and the gate
# FAILS if it proved nothing.
out="$(./native/legit/run_q8_0_gate.sh 2>&1)"
if echo "$out" | grep -q "Q8_0 GATE: PASS"; then
  step "S0.5 Q8_0 dequant (Ridge)" "$(echo "$out" | grep -o 'Q8_0 GATE: PASS.*' | head -1)"
else
  step "S0.5 Q8_0 dequant (Ridge)" "FAIL"; echo "$out" | tail -8 | sed 's/^/      /'; fail=1
fi

# ── S0.6 — IQ2_S dequant on real Ridge tensors (VybForge#10 phase 3) ────────────────
# Gate: native/legit/run_iq2_s_gate.sh. The largest type in the file (160 tensors, 4.25 GiB of
# mid-stack FFN). The gguf package implements Q8_0 but NOT IQ2_S, so the authority here is
# llama.cpp's own dequantize_row_iq2_s, copied verbatim out of the local checkout, compiled
# as-is, and required to agree with our numpy port on whole 4096-element slices before either is
# used to judge the GPU kernel. SKIPs without the model; FAILS if it proved nothing.
out="$(./native/legit/run_iq2_s_gate.sh 2>&1)"
if echo "$out" | grep -q "IQ2_S GATE: PASS"; then
  step "S0.6 IQ2_S dequant (Ridge)" "$(echo "$out" | grep -o 'IQ2_S GATE: PASS.*' | head -1)"
else
  step "S0.6 IQ2_S dequant (Ridge)" "FAIL"; echo "$out" | tail -8 | sed 's/^/      /'; fail=1
fi

# ── S0.7 — Q5_K dequant on real Ridge tensors (VybForge#10 phase 3) ────────────────
# Gate: native/legit/run_q5k_gate.sh — the full-attention layers' q/k/v (51 tensors, 0.80 GiB).
# The 6-bit scale/min unpacking in scales[12] is where a bit-level mistake would hide, so the
# numpy port is checked against llama.cpp's own compiled dequant on whole slices before it is
# used to judge the GPU kernel. SKIPs without the model; FAILS if it proved nothing.
out="$(./native/legit/run_q5k_gate.sh 2>&1)"
if echo "$out" | grep -q "Q5_K GATE: PASS"; then
  step "S0.7 Q5_K dequant (Ridge)" "$(echo "$out" | grep -o 'Q5_K GATE: PASS.*' | head -1)"
else
  step "S0.7 Q5_K dequant (Ridge)" "FAIL"; echo "$out" | tail -8 | sed 's/^/      /'; fail=1
fi

# ── S0.8 — IQ3_S dequant on real Ridge tensors (VybForge#10 phase 3) ───────────────
# Gate: native/legit/run_iq3_s_gate.sh — the edge layers' FFN type (32 tensors, 1.14 GiB), the
# last missing text type. Grid-based like IQ2_S; the numpy port is checked against llama.cpp's own
# compiled dequant on whole slices before it is used to judge the GPU kernel.
out="$(./native/legit/run_iq3_s_gate.sh 2>&1)"
if echo "$out" | grep -q "IQ3_S GATE: PASS"; then
  step "S0.8 IQ3_S dequant (Ridge)" "$(echo "$out" | grep -o 'IQ3_S GATE: PASS.*' | head -1)"
else
  step "S0.8 IQ3_S dequant (Ridge)" "FAIL"; echo "$out" | tail -8 | sed 's/^/      /'; fail=1
fi

# ── S0.9 — BF16 dequant on real vision-tower tensors (VybForge#10 phase 3) ─────────
# Gate: native/legit/run_bf16_gate.sh. BF16 is the mmproj's weight type (110 tensors, 0.85 GiB) —
# not a block quant, so the kernel is a conversion and the independent authority is the python gguf
# package (which does implement BF16). SKIPs when the mmproj is absent.
out="$(./native/legit/run_bf16_gate.sh 2>&1)"
if echo "$out" | grep -q "BF16 GATE: PASS"; then
  step "S0.9 BF16 dequant (vision)" "$(echo "$out" | grep -o 'BF16 GATE: PASS.*' | head -1)"
else
  step "S0.9 BF16 dequant (vision)" "FAIL"; echo "$out" | tail -8 | sed 's/^/      /'; fail=1
fi

# ── S0.2b — tensor core (shape/strides/dtype/broadcast) ─────────────────────────────
# Gate: native/legit/run_tensor_gate.sh (doc/SPIKINGBRAIN.md S0.2b). numpy IS the definition of
# this behaviour, so the table's expectations come from numpy (array strides, ravel_multi_index,
# C_CONTIGUOUS on real views, broadcast_shapes/broadcast_to). Includes refusal and
# non-contiguous cases, and proves its checker can fail on a perturbed expectation.
out="$(./native/legit/run_tensor_gate.sh 2>&1)"
if echo "$out" | grep -q "S0.2b TENSOR GATE: PASS"; then
  step "S0.2b tensor core" "PASS ($(echo "$out" | grep -o 'TSGATE_CASES_OK.*' | head -1))"
else
  step "S0.2b tensor core" "FAIL"; echo "$out" | tail -10 | sed 's/^/      /'; fail=1
fi

# ── S0.2c — the prefill driver re-expressed on both ─────────────────────────────────
# Gate: `make prefill` (doc/SPIKINGBRAIN.md S0.2c). model_driver.vyb now reads its dimensions
# from the model's own config and its buffer sizes from the tensor core, so this target IS the
# generality proof: the same forward, gated, with no dimension literals left in the driver.
# Stream to a log as well as capturing it: this is the slowest step in the battery (~20 min on
# this box) and model_driver prints a per-4-layer heartbeat. Without the tee that whole phase is
# buffered in $out and a working run is indistinguishable from a hang to anyone watching.
P2C_LOG="${TMPDIR:-/tmp}/phase2_prefill.log"
out="$(make -f native/Makefile prefill 2>&1 | tee "$P2C_LOG")"
if echo "$out" | grep -q "PREFILL_HIDDEN_MATCH: OK" && echo "$out" | grep -q "PREFILL_TOP1_MATCH: OK"; then
  step "S0.2c prefill on config dims" "PASS ($(echo "$out" | grep -oE 'maxrel = [0-9.e-]+' | head -1), top1 MATCH)"
else
  step "S0.2c prefill on config dims" "FAIL"; echo "$out" | tail -12 | sed 's/^/      /'; echo "      (full log: $P2C_LOG)"; fail=1
fi

# ── S0.4 — the independent decode oracle (llama.cpp), VybForge#22 ────────────────────
# Every other inference item here compares the GPU against the numpy reference, and both
# implement the same conventions, so agreement proves self-consistency rather than correctness
# (that is how #11 hid). This one compares against llama.cpp on the same GGUF, per token.
# SKIP is a pass with a notice: the gate skips when llama_cpp or the GGUF is absent.
out="$(./native/legit/run_decode_oracle_gate.sh 2>&1)"
last="$(echo "$out" | tail -1)"
if echo "$last" | grep -q "DECODE ORACLE GATE: PASS"; then
  step "S0.4 decode oracle (llama.cpp)" "PASS ($(echo "$out" | grep -cE '[a-z_]+ +PASS \(') cases match llama.cpp)"
elif echo "$last" | grep -q "DECODE ORACLE GATE: SKIP"; then
  step "S0.4 decode oracle (llama.cpp)" "SKIP ($(echo "$out" | grep -m1 'SKIP' | sed 's/^ *[^ ]* *//'))"
else
  step "S0.4 decode oracle (llama.cpp)" "FAIL"; echo "$out" | tail -8 | sed 's/^/      /'; fail=1
fi

# ── P4.1 — Gated DeltaNet layer-op references (VybForge#10 phase 4, item 1) ─────────
# Gate: native/legit/run_gdn_ops_gate.sh (doc/QWEN35-PHASE4.md). The recurrent layer's five verified
# units — the delta-rule recurrence + state read-out, the causal short convolution, the l2 norm /
# RMS norm / gated epilogue, the five projections with the beta/alpha gates, and the whole block wired
# stage by stage (14 dumps) — each checked against the SAME libggml the local llama.cpp is built with
# (a C harness links it and runs the real ops), not against our own maths. Every verifier also reports
# the REJECTED alternative's error (opposite eps convention 7e-2, transposed mul_mat operand ~1.2,
# threshold-less softplus inf, a mis-wired residual/epilogue 8.8e-2/3.0e-1) so the agreements are
# measurements, not tolerances nothing could breach. SKIPs without the llama.cpp checkout.
out="$(./native/legit/run_gdn_ops_gate.sh 2>&1)"
last="$(echo "$out" | tail -1)"
if echo "$last" | grep -q "GDN OPS GATE: PASS"; then
  step "P4.1 Gated DeltaNet op refs" "PASS ($(echo "$last" | sed 's/GDN OPS GATE: PASS //; s/[()]//g'))"
elif echo "$last" | grep -q "GDN OPS GATE: SKIP"; then
  step "P4.1 Gated DeltaNet op refs" "SKIP ($(echo "$last" | sed 's/GDN OPS GATE: SKIP //; s/[()]//g'))"
else
  step "P4.1 Gated DeltaNet op refs" "FAIL"; echo "$out" | tail -10 | sed 's/^/      /'; fail=1
fi

# ── P4.2 — the GDN recurrent layer's GPU kernels (VybForge#10 phase 4, item 1) ───────
# Gate: native/legit/run_gdn_kernel_gate.sh (doc/QWEN35-PHASE4.md). P4.1 pinned the reference to
# ggml's own ops; this is the first Vyb code for the same layer — native/kernels/gdn.vyb implements
# l2norm, the delta-rule step (one thread per head, no barrier or shared memory) and the gated RMS
# epilogue, and the gate compares all 802816 outputs of one token at the 27B geometry against an fp64
# reference, element by element (raw bit patterns, so nothing hides in a decimal dump). Reports the
# rejected read-out axis (1.4e0) as well. SKIPs without CUDA or a Vyb toolchain.
out="$(./native/legit/run_gdn_kernel_gate.sh 2>&1)"
last="$(echo "$out" | tail -1)"
if echo "$last" | grep -q "GDN KERNEL GATE: PASS"; then
  step "P4.2 GDN GPU kernels" "PASS ($(echo "$out" | grep -oE 'elements=[0-9]+' | tail -1))"
elif echo "$last" | grep -q "GDN KERNEL GATE: SKIP"; then
  step "P4.2 GDN GPU kernels" "SKIP ($(echo "$last" | sed 's/GDN KERNEL GATE: SKIP //; s/[()]//g'))"
else
  step "P4.2 GDN GPU kernels" "FAIL"; echo "$out" | tail -10 | sed 's/^/      /'; fail=1
fi

# ── P4.3 — the recurrent LAYER wired on the GPU vs ggml's own graph (phase 4, item 1) ──
# Gate: native/legit/run_gdn_layer_gate.sh (doc/QWEN35-PHASE4.md). P4.2 ran the layer's kernels; this
# one wires the whole block in Vyb (native/host/gdn_layer_driver.vyb: rmsnorm, mm_nt, sigmoid_k,
# alpha_gate, conv1d_k, interleave_qkv, silu_k, l2norm, delta_step, norm_gated, add_k) and compares it
# against unit 5's authority — the same block built from the real ggml ops — STAGE BY STAGE, so a
# wiring mistake names the stage. Small non-degenerate geometry (n_embd=512, S=32, H_k=4, H_v=8,
# d_conv=4), one token, state zero. Reports its own negatives (residual on the normed input 7.9e-2,
# no residual 9.5e-1) against a 1e-4 bar. SKIPs without toolchain/libggml/CUDA.
out="$(./native/legit/run_gdn_layer_gate.sh 2>&1)"
last="$(echo "$out" | tail -1)"
if echo "$last" | grep -q "GDN LAYER GATE: PASS"; then
  step "P4.3 GDN layer wiring (GPU)" "$(echo "$out" | grep -oE 'GPU layer wiring \([0-9]+ stages\) +PASS.*' | head -1 | sed 's/GPU layer wiring //')"
elif echo "$last" | grep -q "GDN LAYER GATE: SKIP"; then
  step "P4.3 GDN layer wiring (GPU)" "SKIP ($(echo "$last" | sed 's/GDN LAYER GATE: SKIP //; s/[()]//g'))"
else
  step "P4.3 GDN layer wiring (GPU)" "FAIL"; echo "$out" | tail -10 | sed 's/^/      /'; fail=1
fi

# ── P4.4 — the layer-kind DISPATCH rule (VybForge#10 phase 4, unit 9c step 1) ─────────
# Gate: native/legit/run_layerkind_gate.sh. Unit 9c is the engine integration, and its first
# requirement is a per-layer dispatch decided from the model's TENSOR TABLE rather than from the
# architecture name or a hardcoded block-index list. The rule now lives in one place
# (model_caps::mc_layer_kind, which the capability descriptor counts with) and this gate pins it
# three ways: a selftest table of the name traps (attn_qkv / attn_gate / attn_q_norm are NOT the
# attention marker), the real interleave on both models (Ridge: 16 attention blocks at N%4==3 + 48
# recurrent out of 64 text blocks; Qwen3-4B: 36 attention + 0), and an INDEPENDENT per-layer check
# against the Python inventory's tensor NAMES. It also pins the draft-head exclusion — block 64 is
# in the plan but out of the text count, which a kind-only count gets wrong as "17 attention". SKIPs
# without a Vyb toolchain or the (12 GiB) Ridge download; a gate that proves nothing is a FAIL.
out="$(./native/legit/run_layerkind_gate.sh 2>&1)"
last="$(echo "$out" | tail -1)"
if echo "$last" | grep -q "LAYER-KIND GATE: PASS"; then
  step "P4.4 layer-kind dispatch rule" "$(echo "$last" | sed 's/P4.4 LAYER-KIND GATE: PASS //; s/[()]//g')"
elif echo "$last" | grep -q "LAYER-KIND GATE: SKIP"; then
  step "P4.4 layer-kind dispatch rule" "SKIP ($(echo "$last" | sed 's/P4.4 LAYER-KIND GATE: SKIP //; s/[()]//g'))"
else
  step "P4.4 layer-kind dispatch rule" "FAIL"; echo "$out" | tail -12 | sed 's/^/      /'; fail=1
fi

# ── P4.5 — the recurrent block on the ENGINE's own weight path (phase 4, unit 9c step 2) ──
# Gate: native/legit/run_gdn_engine_gate.sh. P4.3 proved the block can be wired out of Vyb kernels,
# but on a FIXTURE and with `mm_nt` (B = [out,in]). The engine loads each tensor by NAME from the
# GGUF, dequantises it with the same kernels it uses for every layer, and multiplies with `gemm`
# (B = [in,out], the dequant kernels' transposing write). This gate runs
# native/host/gdn_engine_driver.vyb — all ten blk.0 tensors staged from the live GGUF through that
# path — against the same authority, 26 stages over two chained steps, plus a slot-by-slot check of
# the staged ssm_out operand at the addresses gemm reads. SKIPs without toolchain/libggml/CUDA/the
# Ridge download.
out="$(./native/legit/run_gdn_engine_gate.sh 2>&1)"
last="$(echo "$out" | tail -1)"
if echo "$last" | grep -q "GDN ENGINE GATE: PASS"; then
  step "P4.5 GDN engine weight path" "$(echo "$out" | grep -oE 'engine path block \([0-9]+ stages\) +PASS.*' | head -1 | sed 's/engine path block //')"
elif echo "$last" | grep -q "GDN ENGINE GATE: SKIP"; then
  step "P4.5 GDN engine weight path" "SKIP ($(echo "$last" | sed 's/GDN ENGINE GATE: SKIP //; s/[()]//g'))"
else
  step "P4.5 GDN engine weight path" "FAIL"; echo "$out" | tail -12 | sed 's/^/      /'; fail=1
fi

# ── P4.8 — Ridge's attention-block front half, against ggml (unit 10.3a) ───────────────
# Gate: native/legit/run_attn_block_gate.sh. The joint Q+gate split, both per-head RMS norms and rope,
# each stage compared with ggml's own ops, with the alternative readings required to differ (gate-first
# split, one norm over the whole projection, rope over the whole head). Measured: 7 stages at worst
# 3.98e-07, 3 alternatives rejected at 1.20..1.54. Attention, the output gate and wo are NOT covered.
out="$(./native/legit/run_attn_block_gate.sh 2>&1)"
last="$(echo "$out" | tail -1)"
if echo "$last" | grep -q "ATTENTION BLOCK GATE: PASS"; then
  step "P4.8 attention block (front half)" "$(echo "$out" | grep -oE 'PASS \(.*\)' | head -1 | sed 's/PASS (//; s/)$//')"
elif echo "$last" | grep -q "ATTENTION BLOCK GATE: SKIP"; then
  step "P4.8 attention block (front half)" "SKIP ($(echo "$last" | sed 's/ATTENTION BLOCK GATE: SKIP //; s/[()]//g'))"
else
  step "P4.8 attention block (front half)" "FAIL"; echo "$out" | tail -12 | sed 's/^/      /'; fail=1
fi

# ── P4.9 — Ridge's attention block on the ENGINE's own weight path (unit 10 step 5) ────
# Gate: native/legit/run_attn_engine_gate.sh. P4.8 pinned the READING of the attention block against
# ggml on a synthetic fixture; this runs it in the file that will run it — native/host/model_driver.vyb
# in probe mode (VYB_ATTN_PROBE) — staging the model's own Q5_K attn_q/attn_k/attn_v and Q6_K
# attn_output from the GGUF, splitting the joint q+gate, and comparing all TEN stages against
# native/tools/attn_authority.c fed the same weights. Teeth: the staged operands at gemm's own
# addresses, the contiguous GQA grouping (must match) vs round-robin (must miss), and the output gate
# vs raw/no gate. Measured: worst 1.67e-06.
out="$(./native/legit/run_attn_engine_gate.sh 2>&1)"
last="$(echo "$out" | tail -1)"
if echo "$last" | grep -q "ATTN ENGINE GATE: PASS"; then
  step "P4.9 attention block (engine weights)" "$(echo "$out" | grep -oE 'PASS \(.*\)' | head -1 | sed 's/PASS (//; s/)$//')"
elif echo "$last" | grep -q "ATTN ENGINE GATE: SKIP"; then
  step "P4.9 attention block (engine weights)" "SKIP ($(echo "$last" | sed 's/ATTN ENGINE GATE: SKIP //; s/[()]//g'))"
else
  step "P4.9 attention block (engine weights)" "FAIL"; echo "$out" | tail -14 | sed 's/^/      /'; fail=1
fi

# ── P4.10 — Ridge's FFN block on the ENGINE's own weight path (phase 4, W1) ─────────
# Gate: native/legit/run_ffn_engine_gate.sh. Every Ridge block's FFN is IQ2_S (160 tensors, the
# mid-stack) or IQ3_S (32, the edge layers), and both carry their codebook as an ordinary FIFTH
# kernel argument — so they launch through `cuda_launch_n`, not the four-slot `cuda_launch4i` every
# other dequant uses. Before W1 `stage_one` had no case for types 21/22 and fell through to the F32
# reader, silently mis-staging them. This stages the block's three FFN weights + its pre-norm by
# name from the GGUF through the engine's own kernels and compares seven stages against OUR numpy
# ports of the same two dequantizers, at three type boundaries (blk.3 all IQ3_S, blk.7 mixed:
# IQ3_S down with IQ2_S gate/up, blk.19 all IQ2_S). Teeth: the staged operands at gemm's own
# addresses (B[k*N+n]), the two residual mis-wirings, and SiLU dropped — all must MISS. Measured:
# worst 2.3e-11 over 21 stages.
out="$(./native/legit/run_ffn_engine_gate.sh 2>&1)"
last="$(echo "$out" | tail -1)"
if echo "$last" | grep -q "FFN ENGINE GATE: PASS"; then
  step "P4.10 FFN block (engine weights, IQ2_S/IQ3_S)" "$(echo "$out" | grep -oE 'PASS \(.*\)' | head -1 | sed 's/PASS (//; s/)$//')"
elif echo "$last" | grep -q "FFN ENGINE GATE: SKIP"; then
  step "P4.10 FFN block (engine weights, IQ2_S/IQ3_S)" "SKIP ($(echo "$last" | sed 's/FFN ENGINE GATE: SKIP //; s/[()]//g'))"
else
  step "P4.10 FFN block (engine weights, IQ2_S/IQ3_S)" "FAIL"; echo "$out" | tail -14 | sed 's/^/      /'; fail=1
fi

# ── P4.11 — OUR tokenizer vs the captured Ridge oracle (phase 4, W2/W3 prerequisite) ────
# Gate: native/legit/run_ridge_encoder_gate.sh. The oracle fixtures (native/legit/fixtures/llama_ridge)
# carry llama.cpp's OWN /tokenize output; this requires our encoder to reproduce those ids exactly,
# because a different id stream makes every hidden and top1 downstream meaningless while looking like
# a model bug. It also guards a silent upstream trap: stdlib/vllm's build_vocab_from parses a
# pretty-printed vocab.json to id 0 for EVERY token (its read_int stops at the first non-digit) while
# still returning the right token COUNT — measured, 5 ids for "The capital of France is", all 0. The
# check compacts the file first and asserts a known token maps to a non-zero id. Measured: PASS on
# both fixtures once compacted (5 and 20 ids, exactly the oracle's).
out="$(./native/legit/run_ridge_encoder_gate.sh 2>&1)"
last="$(echo "$out" | tail -1)"
if echo "$last" | grep -q "RIDGE ENCODER GATE: PASS"; then
  step "P4.11 encoder vs the Ridge oracle" "$(echo "$out" | grep -oE 'PASS \(.*\)' | head -1 | sed 's/PASS (//; s/)$//')"
elif echo "$last" | grep -q "RIDGE ENCODER GATE: SKIP"; then
  step "P4.11 encoder vs the Ridge oracle" "SKIP ($(echo "$last" | sed 's/RIDGE ENCODER GATE: SKIP //; s/[()]//g'))"
else
  step "P4.11 encoder vs the Ridge oracle" "FAIL"; echo "$out" | tail -14 | sed 's/^/      /'; fail=1
fi

# ── P4.12 — a WHOLE 65-block Ridge forward vs the captured oracle (phase 4, W2/W3) ────
# Gate: native/legit/run_ridge_forward_gate.sh. The probe gates (P4.5/P4.9/P4.10) each isolate ONE
# block and compare it against a reference written here, so none of them can see an error BETWEEN
# layers; this runs every block in one process (the untied head chunked, the embed built from the
# prompt's own rows) and compares per-position top1 under the fixture's margin_bar rule plus the final
# hidden, with provenance checked first so a moved oracle is a FAIL that says "recapture". EXPENSIVE:
# the per-layer staging goes through the per-8-byte H2D helper, so it is tens of minutes per prompt.
out="$(./native/legit/run_ridge_forward_gate.sh 2>&1)"
last="$(echo "$out" | tail -1)"
if echo "$last" | grep -q "RIDGE FORWARD GATE: PASS"; then
  step "P4.12 whole Ridge forward (vs the oracle)" "$(echo "$out" | grep -oE 'PASS \(.*\)' | head -1 | sed 's/PASS (//; s/)$//')"
elif echo "$last" | grep -q "RIDGE FORWARD GATE: SKIP"; then
  step "P4.12 whole Ridge forward (vs the oracle)" "SKIP ($(echo "$last" | sed 's/RIDGE FORWARD GATE: SKIP //; s/[()]//g'))"
else
  step "P4.12 whole Ridge forward (vs the oracle)" "FAIL"; echo "$out" | tail -16 | sed 's/^/      /'; fail=1
fi

# ── P4.13 — the MTP DRAFT HEAD (blk.NTL) vs the same oracle, teacher-forced (phase 4, W4) ────
# Gate: native/legit/run_ridge_mtp_gate.sh. Feeds the draft head the ORACLE's own normalised hidden row
# and the ORACLE's own next token, and requires its per-step top1 to be the oracle's pick for t+2 — so
# the new block is judged with the main pass taken out of the loop. No new oracle capture: the fixtures
# already record a greedy pick at every position. Also runs the checker's off-by-one tooth, which MUST
# miss (a checker that always agrees would make this gate decoration).
out="$(./native/legit/run_ridge_mtp_gate.sh 2>&1)"
last="$(echo "$out" | tail -1)"
if echo "$last" | grep -q "RIDGE MTP GATE: PASS"; then
  step "P4.13 MTP draft head (teacher-forced)" "$(echo "$out" | grep -oE 'PASS \(.*\)' | head -1 | sed 's/PASS (//; s/)$//')"
elif echo "$last" | grep -q "RIDGE MTP GATE: SKIP"; then
  step "P4.13 MTP draft head (teacher-forced)" "SKIP ($(echo "$last" | sed 's/RIDGE MTP GATE: SKIP //; s/[()]//g'))"
else
  step "P4.13 MTP draft head (teacher-forced)" "FAIL"; echo "$out" | tail -16 | sed 's/^/      /'; fail=1
fi

# ── P4.7 — the rope VARIANT kernel on the GPU, against the same op (unit 10.2) ────────
# Gate: native/legit/run_rope_kernel_gate.sh. P4.6 pins the spec; this proves the kernel that will
# rotate Ridge's q/k reproduces it — at n_rot=64 AND (must-not) at n_rot=HD, so a pass cannot come
# from a check that ignores the parameter. Measured: q 8.90e-08 / k 6.97e-08 at n_rot=64, 1.70e+00 at
# n_rot=256. SKIPs without CUDA or the llama.cpp checkout.
out="$(./native/legit/run_rope_kernel_gate.sh 2>&1)"
last="$(echo "$out" | tail -1)"
if echo "$last" | grep -q "ROPE KERNEL GATE: PASS"; then
  step "P4.7 rope kernel (n_rot-gated)" "$(echo "$out" | grep -oE 'PASS \(.*\)' | head -1 | sed 's/PASS (//; s/)$//')"
elif echo "$last" | grep -q "ROPE KERNEL GATE: SKIP"; then
  step "P4.7 rope kernel (n_rot-gated)" "SKIP ($(echo "$last" | sed 's/ROPE KERNEL GATE: SKIP //; s/[()]//g'))"
else
  step "P4.7 rope kernel (n_rot-gated)" "FAIL"; echo "$out" | tail -12 | sed 's/^/      /'; fail=1
fi

# ── P4.6 — Ridge's attention RoPE, against ggml's own op (phase 4, unit 10) ──────────
# Gate: native/legit/run_rope_gate.sh. Phase 4 blamed "IMROPE" for the hybrid attention path; this
# gate asks the real op which (mode, layout) reproduces it and requires exactly one at the f32 floor.
# Measured: mode=neox, NEOX pairing inside the first n_dims (64 of 256), the rest passed through —
# 8.99e-08, runner-up 1.01e+00. The sections are INERT for this architecture, and the engine's rope
# (every head dim, pairs (i, i+HD/2)) is the rejected candidate at 1.478 — so a rope VARIANT is real
# work, not a parameter change. SKIPs without the llama.cpp checkout.
out="$(./native/legit/run_rope_gate.sh 2>&1)"
last="$(echo "$out" | tail -1)"
if echo "$last" | grep -q "ROPE GATE: PASS"; then
  step "P4.6 attention rope (vs ggml)" "$(echo "$out" | grep -oE 'PASS \(winner.*\)' | head -1 | sed 's/PASS (//; s/)$//')"
elif echo "$last" | grep -q "ROPE GATE: SKIP"; then
  step "P4.6 attention rope (vs ggml)" "SKIP ($(echo "$last" | sed 's/ROPE GATE: SKIP //; s/[()]//g'))"
else
  step "P4.6 attention rope (vs ggml)" "FAIL"; echo "$out" | tail -12 | sed 's/^/      /'; fail=1
fi

# ── S0.10 — Q4_K dequant reference (the type with no reference until now) ───────────
# Gate: native/legit/run_q4k_gate.sh. blk.N.attn_qkv, blk.N.attn_gate and blk.N.ssm_out are all Q4_K
# (the model's biggest projections) and Q4_K was the one quant type with no numpy reference and no
# gate — which is what blocked the GDN layer check from taking those weights. native/tools/q4k_ref.py
# is that reference, compared ELEMENT-WISE against the independent gguf package's own Q4_K
# dequantizer on whole real tensors (2 x 31.5 M values, bit-identical). SKIPs without the model and
# FAILS if it proved nothing.
out="$(./native/legit/run_q4k_gate.sh 2>&1)"
last="$(echo "$out" | tail -1)"
if echo "$last" | grep -q "Q4_K GATE: PASS"; then
  step "S0.10 Q4_K dequant reference" "$(echo "$out" | grep -oE 'PASS \(.*\)' | tail -1)"
elif echo "$last" | grep -q "Q4_K GATE: SKIP"; then
  step "S0.10 Q4_K dequant reference" "SKIP ($(echo "$last" | sed 's/Q4_K GATE: SKIP //; s/[()]//g'))"
else
  step "S0.10 Q4_K dequant reference" "FAIL"; echo "$out" | tail -8 | sed 's/^/      /'; fail=1
fi

echo
if [ $fail -eq 0 ]; then echo "PHASE 2 BATTERY: PASS"; else echo "PHASE 2 BATTERY: FAIL"; fi
exit $fail
