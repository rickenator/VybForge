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

echo
if [ $fail -eq 0 ]; then echo "PHASE 2 BATTERY: PASS"; else echo "PHASE 2 BATTERY: FAIL"; fi
exit $fail
