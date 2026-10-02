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
  step "P2.1b build_fullmanifest.vyb" "manifest parity (429 tokens) + pretok boundaries"
else
  step "P2.1b build_fullmanifest.vyb" "FAIL"; echo "$out" | tail -8 | sed 's/^/      /'; fail=1
fi

# P2.1c — _build_kv.py -> _build_kv.vyb (driver assembler)
out="$(./native/legit/run_kvbuild_gate.sh 2>&1)"
if echo "$out" | tail -1 | grep -q "PASS"; then
  step "P2.1c _build_kv.vyb" "driver parity (1247 lines) + front-end"
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

echo
if [ $fail -eq 0 ]; then echo "PHASE 2 BATTERY: PASS"; else echo "PHASE 2 BATTERY: FAIL"; fi
exit $fail
