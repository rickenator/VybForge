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

echo
if [ $fail -eq 0 ]; then echo "PHASE 2 BATTERY: PASS"; else echo "PHASE 2 BATTERY: FAIL"; fi
exit $fail
