#!/usr/bin/env bash
# run_forge_ledger.sh — VybForge -> shared VybOS ledger build-record posting.
# The forge computes a deterministic artifact hash + seals an AUTHENTIC signed
# build record per selected package, POSTS it to the SHARED registry ledger
# (VybOS modules/ledger.vyb, consumed cross-repo via --module-path ../VybOS/modules),
# and proves 12 cross-repo properties: deterministic hashing, authentic build
# record under the forge key, posted-build authenticity (signature-gate covers
# the recorded artifact hash), wrong-key/wrong-pubkey rejection, tamper
# rejection, all-selected-packages listed, and an offline ledger checkpoint.
#
# Usage:  ./native/legit/run_forge_ledger.sh [fixture.json]
# Env:    VYB (default the Vyb build in ~/Projects); VYBOS (VybOS checkout root)
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
VYB="${VYB:-/home/rick/Projects/Vyb/build/vyb}"
VYB_STDLIB="${VYB_STDLIB:-/home/rick/Projects/Vyb/stdlib}"
VYBOS="${VYBOS:-/home/rick/Projects/VybOS}"

fixture="${1:-$root/config/mock-system.json}"
export FORGE_LEGIT_FIXTURE="$fixture"

echo "== VybForge build records -> shared VybOS registry ledger =="
echo "fixture : $fixture"
echo "ledger  : $VYBOS/modules/ledger.vyb"
"$VYB" "$root/native/legit/build-forge-ledger.vyb" \
    --module-path "$root/native/legit" --module-path "$root/native/json" \
    --module-path "$VYBOS/modules"
