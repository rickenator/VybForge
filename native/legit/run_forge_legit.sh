#!/usr/bin/env bash
# run_forge_legit.sh — legitimize the OS packages named in the VybForge
# desired-state fixture with the shared Vyb `chain` core, and write per-package
# legitimization receipts + tokens under an output dir. Pure-Vyb (JIT); the
# only I/O is reading the fixture and writing the receipts.
#
# Usage:  ./native/legit/run_forge_legit.sh [fixture.json] [receipt-dir]
# Env:    VYB (default the Vyb build in ~/Projects)
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
VYB="${VYB:-/home/rick/Projects/Vyb/build/vyb}"

fixture="${1:-$root/config/mock-system.json}"
outdir="${2:-$root/native/legit/receipts}"

mkdir -p "$outdir"

export FORGE_LEGIT_FIXTURE="$fixture"
export FORGE_LEGIT_OUT="$outdir"

echo "== VybForge package legitimization =="
echo "fixture : $fixture"
echo "receipts: $outdir/"
"$VYB" "$root/native/legit/forge_legit.vyb" --module-path "$root/native/json"
echo
echo "== receipts =="
ls -1 "$outdir" 2>/dev/null || true
