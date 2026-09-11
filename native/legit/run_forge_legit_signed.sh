#!/usr/bin/env bash
# run_forge_legit_signed.sh — AUTHENTIC (signed) OS-package provenance:
# dogfoods native/legit/forge_legit_signed.vyb via the proving entry
# build-forge-legit-signed.vyb. Each selected package's provenance is sealed
# into a SignedChain (every block Ed25519-signed, genesis signer == forge key,
# authorized key rotation) and 8 authenticity checks run.
#
# Usage:  ./native/legit/run_forge_legit_signed.sh [fixture.json]
# Env:    VYB (default the Vyb build in ~/Projects)
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
VYB="${VYB:-/home/rick/Projects/Vyb/build/vyb}"
VYB_STDLIB="${VYB_STDLIB:-/home/rick/Projects/Vyb/stdlib}"

fixture="${1:-$root/config/mock-system.json}"

export FORGE_LEGIT_FIXTURE="$fixture"

echo "== VybForge AUTHENTIC package legitimization (signed) =="
echo "fixture : $fixture"
"$VYB" "$root/native/legit/build-forge-legit-signed.vyb" \
    --module-path "$root/native/legit" --module-path "$root/native/json"
