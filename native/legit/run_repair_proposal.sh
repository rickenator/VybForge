#!/usr/bin/env bash
# run_repair_proposal.sh — VybForge model-boundary closure for #7.
#
# Proves the full PATH: propose_repair.vyb (Vyb-native driver; deterministic
# mock by default, no model needed) emits a schema-constrained PatchProposal;
# validate_proposal.vyb runs it through the REAL VybOS repair core (apply ->
# gates -> guardrail -> promote -> seal) and reports the decision. A live model
# (ollama / openai-chat / openai-responses) replaces mock in production; the
# validator code path is identical, so CI/demo needs no GPU.
#
# Live backends (all via the same driver, VYBFORGE_* env vars):
#   VYBFORGE_BACKEND=ollama|openai-chat|openai-responses
#   VYBFORGE_ENDPOINT=http://127.0.0.1:11434 VYBFORGE_MODEL=qwen3:8b
#   VYBFORGE_API_KEY=...   VYBFORGE_STRUCTURED_OUTPUT=json_schema|json_object|prompt
#
# Usage:  ./native/legit/run_repair_proposal.sh
# Env:    VYB (default the Vyb build); VYBOS (VybOS checkout; default ~/Projects/VybOS)
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
VYB="${VYB:-/home/rick/Projects/Vyb/build/vyb}"
VYB_STDLIB="${VYB_STDLIB:-/home/rick/Projects/Vyb/stdlib}"
VYBOS="${VYBOS:-/home/rick/Projects/VybOS}"
fixture="$root/native/legit/fixtures/repair_pristine.c"
work="$(mktemp -d)"
trap 'rm -rf "$work"' EXIT

echo "== VybForge #7 model boundary (deterministic, no model required) =="
echo "fixture : $fixture"

# 1. deterministic mock proposal (a real model would fill this slot)
VYB_STDLIB="$VYB_STDLIB" \
VYBFORGE_BACKEND=mock \
VYBFORGE_PRISTINE="$fixture" \
VYBFORGE_REASON="add() computes a-b not a+b (sign bug)" \
VYBFORGE_CLS=generated \
VYBFORGE_GENERATOR=qwen3-mock \
VYBFORGE_OUT="$work/proposal.json" \
  "$VYB" "$root/tools/propose_repair.vyb" --module-path "$root/tools" --module-path "$root/native/json"

# 2. validate the proposal through the VybOS repair core — every decision path
echo "  ACCEPT (low-risk, gates pass):"
PROP_PROPOSAL="$work/proposal.json" PROP_PRISTINE="$fixture" PROP_TESTS=pass PROP_INTEGR=pass PROP_RISK=low \
  "$VYB" "$root/tools/validate_proposal.vyb" --module-path "$root/native/json" --module-path "$VYBOS/modules" 2>&1 | grep MODEL-BOUNDARY
echo "  REJECT (build/test gate fails):"
PROP_PROPOSAL="$work/proposal.json" PROP_PRISTINE="$fixture" PROP_TESTS=fail PROP_INTEGR=pass PROP_RISK=low \
  "$VYB" "$root/tools/validate_proposal.vyb" --module-path "$root/native/json" --module-path "$VYBOS/modules" 2>&1 | grep MODEL-BOUNDARY
echo "  HUMAN-REQUIRED (high-risk surface):"
PROP_PROPOSAL="$work/proposal.json" PROP_PRISTINE="$fixture" PROP_TESTS=pass PROP_INTEGR=pass PROP_RISK=high \
  "$VYB" "$root/tools/validate_proposal.vyb" --module-path "$root/native/json" --module-path "$VYBOS/modules" 2>&1 | grep MODEL-BOUNDARY

echo
echo "== VybForge #7 model boundary: ok =="
