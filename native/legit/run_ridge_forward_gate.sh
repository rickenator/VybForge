#!/usr/bin/env bash
# P4.12 the WHOLE Ridge model vs the captured oracle (VybForge#10 phase 4, W2/W3).
#
# The acceptance test for the mission's item 3. Every probe gate in this repo (P4.5/P4.9/P4.10) compares
# the engine against a reference written here, which proves a block in isolation and says nothing about
# an error BETWEEN layers. This runs all 65 blocks in one process and compares the result against
# llama.cpp on the same GGUF, through the fixtures native/tools/ridge_oracle_capture.py froze:
#
#   * per-position top1 under the fixture's own rule — EQUALITY where the oracle's top-2 margin clears
#     margin_bar, MEMBERSHIP of {top1,top2} below it;
#   * the final hidden, with our output_norm applied first (ours is pre-norm, the oracle's /embeddings
#     output is post-norm — comparing across that stage reads ~0.7 cosine on a CORRECT forward, so the
#     un-normalized cosine is printed as a required negative);
#   * provenance FIRST: a llama-server version or GGUF id mismatch is a FAIL saying "recapture".
#
# EXPENSIVE. One full 64-block run per prompt, and the per-layer weight staging goes through the
# per-8-byte H2D helper, so expect tens of minutes. Do not kill it on elapsed time alone (see §4).
#
# SKIPs (never PASSes) without the toolchain, the Ridge model, the inventory / tensor index / rope
# table, the tokenizer dir (artifacts/ridge-tokenizer) or the oracle fixtures.
#
# To CHECK a forward that was already run by hand (rather than pay for another one):
#   VYBFORGE_RF_SKIP_RUN=1 VYBFORGE_RF_LOG=native/out/ridge_run.log ./native/legit/run_ridge_forward_gate.sh
# The log's own PROMPT_SRC must match the fixture's prompt, or the verifier refuses: native/out is
# shared by every run, so an unprovenanced log would compare another prompt's numbers.
set -u
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$root"

step() { printf '%-46s %s\n' "$1" "$2"; }
echo "Ridge whole-model gate — $(date '+%F %T')"
echo

py=""
for cand in "${RIDGEF_PY:-}" "$root/.venv/bin/python" python3; do
  if [ -n "$cand" ] && env -u PYTHONPATH "$cand" -c 'import numpy' >/dev/null 2>&1; then py="$cand"; break; fi
done
if [ -z "$py" ]; then
  step "reference interpreter" "FAIL (no python with numpy)"
  echo; echo "RIDGE FORWARD GATE: FAIL"; exit 1
fi

work="${VYBFORGE_RIDGE_FWD_OUTDIR:-$root/native/out/ridge_forward}"
mkdir -p "$work"
log="$work/verify.log"

env -u PYTHONPATH "$py" native/tools/ridge_forward_verify.py "$@" >"$log" 2>&1
rc=$?
grep -E '^RIDGE_FORWARD_VERIFY' "$log" | sed 's/^/      /'

if grep -q '^RIDGE_FORWARD_VERIFY_SKIP' "$log"; then
  step "whole Ridge forward" "SKIP ($(grep -m1 '^RIDGE_FORWARD_VERIFY_SKIP' "$log" | sed 's/^RIDGE_FORWARD_VERIFY_SKIP //'))"
  echo; echo "RIDGE FORWARD GATE: SKIP (nothing ran)"; exit 0
fi

if grep -q '^RIDGE_FORWARD_VERIFY_DONE' "$log" && [ "$rc" = "0" ]; then
  c="$(grep -oE 'hidden_cos=[0-9.]+' "$log" | tail -1 | sed 's/hidden_cos=//')"
  n="$(grep -oE '^RIDGE_FORWARD_VERIFY_SUMMARY fixture=[^ ]+' "$log" | wc -l)"
  step "whole Ridge forward ($n prompt(s))" "PASS (hidden cos $c, per-position top1 == the oracle)"
  grep -oE '^RIDGE_FORWARD_VERIFY_SUMMARY.*' "$log" | sed 's/^/      /'
  echo; echo "RIDGE FORWARD GATE: PASS"
  exit 0
fi

step "whole Ridge forward" "FAIL (see $log)"
echo; echo "RIDGE FORWARD GATE: FAIL"
exit 1
