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
  full_ok=1
else
  step "whole Ridge forward" "FAIL (see $log)"
  full_ok=0
fi

# ── the CAUSAL PATH, on the same fixture, at shorter prefixes ───────────────────────────────────
# A causal model's token-k hidden cannot depend on the tokens that FOLLOW it, so the fixture also
# covers the first k positions: running the prompt at k = 1, 2, 3 is an oracle-backed check of the
# causal path, and it is the check that catches a per-layer state buffer sized for one layer (the
# S=1/S=2 `GDN_ERR conv` crash) or any cross-token contamination. The verifier proves each prefix's
# IDS are the fixture's first k ids, so a prefix that tokenizes differently FAILS loudly instead of
# silently comparing the wrong positions. Skipped when VYBFORGE_RF_SKIP_RUN is set (there is then
# only one log to read, and it belongs to one prompt).
pfx_ok=1
if [ -z "${VYBFORGE_RF_SKIP_RUN:-}" ]; then
  IFS='|' read -ra PFXS <<< "${VYBFORGE_RIDGE_PREFIXES:-The|The capital|The capital of}"
  i=0
  for px in "${PFXS[@]}"; do
    i=$((i + 1))
    logp="$work/verify_prefix$i.log"
    env -u PYTHONPATH VYBFORGE_RF_PROMPT="$px" "$py" native/tools/ridge_forward_verify.py "$@" >"$logp" 2>&1
    prc=$?
    pc="$(grep -oE 'hidden_cos=[0-9.]+' "$logp" | tail -1 | sed 's/hidden_cos=//')"
    pn="$(grep -oE 'per-position top1: [0-9]+/[0-9]+' "$logp" | tail -1 | sed 's/per-position top1: //')"
    if [ "$prc" = 0 ] && grep -q '^RIDGE_FORWARD_VERIFY_DONE' "$logp"; then
      step "prefix [${px}]" "PASS (top1 $pn, hidden cos $pc)"
    else
      step "prefix [${px}]" "FAIL (see $logp)"
      grep -E '^RIDGE_FORWARD_VERIFY_FAIL' "$logp" | sed 's/^/      /'
      pfx_ok=0
    fi
  done
fi

if [ "$full_ok" = 1 ] && [ "$pfx_ok" = 1 ]; then
  echo; echo "RIDGE FORWARD GATE: PASS"
  exit 0
fi
echo; echo "RIDGE FORWARD GATE: FAIL"
exit 1
