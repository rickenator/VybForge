#!/usr/bin/env bash
# W4b — the CHAINED draft head (VybForge#10 phase 4). VYB_MTP_CHAIN=<N>, default off.
#
# Two steps, and they are different kinds of evidence:
#
#   1. THE TOOTH — the chain's per-step batching is fed the TEACHER-FORCED inputs (the oracle's hidden
#      row and the prompt's own next token) and its per-step top1 must be IDENTICAL to the
#      teacher-forced mode's own picks read out of prefill_top1_vyb.txt. Bit-for-bit equality is the
#      criterion because both runs embed the same inputs through the same arithmetic; a mismatch means
#      the loop is wrong (accumulating rows, batch length k+1, per-row rope position, scoring row k
#      alone), not that the draft head drafts differently. This step is a real PASS/FAIL gate.
#
#   2. THE MEASUREMENT — the chain's acceptance against the oracle, per fixture, reported. This is a
#      NUMBER, not a verdict, and deliberately so: P4.13's acceptance FLOOR (doc/QWEN35-MTP-HARVEST.md
#      §10) belongs to the teacher-forced criterion, while the chain's acceptance-vs-N is a whole-run
#      quantity — ONE early miss zeroes it — so no threshold on it would be meaningful. The chain's
#      per-step quantity is the conditional survival the sweep reports, which is why the sweep exists.
#      The gate fails if the run did not complete (or if the tooth failed), never because acceptance is low.
#
# VYBFORGE_MTP_CHAIN_SWEEP=1 replaces the measurement with the per-anchor survival sweep (the chain's
# conditional acceptance given a verified prefix) — the decay curve the cost model needs, and the only
# way to see it at all, since a single early disagreement makes every longer N read as 0.
#
# SKIPs (never PASSes) without the toolchain, the model, the tensor index, the rope table or a fixture.
set -u
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$root"

step() { printf '%-52s %s\n' "$1" "$2"; }
echo "Ridge MTP CHAINED-draft gate — $(date '+%F %T')"
echo

py=""
for cand in "${RIDGEF_PY:-}" "$root/.venv/bin/python" python3; do
  if [ -n "$cand" ] && env -u PYTHONPATH "$cand" -c 'import numpy' >/dev/null 2>&1; then py="$cand"; break; fi
done
if [ -z "$py" ]; then
  step "reference interpreter" "FAIL (no python with numpy)"
  echo; echo "RIDGE MTP CHAIN GATE: FAIL"; exit 1
fi

work="${VYBFORGE_RIDGE_MTP_OUTDIR:-$root/native/out/ridge_mtp}"
mkdir -p "$work"
log="$work/chain_verify.log"
fixtures="${VYBFORGE_MTP_FIXTURES:-the_capital_of_france_is 1_2_3_4_5_6_7}"

# ---- 1. the tooth: the chain mechanism must reproduce the teacher-forced picks exactly -------------
env -u PYTHONPATH "$py" native/tools/ridge_mtp_chain_verify.py --tooth $fixtures >"$log" 2>&1
rc=$?
grep -E '^RIDGE_MTP_CHAIN_TOOTH' "$log" | sed 's/^/      /'
if grep -q '^RIDGE_MTP_CHAIN_SKIP' "$log"; then
  step "chain mechanism vs teacher-forced" "SKIP ($(grep -m1 '^RIDGE_MTP_CHAIN_SKIP' "$log" | sed 's/^RIDGE_MTP_CHAIN_SKIP //'))"
  echo; echo "RIDGE MTP CHAIN GATE: SKIP (nothing ran)"; exit 0
fi
if [ "$rc" != "0" ]; then
  step "chain mechanism vs teacher-forced" "FAIL (see $log)"
  echo; echo "RIDGE MTP CHAIN GATE: FAIL"; exit 1
fi
step "chain mechanism vs teacher-forced" "PASS ($(grep -c '^RIDGE_MTP_CHAIN_TOOTH_OK' "$log") fixture(s) identical)"

# ---- 2. the measurement: acceptance vs N (and, with the sweep knob, per-anchor survival) ----------
env -u PYTHONPATH "$py" native/tools/ridge_mtp_chain_verify.py $fixtures >"$log" 2>&1
rc=$?
grep -E '^RIDGE_MTP_CHAIN' "$log" | sed 's/^/      /'
if grep -q '^RIDGE_MTP_CHAIN_FAIL' "$log" || [ "$rc" != "0" ]; then
  step "chained draft measurement" "FAIL (see $log)"
  echo; echo "RIDGE MTP CHAIN GATE: FAIL"; exit 1
fi
step "chained draft measurement" "ran ($(grep -c '^RIDGE_MTP_CHAIN_SUMMARY' "$log") fixture(s); the numbers above are reported, not judged — no acceptance bar is set)"
echo; echo "RIDGE MTP CHAIN GATE: PASS"
exit 0
