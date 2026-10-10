#!/usr/bin/env bash
# P4.13 the MTP DRAFT HEAD vs the captured oracle (VybForge#10 phase 4, W4).
#
# The draft head consumes (the normalised hidden at position t, the embedding of token t+1) and predicts
# t+2. The fixtures already record the oracle's greedy pick at every position, so this needs no new
# capture — it feeds the head the ORACLE's own hidden row and the ORACLE's own next token and requires
# its per-step top1 to agree with the oracle's pick under the fixture's own rule (equality where the
# top-2 margin clears `margin_bar`, membership of {top1,top2} below it). Using the oracle's hidden takes
# the main pass out of the loop, so a disagreement is this block's, not accumulation.
#
# THE CRITERION IS A RULE-AWARE FLOOR, NOT EXACT MATCH. The head is an approximate separate head reusing
# the main model's output.weight — not a copy of the main path — so its per-step agreement with the
# oracle is below 100% BY CONSTRUCTION, and an exact-match criterion could only ever report FAIL on a
# correct engine (it did, for weeks). The floors are pinned per fixture and DERIVED from measurement:
# 17/19 measured on the counting fixture → floor 15/19; 4/4 on the capital one → floor 3/4. The
# derivation, the slack's calibration and what the floor must and must not absorb are in
# native/tools/ridge_mtp_verify.py's THE ACCEPTANCE CRITERION block and doc/QWEN35-MTP-HARVEST.md §9.
#
# STEP 1 IS THE FLOOR'S OWN TOOTH, and it is what makes this a gate rather than a printout: the same
# criterion is run against WRONG inputs, which must land BELOW the floor. Two of them, because they are
# two different defect classes (measured on the counting fixture, floor 15/19):
#   * the hidden ZEROED (1/19) — the head's whole h path broken, which is what a wrong operand or a
#     mis-staged weight looks like;
#   * the hidden rows shifted BACKWARDS by one (10/19) — the off-by-one row pairing this gate exists for.
# A floor nobody has watched fail is decoration. Those runs' verdicts are inverted on purpose (rc 0 =
# the criterion caught it), and they run FIRST so the good picks are what is left in
# native/out/prefill_top1_vyb.txt afterwards.
# NOT a defect, and measured so nobody re-opens it: shifting the rows FORWARD by one does NOT fall
# (19/19 on the counting fixture, 4/4 on the capital one) — the head uses h as context rather than as an
# exact-position lookup. See doc/QWEN35-MTP-HARVEST.md §9 for that measurement and its limits.
#
# NOT a substitute for P4.12: this gates the draft head alone (with the oracle's hidden as input); P4.12
# gates the main 65-block pass, and "the draft head does not change the main logits" is exactly what
# P4.12 staying green proves.
#
# SKIPs (never PASSes) without the toolchain, the model, the tensor index, the rope table or the fixture.
set -u
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$root"

step() { printf '%-46s %s\n' "$1" "$2"; }
echo "Ridge MTP draft-head gate — $(date '+%F %T')"
echo

py=""
for cand in "${RIDGEF_PY:-}" "$root/.venv/bin/python" python3; do
  if [ -n "$cand" ] && env -u PYTHONPATH "$cand" -c 'import numpy' >/dev/null 2>&1; then py="$cand"; break; fi
done
if [ -z "$py" ]; then
  step "reference interpreter" "FAIL (no python with numpy)"
  echo; echo "RIDGE MTP GATE: FAIL"; exit 1
fi

work="${VYBFORGE_RIDGE_MTP_OUTDIR:-$root/native/out/ridge_mtp}"
mkdir -p "$work"
log="$work/verify.log"
tlog="$work/floor_tooth.log"
fixtures="${VYBFORGE_MTP_FIXTURES:-the_capital_of_france_is 1_2_3_4_5_6_7}"

# ---- 1. the floor's own teeth: WRONG inputs must land below it (they run first, see above) ---------
tooth_fx="${VYBFORGE_MTP_TOOTH_FIXTURES:-1_2_3_4_5_6_7}"
for mode in "VYBFORGE_MTP_HIDDEN_ZERO=1" "VYBFORGE_MTP_HIDDEN_SHIFT=-1"; do
  tlog="$work/floor_tooth_$(printf '%s' "$mode" | tr -dc 'A-Za-z0-9_').log"
  env -u PYTHONPATH $mode "$py" native/tools/ridge_mtp_verify.py $tooth_fx >"$tlog" 2>&1
  trc=$?
  grep -E '^RIDGE_MTP_VERIFY_(ACCEPT|CONTROL)' "$tlog" | sed 's/^/      /'
  if grep -q '^RIDGE_MTP_VERIFY_SKIP' "$tlog"; then
    step "acceptance floor vs a wrong input" "SKIP ($(grep -m1 '^RIDGE_MTP_VERIFY_SKIP' "$tlog" | sed 's/^RIDGE_MTP_VERIFY_SKIP //'))"
    echo; echo "RIDGE MTP GATE: SKIP (nothing ran)"; exit 0
  fi
  if ! { grep -q '^RIDGE_MTP_VERIFY_CONTROL_OK' "$tlog" && [ "$trc" = "0" ]; }; then
    step "acceptance floor vs a wrong input" "FAIL ($mode: see $tlog)"
    echo; echo "RIDGE MTP GATE: FAIL"; exit 1
  fi
done
step "acceptance floor vs wrong inputs" "PASS (zeroed and backward-shifted hidden both land below it)"

# ---- 2. the head against the oracle, judged by that floor ------------------------------------------
env -u PYTHONPATH "$py" native/tools/ridge_mtp_verify.py $fixtures "$@" >"$log" 2>&1
rc=$?
grep -E '^RIDGE_MTP_VERIFY' "$log" | sed 's/^/      /'
if grep -q '^RIDGE_MTP_VERIFY_SKIP' "$log"; then
  step "MTP draft head" "SKIP ($(grep -m1 '^RIDGE_MTP_VERIFY_SKIP' "$log" | sed 's/^RIDGE_MTP_VERIFY_SKIP //'))"
  echo; echo "RIDGE MTP GATE: SKIP (nothing ran)"; exit 0
fi
if ! { grep -q '^RIDGE_MTP_VERIFY_DONE' "$log" && [ "$rc" = "0" ]; }; then
  step "MTP draft head vs the oracle" "FAIL (see $log)"
  echo; echo "RIDGE MTP GATE: FAIL"; exit 1
fi
step "MTP draft head vs the oracle" "PASS (rule-aware floor: $(grep -oE 'fixture=[^ ]+ agree=[0-9]+/[0-9]+ floor=[0-9]+/[0-9]+' "$log" | tr '\n' '; '))"
# The checker's off-by-one tooth runs INSIDE the verifier on the same completed run (per fixture), so
# there is no second invocation here — its line is printed above.
echo; echo "RIDGE MTP GATE: PASS"
exit 0
