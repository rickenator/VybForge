#!/usr/bin/env bash
# P4.13 the MTP DRAFT HEAD vs the captured oracle (VybForge#10 phase 4, W4).
#
# The draft head consumes (the normalised hidden at position t, the embedding of token t+1) and predicts
# t+2. The fixtures already record the oracle's greedy pick at every position, so this needs no new
# capture — it feeds the head the ORACLE's own hidden row and the ORACLE's own next token and requires
# its per-step top1 to be the oracle's pick for t+2 under the fixture's rule. Using the oracle's hidden
# takes the main pass out of the loop, so a mismatch is this new block's, not accumulation.
#
# NOT a substitute for P4.12: this gates the draft head alone (with the oracle's hidden as input); P4.12
# gates the main 64-block pass, and "the draft head does not change the main logits" is exactly what
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

env -u PYTHONPATH "$py" native/tools/ridge_mtp_verify.py "$@" >"$log" 2>&1
rc=$?
grep -E '^RIDGE_MTP_VERIFY' "$log" | sed 's/^/      /'
if grep -q '^RIDGE_MTP_VERIFY_SKIP' "$log"; then
  step "MTP draft head" "SKIP ($(grep -m1 '^RIDGE_MTP_VERIFY_SKIP' "$log" | sed 's/^RIDGE_MTP_VERIFY_SKIP //'))"
  echo; echo "RIDGE MTP GATE: SKIP (nothing ran)"; exit 0
fi
if ! { grep -q '^RIDGE_MTP_VERIFY_DONE' "$log" && [ "$rc" = "0" ]; }; then
  step "MTP draft head" "FAIL (see $log)"
  echo; echo "RIDGE MTP GATE: FAIL"; exit 1
fi
agree="$(grep -oE 'agree=[0-9]+/[0-9]+' "$log" | tail -1)"
step "MTP draft head vs the oracle" "PASS ($agree steps, rope_pos=$(grep -oE 'rope_pos=[0-9]+' "$log" | tail -1 | cut -d= -f2))"

# ── the CHECKER's tooth: the same run compared OFF BY ONE must MISS ─────────────────────────────
# Without this, a checker that always agrees would make the gate decoration. It costs no GPU run.
neg="$work/verify_offby.log"
env -u PYTHONPATH VYBFORGE_MTP_SKIP_RUN=1 VYBFORGE_MTP_OFFBY=1 "$py" native/tools/ridge_mtp_verify.py "$@" >"$neg" 2>&1
if grep -q '^RIDGE_MTP_VERIFY_DONE' "$neg"; then
  step "tooth (compared off by one)" "FAIL (the checker agrees even when the comparison is shifted)"
  echo; echo "RIDGE MTP GATE: FAIL"; exit 1
fi
step "tooth (compared off by one)" "PASS (missed, as it must)"
echo; echo "RIDGE MTP GATE: PASS"
exit 0
