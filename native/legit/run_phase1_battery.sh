#!/usr/bin/env bash
# run_phase1_battery.sh — Phase-1 cleanup evidence in one command.
#
# Runs every P1.x acceptance check and prints a one-line verdict per step, so the
# "Phase 1 complete" claim in doc/PYTHON-CLEANUP.md can be re-produced rather than
# trusted. Python is used only where the gate itself is an oracle comparison.
#
# NOTE: if a step that was green suddenly fails with no source change, check the
# Vyb toolchain first — `build/vyb` gets rebuilt out from under a session by the
# implementation agent, and a half-finished rebuild makes Vyb programs fail in
# ways that look like repo bugs. Confirm with:
#     ls -l --time-style=full-iso $VYB && git -C $(dirname $(dirname $VYB)) log --oneline -1
# then re-run.
#
# Usage: ./native/legit/run_phase1_battery.sh
set -uo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
. "$root/vybenv.sh" || exit 1   # VYBHOME / VYB / VYB_STDLIB (VybForge#15, rickenator/Vyb#424)
export VYB_STDLIB
PY="${PY:-}"
if [ -z "$PY" ]; then
  if [ -x "$root/.venv/bin/python" ]; then PY="$root/.venv/bin/python"; else PY=python3; fi
fi
fail=0
step() { printf '%-46s %s\n' "$1" "$2"; }

cd "$root"

echo "Phase 1 battery — $(date '+%F %T')"
echo "toolchain: $VYB  ($(git -C "$(dirname "$(dirname "$VYB")")" log --oneline -1 2>/dev/null || echo 'not a git checkout'))"
mtime="$(date -r "$VYB" '+%F %T')"
echo "binary mtime: $mtime"
echo

out="$(make -f native/Makefile schemacheck 2>&1)"
if echo "$out" | grep -q "SCHEMA_TEST: OK"; then step "P1.1 schemacheck" "$(echo "$out" | grep -o 'SCHEMA_TEST: OK.*')"; else step "P1.1 schemacheck" "FAIL"; fail=1; fi

out="$("$VYB" tools/apply_interview.vyb --module-path native/json 2>&1)"
if echo "$out" | grep -q "reproduces spec"; then step "P1.2 apply_interview" "ok (reproduces spec)"; else step "P1.2 apply_interview" "FAIL"; echo "$out" | tail -4 | sed 's/^/      /'; fail=1; fi

n="$(./native/legit/run_repair_proposal.sh 2>&1 | grep -cE 'MODEL-BOUNDARY: (ACCEPT|REJECT|HUMAN-REQUIRED)')"
if [ "$n" = "3" ]; then step "P1.3 repair boundary" "ok (3/3 decisions)"; else step "P1.3 repair boundary" "FAIL (only $n/3)"; fail=1; fi

for pair in "P1.4:run_configurator_gate.sh" "P1.5:run_interview_gate.sh" "P1.6:run_coerce_gate.sh" "P1.7:run_gguf_gate.sh"; do
  tag="${pair%%:*}"; script="${pair##*:}"
  out="$(./native/legit/$script 2>&1)"
  if echo "$out" | tail -1 | grep -q "PASS"; then
    step "$tag ${script#run_}" "$(echo "$out" | tail -1)"
  else
    step "$tag ${script#run_}" "FAIL"; echo "$out" | tail -8 | sed 's/^/      /'; fail=1
  fi
done

out="$("$VYB" native/json/test_json.vyb --module-path native/json 2>&1)"
if echo "$out" | grep -q "DONE"; then step "json unit test" "DONE"; else step "json unit test" "FAIL"; fail=1; fi

out="$("$PY" native/gdecode/verify_contract.py 2>&1)"
if echo "$out" | tail -1 | grep -q "ALL_OK"; then step "contract verifier" "ALL_OK"; else step "contract verifier" "FAIL"; fail=1; fi

if grep -lE "\bpython" run.sh run-vyb.sh vyb-run.sh >/dev/null 2>&1; then
  step "production run scripts" "FAIL (python referenced)"; fail=1
else
  step "production run scripts" "0 python (run.sh, run-vyb.sh, vyb-run.sh)"
fi
if ls tools/*.py app/*.py tests/*.py native/gguf/mk_fixture.py native/gdecode/coerce_contract.py >/dev/null 2>&1; then
  step "ported driver dirs" "FAIL (python driver still present)"; fail=1
else
  step "ported driver dirs" "no python left in tools/ app/ tests/"
fi

echo
# The P1.2/P1.5 steps exercise the real downstream path, which renders the tracked
# example artifacts out/spec.json + out/system.vyb (and the compiled bin) from the
# gate's fixture patches. Put them back so running the battery leaves the tree as it
# found it — the evidence is the assertions above, not the example contents.
if git -C "$root" rev-parse --git-dir >/dev/null 2>&1; then
  git -C "$root" restore out/ 2>/dev/null || true
fi
if [ $fail -eq 0 ]; then echo "PHASE 1 BATTERY: PASS"; else echo "PHASE 1 BATTERY: FAIL"; fi
exit $fail
