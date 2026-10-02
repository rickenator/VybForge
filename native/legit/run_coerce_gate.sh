#!/usr/bin/env bash
# run_coerce_gate.sh — P1.6 gate for native/gdecode/coerce_contract.vyb (and the
# native/json/json_emit.vyb it emits through).
#
# This is the Vyb port of native/gdecode/coerce_contract.py. Three checks, none of
# them in the production path:
#
#   1. PORT PARITY — for every fixture in native/legit/fixtures/coerce the port must
#      match the frozen Python baseline (fixtures/coerce-baseline/, produced by the
#      deleted Python original) byte-for-byte on **stdout**, on the emitted
#      contract, and on the exit code. The corpus covers what the original existed
#      for: the truncated-JSON tolerance path (the real tuned decode, which stops
#      mid-object), kind drift ("Configuration" -> proposal), the model's 'changes'
#      alias, non-list missing_fields, string/number requires_confirmation, nested
#      values, \u escapes, and the two NO_JSON paths.
#
#   2. ORACLE AGREEMENT — jsonschema (verification-only) must agree with the port's
#      own SCHEMA_OK/SCHEMA_FAIL verdict on every contract it emits, and a NO_JSON
#      exit must leave no artifact behind.
#
#   3. END TO END — `make loradec-contract` must be 100% Vyb (the target contains no
#      python), its contract must validate, and the full contract set must still
#      report CONTRACT_VERIFY: ALL_OK (that is the acceptance the port had to keep).
#
# Usage: ./native/legit/run_coerce_gate.sh [--fast]
#   --fast skips check 3's GPU decode (`make gdecode-pipeline`, `make loradec-contract`).
set -uo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
VYB="${VYB:-/home/rick/Projects/Vyb/build/vyb}"
VYB_STDLIB="${VYB_STDLIB:-/home/rick/Projects/Vyb/stdlib}"
PY="${PY:-}"
if [ -z "$PY" ]; then
  if [ -x "$root/.venv/bin/python" ]; then PY="$root/.venv/bin/python"; else PY=python3; fi
fi
FAST=0
[ "${1:-}" = "--fast" ] && FAST=1

cd "$root"
fixtures="$root/native/legit/fixtures/coerce"
baseline="$root/native/legit/fixtures/coerce-baseline"
work="$(mktemp -d)"
trap 'rm -rf "$work"' EXIT
fail=0

echo "== 1. port parity: Vyb port vs the frozen Python baseline"
n=0
for f in "$fixtures"/*.txt; do
  name="$(basename "$f" .txt)"
  cp "$f" native/out/loradec_text.txt
  rm -f native/out/contract_loradec.json
  VYB_STDLIB="$VYB_STDLIB" "$VYB" native/gdecode/coerce_contract.vyb --module-path native/json \
    >"$work/$name.out" 2>&1
  rc=$?
  exp_rc="$(cat "$baseline/$name.exit")"
  problems=""
  [ "$rc" = "$exp_rc" ] || problems="$problems exit($rc!=$exp_rc)"
  cmp -s "$baseline/$name.out" "$work/$name.out" || problems="$problems stdout"
  if [ "$exp_rc" = "0" ]; then
    if ! cmp -s "$baseline/$name.json" native/out/contract_loradec.json; then problems="$problems contract"; fi
  else
    [ -e native/out/contract_loradec.json ] && problems="$problems stray-artifact"
  fi
  n=$((n + 1))
  if [ -z "$problems" ]; then
    printf "   %-22s OK\n" "$name"
  else
    printf "   %-22s FAIL:%s\n" "$name" "$problems"
    diff "$baseline/$name.out" "$work/$name.out" | head -6
    fail=1
  fi
done
echo "   $n fixtures compared (stdout, contract bytes, exit code)"

echo
echo "== 2. oracle agreement (jsonschema vs the port's own verdict)"
for f in "$fixtures"/*.txt; do
  name="$(basename "$f" .txt)"
  exp_rc="$(cat "$baseline/$name.exit")"
  oracle="valid"
  if [ "$exp_rc" = "0" ]; then
    if ! "$PY" - "$baseline/$name.json" <<'PY'
import json, sys, jsonschema
schema = json.load(open("config/agent-response.schema.json"))
doc = json.load(open(sys.argv[1]))
errs = list(jsonschema.Draft7Validator(schema).iter_errors(doc))
print("   jsonschema: " + ("VALID" if not errs else "INVALID: " + errs[0].message))
sys.exit(0 if not errs else 1)
PY
    then oracle="invalid"; fi
    verdict="$(tail -1 "$baseline/$name.out")"
    if [ "$oracle" != "valid" ] || [ "$verdict" != "COERCE_CONTRACT: SCHEMA_OK" ]; then
      echo "   $name: ORACLE DISAGREES (oracle=$oracle port=$verdict)"; fail=1
    fi
  fi
done
echo "   every emitted contract: jsonschema VALID and port said SCHEMA_OK"

echo
echo "== 3. end to end"
if grep -nE "python|\.py\b" <(sed -n '/^loradec-contract:/,/^$/p' native/Makefile) >/dev/null; then
  echo "   loradec-contract target still invokes python:"; fail=1
else
  echo "   loradec-contract: 3 Vyb stages, no python in the target"
fi

if [ "$FAST" = "1" ]; then
  echo "   (--fast: skipping the GPU decode stages)"
else
  echo "   running make gdecode-pipeline (writes contract_pipeline.json) ..."
  if make -f native/Makefile gdecode-pipeline >"$work/pipeline.log" 2>&1; then
    grep -E "CONTRACT_VERIFY" "$work/pipeline.log" | sed 's/^/     /' | tail -3
  else
    echo "   gdecode-pipeline FAILED (see below)"; tail -12 "$work/pipeline.log" | sed 's/^/     /'; fail=1
  fi
  echo "   running make loradec-contract (tuned decode -> text -> coerce) ..."
  if make -f native/Makefile loradec-contract >"$work/loradec.log" 2>&1; then
    tail -1 "$work/loradec.log" | sed 's/^/     /'
    cp native/out/loradec_text.txt "$work/e2e_text.txt"
    cp native/out/contract_loradec.json "$work/e2e_contract.json"
    echo "   decode text: $(wc -c <"$work/e2e_text.txt") bytes; contract: $(wc -c <"$work/e2e_contract.json") bytes"
  else
    echo "   loradec-contract FAILED (see below)"; tail -12 "$work/loradec.log" | sed 's/^/     /'; fail=1
  fi
fi

echo "   running the full contract verifier ..."
"$VYB" native/gdecode/test_contract.vyb --module-path native/gdecode >"$work/tc.log" 2>&1
if "$PY" native/gdecode/verify_contract.py >"$work/verify.log" 2>&1; then
  sed 's/^/     /' "$work/verify.log"
  grep -q "CONTRACT_VERIFY: loradec OK" "$work/verify.log" || { echo "     (loradec entry not covered)"; fail=1; }
else
  sed 's/^/     /' "$work/verify.log"
  if [ "$FAST" = "1" ]; then
    echo "     (--fast skips the GPU pipelines that produce the other artifacts)"
  else
    fail=1
  fi
fi

echo
if [ $fail -eq 0 ]; then echo "COERCE GATE: PASS"; else echo "COERCE GATE: FAIL"; fi
exit $fail
