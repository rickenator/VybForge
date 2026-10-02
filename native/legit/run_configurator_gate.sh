#!/usr/bin/env bash
# run_configurator_gate.sh — P1.4 gate for app/configurator.vyb.
#
# Proves the Vyb-native interviewer is a faithful port of the (now deleted)
# Python driver app/configurator.py, with no model and no GPU:
#
#   * request fidelity — for each backend the Vyb driver's request PATH,
#     Authorization header and BODY are byte-identical to the Python driver's,
#     captured in native/legit/fixtures/configurator-baseline/ (produced by the
#     Python driver before it was removed; see doc/PYTHON-CLEANUP.md P1.4).
#   * response fidelity — both drivers turn the stub's reply into the same
#     contract JSON value. The stub answers the Responses shape with a
#     `reasoning` item first, so the driver must skip it and take the typed
#     `output_text` item.
#   * contract validity — the emitted contract validates against
#     config/agent-response.schema.json.
#
# The stub (native/legit/stub_backend.py) is a verification oracle, not
# production code: it records the request it receives and returns a fixed reply.
#
# Usage: ./native/legit/run_configurator_gate.sh
set -uo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
VYB="${VYB:-/home/rick/Projects/Vyb/build/vyb}"
VYB_STDLIB="${VYB_STDLIB:-/home/rick/Projects/Vyb/stdlib}"
PY="${PY:-}"
if [ -z "$PY" ]; then
  if [ -x "$root/.venv/bin/python" ]; then PY="$root/.venv/bin/python"; else PY=python3; fi
fi
BASELINE="$root/native/legit/fixtures/configurator-baseline"
STUB="$root/native/legit/stub_backend.py"
WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT
PORT=8096
INPUT="Set up the appliance: hostname vyb-appliance-2, timezone America/New_York"

cd "$root"
fail=0
for backend in ollama openai-chat openai-responses; do
  rec="$WORK/rec-$backend"; mkdir -p "$rec"
  "$PY" "$STUB" "$rec" $PORT & stub=$!
  sleep 0.4
  if [ "$backend" = openai-chat ]; then ep="http://127.0.0.1:$PORT/v1"; else ep="http://127.0.0.1:$PORT"; fi

  echo "$INPUT" | env VYBFORGE_BACKEND=$backend VYBFORGE_ENDPOINT=$ep \
    VYBFORGE_MODEL=qwen3-test VYBFORGE_API_KEY=test-key \
    VYBFORGE_STRUCTURED_OUTPUT=json_schema VYB_STDLIB="$VYB_STDLIB" \
    "$VYB" app/configurator.vyb --module-path tools --module-path native/json \
    >"$WORK/$backend.stdout" 2>"$WORK/$backend.stderr"
  sleep 0.4; kill $stub 2>/dev/null; wait $stub 2>/dev/null

  echo "=== $backend"
  echo -n "  path : "
  if diff -q "$rec/0.path" "$BASELINE/$backend.path" >/dev/null 2>&1; then echo "same ($(cat "$BASELINE/$backend.path"))"
  else echo "DIFFER vyb=$(cat "$rec/0.path" 2>/dev/null) baseline=$(cat "$BASELINE/$backend.path")"; fail=1; fi
  echo -n "  auth : "
  if diff -q "$rec/0.auth" "$BASELINE/$backend.auth" >/dev/null 2>&1; then echo "same ('$(cat "$BASELINE/$backend.auth")')"
  else echo "DIFFER vyb='$(cat "$rec/0.auth")' baseline='$(cat "$BASELINE/$backend.auth")'"; fail=1; fi
  echo -n "  body : "
  if diff -q "$rec/0.body" "$BASELINE/$backend.body" >/dev/null 2>&1; then echo "byte-identical ($(wc -c <"$rec/0.body") bytes)"
  else echo "DIFFER ($(wc -c <"$rec/0.body") vs $(wc -c <"$BASELINE/$backend.body") bytes)"; fail=1; fi
  echo -n "  json : "
  if "$PY" - "$WORK/$backend.stdout" "$BASELINE/$backend.stdout" <<'PY'
import json, sys
def contract(text):
    # the Python original printed its banner and "you> " prompt on stdout; the
    # Vyb port keeps stdout pure JSON, so strip them before comparing values
    keep = [l for l in text.splitlines() if not l.startswith("VybForge configurator using")]
    return json.loads("\n".join(keep).replace("you> ", ""))
a, b = contract(open(sys.argv[1]).read()), contract(open(sys.argv[2]).read())
print("same contract value" if a == b else "DIFFERENT CONTRACT VALUE")
sys.exit(0 if a == b else 1)
PY
  then :; else fail=1; fi
  echo -n "  schema: "
  if "$PY" - "$WORK/$backend.stdout" <<'PY'
import json, sys, jsonschema
schema = json.load(open("config/agent-response.schema.json"))
jsonschema.Draft7Validator(schema).validate(json.load(open(sys.argv[1])))
print("contract schema-valid")
PY
  then :; else echo "INVALID"; fail=1; fi
done

echo
if [ $fail -eq 0 ]; then echo "CONFIGURATOR GATE: PASS"; else echo "CONFIGURATOR GATE: FAIL"; fi
exit $fail
