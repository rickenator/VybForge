#!/usr/bin/env bash
# run_interview_gate.sh — P1.5 gate for tools/interview_infer.vyb and
# tools/configurator_repl.vyb.
#
# No model, no GPU, no torch:
#
#   * interview_infer — against the recording stub (STUB_MODE=changes) the tool
#     must append one JSONL patch line per proposed_change, byte-identical to
#     the Python original's emission rule
#     (json.dumps({"path","op","value","reason"}, separators=(",", ":"))).
#   * the applier consumes them — tools/apply_interview.vyb must turn the emitted
#     patches into out/spec.json + out/system.vyb (the real downstream path).
#   * configurator_repl — one Q/A round prints exactly one contract on stdout.
#
# Usage: ./native/legit/run_interview_gate.sh
set -uo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
. "$root/vybenv.sh" || exit 1   # VYBHOME / VYB / VYB_STDLIB (VybForge#15, rickenator/Vyb#424)
PY="${PY:-}"
if [ -z "$PY" ]; then
  if [ -x "$root/.venv/bin/python" ]; then PY="$root/.venv/bin/python"; else PY=python3; fi
fi
STUB="$root/native/legit/stub_backend.py"
WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT
PORT=8095
GOAL="set the appliance hostname to vyb-appliance-2 and add the curl package"

cd "$root"
fail=0

echo "== interview_infer (stub STUB_MODE=changes)"
STUB_MODE=changes "$PY" "$STUB" "$WORK" $PORT & stub=$!
sleep 0.4
VYBFORGE_BACKEND=openai-chat VYBFORGE_ENDPOINT="http://127.0.0.1:$PORT/v1" \
VYBFORGE_MODEL=test-model VYBFORGE_PROMPT_IN="$GOAL" VYBFORGE_OUT="$WORK/patches.jsonl" \
VYB_STDLIB="$VYB_STDLIB" \
  "$VYB" tools/interview_infer.vyb --module-path tools --module-path native/json >"$WORK/infer.out" 2>"$WORK/infer.err"
sleep 0.4; kill $stub 2>/dev/null; wait $stub 2>/dev/null

echo "  patches emitted:"
sed 's/^/    /' "$WORK/patches.jsonl" 2>/dev/null
echo -n "  emission : "
if "$PY" - "$WORK/patches.jsonl" <<'PY'
import json, sys
expected = [
    json.dumps({"path": "hostname", "op": "replace", "value": "vyb-appliance-2",
                "reason": "user asked for this hostname"}, separators=(",", ":")),
    json.dumps({"path": "pkgs", "op": "add",
                "value": {"name": "curl", "version": "8.5.0", "source": "repo"},
                "reason": "needed to fetch the payload"}, separators=(",", ":")),
]
got = [l for l in open(sys.argv[1]).read().splitlines() if l.strip()]
if got == expected:
    print(f"byte-identical to the Python emission rule ({len(got)} lines)")
else:
    print("DIFFERS"); print("  expected:", expected); print("  got     :", got); sys.exit(1)
PY
then :; else fail=1; fi

echo "  applier  : "
VYBFORGE_PATCHES="$WORK/patches.jsonl" "$VYB" tools/apply_interview.vyb --module-path native/json \
  >"$WORK/apply.out" 2>"$WORK/apply.err"
if [ -s out/spec.json ] && [ -s out/system.vyb ] && grep -q "reproduces spec" "$WORK/apply.out"; then
  echo "    consumed $(wc -l <"$WORK/patches.jsonl") patch line(s) -> out/spec.json + out/system.vyb"
  "$PY" -c "import json;d=json.load(open('out/spec.json'));print('    spec hostname:',d.get('system',{}).get('hostname') or d.get('hostname') or '(nested)')" 2>/dev/null || true
else
  echo "    FAILED (see $WORK/apply.out / $WORK/apply.err)"; sed 's/^/      /' "$WORK/apply.err" | head -5; fail=1
fi

echo
echo "== configurator_repl (one Q/A round against the stub)"
"$PY" "$STUB" "$WORK/repl" $PORT & stub=$!
sleep 0.4
mkdir -p "$WORK/repl"
echo "set the appliance hostname" | VYBFORGE_BACKEND=openai-chat VYBFORGE_ENDPOINT="http://127.0.0.1:$PORT/v1" \
  VYBFORGE_MODEL=test-model VYB_STDLIB="$VYB_STDLIB" \
  "$VYB" tools/configurator_repl.vyb --module-path tools --module-path native/json \
  >"$WORK/repl.out" 2>"$WORK/repl.err"
sleep 0.4; kill $stub 2>/dev/null; wait $stub 2>/dev/null
echo -n "  stdout   : "
if "$PY" - "$WORK/repl.out" <<'PY'
import json, sys
lines = [l for l in open(sys.argv[1]).read().splitlines() if l.strip()]
doc = json.loads(lines[0])
assert doc["kind"] == "question", doc
print("one contract, schema shape ok (kind=%s), stdout lines=%d" % (doc["kind"], len(lines)))
PY
then :; else fail=1; fi

echo
if [ $fail -eq 0 ]; then echo "INTERVIEW GATE: PASS"; else echo "INTERVIEW GATE: FAIL"; fi
exit $fail
