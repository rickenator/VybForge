#!/usr/bin/env bash
# run_gguf_gate.sh — P1.7 gate for native/gguf/mk_fixture.vyb.
#
# The fixture generator used to be the Python script native/gguf/mk_fixture.py.
# This gate proves the Vyb port is a drop-in for it:
#
#   1. BYTE PARITY — the port's file must be byte-identical to the frozen
#      Python-generated fixture (native/legit/fixtures/gguf/test.gguf.baseline,
#      sha256 checked below) and its report line must be byte-identical too.
#   2. ORACLE — the Python verifier that asserts the *expected* fixture layout
#      (header, kv, tensor index, offsets) must still pass on the port's file, and
#      the Vyb q4_0 dequant slice must still run green over it.
#   3. TARGET — `make gguf` (which now generates through the Vyb port) must be
#      green, and the Makefile must no longer mention the Python generator.
#
# Usage: ./native/legit/run_gguf_gate.sh
set -uo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
. "$root/vybenv.sh" || exit 1   # VYBHOME / VYB / VYB_STDLIB (VybForge#15, rickenator/Vyb#424)
PY="${PY:-}"
if [ -z "$PY" ]; then
  if [ -x "$root/.venv/bin/python" ]; then PY="$root/.venv/bin/python"; else PY=python3; fi
fi
fixture="$root/native/legit/fixtures/gguf"
# sha256 of the Python-generated fixture this port must reproduce exactly.
want_sha="f23fc38494499a0f27ec1bcb4ee162db893c22be11b33ee17e2a8779ef0a2191"
fail=0

cd "$root"

echo "== 1. byte parity: Vyb generator vs the frozen Python output"
got_sha="$(sha256sum "$fixture/test.gguf.baseline" | cut -d' ' -f1)"
if [ "$got_sha" != "$want_sha" ]; then
  echo "   baseline fixture sha changed: $got_sha != $want_sha"; fail=1
fi
rm -f native/gguf/test.gguf
VYB_STDLIB="$VYB_STDLIB" "$VYB" native/gguf/mk_fixture.vyb >/tmp/gguf_gate.out 2>&1
rc=$?
if [ "$rc" != "0" ]; then
  echo "   generator exited $rc:"; sed 's/^/     /' /tmp/gguf_gate.out; fail=1
elif cmp -s "$fixture/test.gguf.baseline" native/gguf/test.gguf; then
  echo "   fixture: byte-identical ($(wc -c <native/gguf/test.gguf) bytes, sha256 $(sha256sum native/gguf/test.gguf | cut -c1-16)…)"
else
  echo "   fixture DIFFERS:"; cmp "$fixture/test.gguf.baseline" native/gguf/test.gguf | sed 's/^/     /'; fail=1
fi
if cmp -s "$fixture/mk_fixture.python.out" /tmp/gguf_gate.out; then
  echo "   report:  byte-identical ($(cat /tmp/gguf_gate.out))"
else
  echo "   report DIFFERS:"; diff "$fixture/mk_fixture.python.out" /tmp/gguf_gate.out | sed 's/^/     /'; fail=1
fi

echo
echo "== 2. oracle: the Python verifier still accepts the port's fixture"
"$VYB" native/gguf/parse_gguf.vyb >/dev/null 2>&1 || { echo "   parse_gguf.vyb FAILED"; fail=1; }
if out="$("$PY" native/gguf/verify_gguf.py 2>&1)"; then
  echo "   $(echo "$out" | tail -1)"
else
  echo "   verify_gguf.py FAILED:"; echo "$out" | tail -5 | sed 's/^/     /'; fail=1
fi
if out="$("$VYB" native/gguf/dequant_gguf.vyb 2>&1)"; then
  echo "   q4_0 dequant: $(echo "$out" | tail -1)"
else
  echo "   dequant_gguf.vyb FAILED:"; echo "$out" | tail -5 | sed 's/^/     /'; fail=1
fi

echo
echo "== 3. target: make gguf"
if grep -n "mk_fixture\.py" native/Makefile >/dev/null; then
  echo "   Makefile still invokes the Python generator:"; fail=1
else
  echo "   Makefile references no mk_fixture.py"
fi
if out="$(make -f native/Makefile gguf 2>&1)"; then
  echo "$out" | grep -E "GGUF_PARSE_VERIFY|wrote " | sed 's/^/   /'
  echo "   make gguf: green"
else
  echo "   make gguf FAILED:"; echo "$out" | tail -8 | sed 's/^/     /'; fail=1
fi

echo
if [ $fail -eq 0 ]; then echo "GGUF FIXTURE GATE: PASS"; else echo "GGUF FIXTURE GATE: FAIL"; fi
exit $fail
