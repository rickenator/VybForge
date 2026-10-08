#!/usr/bin/env bash
# run_kvbuild_gate.sh — Phase 2 gate for native/train/_build_kv.vyb (P2.1c).
#
# The Python it replaces (native/train/_build_kv.py) rebuilds kvresp_train_kv.vyb from the
# committed kvresp_train.vyb by (A) inserting the two per-token helpers, (B) extending the
# ASLB declaration + adding the CK/CV/RSHI allocations, and (C) swapping the batched layer
# forward for the per-token response loop. Checks:
#
#   1. PARITY   — the port must reproduce the oracle's output byte-for-byte, pinned by
#                 native/legit/fixtures/kvresp_train_kv_baseline.sha256
#                 (86256cd6…, 103,912 B, 1,249 lines), and report the same four stdout lines
#                 (line count + the three structural checks all True).
#   2. FRONT-END— `--emit-llvm` on the generated driver must complete semantic analysis
#                 (codegen for the CUDA driver succeeds; linking is not attempted).
#   3. ORACLE   — if a `python3` is on PATH (the oracle is stdlib-only, no venv needed), run
#                 it in a scratch tree and require byte-identity with the port AND that its
#                 hash still matches the frozen baseline (a stale fixture then shows up).
#   4. NO CLOBBER — the committed native/train/kvresp_train_kv.vyb must be untouched.
#
# Recorded drift, not reconciled: the committed native/train/kvresp_train_kv.vyb has 1,291
# lines while today's generator emits 1,247 — it came from an older revision of the source
# or was edited. Same class as the gen_kv_train (513, 429) drift; see doc/P2-TRAINER-PLAN.md.
#
# Usage: ./native/legit/run_kvbuild_gate.sh
set -uo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
. "$root/vybenv.sh" || exit 1

WORK="${TMPDIR:-/tmp}/kvbuild-gate.$$"
mkdir -p "$WORK/native/train"
fail=0
step() { printf '%-52s %s\n' "$1" "$2"; }

cd "$root"
fixture="native/legit/fixtures/kvresp_train_kv_baseline.sha256"
want_sha=""; want_size=""
if [ -s "$fixture" ]; then
  want_sha="$(awk 'NR==1{print $1}' "$fixture")"
  want_size="$(awk 'NR==1{print $2}' "$fixture")"
else
  step "baseline fixture" "MISSING ($fixture)"; echo; echo "KVBUILD GATE: FAIL"; exit 1
fi
committed_before="$(sha256sum native/train/kvresp_train_kv.vyb | cut -d' ' -f1)"

# 1. parity
VYBFORGE_KVBUILD_DST="$WORK/gen.vyb" "$VYB" native/train/_build_kv.vyb >"$WORK/log" 2>&1
rc=$?
if [ $rc -ne 0 ]; then
  step "_build_kv.vyb" "FAIL (exit $rc)"; tail -4 "$WORK/log" | sed 's/^/      /'; fail=1
else
  got_lines="$(grep -o 'lines [0-9]*' "$WORK/log" | grep -o '[0-9]*')"
  ok=1
  for want in "helpers ok: True" "per-token fwd ok: True" "tail ok: True"; do
    grep -q "^$want\$" "$WORK/log" || { step "stdout: $want" "FAIL"; ok=0; fail=1; }
  done
  [ "$got_lines" = "1249" ] || { step "stdout line count" "FAIL (got ${got_lines:-none}, want 1249)"; ok=0; fail=1; }
  got_sha="$(sha256sum "$WORK/gen.vyb" | cut -d' ' -f1)"
  got_size="$(wc -c < "$WORK/gen.vyb")"
  if [ "$got_sha" = "$want_sha" ] && [ "$got_size" = "$want_size" ]; then
    [ $ok -eq 1 ] && step "generated driver parity" "byte-identical ($got_size B, 1249 lines, ${got_sha:0:8}…)"
  else
    step "generated driver parity" "FAIL (got $got_size B ${got_sha:0:8}…, want $want_size B ${want_sha:0:8}…)"
    fail=1
  fi
fi

# 2. front-end (semantics + LLVM codegen; no CUDA link)
if "$VYB" --emit-llvm "$WORK/gen.vyb" >"$WORK/ll.log" 2>&1 && grep -q "Semantic analysis completed" "$WORK/ll.log"; then
  step "generated driver front-end" "semantic analysis + codegen ok"
else
  step "generated driver front-end" "FAIL"; tail -4 "$WORK/ll.log" | sed 's/^/      /'; fail=1
fi

# 3. optional live oracle cross-check (stdlib python3; runs in a scratch tree)
if command -v python3 >/dev/null 2>&1; then
  ln -sf "$root/native/train/kvresp_train.vyb" "$WORK/native/train/kvresp_train.vyb"
  if (cd "$WORK" && python3 "$root/native/train/_build_kv.py" >"$WORK/py.log" 2>&1); then
    py_sha="$(sha256sum "$WORK/native/train/kvresp_train_kv.vyb" | cut -d' ' -f1)"
    if [ "$py_sha" != "$want_sha" ]; then
      step "oracle cross-check (python3)" "FAIL (oracle moved; fixture stale)"
      fail=1
    elif cmp -s "$WORK/native/train/kvresp_train_kv.vyb" "$WORK/gen.vyb"; then
      step "oracle cross-check (python3)" "ok (port == fresh oracle run)"
    else
      step "oracle cross-check (python3)" "FAIL (port differs from oracle)"; fail=1
    fi
  else
    step "oracle cross-check (python3)" "skipped (oracle run failed)"; tail -3 "$WORK/py.log" | sed 's/^/      /'
  fi
else
  step "oracle cross-check (python3)" "skipped (no python3 on PATH)"
fi

# 4. nothing tracked was touched
if [ "$(sha256sum native/train/kvresp_train_kv.vyb | cut -d' ' -f1)" = "$committed_before" ]; then
  step "committed driver unmodified" "ok"
else
  step "committed driver unmodified" "FAIL"; fail=1
fi

rm -rf "$WORK"
echo
if [ $fail -eq 0 ]; then echo "KVBUILD GATE: PASS"; else echo "KVBUILD GATE: FAIL"; fi
exit $fail
