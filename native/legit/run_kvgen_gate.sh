#!/usr/bin/env bash
# run_kvgen_gate.sh — Phase 2 gate for native/train/gen_kv_train.vyb (P2.1a).
#
# The Python it replaces (native/train/gen_kv_train.py) rewrites the committed S=93
# driver into a driver for another (S, NCTX). Checks:
#
#   1. REGRESSION IDENTITY — `93 9` must rewrite native/train/kvresp_train_kv.vyb back
#      to itself, byte-for-byte (sha256 7f738e41…, 107,836 B). That is the property the
#      Python had and the only self-contained oracle in-tree.
#   2. FROZEN BASELINE — `513 429` must produce exactly what the Python produced:
#      native/legit/fixtures/kvresp_train_p429_S513_N429.vyb (75ffccfd…, 108,165 B),
#      captured from `python3 native/train/gen_kv_train.py 513 429 <out>` before the
#      port. Every replacement in that path is a no-op-or-rewrite sequence, so a
#      dropped or reordered substitution can only show up as a byte difference here.
#   3. ORACLE CROSS-CHECK (optional; reference validation only, per the Oracle policy)
#      — if a `python3` is on PATH, re-derive both parameterizations with it and require
#      the port to agree. Skipped with a note when absent; the fixture still holds.
#   4. NO CLOBBER — the template must be unchanged by the run.
#
# Recorded drift, deliberately not reconciled: the committed
# native/train/kvresp_train_p429.vyb (80f82e34…, 104,685 B) is NOT what this generator
# produces for (513, 429) — it predates the current script. Same class of
# generator-vs-artifact drift as the _build_kv.py line count; see doc/P2-TRAINER-PLAN.md.
#
# Usage: ./native/legit/run_kvgen_gate.sh
set -uo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
. "$root/vybenv.sh" || exit 1

WORK="${TMPDIR:-/tmp}/kvgen-gate.$$"
mkdir -p "$WORK"
fail=0
step() { printf '%-52s %s\n' "$1" "$2"; }

cd "$root"
tmpl="native/train/kvresp_train_kv.vyb"
fixture="native/legit/fixtures/kvresp_train_p429_S513_N429.vyb"
tmpl_sha_before="$(sha256sum "$tmpl" | cut -d' ' -f1)"

gen() { # S NC out
  VYBFORGE_KV_S="$1" VYBFORGE_KV_NCTX="$2" VYBFORGE_KV_OUT="$3" \
    "$VYB" native/train/gen_kv_train.vyb >"$3.log" 2>&1
}

# 1. regression identity
if gen 93 9 "$WORK/r93.vyb"; then
  if cmp -s "$WORK/r93.vyb" "$tmpl"; then
    step "regression identity (93 9)" "byte-identical ($(wc -c < "$tmpl") B)"
  else
    step "regression identity (93 9)" "FAIL (template not reproduced)"; fail=1
  fi
else
  step "regression identity (93 9)" "FAIL (exit $?)"; tail -3 "$WORK/r93.vyb.log" | sed 's/^/      /'; fail=1
fi

# 2. frozen baseline
if [ ! -s "$fixture" ]; then
  step "frozen baseline fixture" "MISSING ($fixture)"; fail=1
elif gen 513 429 "$WORK/r429.vyb"; then
  if cmp -s "$WORK/r429.vyb" "$fixture"; then
    step "frozen baseline (513 429)" "byte-identical ($(wc -c < "$fixture") B)"
  else
    step "frozen baseline (513 429)" "FAIL"; fail=1
  fi
else
  step "frozen baseline (513 429)" "FAIL (exit $?)"; tail -3 "$WORK/r429.vyb.log" | sed 's/^/      /'; fail=1
fi

# 3. optional oracle cross-check (reference validation, stdlib only)
if command -v python3 >/dev/null 2>&1; then
  ok=1
  while read -r s nc tag; do
    python3 native/train/gen_kv_train.py "$s" "$nc" "$WORK/py_$tag.vyb" >/dev/null 2>&1 || ok=0
    cmp -s "$WORK/py_$tag.vyb" "$WORK/$tag.vyb" || ok=0
  done <<'PAIRS'
93 9 r93
513 429 r429
PAIRS
  if [ $ok -eq 1 ]; then step "oracle cross-check (python3, stdlib)" "ok"
  else step "oracle cross-check (python3, stdlib)" "FAIL (port disagrees with oracle)"; fail=1; fi
else
  step "oracle cross-check (python3, stdlib)" "skipped (no python3 on PATH)"
fi

# 4. template untouched
if [ "$(sha256sum "$tmpl" | cut -d' ' -f1)" = "$tmpl_sha_before" ]; then
  step "template unmodified by generator" "ok"
else
  step "template unmodified by generator" "FAIL"; fail=1
fi

# informational: the recorded drift
if [ -s native/train/kvresp_train_p429.vyb ]; then
  step "note: committed kvresp_train_p429.vyb" \
       "$(wc -c < native/train/kvresp_train_p429.vyb) B != $(wc -c < "$fixture" 2>/dev/null) B fixture (pre-existing drift)"
fi

rm -rf "$WORK"
echo
if [ $fail -eq 0 ]; then echo "KVGEN GATE: PASS"; else echo "KVGEN GATE: FAIL"; fi
exit $fail
