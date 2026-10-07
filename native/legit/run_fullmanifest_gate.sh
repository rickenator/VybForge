#!/usr/bin/env bash
# run_fullmanifest_gate.sh — Phase 2 gate for native/train/build_fullmanifest.vyb (P2.1b)
# plus the pre-tokenizer boundary test that this step's debugging produced.
#
# The Python it replaces (native/train/build_fullmanifest.py) embeds the manifest text,
# tokenizes it with the Qwen3 tokenizer from the committed LoRA adapter, prints the token
# count, and writes the text plus the ids as a little-endian <i8 array. native/out/ is
# untracked, so the oracle's output is frozen under native/legit/fixtures/ (captured
# 2026-10-02, refreshed 2026-10-07 after the corpus purge: sha256 82558693… text / c3efa262… ids):
#
#   1. PARITY      — the Vyb program must reproduce both fixture files byte-for-byte and
#                    report the same 425 tokens.
#   2. PRETOK      — native/tokenizer/test_pretok_boundary.vyb: the Qwen2 pre-tokenizer
#                    boundary cases (newline after punctuation, leading tab, CRLF, ...)
#                    against transformers ids. This is the bug this step found: `):` + `\n`
#                    was unreachable, worth exactly one token in the manifest context.
#   3. ORACLE      — if .venv/bin/python can import transformers, re-derive the fixture from
#                    the oracle and require agreement (reference validation only).
#
# Usage: ./native/legit/run_fullmanifest_gate.sh
set -uo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
. "$root/vybenv.sh" || exit 1

WORK="${TMPDIR:-/tmp}/fm-gate.$$"
mkdir -p "$WORK"
fail=0
step() { printf '%-52s %s\n' "$1" "$2"; }

cd "$root"
fixture_txt="native/legit/fixtures/fullmanifest.txt"
fixture_bin="native/legit/fixtures/fullmanifest_ids.bin"
for f in "$fixture_txt" "$fixture_bin"; do
  [ -s "$f" ] || { step "fixture $f" "MISSING"; fail=1; }
done
[ $fail -eq 0 ] || { echo "FULLMANIFEST GATE: FAIL"; exit 1; }

# 1. parity
VYBFORGE_FM_TXT="$WORK/fm.txt" VYBFORGE_FM_BIN="$WORK/fm.bin" \
  "$VYB" native/train/build_fullmanifest.vyb --module-path native/tokenizer --module-path native/json \
  >"$WORK/log" 2>&1
if [ $? -ne 0 ]; then
  step "build_fullmanifest.vyb" "FAIL (exit $?)"; tail -4 "$WORK/log" | sed 's/^/      /'; fail=1
else
  tok="$(grep -o 'manifest context tokens = [0-9]*' "$WORK/log" | grep -o '[0-9]*$')"
  ok=1
  cmp -s "$WORK/fm.txt" "$fixture_txt" || { step "fullmanifest.txt byte parity" "FAIL"; ok=0; fail=1; }
  cmp -s "$WORK/fm.bin" "$fixture_bin" || { step "fullmanifest_ids.bin byte parity" "FAIL"; ok=0; fail=1; }
  [ "$tok" = "425" ] || { step "token count" "FAIL (got ${tok:-none}, want 425)"; ok=0; fail=1; }
  [ $ok -eq 1 ] && step "manifest parity" "byte-identical, 425 tokens ($(wc -c < "$fixture_txt") B + $(wc -c < "$fixture_bin") B)"
fi

# 2. pre-tokenizer boundary test (the bug class this step surfaced)
out="$("$VYB" native/tokenizer/test_pretok_boundary.vyb --module-path native/tokenizer 2>&1)"
if echo "$out" | grep -q "PRETOK BOUNDARY: PASS"; then
  step "pre-tokenizer boundary cases" "$(echo "$out" | grep PASS | sed 's/PRETOK BOUNDARY: //')"
else
  step "pre-tokenizer boundary cases" "FAIL"; echo "$out" | tail -6 | sed 's/^/      /'; fail=1
fi

# 3. optional oracle cross-check
if [ -x .venv/bin/python ] && .venv/bin/python -c "import transformers" >/dev/null 2>&1; then
  if .venv/bin/python native/train/build_fullmanifest.py >/dev/null 2>&1; then
    if cmp -s native/out/fullmanifest.txt "$fixture_txt" && cmp -s native/out/fullmanifest_ids.bin "$fixture_bin"; then
      step "oracle cross-check (.venv transformers)" "ok (fixture == fresh oracle run)"
    else
      step "oracle cross-check (.venv transformers)" "FAIL (fixture stale vs oracle)"; fail=1
    fi
  else
    step "oracle cross-check (.venv transformers)" "skipped (oracle run failed)"
  fi
else
  step "oracle cross-check (.venv transformers)" "skipped (no venv/transformers)"
fi

rm -rf "$WORK"
echo
if [ $fail -eq 0 ]; then echo "FULLMANIFEST GATE: PASS"; else echo "FULLMANIFEST GATE: FAIL"; fi
exit $fail
