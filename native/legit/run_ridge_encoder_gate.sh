#!/usr/bin/env bash
# P4.11 our tokenizer vs the captured Ridge oracle — the prerequisite for every forward gate.
#
# A whole-model comparison is only as good as the token stream it starts from: if our encoder emits
# different ids than llama.cpp for the same text, every hidden and top1 downstream is comparing a
# different input, and it reads as a model bug. The oracle fixtures carry llama's OWN `/tokenize`
# output for their prompts, and this runs our encoder (stdlib/vllm, via native/llm/llm_encode_probe.vyb)
# against them and requires exact equality.
#
# It also guards a silent upstream trap: stdlib/vllm's `build_vocab_from` parses a PRETTY-PRINTED
# vocab.json to id 0 for EVERY token (its `read_int` stops at the first non-digit, and the official
# Qwen/Qwen3.8-27B repo ships the file pretty-printed). The token count still comes out right, so
# nothing errors. The check compacts the file first and asserts a known token maps to a non-zero id.
#
# SKIPs (never PASSes) without the toolchain, the fixtures, or the tokenizer files in
# artifacts/ridge-tokenizer (gitignored; the fetch command is printed when they are missing) — a
# fresh clone has none of them and a step that proved nothing must not read as green.
set -u
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$root"

step() { printf '%-46s %s\n' "$1" "$2"; }
echo "Ridge encoder gate — $(date '+%F %T')"
echo

py=""
for cand in "${RIDGEENC_PY:-}" "$root/.venv/bin/python" python3; do
  if [ -n "$cand" ] && env -u PYTHONPATH "$cand" -c 'import numpy' >/dev/null 2>&1; then py="$cand"; break; fi
done
if [ -z "$py" ]; then
  step "reference interpreter" "FAIL (no python with numpy)"
  echo; echo "RIDGE ENCODER GATE: FAIL"; exit 1
fi

work="${VYBFORGE_RIDGE_ENC_OUTDIR:-$root/native/out/ridge_encoder}"
mkdir -p "$work"
log="$work/check.log"

env -u PYTHONPATH "$py" native/tools/ridge_encoder_check.py >"$log" 2>&1
rc=$?
grep -E '^RIDGE_ENCODER_CHECK' "$log" | sed 's/^/      /'

if grep -q '^RIDGE_ENCODER_CHECK_SKIP' "$log"; then
  step "encoder vs the oracle" "SKIP ($(grep -m1 '^RIDGE_ENCODER_CHECK_SKIP' "$log" | sed 's/^RIDGE_ENCODER_CHECK_SKIP //'))"
  echo; echo "RIDGE ENCODER GATE: SKIP (nothing ran)"; exit 0
fi

if grep -q '^RIDGE_ENCODER_CHECK_DONE' "$log" && [ "$rc" = "0" ]; then
  n="$(grep -cE '^RIDGE_ENCODER_CHECK .* OK <' "$log")"
  step "encoder vs the oracle ($n prompts)" "PASS (ids == llama.cpp's, exactly)"
  echo; echo "RIDGE ENCODER GATE: PASS"
  exit 0
fi

step "encoder vs the oracle" "FAIL (see $log)"
echo; echo "RIDGE ENCODER GATE: FAIL"
exit 1
