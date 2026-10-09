#!/usr/bin/env bash
# P4.8 Ridge's attention-block stages, against ggml's own ops — VybForge#10 phase 4, unit 10 step 3a.
#
# The reconnaissance flagged the JOINT Q+gate projection as the substantial attention-path risk:
# `attn_q.weight` is one matrix carrying, per head, a 256-dim q block then a 256-dim gate block, and
# every later stage depends on splitting it as llama.cpp does. This gate compares each stage of the
# block's front half (the split, both per-head RMS norms, rope) against native/tools/attn_authority.c,
# which builds the stages with ggml_view_3d / ggml_rms_norm / ggml_rope_multi — the same ops llama.cpp
# runs — and it REQUIRES the alternative readings to differ: gate-first split, one norm across the whole
# projection, rope over the whole head. A check that cannot tell those apart proves nothing.
#
# Scope: attention, the output gate (attn * sigmoid(gate)) and `wo` are NOT covered here yet.
# SKIPs (never PASSes) without the llama.cpp checkout.
set -u
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$root"

step() { printf '%-46s %s\n' "$1" "$2"; }
echo "attention block gate — $(date '+%F %T')"
echo

LLAMA="${VYBFORGE_LLAMA:-$HOME/Projects/llama.cpp}"
if [ ! -f "$LLAMA/ggml/include/ggml.h" ] || [ ! -e "$LLAMA/build/bin/libggml.so" ]; then
  step "libggml authority" "SKIP (no llama.cpp checkout + build at $LLAMA)"
  echo; echo "ATTENTION BLOCK GATE: SKIP (nothing to measure against)"; exit 0
fi

py=""
for cand in "${ATTN_PY:-}" "$root/.venv/bin/python" python3; do
  if [ -n "$cand" ] && env -u PYTHONPATH "$cand" -c 'import numpy' >/dev/null 2>&1; then py="$cand"; break; fi
done
if [ -z "$py" ]; then
  step "reference interpreter" "FAIL (no python with numpy)"
  echo; echo "ATTENTION BLOCK GATE: FAIL"; exit 1
fi

work="${VYBFORGE_ATTN_OUTDIR:-$root/native/out/attn}"
mkdir -p "$work"
log="$work/verify.log"
env -u PYTHONPATH "$py" native/tools/attn_verify.py >"$log" 2>&1
rc=$?

grep -E '^ATTN_VERIFY' "$log" | sed 's/^/      /'

if grep -q "^ATTN_VERIFY_SKIP" "$log"; then
  step "attention stages vs ggml" "SKIP ($(grep -m1 '^ATTN_VERIFY_SKIP' "$log" | sed 's/^ATTN_VERIFY_SKIP //'))"
  echo; echo "ATTENTION BLOCK GATE: SKIP (nothing ran)"; exit 0
fi

if grep -q "^ATTN_VERIFY_DONE" "$log" && [ "$rc" = "0" ]; then
  summ="$(grep -m1 '^ATTN_VERIFY_DONE' "$log" | sed 's/^ATTN_VERIFY_DONE //')"
  nalt="$(grep -cE 'rejected$' "$log")"
  step "attention stages vs ggml" "PASS ($summ; $nalt alternatives rejected)"
  echo; echo "ATTENTION BLOCK GATE: PASS"
  exit 0
fi

step "attention stages vs ggml" "FAIL (see $log)"
echo; echo "ATTENTION BLOCK GATE: FAIL"
exit 1
