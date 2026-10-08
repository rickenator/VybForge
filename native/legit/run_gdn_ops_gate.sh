#!/usr/bin/env bash
# Gated DeltaNet layer-op references (VybForge#10 phase 4, item 1's units 1-3).
#
# What it proves. Before a Vyb kernel for the recurrent layer can be written, the layer's three
# non-trivial pieces are pinned to an authority that is NOT our own code: the same libggml that the
# local llama.cpp is built with. Each unit is a C harness that links it and calls the real op
# (`ggml_recurrent`, `ggml_ssm_conv`, `ggml_l2_norm`/`ggml_rms_norm`/the gated epilogue) plus a numpy
# port, and the check compares the two — so agreement is "our port matches the implementation the
# model actually runs", not "two of our implementations agree" (the VybForge#22 mistake).
#
#   unit 1  native/tools/gdn_authority.c  + gdn_verify.py    recurrence + state read-out
#   unit 2  native/tools/conv_authority.c + conv_verify.py   causal depthwise short convolution
#   unit 3  native/tools/norm_authority.c + norm_verify.py   l2 norm, RMS norm, gated epilogue
#
# The norm and conv verifiers also report the OPPOSITE convention's error, because a check that
# cannot fail proves nothing: eps-after-sqrt vs eps-inside-sqrt differ by 7e-2 and 7e-3 here, so the
# 1e-6 agreement is a real measurement rather than a tolerance nothing could breach.
#
# The recurrence verifier refuses multi-token: with T>1 the op runs a second (chunked) kernel whose
# buffer layout is not yet characterised — see doc/QWEN35-PHASE4.md.
#
# Needs the llama.cpp checkout (headers + built libggml) to compile the harnesses; SKIPs without it.
#   VYBFORGE_LLAMA=/path/to/llama.cpp   override the checkout
set -u
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$root"

LLAMA="${VYBFORGE_LLAMA:-$HOME/Projects/llama.cpp}"
fail=0
proven=0
step() { printf '%-46s %s\n' "$1" "$2"; }

echo "Gated DeltaNet op references — $(date '+%F %T')"
echo

if [ ! -f "$LLAMA/ggml/include/ggml.h" ] || [ ! -e "$LLAMA/build/bin/libggml.so" ]; then
  step "libggml authority present" "SKIP (no llama.cpp checkout + build at $LLAMA)"
  echo
  echo "GDN OPS GATE: SKIP (nothing to check against)"
  exit 0
fi

py=""
# env -u PYTHONPATH matters here too: the session's PYTHONPATH shadows the venv's numpy with a
# 3.14 build, so a bare `python -c 'import numpy'` fails while the venv is perfectly fine.
for cand in "${GDN_PY:-}" "$root/.venv/bin/python" python3; do
  if [ -n "$cand" ] && env -u PYTHONPATH "$cand" -c 'import numpy' >/dev/null 2>&1; then py="$cand"; break; fi
done
if [ -z "$py" ]; then
  step "reference interpreter" "FAIL (no python with numpy)"
  echo; echo "GDN OPS GATE: FAIL"; exit 1
fi
echo "llama.cpp: $LLAMA ($(git -C "$LLAMA" log --oneline -1 2>/dev/null || echo 'not a git checkout'))"
echo

work="${VYBFORGE_GDN_OUTDIR:-$root/native/out/gdn_ops}"
mkdir -p "$work"

# slug | command | done-token | label
run_unit() {
  local slug="$1" script="$2" done_tok="$3" label="$4"
  shift 4
  local log="$work/$slug.log"
  env -u PYTHONPATH "$py" "$script" "$@" >"$log" 2>&1
  if grep -q "^$done_tok" "$log"; then
    local detail
    detail="$(grep -oE '^(GDN|CONV|NORM)_VERIFY_SUMMARY .*' "$log" | sed 's/^[A-Z_]*SUMMARY //')"
    step "$label" "OK ($detail)"
    proven=$((proven + 1))
  else
    step "$label" "FAIL (see $log)"
    tail -8 "$log" | sed 's/^/      /'
    fail=1
  fi
}

run_unit "unit1" native/tools/gdn_verify.py  GDN_VERIFY_DONE  "unit 1 recurrence + read-out" --tokens 1
run_unit "unit2" native/tools/conv_verify.py CONV_VERIFY_DONE "unit 2 short convolution"
run_unit "unit3" native/tools/norm_verify.py NORM_VERIFY_DONE "unit 3 l2 / rms / gated epilogue"

# The norm check's teeth: report the opposite convention's error so a silently-broken verifier is visible.
if [ -f "$work/unit3.log" ]; then
  opp="$(grep -o 'opposite-convention maxrel=[0-9.e+-]*' "$work/unit3.log" | tr '\n' ' ' | sed 's/  */ /g')"
  [ -n "$opp" ] && step "unit 3 convention discrimination" "$opp (must be >> 1e-6)"
fi

echo
if [ "$fail" = "0" ]; then
  echo "GDN OPS GATE: PASS ($proven units against ggml's own ops)"
else
  echo "GDN OPS GATE: FAIL"
fi
exit $fail
