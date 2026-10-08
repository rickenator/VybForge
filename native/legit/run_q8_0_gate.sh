#!/usr/bin/env bash
# Q8_0 dequant gate (VybForge#10 phase 3, first kernel of the Ridge work list).
#
# What it proves. Q8_0 is the Gated-DeltaNet state path of Qwen3.8-27B Ridge
# (`ssm_alpha`/`ssm_beta`, 96 tensors). This gate checks the GPU kernel's dequant of REAL
# tensors from that file against `native/tools/q8_0_ref.py`, and — the part that makes the
# comparison mean something — that reference is itself checked against the INDEPENDENT python
# gguf package's dequantizer. A reference that only agrees with itself would repeat the #22
# mistake: two implementations sharing conventions can be wrong together.
#
# The compared values are the first 6 elements of four tensors. Tolerance is maxrel 1e-5: the
# Criterion: EXACT, element by element, over the first 4096 elements of each tensor.
#
# Neither side prints decimals any more. Each prints the signed 64-bit integer that the value's
# f64 bits spell (the driver copies the 8 bytes of the output slot into a Vyb Int; the reference
# unpacks '<q'), and native/tools/compare_bits.py compares those integers, reporting the number of
# differing elements and the worst gap in ulp. Agreement is therefore identity, not "within a
# rounding floor": a 1-ulp arithmetic difference is visible, and an O(1) layout/scale/table error
# is impossible to miss.
#
# This replaced a 1e-5 maxrel criterion that measured the DUMPS rather than the kernels — the old
# drivers printed six significant digits, so every gate reported ~4.5e-6 that was exactly the
# reference rounded to six figures (exposed by BF16, which does no arithmetic at all). Measured
# now: 65536 elements across the five quant types, ZERO differing, every tensor exact — which is
# why 0 ulp is the default and not a hopeful tolerance. Triaging a toolchain change that moves the
# last bit: VYBFORGE_MAXULP=<n>.

set -u
MAXULP="${VYBFORGE_MAXULP:-0}"   # 0 ulp = bit-exact (the measured result); raise only to triage
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
. "$root/vybenv.sh" || exit 1

fail=0
proven=0
step() { printf '%-46s %s\n' "$1" "$2"; }
cd "$root"
work="${VYBFORGE_Q8_0_OUTDIR:-$root/native/out/q8_0}"
mkdir -p "$work"

MODEL="${VYBFORGE_RIDGE_GGUF:-$HOME/Models/qwen38-27b-ridge/Qwen3.8-27B-Ridge-3.7bpw.gguf}"
py=""
for cand in "${Q8_0_PY:-}" "$root/.venv/bin/python" python3; do
  if [ -n "$cand" ] && "$cand" -c 'import numpy' >/dev/null 2>&1; then py="$cand"; break; fi
done

echo "Q8_0 dequant gate — $(date '+%F %T')"
echo "toolchain: $VYB"
echo

if [ ! -f "$MODEL" ]; then
  step "model present" "SKIP (not on disk: $MODEL)"
  echo
  echo "Q8_0 GATE: SKIP (no model — nothing to check)"
  exit 0
fi
if [ -z "$py" ]; then
  step "reference interpreter" "FAIL (no python with numpy)"
  echo; echo "Q8_0 GATE: FAIL"; exit 1
fi

# ── 1. the reference, cross-checked against the independent gguf package ──────────────
ref_out="$work/ref.log"
env -u PYTHONPATH "$py" native/tools/q8_0_ref.py >"$ref_out" 2>&1
if ! grep -q "^Q8_0_REF_DONE" "$ref_out"; then
  step "reference (numpy + gguf package)" "FAIL (did not complete)"
  tail -8 "$ref_out" | sed 's/^/      /'
  echo; echo "Q8_0 GATE: FAIL"; exit 1
fi
ntens="$(grep -c '^Q8_0_REF blk' "$ref_out")"
ident="$(grep -o 'identical=[0-9]*' "$ref_out" | tail -1 | cut -d= -f2)"
differ="$(grep -o 'differing=[0-9]*' "$ref_out" | tail -1 | cut -d= -f2)"
if [ "${differ:-1}" != "0" ]; then
  step "reference vs gguf package" "FAIL ($ident identical, $differ differing)"
  echo; echo "Q8_0 GATE: FAIL"; exit 1
fi
if [ "${ident:-0}" = "0" ]; then
  # The independent authority was unavailable: report it, do not imply agreement.
  step "reference vs gguf package" "WARN (gguf package unavailable; numpy is the only authority)"
else
  step "reference vs gguf package" "OK ($ident/$ntens tensors bit-identical)"
fi
proven=$((proven + 1))

# ── 2. the GPU kernel on the same real tensors ───────────────────────────────────────
drv_out="$work/driver.log"
env -u PYTHONPATH "$VYB" native/host/q8_0_load_driver.vyb >"$drv_out" 2>&1
if grep -q "^SKIP" "$drv_out"; then
  step "GPU kernel (q8_0deq)" "SKIP (no CUDA device)"
  echo; echo "Q8_0 GATE: PASS ($proven case: reference only)"
  exit 0
fi
if grep -q "^Q8_0_SKIP_MODEL" "$drv_out"; then
  step "GPU kernel (q8_0deq)" "SKIP (model unreadable)"
  echo; echo "Q8_0 GATE: PASS ($proven case: reference only)"
  exit 0
fi
if ! grep -q "^Q8_0_DRIVER_DONE" "$drv_out"; then
  step "GPU kernel (q8_0deq)" "FAIL (driver did not complete)"
  tail -8 "$drv_out" | sed 's/^/      /'
  fail=1
else
  cmp_out="$work/compare.txt"
  "$py" "native/tools/compare_bits.py" "$drv_out" "native/out/q8_0_ref.txt" "Q8_0" "$cmp_out" >/dev/null
  cat "$cmp_out" | sed 's/^/      /'
  ncmp="$(grep -o '^TENSORS_COMPARED [0-9]*' "$cmp_out" | cut -d' ' -f2)"
  worst="$(grep -o '^WORST_ULP [0-9]*' "$cmp_out" | cut -d' ' -f2)"
  if grep -qE "^(UNREADABLE|STALE_DUMP|LENGTH_MISMATCH|MISMATCH|MISSING_IN_DRIVER|EXTRA_IN_DRIVER)" "$cmp_out"; then
    step "GPU kernel vs reference (exact bits)" "FAIL (tensor set or length mismatch)"
    fail=1
  elif [ "${ncmp:-0}" = "0" ]; then
    step "GPU kernel vs reference (exact bits)" "FAIL (nothing compared)"
    fail=1
  else
    ok="$(env -u PYTHONPATH "$py" -c "import sys; w=int('$worst'); print('1' if w <= $MAXULP else '0')" 2>/dev/null)"
    if [ "$ok" = "1" ]; then
      step "GPU kernel vs reference (exact bits)" "OK ($ncmp tensors, worst $worst ulp <= $MAXULP)"
      proven=$((proven + 1))
    else
      step "GPU kernel vs reference (exact bits)" "FAIL (worst $worst ulp > $MAXULP)"
      fail=1
    fi
  fi
fi

echo
if [ "$fail" = "0" ]; then
  echo "Q8_0 GATE: PASS ($proven cases)"
else
  echo "Q8_0 GATE: FAIL"
fi
exit $fail
