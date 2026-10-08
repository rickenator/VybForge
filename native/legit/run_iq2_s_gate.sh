#!/usr/bin/env bash
# IQ2_S dequant gate (VybForge#10 phase 3 — the largest type in the Ridge file).
#
# What it proves, in two independent steps:
#
#  1. native/tools/iq2_s_ref.py dequantizes real FFN tensors from the 12 GiB GGUF with our numpy
#     port AND with llama.cpp's OWN dequantize_row_iq2_s — the upstream function body, struct and
#     1024-entry grid table copied verbatim out of the local checkout and compiled as-is
#     (native/tools/iq2_s_c_authority.py). The gguf package implements Q8_0 but NOT IQ2_S, so
#     there is no third-party Python check for this type; upstream C is the authority instead.
#  2. native/host/iq2_s_load_driver.vyb runs the GPU kernel on the same tensors (it reads the
#     tensor list from the reference's output, so the two cannot drift apart) and the values are
#     compared here.
#
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
work="${VYBFORGE_IQ2S_OUTDIR:-$root/native/out/iq2_s}"
mkdir -p "$work"

MODEL="${VYBFORGE_RIDGE_GGUF:-$HOME/Models/qwen38-27b-ridge/Qwen3.8-27B-Ridge-3.7bpw.gguf}"
py=""
for cand in "${IQ2S_PY:-}" "$root/.venv/bin/python" python3; do
  if [ -n "$cand" ] && "$cand" -c 'import numpy' >/dev/null 2>&1; then py="$cand"; break; fi
done

echo "IQ2_S dequant gate — $(date '+%F %T')"
echo "toolchain: $VYB"
echo

if [ ! -f "$MODEL" ]; then
  step "model present" "SKIP (not on disk: $MODEL)"
  echo; echo "IQ2_S GATE: SKIP (no model — nothing to check)"
  exit 0
fi
if [ -z "$py" ]; then
  step "reference interpreter" "FAIL (no python with numpy)"
  echo; echo "IQ2_S GATE: FAIL"; exit 1
fi

# ── 1. our port vs llama.cpp's own compiled dequant ──────────────────────────────────
ref_out="$work/ref.log"
env -u PYTHONPATH "$py" native/tools/iq2_s_ref.py >"$ref_out" 2>&1
if ! grep -q "^IQ2_S_REF_DONE" "$ref_out"; then
  step "reference (numpy vs llama.cpp C)" "FAIL (did not complete)"
  tail -8 "$ref_out" | sed 's/^/      /'
  echo; echo "IQ2_S GATE: FAIL"; exit 1
fi
ntens="$(grep -c '^IQ2_S_REF blk' "$ref_out")"
ident="$(grep -o 'identical=[0-9]*' "$ref_out" | tail -1 | cut -d= -f2)"
differ="$(grep -o 'differing=[0-9]*' "$ref_out" | tail -1 | cut -d= -f2)"
if [ "${differ:-1}" != "0" ]; then
  step "reference vs llama.cpp C" "FAIL ($ident identical, $differ differing)"
  echo; echo "IQ2_S GATE: FAIL"; exit 1
fi
if [ "${ident:-0}" = "0" ]; then
  step "reference vs llama.cpp C" "WARN (no compiled authority; numpy is the only authority)"
else
  step "reference vs llama.cpp C" "OK ($ident/$ntens tensors bit-identical, whole 4096-element slices)"
fi
proven=$((proven + 1))

# ── 2. the GPU kernel on the same tensors ────────────────────────────────────────────
drv_out="$work/driver.log"
env -u PYTHONPATH "$VYB" native/host/iq2_s_load_driver.vyb >"$drv_out" 2>&1
if grep -qE "^(SKIP|IQ2_S_SKIP_MODEL)" "$drv_out"; then
  step "GPU kernel (iq2sdeq)" "SKIP (no CUDA device or model unreadable)"
  echo; echo "IQ2_S GATE: PASS ($proven case: reference only)"
  exit 0
fi
if ! grep -q "^IQ2_S_DRIVER_DONE" "$drv_out"; then
  step "GPU kernel (iq2sdeq)" "FAIL (driver did not complete)"
  tail -8 "$drv_out" | sed 's/^/      /'
  fail=1
else
  cmp_out="$work/compare.txt"
  "$py" "native/tools/compare_bits.py" "$drv_out" "native/out/iq2_s_ref.txt" "IQ2_S" "$cmp_out" >/dev/null
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
    ok="$(env -u PYTHONPATH "$py" -c "print('1' if float('$worst') <= 1e-5 else '0')" 2>/dev/null)"
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
  echo "IQ2_S GATE: PASS ($proven cases)"
else
  echo "IQ2_S GATE: FAIL"
fi
exit $fail
