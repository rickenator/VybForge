#!/usr/bin/env bash
# run_dtype_gate.sh — S0.1 gate: half-precision storage (f16/bf16) vs an independent authority.
#
# What it proves. Every f16/bf16 conversion an inference kernel can do is compared, on the
# GPU, against expectations produced by numpy/torch (native/dtype/dtcvt_ref.py):
#
#   PART A (must be 0 value differences)
#     widen_f16_all      all 65536 f16 encodings -> f32, byte-exact
#     widen_bf16_all     all 65536 bf16 encodings -> f32, byte-exact
#     real_bf16          REAL bf16 tensor bytes from a real checkpoint
#     real_f16           REAL f16 bytes from a real checkpoint
#     narrow_f16_struct  f32 built from every (sign,exp,mtop10) x the five low-13 classes
#                        that decide the rounding -> f16
#     narrow_rand        seeded random f32 -> f16
#     narrow_bf16_struct the same construction for bf16's 7-bit mantissa -> bf16
#     narrow_rand_bf16   seeded random f32 -> bf16
#
# "0 value differences" means: every input that is not NaN converts byte-exactly, AND no
# NaN input ever comes back as a number. A NaN that keeps a different NaN payload is
# reported as information only — IEEE 754 leaves payload propagation on a conversion
# implementation-defined — but the dangerous case (NaN in, Inf/number out) is a failure.
# The classification is done in the kernel over every element, not inferred from samples.
#
# The two bf16 narrowing cases were red until Vyb#441 was fixed: st_bf16 lowered to
# `lshr 16` + `trunc`, so ~50% of values were one ulp off the authority and a NaN whose
# only set mantissa bits were in the discarded half silently became Inf (measured:
# 0x7f807fff -> 0x7f80). st_bf16 now rounds to nearest even and passes Inf/NaN through
# (`bf.round` / `bf.special` in cgen_expr_kernel.cpp), and all eight cases gate. Torch
# still canonicalises a NaN payload where this reaches for the architectural quiet bit —
# the difference the first paragraph allows.
#
# The comparison happens on the device: only a count and the first mismatches come back,
# so the corpora can be millions of values.
#
# Usage: ./native/legit/run_dtype_gate.sh
# Knobs: DTCVT_PY=<python with numpy/torch>  VYBFORGE_DTCVT_OUTDIR=<corpus/cache dir>
#        VYBFORGE_DTCVT_REAL_BF16 / _REAL_F16 = checkpoint paths (absent => skipped)
set -uo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
. "$root/vybenv.sh" || exit 1

fail=0
step() { printf '%-46s %s\n' "$1" "$2"; }
cd "$root"

work="${VYBFORGE_DTCVT_OUTDIR:-$root/native/out/dtype}"
mkdir -p "$work"
vroot="$(dirname "$(dirname "$VYB")")"
modargs=(--module-path native/dtype --module-path native/tensor --module-path "$vroot/bindings/cuda")

echo "S0.1 dtype gate — $(date '+%F %T')"
echo "toolchain: $VYB  ($(git -C "$vroot" log --oneline -1 2>/dev/null || echo 'not a git checkout'))"
echo "binary mtime: $(date -r "$VYB" '+%F %T')"
echo

# The authority needs numpy (bit machinery) and wants torch (the conversion semantics).
py=""
for cand in "${DTCVT_PY:-}" "$HOME/Projects/VybAIConf/.venv/bin/python" "$root/.venv/bin/python" python3; do
  if [ -n "$cand" ] && "$cand" -c 'import numpy' >/dev/null 2>&1; then py="$cand"; break; fi
done
if [ -z "$py" ]; then step "reference interpreter" "FAIL (no python with numpy)"; exit 1; fi
step "reference interpreter" "$py"

# Kernel: build it the same way the Makefile does (the gate must test the real artifact).
if make -f native/Makefile "$root/native/build/dtcvt.ptx" >"$work/build.log" 2>&1; then
  step "kernel dtcvt.ptx" "built ($(grep -c . "$work/build.log") lines, sm_86 ptxas ok)"
else
  step "kernel dtcvt.ptx" "FAIL"; tail -8 "$work/build.log" | sed 's/^/      /'; fail=1
fi

# dtype table: the module must agree with the container readers' own tables.
env VYBFORGE_DTCVT_MODE=9 "$VYB" native/dtype/dtcvt_probe.vyb "${modargs[@]}" >"$work/dttable.txt" 2>&1
if [ "$(grep -c '^DTCVT_DT ' "$work/dttable.txt")" -ge 15 ]; then
  out="$("$py" native/dtype/dtcvt_ref.py checktable "$work/dttable.txt" "$root" 2>&1)"
  if echo "$out" | grep -q "DTCVT_TABLE_OK"; then step "dtype table vs loaders" "$out"
  else step "dtype table vs loaders" "FAIL"; echo "$out" | tail -4 | sed 's/^/      /'; fail=1; fi
else
  step "dtype table vs loaders" "FAIL (driver did not print the table)"
  tail -8 "$work/dttable.txt" | sed 's/^/      /'; fail=1
fi

run_case() {
  local case="$1"
  local gen mode n vout mm ok
  gen="$("$py" native/dtype/dtcvt_ref.py gen "$case" "$work" 2>&1)"; local grc=$?
  if [ $grc -eq 2 ]; then step "S0.1 $case" "skipped ($(echo "$gen" | grep -o 'DTCVT_REF_SKIP.*'))"; return; fi
  if [ $grc -ne 0 ]; then step "S0.1 $case" "FAIL (reference gen)"; echo "$gen" | tail -4 | sed 's/^/      /'; fail=1; return; fi
  mode="$(echo "$gen" | grep -o 'mode=[0-9]*' | head -1 | cut -d= -f2)"
  n="$(echo "$gen" | grep -o 'n=[0-9]*' | head -1 | cut -d= -f2)"
  vout="$work/$case.vyb.txt"
  if ! env VYBFORGE_DTCVT_MODE="$mode" VYBFORGE_DTCVT_IN="$work/$case.in" \
        VYBFORGE_DTCVT_EXP="$work/$case.exp" \
        "$VYB" native/dtype/dtcvt_probe.vyb "${modargs[@]}" >"$vout" 2>&1; then
    step "S0.1 $case" "FAIL (driver exit)"; tail -6 "$vout" | sed 's/^/      /'; fail=1; return
  fi
  ok="$(grep -c 'DTCVT_OK' "$vout")"
  mm="$(grep -o 'mismatches=[0-9]*' "$vout" | head -1 | cut -d= -f2)"
  npi="$(grep -o 'nan_payload=[0-9]*' "$vout" | head -1 | cut -d= -f2)"
  nxi="$(grep -o 'nan_to_nonnan=[0-9]*' "$vout" | head -1 | cut -d= -f2)"
  vdi="$(grep -o 'value_diff=[0-9]*' "$vout" | head -1 | cut -d= -f2)"
  if [ "$ok" != "1" ]; then
    step "S0.1 $case" "FAIL (no verdict)"; tail -6 "$vout" | sed 's/^/      /'; fail=1; return
  fi
  # A NaN payload difference is not a value error: IEEE 754 leaves payload propagation on
  # a conversion implementation-defined. A NaN that comes back as a NUMBER is an error.
  if [ "$vdi" = "0" ] && [ "$nxi" = "0" ]; then
    step "S0.1 $case" "PASS (n=$n, 0 value differences; ${npi:-0} NaN payload)"
  else
    step "S0.1 $case" "FAIL (value_diff=$vdi nan_to_nonnan=$nxi of $n)"
    "$py" native/dtype/dtcvt_ref.py explain "$case" "$work" "$vout" | grep -E 'DTCVT_EXPLAIN(_HAZARD)? ' | sed 's/^/      /'
    fail=1
  fi
}

# Every case gates: the conversions a half-precision inference path can do, both directions.
run_case widen_f16_all A
run_case widen_bf16_all A
run_case real_bf16 A
run_case real_f16 A
run_case narrow_f16_struct A
run_case narrow_rand A
run_case narrow_bf16_struct A
run_case narrow_rand_bf16 A

echo
if [ $fail -eq 0 ]; then echo "S0.1 DTYPE GATE: PASS"; else echo "S0.1 DTYPE GATE: FAIL"; fi
exit $fail
