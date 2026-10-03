#!/usr/bin/env bash
# run_tensor_gate.sh — S0.2b gate: the host tensor core (shape/strides/dtype/broadcast).
#
# What it proves. native/tensor/tshape.vyb is the piece every driver uses to turn a model's
# dimensions into buffer sizes, element offsets and broadcast views. numpy is the definition of
# that behaviour, so the gate generates a case table WITH numpy's answers (native/tensor/
# ts_ref.py) and compares:
#
#   * strides  — a real float32 array's own strides (rank 0 through 5, the degenerate shapes)
#   * offsets  — np.ravel_multi_index, including an out-of-bounds coordinate (must be refused)
#   * contiguity — .flags["C_CONTIGUOUS"] on real views (row, column, transpose, strided slice)
#   * broadcast — np.broadcast_shapes (refusals included) and np.broadcast_to(...).strides,
#                 whose ZERO strides are exactly the expanded axes
#
# The table is required to contain both a refusal case and a non-contiguous case: a table that
# cannot catch a broken core is decoration. The checker is then fed a deliberately perturbed
# expectation to prove it reports a mismatch instead of always passing.
#
# Knobs: TS_PY=<python with numpy>  VYBFORGE_TS_OUTDIR=<dir>
set -uo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
. "$root/vybenv.sh" || exit 1

fail=0
step() { printf '%-46s %s\n' "$1" "$2"; }
cd "$root"

work="${VYBFORGE_TS_OUTDIR:-$root/native/out/tensor}"
mkdir -p "$work"
mods=(--module-path native/tensor --module-path native/dtype)
ref=native/tensor/ts_ref.py

echo "S0.2b tensor gate — $(date '+%F %T')"
echo "toolchain: $VYB"
echo

py=""
for cand in "${TS_PY:-}" "$root/.venv/bin/python" "$HOME/Projects/VybAIConf/.venv/bin/python" python3; do
  if [ -n "$cand" ] && "$cand" -c 'import numpy' >/dev/null 2>&1; then py="$cand"; break; fi
done
if [ -z "$py" ]; then step "reference interpreter" "FAIL (no python with numpy)"; exit 1; fi
step "reference interpreter" "$py (numpy $("$py" -c 'import numpy;print(numpy.__version__)' 2>/dev/null))"

out="$("$py" "$ref" gen "$work/cases.txt" "$work/expected.json" 2>&1)"
if echo "$out" | grep -q "TSREF_OK"; then
  step "case table (numpy)" "$(echo "$out" | grep -o 'TSREF_CASES=.*' | head -1)"
else
  step "case table (numpy)" "FAIL"
  echo "$out" | tail -4 | sed 's/^/      /'; fail=1
fi

if [ $fail -eq 0 ]; then
  env TS_CASES="$work/cases.txt" "$VYB" native/tensor/ts_probe.vyb "${mods[@]}" >"$work/ts.txt" 2>&1
  if grep -q '^TSDONE' "$work/ts.txt"; then
    step "tensor core probe" "$(grep -c '^TS ' "$work/ts.txt") cases answered"
    res="$("$py" "$ref" check "$work/ts.txt" "$work/expected.json" 2>&1)"
    if echo "$res" | grep -q "TSGATE_OK"; then
      step "core vs numpy" "$(echo "$res" | grep -o 'TSGATE_CASES_OK.*' | head -1)"
    else
      step "core vs numpy" "FAIL"
      echo "$res" | grep -E 'TSGATE_(MISMATCH|FAIL|MISSING|UNKNOWN)' | head -10 | sed 's/^/      /'
      fail=1
    fi

    # the checker must be able to fail
    "$py" "$ref" mutate "$work/expected.json" "$work/perturbed.json" >/dev/null 2>&1
    pres="$("$py" "$ref" check "$work/ts.txt" "$work/perturbed.json" 2>&1)"
    if echo "$pres" | grep -q "TSGATE_FAIL"; then
      step "discrimination: perturbed expectation" "correctly reported $(echo "$pres" | grep -o 'TSGATE_MISMATCH [^ ]*' | head -1)"
    else
      step "discrimination: perturbed expectation" "FAIL (a corrupted expectation still passed — the checker is toothless)"
      fail=1
    fi
  else
    step "tensor core probe" "FAIL"
    tail -6 "$work/ts.txt" | sed 's/^/      /'; fail=1
  fi
fi

echo
if [ $fail -eq 0 ]; then echo "S0.2b TENSOR GATE: PASS"; else echo "S0.2b TENSOR GATE: FAIL"; fi
exit $fail
