#!/usr/bin/env bash
# Q4_K dequant reference gate (the type of the 27B Ridge model's biggest projections).
#
# Q4_K was the one quant type in this repo with no reference and no gate, while blk.N.attn_qkv,
# blk.N.attn_gate and blk.N.ssm_out are all Q4_K (17.7 MB each). The GDN layer check could not take
# those weights without it: the authority is fed the same weights the GPU used, so without an
# independent reference the comparison would be our numpy agreeing with our kernel.
#
# native/tools/q4k_ref.py is that reference: a numpy port of dequantize_row_q4_K (144-byte blocks:
# d, dmin, scales[12] of six-bit scales/mins, qs[128] holding two four-bit quants per byte, the LOW
# nibble belonging to the even sub-block) compared ELEMENT-WISE against the independent `gguf`
# package's Q4_K dequantizer on whole real tensors. Measured on two 31.5 M-value tensors: identical.
#
# SKIPs without the model or the inventory, and FAILS if it proved nothing (no gemm package = no
# authority = not a pass). The GPU kernel comparison is the next step for this file: q4k_ref.txt
# carries the first 4096 values of each tensor as raw f64 bit patterns for exactly that.
set -u
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$root"
step() { printf '%-46s %s\n' "$1" "$2"; }

py=""
for cand in "$root/.venv/bin/python" python3; do
  if [ -n "$cand" ] && env -u PYTHONPATH "$cand" -c 'import numpy' >/dev/null 2>&1; then py="$cand"; break; fi
done
if [ -z "$py" ]; then step "reference interpreter" "FAIL (no python with numpy)"; echo; echo "Q4_K GATE: FAIL"; exit 1; fi

work="${VYBFORGE_Q4K_OUTDIR:-$root/native/out/q4k}"
mkdir -p "$work"
log="$work/ref.log"
env -u PYTHONPATH "$py" native/tools/q4k_ref.py >"$log" 2>&1
rc=$?

grep -E '^Q4K_REF ' "$log" | sed 's/^/      /'

if grep -q "^Q4K_REF_SKIP" "$log"; then
  step "Q4_K reference vs gguf package" "SKIP ($(grep -m1 '^Q4K_REF_SKIP' "$log" | sed 's/^Q4K_REF_SKIP //'))"
  echo; echo "Q4_K GATE: SKIP (no model — nothing to check)"; exit 0
fi
if grep -q "^Q4K_REF_WARN no independent" "$log"; then
  step "Q4_K reference vs gguf package" "FAIL (gguf package unavailable — numpy would be the only authority)"
  echo; echo "Q4_K GATE: FAIL"; exit 1
fi
ident="$(grep -o 'identical=[0-9]*' "$log" | tail -1 | cut -d= -f2)"
differ="$(grep -o 'differing=[0-9]*' "$log" | tail -1 | cut -d= -f2)"
if [ "${differ:-1}" != "0" ] || [ "${ident:-0}" = "0" ] || [ "$rc" != "0" ]; then
  step "Q4_K reference vs gguf package" "FAIL (identical=${ident:-?} differing=${differ:-?})"
  echo "$log" | tail -6 | sed 's/^/      /'
  echo; echo "Q4_K GATE: FAIL"; exit 1
fi
step "Q4_K reference vs gguf package" "PASS ($ident full tensors bit-identical)"
echo; echo "Q4_K GATE: PASS"
exit 0
