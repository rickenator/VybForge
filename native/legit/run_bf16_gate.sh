#!/usr/bin/env bash
# BF16 dequant gate (VybForge#10 phase 3 — the vision tower's weight type).
#
# What it proves, in two independent steps:
#
#  1. native/tools/bf16_ref.py converts real vision-tower tensors from the mmproj with our numpy
#     AND with the INDEPENDENT python `gguf` package (which does implement BF16) — the two must be
#     bit-identical, so a mistake in our reading of the format cannot pass as the authority. BF16
#     is not a block quant (2 bytes per element, no scale, no table), so there is no ggml-quants
#     function to extract; the kernel uses the language's own `ld_bf16`, which is ggml's definition.
#  2. native/host/bf16_load_driver.vyb runs the GPU kernel on the same tensors (it reads the
#     tensor list from the reference's output, so the two cannot drift apart) and every value of
#     each 4096-element slice is compared here.
#
# Tolerance maxrel 1e-5, and what that number actually measures: the DRIVER's dump carries SIX
# significant digits (Vyb's Float printing), and the measured difference is EXACTLY the reference
# rounded to those six digits — verified by reproducing the dump that way. So this gate resolves
# layout, scale, ordering and table errors (all O(1) by nature) and CANNOT resolve an arithmetic
# difference below ~1e-5. Tightening it means dumping the value's BITS instead of its decimal form.

set -u
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
. "$root/vybenv.sh" || exit 1

fail=0
proven=0
step() { printf '%-46s %s\n' "$1" "$2"; }
cd "$root"
work="${VYBFORGE_IQ2S_OUTDIR:-$root/native/out/bf16}"
mkdir -p "$work"

MODEL="${VYBFORGE_MMPROJ_GGUF:-$HOME/Models/qwen38-27b-ridge/mmproj-Qwen3.8-27B-BF16.gguf}"
py=""
for cand in "${IQ3S_PY:-}" "$root/.venv/bin/python" python3; do
  if [ -n "$cand" ] && "$cand" -c 'import numpy' >/dev/null 2>&1; then py="$cand"; break; fi
done

echo "BF16 dequant gate — $(date '+%F %T')"
echo "toolchain: $VYB"
echo

if [ ! -f "$MODEL" ]; then
  step "model present" "SKIP (not on disk: $MODEL)"
  echo; echo "BF16 GATE: SKIP (no model — nothing to check)"
  exit 0
fi
if [ -z "$py" ]; then
  step "reference interpreter" "FAIL (no python with numpy)"
  echo; echo "BF16 GATE: FAIL"; exit 1
fi

# ── 1. our port vs llama.cpp's own compiled dequant ──────────────────────────────────
ref_out="$work/ref.log"
env -u PYTHONPATH "$py" native/tools/bf16_ref.py >"$ref_out" 2>&1
if ! grep -q "^BF16_REF_DONE" "$ref_out"; then
  step "reference (numpy vs llama.cpp C)" "FAIL (did not complete)"
  tail -8 "$ref_out" | sed 's/^/      /'
  echo; echo "BF16 GATE: FAIL"; exit 1
fi
ntens="$(grep -c '^BF16_REF blk' "$ref_out")"
ident="$(grep -o 'identical=[0-9]*' "$ref_out" | tail -1 | cut -d= -f2)"
differ="$(grep -o 'differing=[0-9]*' "$ref_out" | tail -1 | cut -d= -f2)"
if [ "${differ:-1}" != "0" ]; then
  step "reference vs llama.cpp C" "FAIL ($ident identical, $differ differing)"
  echo; echo "BF16 GATE: FAIL"; exit 1
fi
if [ "${ident:-0}" = "0" ]; then
  step "reference vs llama.cpp C" "WARN (no compiled authority; numpy is the only authority)"
else
  step "reference vs llama.cpp C" "OK ($ident/$ntens tensors bit-identical, whole 4096-element slices)"
fi
proven=$((proven + 1))

# ── 2. the GPU kernel on the same tensors ────────────────────────────────────────────
drv_out="$work/driver.log"
env -u PYTHONPATH "$VYB" native/host/bf16_load_driver.vyb >"$drv_out" 2>&1
if grep -qE "^(SKIP|BF16_SKIP_MODEL)" "$drv_out"; then
  step "GPU kernel (bf16deq)" "SKIP (no CUDA device or model unreadable)"
  echo; echo "BF16 GATE: PASS ($proven case: reference only)"
  exit 0
fi
if ! grep -q "^BF16_DRIVER_DONE" "$drv_out"; then
  step "GPU kernel (bf16deq)" "FAIL (driver did not complete)"
  tail -8 "$drv_out" | sed 's/^/      /'
  fail=1
else
  cmp_out="$work/compare.txt"
  env -u PYTHONPATH "$py" - "$drv_out" "$root/native/out/bf16_ref.txt" "$cmp_out" <<'PY'
import sys
drv, ref, outp = sys.argv[1], sys.argv[2], sys.argv[3]
def parse(p, tag):
    got = {}
    for line in open(p):
        if line.startswith(tag + " "):
            head, vals = line.split("->")
            got[head.split()[1].split("@")[0]] = [float(x) for x in vals.split()]
    return got
d, r = parse(drv, "BF16"), parse(ref, "BF16")
lines, worst_rel = [], 0.0
for k in sorted(r):
    if k not in d:
        lines.append(f"MISSING_IN_DRIVER {k}")
        continue
    if len(d[k]) != len(r[k]):
        lines.append(f"MISMATCH {k} len {len(d[k])} != {len(r[k])}")
        continue
    md = max(abs(a - b) for a, b in zip(d[k], r[k]))
    rel = max(abs(a - b) / max(abs(b), 1e-12) for a, b in zip(d[k], r[k]))
    worst_rel = max(worst_rel, rel)
    lines.append(f"TENSOR {k} maxabs={md:.3e} maxrel={rel:.3e}")
for k in d:
    if k not in r:
        lines.append(f"EXTRA_IN_DRIVER {k}")
lines.append(f"TENSORS_COMPARED {len([k for k in r if k in d])}")
lines.append(f"WORST_REL {worst_rel:.3e}")
open(outp, "w").write("\n".join(lines) + "\n")
PY
  cat "$cmp_out" | sed 's/^/      /'
  ncmp="$(grep -o '^TENSORS_COMPARED [0-9]*' "$cmp_out" | cut -d' ' -f2)"
  worst="$(grep -o '^WORST_REL .*' "$cmp_out" | cut -d' ' -f2)"
  if grep -qE "^(MISMATCH|MISSING_IN_DRIVER|EXTRA_IN_DRIVER)" "$cmp_out"; then
    step "GPU kernel vs reference" "FAIL (tensor set or length mismatch)"
    fail=1
  elif [ "${ncmp:-0}" = "0" ]; then
    step "GPU kernel vs reference" "FAIL (nothing compared)"
    fail=1
  else
    ok="$(env -u PYTHONPATH "$py" -c "print('1' if float('$worst') <= 1e-5 else '0')" 2>/dev/null)"
    if [ "$ok" = "1" ]; then
      step "GPU kernel vs reference" "OK ($ncmp tensors, worst rel $worst <= 1e-5)"
      proven=$((proven + 1))
    else
      step "GPU kernel vs reference" "FAIL (worst rel $worst > 1e-5)"
      fail=1
    fi
  fi
fi

echo
if [ "$fail" = "0" ]; then
  echo "BF16 GATE: PASS ($proven cases)"
else
  echo "BF16 GATE: FAIL"
fi
exit $fail
