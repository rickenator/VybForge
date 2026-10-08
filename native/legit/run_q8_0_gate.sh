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
  env -u PYTHONPATH "$py" - "$drv_out" "$root/native/out/q8_0_ref.txt" "$cmp_out" <<'PY'
import sys
drv, ref, outp = sys.argv[1], sys.argv[2], sys.argv[3]
def parse(p, tag):
    got = {}
    for line in open(p):
        if line.startswith(tag + " "):
            parts = line.split("->")
            got[parts[0].split()[1]] = [float(x) for x in parts[1].split()]
    return got
d, r = parse(drv, "Q8_0"), parse(ref, "Q8_0")
lines = []
worst_rel = 0.0
missing = [k for k in r if k not in d]
extra = [k for k in d if k not in r]
for k in sorted(r):
    if k not in d:
        continue
    if len(d[k]) != len(r[k]):
        lines.append(f"MISMATCH {k} len {len(d[k])} != {len(r[k])}")
        continue
    md = max(abs(a - b) for a, b in zip(d[k], r[k]))
    rel = max(abs(a - b) / max(abs(b), 1e-12) for a, b in zip(d[k], r[k]))
    worst_rel = max(worst_rel, rel)
    lines.append(f"TENSOR {k} maxabs={md:.3e} maxrel={rel:.3e}")
for k in missing:
    lines.append(f"MISSING_IN_DRIVER {k}")
for k in extra:
    lines.append(f"EXTRA_IN_DRIVER {k}")
lines.append(f"TENSORS_COMPARED {len(r) - len(missing)}")
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
    ok="$(env -u PYTHONPATH "$py" -c "import sys; w=float('$worst'); print('1' if w <= 1e-5 else '0')" 2>/dev/null)"
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
  echo "Q8_0 GATE: PASS ($proven cases)"
else
  echo "Q8_0 GATE: FAIL"
fi
exit $fail
