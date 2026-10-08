#!/usr/bin/env bash
# GDN recurrent-layer KERNELS on the GPU (VybForge#10 phase 4, item 1, unit 6).
#
# What it proves. The reference work for the recurrent layer (units 1-5) is pinned to ggml's own ops;
# this gate is the first Vyb code for it. `native/kernels/gdn.vyb` implements the three pieces of a
# qwen35 linear-attention layer that are not gemms — `l2norm`, the delta-rule step `delta_step` and the
# gated RMS epilogue `norm_gated` — and `native/host/gdn_driver.vyb` runs them through native/build/gdn.ptx
# on one token at the 27B geometry. `native/tools/gdn_kernel_verify.py` writes the fixture, drives the
# driver, reads every result back as the raw 64-bit pattern of each fp64 element and compares it
# against an fp64 reference (802816 elements over 5 sections).
#
# Why fp64 on both sides: the point of this gate is the kernels, not a precision floor, so the
# reference is the same precision as the kernel and any wiring error stands out. The pure-op gates do
# the opposite — they mirror ggml's f32 bodies — because there the authority IS f32.
#
# The check reports the REJECTED alternative as well: contracting the read-out on the state's first
# axis, which is the mistake this kernel was written with first, is off by 1.4e0 (no-decay 2.0e-1)
# against a 6e-11 pass.
#
# SKIPs (never PASSes) without CUDA or a Vyb toolchain. Gate: run_gdn_kernel_gate.sh
# Env: VYBFORGE_GDNK_MAXREL=<n>   widen the bar while triaging
set -u
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$root"

step() { printf '%-46s %s\n' "$1" "$2"; }
echo "GDN kernel gate — $(date '+%F %T')"
echo

# The Vyb toolchain (VYBHOME/VYB/VYB_STDLIB). Missing means no kernel to run.
if ! . "$root/vybenv.sh" >/dev/null 2>&1; then
  step "Vyb toolchain" "SKIP (vybenv.sh could not resolve VYBHOME)"
  echo; echo "GDN KERNEL GATE: SKIP (no toolchain — nothing to run)"; exit 0
fi
if [ ! -x "$VYB" ]; then
  step "Vyb toolchain" "SKIP (no compiler at $VYB)"
  echo; echo "GDN KERNEL GATE: SKIP (no toolchain — nothing to run)"; exit 0
fi
echo "toolchain: $VYB  ($(git -C "$VYBHOME" log --oneline -1 2>/dev/null || echo 'not a git checkout'))"
echo "binary mtime: $(date -r "$VYB" '+%F %T')"
echo

py=""
for cand in "${GDNK_PY:-}" "$root/.venv/bin/python" python3; do
  if [ -n "$cand" ] && env -u PYTHONPATH "$cand" -c 'import numpy' >/dev/null 2>&1; then py="$cand"; break; fi
done
if [ -z "$py" ]; then
  step "reference interpreter" "FAIL (no python with numpy)"
  echo; echo "GDN KERNEL GATE: FAIL"; exit 1
fi

work="${VYBFORGE_GDNK_OUTDIR:-$root/native/out/gdn_kernel}"
mkdir -p "$work"
log="$work/verify.log"
env -u PYTHONPATH "$py" native/tools/gdn_kernel_verify.py >"$log" 2>&1
rc=$?

grep -E '^GDNK_VERIFY' "$log" | grep -v '^GDNK_VERIFY_SUMMARY' | sed 's/^/      /'

if grep -q "^GDNK_VERIFY_SKIP" "$log"; then
  step "GPU kernels (l2norm/delta_step/norm_gated)" "SKIP ($(grep -m1 '^GDNK_VERIFY_SKIP' "$log" | sed 's/^GDNK_VERIFY_SKIP //'))"
  echo; echo "GDN KERNEL GATE: SKIP (no CUDA device — nothing ran)"; exit 0
fi

if grep -q "^GDNK_VERIFY_DONE" "$log" && [ "$rc" = "0" ]; then
  sum="$(grep -oE 'elements=[0-9]+' "$log" | tail -1)"
  alt="$(grep -oE 'rejected_alternatives=[0-9.e+-]+/[0-9.e+-]+' "$log" | tail -1)"
  step "GPU kernels (l2norm/delta_step/norm_gated)" "PASS ($sum, $alt rejected)"
  echo; echo "GDN KERNEL GATE: PASS"
  exit 0
fi

step "GPU kernels (l2norm/delta_step/norm_gated)" "FAIL (see $log)"
echo; echo "GDN KERNEL GATE: FAIL"
exit 1
