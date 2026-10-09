#!/usr/bin/env bash
# P4.7 Ridge's rope VARIANT kernel on the GPU — VybForge#10 phase 4, unit 10 step 2.
#
# P4.6 pinned the spec against ggml's own op (NEOX inside the first n_dims=64 of each 256-dim head,
# the rest passed through; mode=neox, 8.993e-08). This gate moves that spec onto the GPU:
# native/kernels/rope.vyb's `rope_nrot` runs through native/host/rope_driver.vyb and is compared with
# the same authority. Two launches, and the second is the tooth — n_rot=64 must match the op and
# n_rot=HD (the whole head, which is what the engine's `qwen3rope` does today) must NOT, so a pass
# cannot come from a check that ignores the parameter.
#
# SKIPs (never PASSes) without the llama.cpp checkout or without CUDA; a gate that proved nothing is a
# FAIL.
set -u
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$root"

step() { printf '%-46s %s\n' "$1" "$2"; }
echo "rope kernel gate — $(date '+%F %T')"
echo

LLAMA="${VYBFORGE_LLAMA:-$HOME/Projects/llama.cpp}"
if [ ! -f "$LLAMA/ggml/include/ggml.h" ] || [ ! -e "$LLAMA/build/bin/libggml.so" ]; then
  step "libggml authority" "SKIP (no llama.cpp checkout + build at $LLAMA)"
  echo; echo "ROPE KERNEL GATE: SKIP (nothing to measure against)"; exit 0
fi

py=""
for cand in "${ROPE_PY:-}" "$root/.venv/bin/python" python3; do
  if [ -n "$cand" ] && env -u PYTHONPATH "$cand" -c 'import numpy' >/dev/null 2>&1; then py="$cand"; break; fi
done
if [ -z "$py" ]; then
  step "reference interpreter" "FAIL (no python with numpy)"
  echo; echo "ROPE KERNEL GATE: FAIL"; exit 1
fi

work="${VYBFORGE_ROPE_OUTDIR:-$root/native/out/rope}"
mkdir -p "$work"
log="$work/kernel.log"
env -u PYTHONPATH "$py" native/tools/rope_kernel_verify.py >"$log" 2>&1
rc=$?

grep -E '^ROPE_KERNEL_VERIFY' "$log" | sed 's/^/      /'

if grep -q "^ROPE_KERNEL_VERIFY_SKIP" "$log"; then
  step "rope_nrot vs the op (n_rot=gated)" "SKIP ($(grep -m1 '^ROPE_KERNEL_VERIFY_SKIP' "$log" | sed 's/^ROPE_KERNEL_VERIFY_SKIP //'))"
  echo; echo "ROPE KERNEL GATE: SKIP (nothing ran)"; exit 0
fi

if grep -q "^ROPE_KERNEL_VERIFY_DONE" "$log" && [ "$rc" = "0" ]; then
  step "rope_nrot vs the op (n_rot=gated)" "PASS ($(grep -E 'n_rot=' "$log" | sed 's/^ROPE_KERNEL_VERIFY //' | tr '\n' ';' | sed 's/;$//'))"
  echo; echo "ROPE KERNEL GATE: PASS"
  exit 0
fi

step "rope_nrot vs the op (n_rot=gated)" "FAIL (see $log)"
echo; echo "ROPE KERNEL GATE: FAIL"
exit 1
