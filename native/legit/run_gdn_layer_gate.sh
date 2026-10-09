#!/usr/bin/env bash
# GDN recurrent LAYER wired on the GPU vs ggml's own graph (VybForge#10 phase 4, item 1, unit 7).
#
# What it proves. Unit 5 built one qwen35 linear-attention block out of ggml's own ops (the authority)
# and unit 6 put the layer's non-gemm pieces on the GPU as Vyb kernels. This gate wires the whole
# block in Vyb — native/host/gdn_layer_driver.vyb composes rmsnorm, mm_nt, sigmoid_k, alpha_gate,
# conv1d_k, interleave_qkv, silu_k, l2norm, delta_step, norm_gated and add_k — and compares it
# against that authority STAGE BY STAGE, so a wiring mistake names the stage that is wrong.
#
# Geometry is the 27B Ridge model's own (n_embd=5120, S=128, H_k=16, H_v=48, d_conv=4) and
# ssm_alpha/ssm_beta are the model's real Q8_0 tensors, dequantised on the GPU by the existing q8_0deq
# kernel. It runs TWO chained decode steps and compares both: step 2's conv window and delta-net state
# come out of step 1, so the carry-over a running sequence needs is exercised. Wall clock ~23 s (a
# single-token step is I/O-bound — see native/host/mmnt_bench.vyb). The fixture is generated once and fed to both sides in their
# own precision (authority f32, kernels f64), so the expected agreement is the authority's f32 floor
# (~2e-7 measured) while a wiring error is O(1).
#
# The check reports its own negatives: the same run also prints, for step 2, the residual taken on the
# normed input (7.6e-2) and with no residual at all (9.1e-1), both against a 1e-4 bar.
#
#
# SKIPs (never PASSes) without a Vyb toolchain, without libggml for the authority, or without CUDA.
set -u
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$root"

step() { printf '%-46s %s\n' "$1" "$2"; }
echo "GDN layer gate — $(date '+%F %T')"
echo

if ! . "$root/vybenv.sh" >/dev/null 2>&1 || [ ! -x "$VYB" ]; then
  step "Vyb toolchain" "SKIP (no compiler)"
  echo; echo "GDN LAYER GATE: SKIP (no toolchain — nothing to run)"; exit 0
fi
LLAMA="${VYBFORGE_LLAMA:-$HOME/Projects/llama.cpp}"
if [ ! -f "$LLAMA/ggml/include/ggml.h" ] || [ ! -e "$LLAMA/build/bin/libggml.so" ]; then
  step "libggml authority" "SKIP (no llama.cpp checkout + build at $LLAMA)"
  echo; echo "GDN LAYER GATE: SKIP (nothing to compare against)"; exit 0
fi
echo "toolchain: $VYB  ($(git -C "$VYBHOME" log --oneline -1 2>/dev/null || echo 'not a git checkout'))"
echo

py=""
for cand in "${GDNL_PY:-}" "$root/.venv/bin/python" python3; do
  if [ -n "$cand" ] && env -u PYTHONPATH "$cand" -c 'import numpy' >/dev/null 2>&1; then py="$cand"; break; fi
done
if [ -z "$py" ]; then
  step "reference interpreter" "FAIL (no python with numpy)"
  echo; echo "GDN LAYER GATE: FAIL"; exit 1
fi

work="${VYBFORGE_GDNL_OUTDIR:-$root/native/out/gdn_layer}"
mkdir -p "$work"
log="$work/verify.log"
env -u PYTHONPATH "$py" native/tools/gdn_layer_kernel_verify.py >"$log" 2>&1
rc=$?

grep -E '^GDNL_VERIFY' "$log" | sed 's/^/      /'

if grep -q "^GDNL_VERIFY_SKIP" "$log"; then
  step "GPU layer wiring (2 steps, 26 stages)" "SKIP ($(grep -m1 '^GDNL_VERIFY_SKIP' "$log" | sed 's/^GDNL_VERIFY_SKIP //'))"
  echo; echo "GDN LAYER GATE: SKIP (no CUDA device — nothing ran)"; exit 0
fi

if grep -q "^GDNL_VERIFY_DONE" "$log" && [ "$rc" = "0" ]; then
  nst="$(grep -cE '^GDNL_VERIFY [a-z0-9_]+ +n=' "$log")"
  worst="$(grep -E '^GDNL_VERIFY [a-z0-9_]+ +n=' "$log" | grep -oE 'maxrel=[0-9.e+-]+' | sed 's/maxrel=//' | sort -g | tail -1)"
  step "GPU layer wiring ($nst stages)" "PASS (worst ${worst}, steps=2, rejected 7.6e-2/9.1e-1)"
  echo; echo "GDN LAYER GATE: PASS"
  exit 0
fi

step "GPU layer wiring (2 steps, 26 stages)" "FAIL (see $log)"
echo; echo "GDN LAYER GATE: FAIL"
exit 1
