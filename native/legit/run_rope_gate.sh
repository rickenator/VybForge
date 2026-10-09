#!/usr/bin/env bash
# P4.6 Ridge's attention RoPE, pinned against ggml's own op — VybForge#10 phase 4, unit 10.
#
# What it settles. A Ridge attention layer rotates only n_dims = 64 of each 256-dim head, and phase 4
# has blamed "IMROPE" for this since phase 3. This gate asks the real op (native/tools/rope_authority.c
# links the libggml llama.cpp was built with and calls ggml_rope_multi) which of the candidate
# (mode, layout) pairs reproduces it, and requires exactly one to land at the f32 floor while the others
# stay O(1). Measured: mode=neox with NEOX pairing inside the first n_dims, everything else passed
# through — 8.993e-08 — with the runner-up at 1.01e+00. So the sections are INERT for this
# architecture: `llama_model_rope_type` gives qwen35 NEOX, ggml's `mrope_used` is false, and the
# `[11,11,10,0]` metadata never reaches the kernel. The engine's `qwen3rope` rotates EVERY head dim
# with pairs (i, i + HD/2), which is the `neox_all_dims` candidate — measured at 1.478, i.e. wrong for
# this model, and the reason a rope variant (pairs (k, k + n_dims/2) inside the first n_dims, rest
# passed through) is a real piece of work rather than a parameter change.
#
# SKIPs (never PASSes) without the llama.cpp checkout the authority links; a gate that proved nothing
# is a FAIL.
set -u
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$root"

step() { printf '%-46s %s\n' "$1" "$2"; }
echo "GDN rope gate — $(date '+%F %T')"
echo

LLAMA="${VYBFORGE_LLAMA:-$HOME/Projects/llama.cpp}"
if [ ! -f "$LLAMA/ggml/include/ggml.h" ] || [ ! -e "$LLAMA/build/bin/libggml.so" ]; then
  step "libggml authority" "SKIP (no llama.cpp checkout + build at $LLAMA)"
  echo; echo "ROPE GATE: SKIP (nothing to measure against)"; exit 0
fi

py=""
for cand in "${ROPE_PY:-}" "$root/.venv/bin/python" python3; do
  if [ -n "$cand" ] && env -u PYTHONPATH "$cand" -c 'import numpy' >/dev/null 2>&1; then py="$cand"; break; fi
done
if [ -z "$py" ]; then
  step "reference interpreter" "FAIL (no python with numpy)"
  echo; echo "ROPE GATE: FAIL"; exit 1
fi

work="${VYBFORGE_ROPE_OUTDIR:-$root/native/out/rope}"
mkdir -p "$work"
log="$work/verify.log"
env -u PYTHONPATH "$py" native/tools/rope_verify.py >"$log" 2>&1
rc=$?

grep -E '^ROPE_VERIFY' "$log" | sed 's/^/      /'

if grep -q "^ROPE_VERIFY_SKIP" "$log"; then
  step "rope op vs candidate layouts" "SKIP ($(grep -m1 '^ROPE_VERIFY_SKIP' "$log" | sed 's/^ROPE_VERIFY_SKIP //'))"
  echo; echo "ROPE GATE: SKIP (nothing ran)"; exit 0
fi

if grep -q "^ROPE_VERIFY_DONE" "$log" && [ "$rc" = "0" ]; then
  summ="$(grep -m1 '^ROPE_VERIFY_SUMMARY' "$log" | sed 's/^ROPE_VERIFY_SUMMARY //')"
  nbad="$(grep -cE 'maxrel=[0-9.e+-]+ differs' "$log")"
  step "rope op vs candidate layouts" "PASS ($summ; $nbad alternatives rejected)"
  echo; echo "ROPE GATE: PASS"
  exit 0
fi

step "rope op vs candidate layouts" "FAIL (see $log)"
echo; echo "ROPE GATE: FAIL"
exit 1
