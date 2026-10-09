#!/usr/bin/env bash
# P4.5 the recurrent block on the ENGINE's own weight path — VybForge#10 phase 4, unit 9c step 2.
#
# What it proves, and why P4.3 is not enough. P4.3 wired the block out of Vyb kernels, but it fed
# the driver a FIXTURE and multiplied with `mm_nt`, whose B operand is [out,in]. The engine does not
# work that way: it finds each tensor by NAME in the GGUF, dequantises the packed types on the GPU
# with the kernels it uses for every layer, and multiplies with `gemm` (layer.ptx), whose B operand
# is [in,out] — the layout the dequant kernels' transposing write produces (VybForge#11). This gate
# runs native/host/gdn_engine_driver.vyb, which stages all ten of blk.0's tensors from the live GGUF
# through that path, and compares 13 stages over TWO chained decode steps against unit 5's authority
# (the real ggml ops), on the model's own weights. Measured: worst maxrel 3.9e-06.
#
# The check has teeth in three ways:
#   * the STAGED OPERAND is compared slot by slot against the model's own dequantised ssm_out
#     tensor, at the exact addresses gemm reads (B[k*N+n]) — a good block fed a wrong operand is
#     invisible to stage comparison alone;
#   * the same run prints two mis-wirings computed from the authority's own steps (no residual
#     1.0e+00, residual on the normed input 9.0e-02) against a 1e-4 bar;
#   * NaN fails: the verifier tests `not (r <= bar)`, never `r > bar`.
#
# Wall clock ~40 s: one full-size projection is I/O-bound (native/host/mmnt_bench.vyb).
#
# SKIPs (never PASSes) without a Vyb toolchain, without libggml for the authority, without CUDA, or
# without the Ridge model (a 12 GiB download). A gate that proved nothing is a FAIL.
set -u
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$root"

step() { printf '%-46s %s\n' "$1" "$2"; }
echo "GDN engine-path gate — $(date '+%F %T')"
echo

if ! . "$root/vybenv.sh" >/dev/null 2>&1 || [ ! -x "$VYB" ]; then
  step "Vyb toolchain" "SKIP (no compiler)"
  echo; echo "GDN ENGINE GATE: SKIP (no toolchain — nothing to run)"; exit 0
fi
LLAMA="${VYBFORGE_LLAMA:-$HOME/Projects/llama.cpp}"
if [ ! -f "$LLAMA/ggml/include/ggml.h" ] || [ ! -e "$LLAMA/build/bin/libggml.so" ]; then
  step "libggml authority" "SKIP (no llama.cpp checkout + build at $LLAMA)"
  echo; echo "GDN ENGINE GATE: SKIP (nothing to compare against)"; exit 0
fi
echo "toolchain: $VYB"
echo

py=""
for cand in "${GDNE_PY:-}" "$root/.venv/bin/python" python3; do
  if [ -n "$cand" ] && env -u PYTHONPATH "$cand" -c 'import numpy' >/dev/null 2>&1; then py="$cand"; break; fi
done
if [ -z "$py" ]; then
  step "reference interpreter" "FAIL (no python with numpy)"
  echo; echo "GDN ENGINE GATE: FAIL"; exit 1
fi

work="${VYBFORGE_GDNE_OUTDIR:-$root/native/out/gdn_engine}"
mkdir -p "$work"
log="$work/verify.log"

# the engine's tensor index is generated from the inventory, never hand-written
tsv="${VYBFORGE_RIDGE_TSV:-$root/native/out/ridge_tensors.tsv}"
if [ ! -f "$tsv" ]; then
  if env -u PYTHONPATH "$py" native/tools/inventory_to_tsv.py --out "$tsv" >"$work/tsv.log" 2>&1; then
    step "engine tensor index" "generated ($(grep -oE 'tensors=[0-9]+' "$work/tsv.log" | tail -1))"
  else
    step "engine tensor index" "FAIL"
    tail -4 "$work/tsv.log" | sed 's/^/      /'
    echo; echo "GDN ENGINE GATE: FAIL"; exit 1
  fi
else
  step "engine tensor index" "present ($(wc -l <"$tsv" | tr -d ' ') tensors)"
fi

env -u PYTHONPATH "$py" native/tools/gdn_engine_verify.py >"$log" 2>&1
rc=$?

grep -E '^GDNL_ENGINE_VERIFY' "$log" | sed 's/^/      /'

if grep -q "^GDNL_ENGINE_VERIFY_SKIP" "$log"; then
  step "engine path block (2 steps, 26 stages)" "SKIP ($(grep -m1 '^GDNL_ENGINE_VERIFY_SKIP' "$log" | sed 's/^GDNL_ENGINE_VERIFY_SKIP //'))"
  echo; echo "GDN ENGINE GATE: SKIP (nothing ran)"; exit 0
fi

if grep -q "^GDNL_ENGINE_VERIFY_DONE" "$log" && [ "$rc" = "0" ]; then
  nst="$(grep -cE '^GDNL_ENGINE_VERIFY s[12]_[a-z0-9_]+ +n=' "$log")"
  worst="$(grep -oE 'worst=[0-9.e+-]+' "$log" | tail -1 | sed 's/worst=//')"
  alt="$(grep -oE 'rejected_alternatives=[0-9.e+/.+-]+' "$log" | tail -1 | sed 's/rejected_alternatives=//')"
  step "engine path block ($nst stages)" "PASS (worst ${worst}, steps=2, rejected ${alt})"
  echo; echo "GDN ENGINE GATE: PASS"
  exit 0
fi

step "engine path block (2 steps, 26 stages)" "FAIL (see $log)"
echo; echo "GDN ENGINE GATE: FAIL"
exit 1
