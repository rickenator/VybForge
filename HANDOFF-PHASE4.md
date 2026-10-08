# Phase 4 handoff — Gated DeltaNet reference (VybForge #10, item 1)

Written at the end of a long session whose context window is spent. Everything below is pushed;
`~/Projects/VybForge` main is at `a7273e4`, tree clean. Read `doc/QWEN35-PHASE4.md` for the full
narrative and `doc/SPIKINGBRAIN.md` for the phase-1..3 background.

## The goal, and where phase 4 sits

Phase 4 = the layer wiring that unblocks the 27B. The descriptor names exactly two blockers, both
hardcoded in `native/config/model_caps.vyb`:

    :140  eng_gdn()<Int> -> { return 0 }   // Gated DeltaNet layers (48 of 64)
    :141  eng_mtp()<Int> -> { return 0 }   // native draft head

Item 1 is the recurrent (Gated DeltaNet) layer's maths. It proceeds as: a numpy reference for each
unit, checked against ggml's OWN implementation; only then a Vyb kernel + driver + gate; the
descriptor flips last.

## What is verified (do not re-derive these)

Both use the same technique: a C harness that LINKS the libggml llama.cpp was built with
(`~/Projects/llama.cpp/build/bin`, headers in `ggml/include`) and calls the real op, plus a numpy port
compared against it.

| unit | harness | port/check | measured |
|---|---|---|---|
| Gated DeltaNet recurrence | `native/tools/gdn_authority.c` | `native/tools/gdn_verify.py` | `maxrel out=1.555e-07 state=8.651e-08` |
| short convolution (`SSM_CONV`) | `native/tools/conv_authority.c` | `native/tools/conv_verify.py` | `maxrel 7.606e-08` at d_inner=10240 |

Both figures are the authority's own f32 floor, i.e. agreement, not slack. Also established by data:
the q/k→v head broadcast is **`h % H_k`** (interleaved), and the op's result buffer is scores
(`ne0`-fastest) followed by the new state.

## THE LESSON THAT COST THE MOST — read this first

**ggml's `ne[0]` is the FASTEST axis; numpy's fastest axis is the LAST one.** An ggml tensor of ne
`(S,H,T,B)` is a numpy array of shape `(B,T,H,S)`, and the state `(S_v,S_v,H,B)` is `(B,H,S_v,S_v)`
with the last axis `i` and the one before it `j`. Reading the leading axes as if they were ggml's
makes every tensor transposed.

The tell, and the reason it hid for so long: at the minimal geometry (S=4, H=1, T=1, B=1) every stride
is 1, so a transposed read coincides EXACTLY with a correct one — that case matched to 6e-8 while
nothing at model geometry did. **If a small case passes and the real one fails, suspect axis order
before suspecting the target.** Two conclusions recorded earlier in `doc/QWEN35-PHASE4.md` (a
delta-function scan and an identical-heads test) were artefacts of exactly this and are struck; the
op was right all along. Full write-up: the `vybos-development` skill,
`references/gate-and-probe-discipline.md` items 10 and 11.

## Third unit — DONE (2026-10-08, commit 13fd444)

The l2 norm, the RMS norm and the gated epilogue are verified against ggml's own ops:
`native/tools/norm_authority.c` (modes `l2` / `rms` / `epilogue`) + `native/tools/norm_verify.py`.
Measured at model geometry: l2 and rms **bit-exact** (`maxrel 0.000e+00`), epilogue `8.775e-08`
(SiLU `expf` vs numpy). The two eps conventions are opposite — l2 floors eps AFTER `sqrtf(sum)`, rms
puts eps INSIDE `sqrtf(mean+eps)` — and each verifier prints the opposite convention's error
(7.1e-02 / 6.6e-03) so the agreement is a measurement, not a tolerance nothing could breach.

All three unit verifiers now live in a gate instead of only in a transcript:
`native/legit/run_gdn_ops_gate.sh` (SKIPs without the llama.cpp checkout) is step **P4.1** of
`native/legit/run_phase2_battery.sh`.

## Unit 4 — DONE (fourth unit, same session)

The five projections and the beta/alpha gates are verified against `ggml_mul_mat` / `ggml_sigmoid` /
`ggml_softplus`: `native/tools/mm_authority.c` (+ `mm_verify.py`), at the 27B Ridge geometry read from
the GGUF. Measured: the projections at 4.3e-07..5.3e-07 (the f32 floor of a K=5120 dot), beta
2.7e-06, alpha 2.8e-07 against the real `blk.0.ssm_dt.bias` / `blk.0.ssm_a`.

Three conventions pinned, each with its rejected alternative measured:
* `ggml_mul_mat(w, x)` is `w^T x`, so a weight of ne (K, N) is a numpy (N, K) array. Written the other
  way first, every projection came back at maxrel ~1.3 — the recurrence's axis trap, hit again, and
  now caught by the verifier's transposed-read check (~1.2, must be large).
* `softplus` is `(x > 20.0f) ? x : logf(1.0f + expf(x))` — threshold at 20, no `log1p`; both the
  `log1p` and the threshold-less ports diverge to inf. Not a corner case here: the real `ssm_dt` bias
  reaches 19.25 and 23/192 values land in the threshold branch.
* `ssm_a` is stored pre-exponentiated (`SSM_A_NOSCAN`; qwen35.cpp comments it `-A_log.exp()`), so do
  not re-apply the exp. The verifier reads it from the GGUF and asserts all-negative.

`run_gdn_ops_gate.sh` now runs all five units (still step **P4.1** of the phase-2 battery).

## Unit 5 — DONE (fifth unit, same session)

The whole recurrent block is verified wired stage by stage: `native/tools/layer_authority.c` builds it
from the real ops in one graph (attn_norm → wqkv/wqkv_gate → beta/alpha gates → conv + SiLU → q|k|v
views → l2 norms → gated_delta_net → gated epilogue → ssm_out → attn residual) and dumps 14 named
stages; `native/tools/layer_verify.py` assembles the four earlier ports in the same order and compares
each stage. Measured at the 27B geometry with T=1, zero state and the model's real ssm_dt/ssm_a: every
stage at the f32 floor (worst 5.58e-07 on `y`, layer_out 1.21e-07), and the check has teeth — dropping
the epilogue's SiLU(z) costs 2.95e-01, taking the residual on the normed input 8.79e-02.

`run_gdn_ops_gate.sh` now runs all five units (still step **P4.1** of the phase-2 battery).

Not covered by unit 5: the FFN half of the block (dense path, already gated in phase 2), a NON-empty
delta-net/conv state (one step from a fresh sequence only), and real weights (synthetic on purpose —
op semantics are what is under test).

## IMMEDIATE NEXT STEP (unit 6)

**The Vyb kernel for the recurrence + driver + gate.** The reference work for item 1 is complete:
five units, all passing against the real ops. The next piece of the descriptor's blocker
(`eng_gdn()` in `native/config/model_caps.vyb:140`) is Vyb code, not reference code:

1. the kernel (`native/kernels/`): the l2 norm, the delta-rule step and the gated epilogue are the
   natural first targets — they are elementwise/reduction work with no matmul, so they exercise the
   Vyb CUDA path without needing a GEMM; the projections and the conv can reuse the existing gemm and
   FIR kernels (check what `native/kernels/` already has before writing anything).
2. the driver (`native/host/`) + a `native/legit/run_*_gate.sh` in the shape of the quant gates
   (SKIP without model/GPU, FAIL if it proved nothing), wired into the phase-2 battery as P4.2.
3. then the layer wiring in the driver, then flip `eng_gdn()`.

Gate discipline to reuse from the five units: compile/run against the LOCAL authority, dump results
and compare per stage, and print the rejected alternative's error so a check that cannot fail is
visible.

## After that (do not start before unit 6 closes)

* **The prefill/multi-token path.** The op has TWO kernels: with one token it runs the sequential rule
  the port implements; with several it runs a chunked one that fills the buffer differently — proven
  by feeding identical tokens at T=1,2,3 and watching token 0's output change, which no causal
  recurrence can do. Until that is characterised, the reference must refuse multi-token comparison
  (`--tokens 1`), and no real prompt can run.
* Then `eng_mtp()` (the 65th block + the `d2t` tensor) — smaller, but buys no tokens alone.

## Environment / gotchas

* Real runs need `env -u PYTHONPATH` (the session's PYTHONPATH shadows the repo `.venv`).
* Harnesses must be compiled against the built libggml:
  `-I ~/Projects/llama.cpp/ggml/include -L ~/Projects/llama.cpp/build/bin -lggml -lggml-base -lggml-cpu -Wl,-rpath,<that bin>`.
* `llama-cli` **dumps core** on this box (even on `--help`). `llama-server` is the working entry point
  for anything end-to-end; `llama.cpp` at `4df29be4` supports `qwen35`.
* `~/Projects/llama.cpp` has 27 uncommitted lines in `src/models/qwen35.cpp` that are NOT ours (the
  sibling `build-fastmtp` tree suggests MTP experimentation) — leave them alone.
* A guarded debug patch for the recurrent op is saved at `native/tools/ggml_gdn_debug.patch`
  (`GGML_GDN_DEBUG=1`, dumps tensors and the kernel's intermediates). Apply it to
  `ggml/src/ggml-cpu/ops.cpp` and rebuild `ggml-cpu` when one more instrumented run is needed; that
  build currently sits pristine and rebuilt clean.
* `native/tools/gdn_ref.py` still carries the OLD transposed indexing (from before the root cause was
  found). Treat `gdn_verify.py` as authoritative; retire `gdn_ref.py` or rewrite it onto the same
  convention when convenient. Its stale numbers in `doc/QWEN35-PHASE4.md` are labelled as such.
* Neither verifier is wired into the Makefile or the Phase-2 battery yet — they are one-off scripts.
  Wiring them so they stay green is a small, worthwhile task.
