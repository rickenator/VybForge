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

## Unit 6 — DONE (sixth unit, same session): the kernels run on the GPU

`native/kernels/gdn.vyb` (l2norm, delta_step, norm_gated) + `native/host/gdn_driver.vyb` +
`native/tools/gdn_kernel_verify.py`, gated by `native/legit/run_gdn_kernel_gate.sh` (step **P4.2**,
SKIP without CUDA/toolchain; `make -f native/Makefile gdn` also runs it). One token at the 27B
geometry, fp64 both sides, all 802816 outputs compared element by element via raw bit patterns:

    q_norm 3.5e-16   k_norm 5.2e-16   gdn_out 6.3e-11   state_out 5.4e-11   epi 1.6e-11

The read-out axis is the trap here and it is now the printed negative: the op's read-out is `m @ q`
(contraction on the state's SECOND axis), so `out[j] = scale*(gexp*sum_i state[j,i]*q[i] + delta[j]*kq)`.
Contracting on the first axis instead gave maxrel 1.45e0 on every output while `state_out` stayed
correct — the per-stage comparison localised it in one run. Gate prints it (plus a no-decay variant,
1.98e-01) every run.

Not covered by unit 6: no carry-over of state across calls (one token from a state read off disk),
synthetic fp64 weights, no projections/conv (existing kernels), and no performance work (one thread
per head).

## Unit 7 — DONE (seventh unit, same session): the whole block wired on the GPU

`native/host/gdn_layer_driver.vyb` composes the block out of the Vyb kernels (rmsnorm, mm_nt,
sigmoid_k, alpha_gate, interleave_qkv, conv1d_k, silu_k, l2norm, delta_step, norm_gated, add_k) in
build_layer_attn_linear's order and dumps 12 stages; `native/tools/gdn_layer_kernel_verify.py` feeds
one fixture to it and to unit 5's authority (real ggml ops) and compares stage by stage. Gate:
`native/legit/run_gdn_layer_gate.sh`, step **P4.3**; `make -f native/Makefile gdn-layer`.

Small non-degenerate geometry (n_embd 512, S 32, H_k 4, H_v 8, d_conv 4), one token, state zero, real
ssm_a/ssm_dt: every stage 8.8e-08..3.0e-07 against the authority's f32 floor, with the negatives
printed (residual on the normed input 7.85e-02, no residual 9.45e-01).

Three lessons, all found by the per-stage split in one run each:
* the conv's memory order is FRAME-fastest: ne (ncs, qkv_dim) is `frame + ncs*ch`, the weight's ne
  (d_conv, qkv_dim) is `j + d_conv*ch` (numpy (qkv_dim, ncs) / (qkv_dim, d_conv));
* a projection cannot write the conv buffer directly — `build_conv_state`'s concat interleaves the
  token frames per channel, hence `interleave_qkv` and a conv buffer that starts as zeros;
* a NaN must FAIL a gate: `r > MAXREL` is False for NaN, so a NaN-filled stage printed DIFFERS and the
  gate still reported DONE. It is `not (r <= MAXREL)` now.

## Unit 8 — state carry-over DONE (same session)

The layer driver runs two chained decode steps and the gate compares both (26 stages, 2.4e-08..2.4e-07):
step 2's conv window comes from step 1's qkv via a new `conv_shift` kernel, and step 2's delta-net
state is step 1's new state (the state buffers swap each step). The authority runs the same sequence
as two chained single-token steps. Gate: `run_gdn_layer_gate.sh` (P4.3, now 2 steps).

Trap worth remembering: the fixture grew to two tokens but `DX` was still allocated for one, so an
8192-byte upload into a 4096-byte buffer clobbered its neighbours and made step 1 read zeros — no
device error, just wrong numbers. Device allocation sizes must move with the fixture.

## Unit 9 — DONE for the Q8_0 pair: the layer check runs at model geometry on real weights

`native/tools/gdn_layer_kernel_verify.py` now runs at the 27B's own geometry (n_embd 5120, S 128,
H_k 16, H_v 48, d_conv 4) and `ssm_alpha`/`ssm_beta` are the model's own Q8_0 tensors, read raw from
the GGUF and dequantised on the GPU by the existing `q8_0deq` kernel. 26 stages over two chained
steps, worst maxrel 7.866e-07, gate wall clock 23 s. The driver's launch grids are computed from the
dims now, so the geometry is a constant, not a set of literals.

Measured before changing anything (`native/host/mmnt_bench.vyb`): one full-size 5120 -> 10240 f64
projection — load + 419 MB upload + kernel — is 0.53 s. A single-token step is a matrix-vector product
and is I/O-bound; the ~270 GFLOP estimate that had kept this dormant describes a PREFILL. So no
cluster is needed for this check — the 3090 here is fine — and the Sparks are the right place for the
prefill case (unit 9b) if it needs scheduling.

Also real now: `attn_norm` and `ssm_conv1d` (the model stores both as F32, read raw and used
directly). Four of the layer's weight sets are the model's own.

Q4_K IS NO LONGER THE BLOCKER: `native/tools/q4k_ref.py` (a numpy port of dequantize_row_q4_K) is now
compared element-wise against the independent `gguf` package's Q4_K dequantizer on whole real tensors
— 2 x 31.5 M values, bit-identical — and gated as **S0.10** of the phase-2 battery
(`run_q4k_gate.sh`). The bug worth remembering: the LOW nibble of each `qs` byte belongs to the EVEN
sub-block (2g) and the high nibble to 2g+1; swapped, every value was wrong.

WIRED, and the layer now runs on the model's OWN weights throughout: the three Q4_K projections
(`blk.0.attn_qkv`, `blk.0.attn_gate`, `blk.0.ssm_out`) are uploaded as raw GGUF bytes and dequantised
on the GPU by the existing `q4kdeq` kernel, with the authority fed the same values from `q4k_ref.py`.
Four weight sets are real F32/Q8_0 and three are real Q4_K; nothing in the layer is synthetic any
more. Measured: 26 stages over two chained steps pass — the Q4_K-driven stages at 4.2e-07 (attn_qkv),
8.6e-08/1.7e-06 (ssm_out) — and this also exercises `q4kdeq` on whole real tensors against a reference
that is independently checked (the S0.10 GPU side, as a side effect).

## Unit 9c — NOT STARTED, and deliberately: what it needs, where the hooks are

I stopped short of the engine work rather than starting it half-way. `native/host/model_driver.vyb`
is a 660-line full-model driver for the Qwen3-4B path (36 layers, per-tensor quant staging, one
`stage_one(...)` call per weight at fixed lines ~544-560, kernels from `layer.vyb`/`qwen3.vyb`), and a
recurrent branch is a real piece of engine surgery plus a full 27B run to verify it — not something to
begin at the end of a session's context budget with no way to check the result.

What it needs:

1. **A per-layer dispatch, decided from the tensor table, not the model name**: a recurrent layer has
   `blk.N.attn_qkv.weight`, `blk.N.attn_gate.weight`, `blk.N.ssm_alpha.weight`, `blk.N.ssm_beta.weight`,
   `blk.N.ssm_out.weight`, `blk.N.ssm_conv1d.weight`, `blk.N.ssm_norm.weight`, `blk.N.ssm_a`,
   `blk.N.ssm_dt.bias` where an attention layer has `attn_q/k/v/output`. `model_caps.vyb` already counts
   exactly this way, so the same test belongs in the driver.
2. **The six projections through the existing quant path**: `stage_one(...)` per tensor, then the
   quant gemm the attention layers already use (attn_qkv 5120->10240 Q4_K, attn_gate 5120->6144 Q4_K,
   ssm_beta/ssm_alpha 5120->48 Q8_0, ssm_out 6144->5120 Q4_K) — `mm_nt` is the f64 reference form of
   the same product, so the layer check's numbers are the target.
3. **The state cache, per layer per sequence**: the conv state is `(d_conv-1) * conv_channels` values
   (3 x 10240) and the delta-net state `S * S * H_v` (128 x 128 x 48 = 786432) — llama.cpp calls these
   `n_embd_r()`/`n_embd_s()`; the layer driver in this repo already exercises both, including the
   per-step `conv_shift` + `interleave_qkv` and the state-buffer swap.
4. **The kernel sequence per layer**: rmsnorm(attn_norm) -> the projections -> sigmoid/alpha_gate ->
   conv -> q|k|v split -> l2norm -> delta_step -> norm_gated -> ssm_out -> residual, i.e. exactly
   `gdn_layer_driver.vyb`'s chain, now reading real weights.
5. **Verification, two levels**: per layer against `layer_authority` (already written, and the GPU
   layer check is already at model geometry on the model's own weights), then the whole model's logits
   against llama.cpp on the same GGUF — the same "independent oracle" standard as the S0.4 decode
   oracle, because per-layer agreement cannot see a wiring error between layers.

**And a sequencing fact that changes the finish line**: the Ridge file has 65 blocks and the 65th is
the MTP head, so `eng_gdn()` alone does not make it runnable — `eng_mtp()` (item 2) must flip too, or
the descriptor needs a documented "run without the draft head" profile. The caps gate asserts the
refusal list SHRINKS as well as the new count, so it will say which of the two is still refusing.

## `eng_gdn()` is STILL 0, deliberately

Do not flip it yet. The descriptor's `eng_gdn()` decides whether the caps gate reports
`UNSUPPORTED_LAYER_KIND gated-deltanet` for the Ridge model, and the engine path still has no recurrent
layer: no multi-token/prefill, no state carry-over across a sequence, no quantized weight loading, and
nothing wired into `model_driver`/the chat server. Flipping it would make the descriptor claim a
capability the engine cannot deliver.

## IMMEDIATE NEXT STEP (unit 9) — what the flip still needs

State carry-over is done (unit 8). Three things remain before `eng_gdn()` can honestly flip:

1. **Quantized weights**: the mechanism is in place (unit 9, above); what is missing is a projection
   kernel fast enough to run the layer check at the model's real geometry, or routing the projections
   through the quant gemm path `layer_driver.vyb` already uses.
2. **The multi-token/prefill path** — the op's chunked kernel, still uncharacterised (see below); a
   real prompt cannot run without it, and prefill is what makes 48 layers affordable.
3. **Engine integration**: a recurrent branch in `model_driver.vyb`'s per-layer loop, then the flip and
   a caps-gate re-run (the gate asserts the refusal list SHRINKS as well as the new count, skill
   gate-and-probe §6), plus re-running P4.1-P4.3.

## After that (do not start before unit 9 closes)

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
