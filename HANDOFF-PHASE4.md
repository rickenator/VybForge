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

## IMMEDIATE NEXT STEP (option 1)

Finish the third unit: **the l2 normalisation and the gated RMS epilogue.** The semantics are already
pinned down, read first-hand from `ggml/src/ggml-cpu/ops.cpp:4198-4206`:

    ggml_float sum = 0.0;
    for (i00 < ne00) sum += (ggml_float)(xi * xi);
    const float scale = 1.0f/fmaxf(sqrtf(sum), eps);

i.e. **eps is a floor applied AFTER the sqrt**, the accumulation is in **double**, the reciprocal is
`sqrtf` (f32) with an f32 `fmaxf`. A port that puts eps inside the sqrt, or accumulates in f32, will
disagree exactly where the norm is small.

Concretely, mirroring the two existing units:

1. `native/tools/norm_authority.c` — one harness, two modes, because both ops take `(x, eps)`:
   `ggml_l2_norm` (`ggml/include/ggml.h:1407`) and `ggml_rms_norm` (`:1381`). Copy the shape of
   `conv_authority.c` (read floats from a file, build the tensor, call the op, `ggml_set_output`,
   `ggml_new_graph` + `ggml_build_forward_expand` + `ggml_graph_compute_with_ctx`, write the result).
2. `native/tools/norm_verify.py` — the port and the comparison, modelling the l2 norm **per head over
   the 128 state dimensions** (`qwen35.cpp:440-443`) and the epilogue as
   `RMSNorm(output, ssm_norm) * SiLU(z)` (`qwen35.cpp:257-266`).
3. Then commit both, add a doc entry to `doc/QWEN35-PHASE4.md`, and continue outward: the projections
   (`attn_qkv`, `attn_gate`, `ssm_beta`, `ssm_alpha`, `ssm_out` — ordinary `mul_mat`s), then the layer
   wiring, then the Vyb kernel + driver + gate, then flip `eng_gdn()`.

## After that (do not start before item 1 closes)

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
