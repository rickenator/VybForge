# Phase 4 — make the ENGINE run the Ridge 27B (hybrid Gated-DeltaNet + attention)

This file is the mission brief: it should be enough to finish the job without re-deriving anything.
Everything stated as measured below was measured; everything stated as not done is not done. Read it
top to bottom once, then work the W-numbered items in order.

Read alongside it: `doc/QWEN35-PHASE4.md` — the phase narrative, unit by unit, including every
measurement AND every refuted hypothesis (units 1-9, then unit 10 steps 1-5). Where this file and the
narrative disagree, the narrative is the record and this file is the plan; fix this file.

---

## 1. MISSION — the finish line, precisely

Make the engine run `qwen35` (Qwen3.8-27B "Ridge"). Done means ALL of:

1. **`eng_gdn()` = 1** in `native/config/model_caps.vyb` and the caps gate reports the Ridge model
   without `UNSUPPORTED_LAYER_KIND gated-deltanet` — i.e. a full 64-block forward runs, with the 48
   recurrent layers carrying their conv window and delta-net state across the sequence.
2. **`eng_mtp()` = 1** and the draft head (blk.64) is verified — the refusal
   `UNSUPPORTED_CAPABILITY mtp nextn_predict_layers=1` is gone. The descriptor refuses BOTH by design:
   a 65-block file is not runnable while either is missing.
3. **A whole-model Ridge run is verified against an INDEPENDENT oracle** — llama-server on the same
   GGUF, temp 0 — using the S0.4 standard: final hidden + per-position top1 vs the oracle, and a real
   prompt whose top-2 margin is decisive. Per-layer agreement cannot see an error BETWEEN layers, so
   this is the acceptance test, not the per-layer gates.
4. The phase-2 battery green, including the new steps for whatever lands.
5. `eng_vision()` stays 0: the 27B text GGUF has no vision tower (`mmproj-*.gguf` is a separate file),
   so text-only runs need nothing from it.

NOT in scope for this mission: macOS, performance work, prefill-scale distribution (a single-token
step is I/O-bound; only prefill is throughput-bound and the 3090 handles it), and speculative decoding
as a *feature* (item 2 only requires the MTP block to be implemented and verified, since the
descriptor claims it).

## 2. STATE — what is verified, and what is not

### The model

    ~/Models/qwen38-27b-ridge/Qwen3.8-27B-Ridge-3.7bpw.gguf   12.6 GB   (VYBFORGE_RIDGE_GGUF overrides)
    arch=qwen35  65 blocks = 64 text + 1 MTP
    D=5120  H=24  KVH=4  HD=256  FF=17408  VOCAB=248320  CTX=262144
    rope: base 1e7, dimension_count 64 (rotate 64 of each 256-dim head), sections [11,11,10,0] (INERT)
    eps=1e-6  attention 16 of 64 blocks, `blk.N` with N % 4 == 3 (`full_attention_interval=4`)
    ssm: state_size 128, conv_kernel 4, inner_size 6144, time_step_rank 48, group_count 16
    nextn_predict_layers=1     UNTIED head: `output.weight` (Q6_K, 5120x248320) is present

Tensor types (from `native/gguf/ridge-3.7bpw-inventory.tsv`, which is generated, never hand-edited):

    role/type            count   where                                   engine staging
    gdn_state/F32         192    ssm_conv1d, ssm_norm, ssm_dt, ssm_a      load_norm            OK
    gdn_state/Q8_0         96    ssm_beta, ssm_alpha                       load_quant(q8fn)     OK
    gdn_state/Q4_K         48    ssm_out                                   load_quant(q4fn)     OK
    gdn_mixer/Q4_K         96    attn_qkv, attn_gate                       load_quant(q4fn)     OK
    attention/F32          96    attn_norm, attn_q_norm, attn_k_norm       load_norm            OK
    attention/Q5_K         48    attn_q (JOINT q+gate), attn_k, attn_v     stage_one ty==13     OK (step 5)
    attention/Q6_K         16    attn_output                               stage_one ty==14     OK
    norm/F32               65    per-block norms                           load_norm            OK
    ffn/IQ2_S             160    blk.4-11 gate/up, blk.12-59 all three     *** MISSING ***      -> W1
    ffn/IQ3_S              32    blk.0-3 all three, blk.4-11 down,
                                 blk.60-63 all three                       *** MISSING ***      -> W1
    mtp/Q5_K                3    blk.64.attn_q (joint), attn_k, attn_v     stage_one ty==13     OK
    mtp/Q6_K                5    blk.64.attn_output, ffn_gate/up/down,
                                 nextn.eh_proj                             stage_one ty==14     OK
    mtp/F32                 7    blk.64 norms incl. nextn.enorm/hnorm/
                                 shared_head_norm                          load_norm            OK
    embed_head/Q6_K         2    token_embd, output.weight                 see W2               W2

`IQ2_S` is 82 bytes per 256 values, `IQ3_S` 110, `Q5_K` 176, `Q6_K` 210, `Q4_K` 144. The FFN type
boundary is NOT by block kind: blocks 0-3 and 60-63 are all IQ3_S, blocks 4-11 have IQ3_S `ffn_down`
with IQ2_S gate/up, blocks 12-59 are all IQ2_S.

### The descriptor's own verdict right now

    ./native/legit/run_caps_gate.sh        # or: make -f native/Makefile caps
    CAPS_LAYOUT=hybrid-recurrent  N_LAYERS=65 TEXT_LAYERS=64 ATTN_LAYERS=16 GDN_LAYERS=48 MTP_LAYERS=1
    CAPS_NEXTN_TENSORS=4  ATTN_INTERVAL=4  VISION=0
    CAPS_TYPES=F32:360,Q8_0:96,Q4_K:144,Q5_K:51,Q6_K:23,IQ3_S:32,IQ2_S:160
    CAPS_UNSUPPORTED=UNSUPPORTED_LAYER_KIND gated-deltanet layers=48,
                     UNSUPPORTED_CAPABILITY mtp nextn_predict_layers=1
    CAPS_VERDICT=UNSUPPORTED

Two things to know about that output. (a) There is NO quant-type refusal, because `eng_type()` in
`native/config/model_caps.vyb` returns 1 for IQ2_S/IQ3_S — it means "a kernel exists", and the kernels
do exist and are gated. But the model PATH cannot stage those types yet (W1). The descriptor therefore
over-claims relative to staging, and nothing currently catches that; see W1's last bullet.
(b) `eng_gdn()`/`eng_mtp()` are the only refusals, which is exactly the finish line in §1.

### Verified — do not re-derive (each one has a gate; numbers are the recorded measurements)

| gate | step | proves | measured |
|---|---|---|---|
| `run_gdn_ops_gate.sh` | P4.1 | the GDN op references (recurrence, conv, l2/rms norms, gates, projections) against ggml's own ops | maxrel 1.6e-07..5e-07 at model geometry |
| `run_gdn_kernel_gate.sh` | P4.2 | the GDN GPU kernels | 8.0e+05 outputs, worst 6.3e-11 (fp64 both sides) |
| `run_gdn_layer_gate.sh` | P4.3 | the recurrent block WIRED out of Vyb kernels, 2 chained steps | 26 stages, worst 2.4e-07 |
| `run_layerkind_gate.sh` | P4.4 | the per-block KIND dispatch (`mc_layer_kind`) on both models vs an independent parser | per block, both models |
| `run_gdn_engine_gate.sh` | P4.5 | the recurrent block on the ENGINE's weight path (GGUF by name + its own dequant + `gemm`) | 26 stages / 2 steps, worst 3.9e-06 |
| `run_rope_gate.sh` | P4.6 | the rope SPEC: NEOX inside n_dims=64, 11 alternatives rejected | 8.993e-08 |
| `run_rope_kernel_gate.sh` | P4.7 | `rope_nrot` on the GPU at n_rot=64 AND (must-miss) n_rot=HD | 8.9e-08 / 7.0e-08 vs 1.703 |
| `run_attn_block_gate.sh` | P4.8 | Ridge's attention block's ten stages against ggml (synthetic fixture), incl. the GQA-replicated case | worst 3.9e-07..4.6e-07, 6-7 alternatives rejected |
| `run_attn_engine_gate.sh` | P4.9 | the same block on the ENGINE's weight path (blk.3, blk.7), ten stages, real Q5_K/Q6_K weights | worst 1.668e-06 / 1.061e-06 |
| `make prefill` | — | the dense Qwen3-4B regression (the path every engine change must not disturb) | bit-identical: maxrel 3.393e-04, top1 [55286, 576] |
| S0.5/S0.6/S0.7/S0.8/S0.9/S0.10 | — | Q8_0, IQ2_S, Q5_K, IQ3_S, BF16, Q4_K dequant kernels vs the compiled upstream C | byte-exact (0 ulp) |
| `run_caps_gate.sh` | S0.2e | the capability descriptor, 4 cases | see above |

### NOT done — this is the work list

* **W1** the FFN of every Ridge block cannot be STAGED (IQ2_S 160 + IQ3_S 32). Kernels gated; the
  staging path lacks the types, and those two kernels take a GRID argument so the 4-arg loader cannot
  launch them. Until W1, an attention probe must stop at `out` (it does, deliberately) and no whole
  Ridge layer can run.
* **W2** the UNTIED lm_head and the 248320-row embedding table. `output.weight` exists; the driver
  uses `token_embd` as if tied. A full f64 dequant of either table is 10.2 GB → both must be chunked.
* **W3** no whole-model Ridge forward has ever run; the loop has only run single blocks in probe mode.
* **W4** the MTP head (blk.64) is not implemented at all.
* **W5** `eng_gdn()`/`eng_mtp()` are still 0 (correctly — see the rule in §5).
* **W6** (not required for the flip) multi-token / prefill for the recurrent op: the op's chunked
  kernel is UNCHARACTERISED, and only single-token steps are verified. A prompt can still be run as S
  single-token steps, which is what the loop does today.

## 3. THE WORK, in order of dependency

### W1 — stage IQ2_S (82 B/256) and IQ3_S (110 B/256): this unblocks the FFN

Facts you need, all checked:

* The kernels exist and are BYTE-EXACT gated: `native/kernels/iq2s.vyb` (`iq2sdeq`) and
  `native/kernels/iq3s.vyb` (`iq3sdeq`), gates `run_iq2_s_gate.sh` / `run_iq3_s_gate.sh`.
* Their signature is FIVE arguments — `iq2sdeq(grid, q, o, n, z)` — with the codebook grid as an
  ordinary pointer argument, so they are launched with `cuda_launch_n` (Vyb#476), NOT
  `cuda_launch4i`. **`model_driver.vyb` declares only `cuda_launch4i` today**: add
  `cuda_launch_n(f<loc<CVoid>>, gx, gy, gz, bx, by, bz, kernelParams<loc<CVoid>>, nargs<CUInt>)<CInt>`
  to its `extern "C"` and write a staging helper that builds the cell-address array (the pattern is in
  `native/host/iq3_s_load_driver.vyb`, and gate-and-probe-discipline §8 has the shape).
* The grid images are files: `native/out/iq2s_grid.bin` (8192 B) and `native/out/iq3s_grid.bin`
  (2048 B), written by `native/tools/iq2_s_ref.py` / `iq3_s_ref.py` (mechanically extracted from
  upstream by `gen_iq2s_tables.py`, commit recorded in the header). The driver must length-check them
  and upload each ONCE (they are the same for every tensor).
* `z` keeps the same meaning as the other dequant kernels: `z > 0` transposes the whole-matrix write,
  and the value passed is the tensor's IN dim (`packed_bytes`-style numel is the N dimension). Same
  convention as `q4kdeq`/`q5kdeq`/`q6kdeq` — read those comments before writing the call.
* `packed_bytes` in `model_driver.vyb` needs `if (ty == 22) { return numel * 82 / 256 }` and
  `if (ty == 21) { return numel * 110 / 256 }`.
* `stage_one` needs two more cases, and because the launcher differs, give the grid kernels their own
  helper rather than another positional argument to `stage_one` (its 13 call sites already carry
  `q4fn, q5fn, q6fn, efn`; a 5th function pointer plus a grid pointer is where that signature stops
  being readable). Suggested: `load_quant_grid(path, dpk, dout, off, nb, numel, qfn, gridDev, inz)`.
* The reference for the FFN check is OUR ports (`iq2_s_ref.iq2_s_ours(raw)`,
  `iq3_s_ref.iq3_s_ours(raw, kmask, grid)`), each verified element-wise against llama.cpp's own
  compiled dequantizer in its gate. Do NOT use the python `gguf` package for these types — it has them
  in its enum and no dequantizer, and "our numpy agrees with our numpy" is exactly the self-consistency
  the authority rule forbids.

Acceptance: extend the probe gate with the FFN. Cheapest honest version — a `VYB_FFN_PROBE=<layer>`
(or an FFN section inside `VYB_ATTN_PROBE`) that stages the block's three FFN weights + its
`post_attention_norm`, runs attn_post_norm → gate/up → silu → down → residual, and dumps the stages
(`ffn_xn`, `ffn_gate`, `ffn_up`, `ffn_silu`, `ffn_down`, `block_out`). Compare against a numpy
reference assembled from the model's OWN dequantised tensors (the GDN/attention verifiers are the
template: `native/tools/gdn_engine_verify.py`, `native/tools/attn_engine_verify.py`), with
(a) an OPERAND probe at the addresses `gemm` reads (`B[k*N+n]`), (b) the residual mis-wirings printed
as negatives, (c) `not (r <= bar)` so NaN fails. Then wire it as P4.10 and prove it can FAIL (§21).
Bar: ~1e-4 is right here, because the authority's operands come from our ports at f32 while the engine
is f64 — say so in the gate header instead of implying an exact-path tolerance.

Last bullet, and it is a real gap: when W1 lands, `eng_type(21)`/`eng_type(22)` are *finally* true of
the model path as well as of the kernels. Until then the descriptor over-claims; record that in the
gate header, and consider a caps-probe line that distinguishes "kernel exists" from "the model path can
stage it" (the probe reads `eng_type` only, so today it cannot see the difference).

### W2 — the UNTIED head and the chunked embedding table (needed for W3's oracle comparison)

* `model_config` already derives this: `ModelConfig.tied` is 0 when the tensor table has
  `output.weight` (`native/config/model_config.vyb`, the `tied(output.weight present)` branch). Ridge
  is therefore tied=0 and `output.weight` is Q6_K 5120x248320. The driver currently ignores `cfg.tied`
  and dequantizes `token_embd` into `DE`, using it for BOTH the embed gather and the lm_head.
* Arithmetic that decides the design: a full f64 table is `248320 * 5120 * 8 = 10.17 GB` per table —
  two of them plus the 3 GB of layer weights plus the 402 MB GDN state cache will not fit a 24 GB
  card. So:
  * **embed**: dequantize only the rows the prompt needs. One token row = 5120 values = 20 Q6_K blocks
    of 210 B = 4200 bytes. A tiny kernel (or a `q6kdeq` launch over just those blocks with the right
    offsets) builds the S rows into a `S*D` f64 buffer; the dense path's `embed` gather kernel can then
    stay as it is, or be bypassed for a per-row build. Do NOT allocate `DE` for Ridge.
  * **lm_head**: dequantize a VOCAB CHUNK into a reusable f64 buffer and run `gemm` over it, chunk by
    chunk (16384 rows → 671 MB). The existing `logits_slice` kernel assumes a fully dequantized table,
    so either extend it to take a chunk or drive `gemm` per chunk and keep the argmax running — state
    which you did. This is also the shape the MTP head's shared head needs (W4).
* Acceptance: the numeric gate on a real prompt — per-position top1 + final hidden vs `llama-server`
  at `-ngl 0` temp 0 (see §6 for the oracle recipe). Pin the oracle's identity (llama.cpp revision +
  GGUF id) in the fixture and make a provenance mismatch a FAIL that says "recapture required".

### W3 — the whole-model Ridge forward

With W1 and W2 in place the loop has everything:

* the per-layer dispatch is already the shared rule (P4.4) and the loop asks it (`lkc` in
  `model_driver.vyb`);
* the recurrent branch is staged and verified per layer (P4.5) and carries conv + delta-net state
  across S single-token steps (the loop's `for (gst in 0..S-1)`), with the 2-slot-per-layer state
  cache allocated and zeroed per sequence;
* the attention branch runs the whole block (P4.9) — extend it past `out` once W1 lands, so a whole
  Ridge layer is covered end to end;
* the FFN branch is the SAME code the dense model uses, and it is already exercised by `make prefill`.

What has never happened: all 64 blocks in one process. Expect the class of failure in
gate-and-probe-discipline §28 — "a hybrid model exercises the loader paths a dense one leaves dormant":
buffers sized from the wrong template layer, a tensor name that differs by architecture, a device
buffer whose name shadows a geometry value, a large cache zeroed through the 8-byte helper.

Acceptance: `MODEL_PREFILL_DONE` + the oracle comparison of W2, on a real prompt, with the hidden
comparison as the primary signal (the synthetic `[0,1]` prefix is a coin flip for top1 — see §5).

### W4 — the MTP head (blk.64): `eng_mtp()`'s evidence

The block is a FULL attention block plus an FFN plus the NextN-specific tensors, and llama.cpp's own
`graph_mtp` is the recipe — copy it with file:line in your harness (`llama.cpp/src/models/qwen35.cpp`,
`graph_mtp`, lines 498-671):

    e        = rms(embed(next_token_ids), nextn.enorm)        # the SHARED token_embd (no nextn.embed_tokens here)
    h        = rms(hidden, nextn.hnorm)
    concat   = concat(e, h, ne0)                              # 2*D, EMBEDDING FIRST, then hidden  (line 559)
    cur      = eh_proj @ concat                               # eh_proj ne (2*D, D) -> numpy (D, 2*D)
    inpSA    = cur
    cur      = attn_norm(cur)  -> wq (JOINT q+gate: q FIRST, gate SECOND) -> attn_q_norm per head
                               -> wk -> attn_k_norm per head -> wv
                               -> partial rope (n_dims 64, same sections, same base)
                               -> GQA attention (kv = h/(H/KVH), same kernel)
                               -> cur * sigmoid(gate) -> wo
    cur      = cur + inpSA
    cur      = cur + ffn(attn_post_norm(cur))                 # SILU, parallel gate/up   (lines 621-632)
    logits   = shared_head_norm(cur) @ output.weight          # nextn.shared_head_norm here; head = model.output

* **`d2t` does not exist in this file.** It is a DRAFT-VOCAB TRIM table (`n_vocab_out` rows) that
  llama.cpp only uses when the GGUF is MTP-ONLY (`mtp_only`) and only if `output.weight` is present
  (lines 40-61, 653-667). An earlier revision of this handoff called the draft head "the 65th block +
  the d2t tensor" — that is wrong for this model, and a reader who went looking for `d2t` would waste
  a session. The file has exactly four `nextn.*` tensors: `eh_proj`, `enorm`, `hnorm`,
  `shared_head_norm` (which is what `CAPS_NEXTN_TENSORS=4` counts).
* All of blk.64's types are already stageable after step 5 (Q5_K joint attn_q + Q6_K + F32) — W1 is
  NOT a prerequisite for the MTP head, only W2's chunked head is.
* Acceptance: an oracle comparison of the head's logits for a given (hidden, next token) pair against
  llama.cpp. `llama-server`'s speculative/MTP path is the natural oracle if this revision exposes it
  (`~/Projects/llama.cpp` at 4df29be4 has `--draft`/speculative machinery and `graph_mtp` as
  `LLM_GRAPH_TYPE_DECODER_MTP`); if the server cannot emit it directly, build a small C harness that
  links libggml and calls the same graph — the harness pattern is `native/tools/attn_authority.c`.
  Verify the draft head does not change the MAIN logits (it must not: it is not in the main pass).
* Wire it as a gate + battery step and prove it can FAIL, as always.

### W5 — flip the descriptor, last

`native/config/model_caps.vyb`:

    eng_gdn()<Int> -> { return 1 }      // only after W3's oracle comparison is green
    eng_mtp()<Int> -> { return 1 }      // only after W4 is green
    eng_vision()<Int> -> { return 0 }   // stays: the text GGUF has no vision tower

Then re-run `run_caps_gate.sh`. Its assertions are on the REFUSAL LIST, not just the count (S0.2e /
unit-19): the two names above must disappear and nothing else may appear. Read the step that fails if
one does — it names the flag still refusing, so never hunt for anything else first.

The rule that keeps this honest: **a descriptor flag claims a whole capability at once** — weight
loading, per-sequence state carry-over, the multi-token/prefill path, the engine wiring, and (for MTP)
the whole draft head. "Our kernels reproduce the layer to 1e-6" is evidence about KERNELS and says
nothing about those. Flip on the oracle comparison, not on the per-layer gates.

### W6 — the multi-token/prefill path for the recurrent op (not on the critical path)

The op has TWO kernels: one token runs the sequential rule the port implements; several run a chunked
one that fills the buffer differently. **This is NOT YET CHECKED** — two earlier claims about it (that
token 0's output depends on later tokens; that identical slices gave different per-head results) were
ARTEFACTS of a transposed read and are WITHDRAWN; do not repeat them. Characterise it with the
`gdn_authority.c` harness at T=1,2,3 with distinguishable inputs, then extend the reference. Until
then the port/reference must refuse multi-token comparison and the engine runs S single-token steps —
which is correct, just slower.

## 4. ENVIRONMENT — the exact setup

    repo      ~/Projects/VybForge        (this file's repo; ships to `main` directly)
    compiler  ~/Projects/Vyb             (VYBHOME; `build/vyb`, `stdlib/`)
    llama.cpp ~/Projects/llama.cpp       (4df29be4, supports qwen35; the ORACLE, not a dependency)

* `. ./vybenv.sh` resolves `$VYB`, `$VYB_STDLIB`, `$VYBHOME` and the model paths; `VYBHOME=<other
  checkout> ./native/legit/run_x_gate.sh` exercises an unmerged compiler fix in a worktree without
  touching the main checkout.
* **Run every python oracle with `env -u PYTHONPATH`** — the agent session's PYTHONPATH shadows the
  repo `.venv`. A gate that says "transformers not available" or dies on
  `numpy._core._multiarray_umath` is this, not a broken venv. Prefer
  `env -u PYTHONPATH make -f native/Makefile <target>`.
* `make -f native/Makefile verify` builds every kernel in `KERNELS` (kernels are artifacts of the
  compiler that emitted them; a toolchain stamp guards reuse). A kernel used by a driver must be in
  `KERNELS` or a clean build lacks its `.ptx`.
* Derived inputs, all regenerable, never hand-edited:
  * `native/gguf/ridge-3.7bpw-inventory.tsv` — `native/gguf/ridge_inventory.py` (name/dims/type/role/
    offset/bytes + per-type summary). Offsets are ABSOLUTE file offsets.
  * `native/out/ridge_tensors.tsv` — `native/tools/inventory_to_tsv.py` (what the driver parses).
  * `native/out/ridge_invfreq.bin` — `native/tools/gen_invfreq.py 1e7 64 <out>` (32 f64 entries; the
    entry COUNT is n_dims/2, so the table's length IS n_dims).
  * `native/out/iq2s_grid.bin` (8192 B) / `native/out/iq3s_grid.bin` (2048 B) — written by
    `iq2_s_ref.py` / `iq3_s_ref.py`.
* Wrapper laws: `main()`'s return value is PRINTED, not the exit code (a driver that returns 60 still
  exits 0) — parse the driver's `*_DONE` line or, better, gate on the verifier's exit status.
* `llama-cli` dumps core on this box (even `--help`); `llama-server` is the working entry point.
* `~/Projects/llama.cpp` has 27 uncommitted lines in `src/models/qwen35.cpp` that are NOT ours
  (sibling `build-fastmtp` tree suggests MTP experimentation) — leave them alone, and remember the
  tree is not pristine when you build an authority from it.
* The conda plugin crash-report printed by every `python`/`vyb` start on this box is benign noise.
* **ONE GPU JOB AT A TIME.** Two CUDA drivers serialise on the card and each busy-waits at 100% CPU:
  a driver whose own phase normally writes its output in ~2 min wrote nothing for 33 minutes while a
  second driver held the card, and wrote 15 s after it was killed. CPU time 1:1 with wall clock proves
  a process is RUNNING, never that it is progressing.
* Budget: the 3090 is 24 GB. A full-size attention weight is 503 MB f64 (12288x5120), the GDN paper
  weight set ~1.2 GB, the GDN state cache 402 MB, the dense FFN weights 713 MB each. The dense path's
  `DE` (10.2 GB for Ridge) is the one allocation that does not fit — see W2.
* Timings worth keeping: `make prefill` varies from ~1 min to ~20 min run to run (page cache, GPU
  state) — never kill it on elapsed time alone, and a wrong kill records a FAILED gate in the battery
  (`Error 143`). The GDN engine gate is ~40 s; the attention engine gate is a few minutes.

## 5. GATE AND EVIDENCE CONVENTIONS (short)

* Three levels of evidence for a new capability, in this order: (1) an OP authority — link the built
  libggml and call the real op, or an upstream function extracted verbatim and compiled; (2) the
  engine's own path in probe mode with per-stage dumps and REQUIRED negatives; (3) a whole-model
  comparison against llama-server. Never let level 2 be a side harness: put the probe in the file that
  will run it (`§26`), and delete the harness in the same change.
* Comparative teeth are mandatory: every stage check must print the error of the plausible mis-wiring
  and FAIL if it is not distinguished. `not (r <= bar)` for float comparisons, always.
* Prove the gate can FAIL (§21): mutate the implementation, re-run, revert, re-run. Never bake the
  mutation in.
* A gate that neither runs nor can disagree with the bug is not a gate: regenerate golds inside the
  target, assert the producer's completion marker, and point comparators at data artifacts, not logs.
* Landing a capability flag is the LAST step; if the budget runs out before a piece is verified, land
  the PLAN plus a written reason, not a half-modified engine.
* Workflow: do NOT commit or push without an explicit go-ahead; batch related checkpoints into one
  verified commit; never `git add -A`; commit messages are `scope: what now happens` in the
  imperative, and a commit that claims a gate verdict must be written AFTER that verdict returns.
  Handoff files get their own `handoff:` commit.

## APPENDIX — the accumulated record (chronological; headings marked HISTORIC are kept only
## so the record is not rewritten)

Below: the phase narrative as it accumulated (goal, per-unit measurements, the traps that cost the
most, and the earlier step-5 plan, now superseded by §3 above and kept only so the record is not
rewritten). The authoritative detailed narrative lives in `doc/QWEN35-PHASE4.md`; the skill lessons in
the `vybos-development` skill (`references/gate-and-probe-discipline.md`, §1-42) and the
`vybforge-gpu-model` skill (`references/hybrid-engine-wiring.md`, `model-config-contract.md`,
`authority-op-verification.md`, `llama-oracle-verification.md`).

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

## Unit 9c — steps 1 and 2 DONE; step 3 (the loop) not started

`native/host/model_driver.vyb` is a 660-line full-model driver for the Qwen3-4B path (36 layers,
per-tensor quant staging, one `stage_one(...)` call per weight at fixed lines ~544-560, kernels from
`layer.vyb`/`qwen3.vyb`). The recurrent branch is now built and verified on its own; what is left is
the loop surgery plus a full 27B run.

What it needs:

1. **A per-layer dispatch, decided from the tensor table, not the model name** — **DONE, commit 3f34465.**
   `model_caps::mc_layer_kind(name)` is that test and the descriptor counts with it; the engine's loop
   asks the same function instead of re-deriving the answer. Gate P4.4
   (`native/legit/run_layerkind_gate.sh`) proves it per block on both models against an independent
   parser. A recurrent layer has `blk.N.attn_qkv.weight`, `blk.N.attn_gate.weight`,
   `blk.N.ssm_alpha.weight`, `blk.N.ssm_beta.weight`, `blk.N.ssm_out.weight`,
   `blk.N.ssm_conv1d.weight`, `blk.N.ssm_norm.weight`, `blk.N.ssm_a`, `blk.N.ssm_dt.bias` where an
   attention layer has `attn_q/k/v/output` — and note the two traps the gate pins: `attn_qkv` and
   `attn_gate` carry the `attn_` prefix but are RECURRENT weights.
2. **The six projections through the existing quant path** — **DONE, commit cd78628.** Gate P4.5
   (`native/legit/run_gdn_engine_gate.sh`, `native/host/gdn_engine_driver.vyb`) stages every blk.0
   tensor from the live GGUF through the engine's own path and runs the chain with layer.ptx's
   `gemm`, 26/26 stages over two chained steps, worst 3.903e-06 against unit 5's authority. Two traps
   it recorded: `delta_step` and the `ssm_out` gemm share the `P2` param block (alpha lives at
   `P2+48`, which `delta_step` overwrites with a pointer), and the dequant kernels' `z` argument is
   the tensor's IN dim, which is what makes the staged B `[in,out]` for `gemm`.
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

## (HISTORIC) `eng_gdn()` was 0 until the flip — the rule still applies, the state is stale

Do not flip it yet. The descriptor's `eng_gdn()` decides whether the caps gate reports
`UNSUPPORTED_LAYER_KIND gated-deltanet` for the Ridge model, and the engine path still has no recurrent
layer: no multi-token/prefill, no state carry-over across a sequence, no quantized weight loading, and
nothing wired into `model_driver`/the chat server. Flipping it would make the descriptor claim a
capability the engine cannot deliver.

## (HISTORIC) what the flip needed at unit 9 — superseded by the brief's §1/§3

State carry-over is done (unit 8). Three things remain before `eng_gdn()` can honestly flip:

1. **Quantized weights**: the mechanism is in place (unit 9, above); what is missing is a projection
   kernel fast enough to run the layer check at the model's real geometry, or routing the projections
   through the quant gemm path `layer_driver.vyb` already uses.
2. **The multi-token/prefill path** — the op's chunked kernel, still uncharacterised (see below); a
   real prompt cannot run without it, and prefill is what makes 48 layers affordable.
3. **Engine integration**: a recurrent branch in `model_driver.vyb`'s per-layer loop, then the flip and
   a caps-gate re-run (the gate asserts the refusal list SHRINKS as well as the new count, skill
   gate-and-probe §6), plus re-running P4.1-P4.3.

## (HISTORIC) 'after that' list from unit 9 — the live list is the brief's §3

* **The prefill/multi-token path.** The op has TWO kernels: with one token it runs the sequential rule
  the port implements; with several it runs a chunked one that fills the buffer differently — proven
  by feeding identical tokens at T=1,2,3 and watching token 0's output change, which no causal
  recurrence can do. Until that is characterised, the reference must refuse multi-token comparison
  (`--tokens 1`), and no real prompt can run.
* Then `eng_mtp()` (the 65th block + the `d2t` tensor) — smaller, but buys no tokens alone.

## (HISTORIC) environment notes as first written — the brief's §4 supersedes this

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

## APPENDIX D — the step-5 plan as written before it was done (SUPERSEDED: step 5 landed; see the
## brief's §1-§3 for the live plan; kept only so the record is not rewritten)


The authority side is DONE and gated: the attention block's ten stages (joint Q+gate split, per-head q/k
RMS norms, rope over 64 of 256 dims, causal attention, the output gate, `wo`), verified three ways —
hand-rolled (4.597e-07), flash at S=6 (3.887e-07), and GQA 6-over-2 replicated (3.493e-07), each with its
misreadings rejected. Rope's `n_rot` is wired in the engine and the dense prefill is bit-identical.

What is left is the ENGINE side, and the reconnaissance is done — the pieces exist, so this is assembly,
not discovery:

* The attention weights are ALREADY staged per layer in `model_driver.vyb` (attn_q/k/v/output at the
  `stage_one` calls around line 1005, plus attn_q_norm/attn_k_norm and attn_norm) — a probe does not need
  new staging, only a dump path.
* Mirror the recurrent probe's shape: `VYB_GDN_PROBE=<layer>` reads its fixture input from
  `$VYB_GDN_X` (default `native/build/gdn_engine_x.bin`), runs the block, and dumps stages with
  `dump_section(...)` under a `gdn_engine_verify.py`-style verifier. The attention probe should do the
  same: `VYB_ATTN_PROBE=<layer>`, a fixture hidden state from `$VYB_ATTN_X`, and stage dumps named after
  the authority's (`qg`, `q_pre`, `gate_pre`, `q_norm`, `q_rope`, `k_norm`, `k_rope`, `attn`, `gated`,
  `out`).
* The engine needs the GQA REPLICATION itself: Ridge is 24 query heads over 4 kv heads, so `run_layer_attn`
  must replicate K/V to the query-head count (the same design the authority now uses and that is verified —
  see doc/QWEN35-PHASE4.md unit 10 step 4q). The authority's replicated fixture is the reference for it.
* Weights are quantized in the GGUF while the authority's fixture is f32, so the comparison bar must
  account for dequantization the way `gdn_engine_verify.py` does (it reads the model's real tensors and
  uses a tolerance reflecting the quantization, not the 1e-4 used for exact paths).
* Then extend `native/legit/run_attn_block_gate.sh` (three cases today) with the engine case, so the
  battery covers the block end to end rather than only its authority.

Order I would take it: (1) probe branch that runs layer L's attention on a fixture and dumps the ten
stages; (2) `native/tools/attn_engine_verify.py` comparing the dumps against the authority's, with the
same teeth; (3) the K/V replication for GQA in `run_layer_attn`; (4) wire the gate + battery; (5) only
then run a whole Ridge model.
