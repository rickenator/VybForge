# HANDOFF — RESUME HERE (tracked on main; a running log, not a spec — keep it out of unrelated commits)

> **Historical log (superseded for anything Python-related).** This file is the
> running handoff from the inference-substrate phase. For the current state of the
> Python cleanup — what was ported to Vyb, what was deleted, and the oracle policy
> that says the remaining `verify_*.py` / `*_ref.py` files are deliberately KEPT —
> read `doc/PYTHON-CLEANUP.md` first. Two claims below are now stale: the
> `tests/test_schema.py` original was deleted (its Vyb port `tests/test_schema.vyb`
> carries `make schemacheck`), and the `test_backends.py` port is not a follow-up —
> its guards moved into `native/legit/run_configurator_gate.sh` and the file is gone.

**Project:** VybForge — repo `rickenator/VybForge`, local checkout
`<VybForge checkout>`, branch `main`. Address the user neutrally.
**Model:** `<model path>` (2.5GB, real). **Compiler:**
`<Vyb checkout>/build/vyb` (now includes #209 fix). GPU: RTX 3090. venv `.venv`.
Conventions unchanged (runtime pure Vyb; handoffs = GitHub issues on rickenator/Vyb).

## Status of the two issues filed earlier
- **rickenator/Vyb#209** (cuModuleLoadData intermittent INVALID_PTX / String→CString
  NUL hazard) — REAL compiler bug, **fixed upstream**. Drivers already carry the
  bounded retry; safe to keep.
- **rickenator/Vyb#210** (claimed "launch-drop") — **NOT a Vyb bug, CLOSED.** Root
  cause was MY driver: a `freedom{... return 0}` inside the `emb_dequant` chunked
  loop returned the whole function after chunk 0, so only ~26 token_embd rows
  dequantized (DE rows >25 = 0) — which faked "kernels truncating after many
  launches". Fixed by capturing the error in a function-scope var and returning
  only after the loop. Probe already proved the math; now the full driver matches.

## DONE & VERIFIED — real Qwen3-4B prefill + tied lm_head, all on-GPU, pure Vyb
`make -f native/Makefile prefill` → both gates green:
```
PREFILL_HIDDEN_MATCH: OK   maxrel ~4.9e-6   bad=0
PREFILL_TOP1_MATCH: OK     top1 = [31784, 31784] == numpy
```
(also `make layer0` → LAYER0_MATCH OK, ~5e-6).
Key files:
- `native/gguf/layer0_ref.py` (layer-0 + shared dequant/math), `native/gguf/prefill_ref.py`
  (full-36 numpy contract), `native/host/layer0_driver.vyb`, `native/host/model_driver.vyb`
  (per-layer staged loop + ping-pong + `emb_dequant` chunked dequant + `logits_slice`
  chunked kernel + `argmax_row`), `native/kernels/qwen3.vyb`, `native/tools/verify_*.py`,
  Makefile `layer0`/`prefill` targets.

## Gotchas (verified)
- Inside a Vyb `freedom{}` block, `return` exits the ENCLOSING FUNCTION — never put
  `return 0` in the body of a LOOP you intend to continue (that silently truncated
  `emb_dequant` to chunk 0). Capture an error var and return after the loop.
- `cuModuleLoadData` is flaky (#209) → bounded retry with fresh read.
- `&&` works fine (earlier claim wrong); `from`/`to` reserved; top-level fns before
  main; extern/loc/from only inside `freedom{}`; Q6_K ffn_down packed 20.4MB >
  Q4_K ffn_gate 14MB (size DPK to max, computed at runtime); roped-k dest index is
  `Kd+g2` not `Kd+gid`.

## NEXT SESSION — do this (in order)
1. **REAL DECODE: SOLVED (2026-08-28).** Three layer-4/+ root causes, all found & fixed in
   `decode_driver.vyb` (real 36-layer greedy decode now produces numpy-matching tokens):
   (a) `vmath.vyb` `vexp` data-dependent loop hung on NaN (`(NaN+0.5) as Int` ~9.9e13; NaN
   bypasses the ±20 clamps; probe native/host/probe_nan.vyb). Fixed: clamp np to ±2048.
   (b) **Mixed per-layer quantization**: `attn_v` & `ffn_down` are Q6_K (ty14) on layers 0-3
   & 31-35, Q4_K (ty12) in between. decode_driver used uniform K*L stride + one kernel from
   blk.0 → layer 4 mis-dequantized → NaN. Fixed: per-layer cumulative packed offsets
   (ov2/os2/ov6/os6) + per-layer kernel dispatch (q4fn/q6fn by ty). Verified weights reproduce
   numpy (native/host/wcheck_driver.vyb + native/gguf/wcheck_ref.py).
   (c) **RMSNorm weights are per-layer too**: decode loaded blk.0's attn_norm/attn_q_norm/
   attn_k_norm/ffn_norm for ALL layers → final hidden uncorrelated with numpy (corr 0.009,
   wrong tokens). Fixed: load layer L's 4 norms each step (mirrors stage_one). After all three
   fixes: step-0 hidden matches numpy to 5e-6, first gen token 31784 == reference.
   **ALSO FOUND: `native/gguf/decode_ref.py` (numpy ref) itself had a greedy bug — it
   carried the PREVIOUS step's hidden output forward (`x=concat([x_hidden, emb(nxt)])`)
   instead of re-embedding the full prefix each step, so decode_ids_ref.txt
   [0,1,31784,12001,...] was INVALID ground truth (decode_ref's own functions give
   step-1 argmax 73305, matching the GPU, not 12001). FIXED decode_ref to recompute
   the full prefix each step. The Vyb GPU decode — correct recompute-full-prefix —
   produced [0,1,31784,73305,43790,1888,86805,31784] and the corrected ref now confirms it
   token-for-token (**DECODE_REAL_MATCH: OK** via `make -f native/Makefile decode-real`).
   **REAL DECODE IS DONE & COMMITTED (aee4408).**
2. **KV-CACHE DECODE DONE (2026-08-28):** `native/host/kv_driver.vyb` — build the seed forward once
   (run_layer, caches each layer's roped-K/V into per-layer `[MAXS,NKV]` slabs KC/VC), then each gen step
   `run_kv` computes ONLY the new position P (1-token rmsnorm/gemm/rope at absolute pos via qwen3rope's new
   +224 POS param, stores k/v in cache, existing attn kernel with S=P+1 attends the new q over cached
   K/V, reads ctx row P). `make decode-kv` → DECODE_REAL_MATCH: OK, same tokens as recompute/numpy.
   GOTCHA: single-row rmsnorm/rope S-param is **1**, not 0 — S=0 launches zero threads and collapses the
   stream to a constant token (61619 everywhere).
3. **TRAINING PHASE (NEW DIRECTION, 2026-08-28):** the current direction requires the LoRA training itself to be Vyb-native
   on GPU (no torch) — the true no-Python path — full Qwen3-4B research-scale, PLUS an expanded
   interview corpus. Design doc `training/CORPUS-STRATEGY.md`: **facts live in context (a capabilities
   manifest in the system prompt), behavior lives in the weights**, so a new VybOS capability = edit the
   manifest, NOT a retrain. Also: goal-driven WHOLE-STACK interviews ("I want a graphical desktop" →
   decompose the full stack, manifest-grounded, surface unbuildable subsystems as open decisions).
   Pyright-note: tests are verification (stays Python); the SHIPPED runtime must be pure Vyb.
   **`tests/test_schema.py` → Vyb PORT DONE & passing** (`tests/test_schema.vyb`, `make schemacheck` →
   SCHEMA_TEST: OK, 1440 records; python original stays green). `test_backends.py` port is a follow-up
   (needs the Vyb configurator's payload-shaping to exist).
   **T0 DATA DONE (committed 539dfb8):** `training/generate_dataset.vyb` rewritten to a **720-record /
   20-topic** manifest-grounded, goal-driven corpus. 4 manifest variants (0-3); v3 (wayland-desktop
   "derived" future) appears ONLY in eval → transfer test; train = manifests {0,1,2}. Whole-stack goal
   decompositions (desktop/kiosk/CI-runner/gateway/rescue) stay manifest-grounded. All 720 schema-valid;
   Vyb schemacheck + python test_schema both green.
   **T1 (GROUND TRUTH) DONE (committed 8cb8301):** `native/train/encode_corpus.vyb` reads corpus records,
   encodes each assistant response with the Vyb Qwen3 tokenizer -> input ids + shifted CE labels (-1 ignore);
   verified ENCODE_CORPUS_VERIFY: OK (4/4 exact vs transformers). `make encode-corpus`.
   **T2 (FORWARD + ACTIVATION CACHE) DONE (committed 0514369):** `native/train/train_fwd.vyb` runs a
   layer-0 forward and dumps the activations backward needs (XN, DQr, DKr, KV, DCtx, X1N, Hu); verified
   vs numpy (train_fwd_ref.py) at max-rel <5e-6, corr 1.0. `make fwd-cache` (gate tol 2e-5). The transient
   buffers ARE the cache at NLAYER=1; multi-layer = copy into per-layer slabs (mechanical next step).
   **T3 BACKWARD SET DONE (committed 8435e32..49ea3bb):** `native/kernels/backprop.vyb` has all 8 backward
   kernels, each gated against an analytic numpy ref at ~5e-6 (`make bwd-probe/lora/norm/rope/attn`): dsilu,
   dresid, dgemm_at/bt (LoRA), drmsnorm, dnormw, drope, dattn (softmax backward; `atomic_add_f64`).
   **T4 LoRA apply + AdamW DONE (committed 017e933):** `gemmf`+`addscaled` (h=x@W+alpha_r*(x@U@V)) and
   `adamw` (bias-corrected momentum + weight decay) — verified vs analytic (`make t4-lora`). Gotchas:
   `fn`/`at`/`as` reserved; single-module rule (load ONE .ptx per process); `grid-type>=` spacing.
   **T5 (LoRA training loop) DONE & VERIFIED (committed aaf6dbe):** `native/train/t5_probe.vyb` runs the full
   training loop on GPU in pure Vyb (forward gemmf+addscaled -> MSE-grad sub+scalec -> LoRA backward
   dgemm_at/bt -> AdamW on U/V, W frozen), computing its OWN loss: 3.2415->2.735->2.1126->1.4961->1.0303,
   matching numpy at every step; U/V ~4e-6 corr 1.0 (`make t5-lora`). The long "step-2 vU anomaly" was NOT
   a Vyb/kernel/compiler bug — the standalone repro confirmed the kernel exact; the real culprit was a BUG
   IN THE NUMPY REFERENCE (t5_ref.py: "m = b1*m + (1-b1)*g" rebinds the local, never updating the persistent
   mU/vU arrays, so its moments never accumulated -> wrong step-2+ refs). Fixed with in-place "mU[:] = ...".
   The Vyb loop was correct throughout.
   **REAL QUEN3-LAYER MILESTONE (M2) — forward/backward done & verified:**
   - **M2a (5396312):** combined training kernel module `native/kernels/train.vyb` (one .ptx: forward
     layer kernels + all backward/training kernels + qwen3rope + q4kdeq/q6kdeq/f32expand). Single-module rule.
   - **M2b (2ea7451):** `native/train/train_layer.vyb` real blk.0 forward WITH LoRA on all 7 projections +
     full activation cache, vs numpy at 5e-6 (`make train-layer`).
   - **M2c (6eafb99):** `native/train/train_layer_back.vyb` real blk.0 BACKWARD chain producing dU/dV for
     all 7 projections + dx, vs the FD-validated numpy oracle (train_layer_back_ref.py) at ~5e-6
     (`make train-layer-back`). The oracle was FD-validated (m2c_fdcheck/fdq) after fixing silu' (needs
     sigmoid(g)) and RMSNorm-backward (pre-norm input). Vyb gotchas: zero accumulators with `zbuf`
     (addscaled(x,src,0) leaves cuMemAlloc garbage); watch for duplicate `+=` accumulator lines
     silently double-counting.
   **M2d DONE (2026-08-29):** `native/train/train_layer_train.vyb` — full teacher-forced   real blk.0 LoRA TRAINING LOOP: forward (M2b) + MSE grad dG=xo-t + verified backward (M2c,
   7 projections) + AdamW on all 14 U/V (bias-corrected moments, W frozen), overfit ONE fixed
   target over 20 steps. Per-step loss dumped to native/out/m2d_loss.txt, gated vs the numpy
   loop oracle (train_layer_train_ref.py, in-place Adam state) at `make train-layer-train` →
   **M2D_TRAIN_VERIFY: OK** (maxrel 1.146e-05 /l0, corr 1.0, loss 0.477494 → 0.474039 descends).
   **GOTCHA fixed (kernel bug, vmath.vyb):** Vyb Newton `vsqrt(0)` = 0.5*(0 + 0/0) = NaN — at
   step-1 AdamW any gradient element ==0.0 makes vhat=g²=0 → U/V gain NaN (~2% of the 5120 elts).
   numpy sqrt(0)=0. Fixed: `vsqrt` now returns 0 for x<=0 (feeds mean+eps>0 elsewhere, safe).
   ALSO: cuMemAlloc Adam moment buffers are garbage → MUST zero all 14 m + 14 v with zbuf before
   the loop or step-1 AdamW reads NaN moments.
   **NEXT = M2e:** scale the full-36-layer forward+backward+AdamW over the REAL corpus slice
   into a full training loop; then the whole-stack interview. (Careful with numpy refs: use
   in-place array writes for stateful cross-step arrays.)
   **M2e FULL-FORWARD+ACT-CACHE DONE (2026-08-29, commit 56c4cd1):** 36-layer LoRA forward
   (all 7 proj/layer) streaming weights per layer; `train_full_ref.py` oracle caches 25
   activations/layer + hidden; `train_full.vyb` GPU driver; `make train-full-forward` →
   M2E1_VERIFY: OK (hidden 4.9e-6, 900/900 per-layer tensors). LoRA U/V per layer seeds
   1000+L → m2e_l{L}_{UV}.bin; activations m2e_L{L}_{t}.bin.
   **M2e FULL-BACKWARD DONE (2026-08-29, commit d0c4aa7):** 36-layer backward chained L35→0
   from a fixed upstream grad (seed 67) producing per-layer dU/dV (14) + dx.
   `train_full_back_ref.py` (numpy oracle) + `train_full_back.vyb` (GPU, backward block
   generated from M2d) + `make train-full-back` → M2E2_VERIFY: OK (540/540). Root cause:
   oracle x_in was flat (x_in[s_] scalar) → .reshape(S,D) fix; GPU was CORRECT all along.
   GOTCHA: descending `for(in 35..0)` runs ZERO iters in Vyb → use ascending + L=MAXL-1-Li.
   **NEXT = M2e-3:** full-36-layer LoRA TRAINING loop on the corpus total.
   **M2e-3 ORACLE DONE (2026-08-29, commit f4495a9):** `train_full_loop_ref.py` = full-stack
   N-step training loop (forward w/ LoRA all layers -> MSE grad at output -> backward 35..0 ->
   AdamW on all 36 layers' U/V, in-place moments; preloads GGUF weights ONCE). + verify gate.
   **M2e KEY SCIENTIFIC FINDING (vanishing gradient):** for a deep real-Qwen3 stack the backward
   grad decays |dG|@out~1.4e3 → |dU|@L0~4e-13 over 36 real RMSNorms — the TOP layers carry ~all
   of a deep LoRA's learning signal, lower layers ~0. So MSE-overfit to a FIXED random target does
   NOT descend on the full stack (M2d's single-layer overfit worked because one layer can reach t).
   This is correct deep-net behavior (why real LoRA fine-tune works: head/norm/top layers take the
   gradient; lower layers stay ~frozen) and true for BOTH numpy and the GPU (M2e-2 confirmed both
   agree on the 1e-13 chain). IMPLICATION for M2e-3 GPU loop: gate on GPU-loss == oracle-loss
   (loop correctness), NOT on descent; the PRODUCTION corpus objective (next-token CE, where the
   HEAD gradient is non-trivial) is what will show real descent.**
   **NEXT = M2e-3 GPU loop driver:** `train_full_loop.vyb` replicating the oracle (per-step fwd+
   bwd+AdamW, device-resident per-layer activation cache for backward — do NOT dump 25 txt/layer
   per step, that's ~20min/step). Gate = per-step loss match. Then swap the MSE-overfit target for
   a real corpus next-token CE slice to get descent + actual corpus training.
   **M2e-3 GPU LOOP DRIVER DONE (2026-08-29, commit 44734b8):** `train_full_loop.vyb` = monolithic
   loop: per step forward-36 (LoRA from LSLB slab, activations cached to per-layer ASLB via new
   dcopy kernel), loss = MSE(xo-t) readback, backward 35..0 (load slab + weights + LoRA, M2d bwd
   block, dx chains dX1la->DG), AdamW per-layer (MSLB/VSLB moments). + numpy ref + verify gate +
   `make train-full-loop`. **Verified: GPU step-1 loss=319.268 == oracle 319.2683 — loop computes
   correctly.** NB: ~40 min/step (weight streaming is per-8-byte H2D from the 2.5GB GGUF), so run
   NSTP small / in background; the deep-stack vanishing gradient keeps MSE-overfit loss FLAT
   (~319.x) so descent is NOT the gate — loss-match is. REAL descent = corpus next-token CE (next).
   **ANALYTIC-BACKWARD BUG FOUND + FIXED (2026-08-29, commits 221023b, a0b8dc1):** independent
   finite-difference probe (train_full_fd_ref.py) exposed that the analytic backward (numpy AND
   GPU share it) RETURNED ONLY the rmsnorm/attention path in the chained dx, DROPPING the
   residual-identity term dx1 from the `x1 = x + o` skip (dx1 = dG + ffn-norm-back). Over 36
   residual layers this under-counts lower-layer gradients by ~19 orders (analytic L0 dU_q[0,0]
   was ~1e-18; FD truth 7.62). This is why "M2e-2 540/540" and step-1 loss-match held (self-
   consistent analytic) but the loop couldn't train. FIX: dx = attention_path + dx1, applied to
   train_full_loop_ref.py + train_full_back_ref.py + train_full_loop.vyb chain. **FD-verified 8/8
   sampled adapters (U+V, L0/L17/L35) at rel <= 1.3e-4; numpy loop now DESCENDS 319.27->248.95
   over 5 steps (was flat 319).** Remainder: re-run GPU loop to confirm per-step loss matches the
   NEW descending oracle (step-1..5 == oracle's 319.3,313.4,292.0,266.5,249.0).
   **NEXT (bulk-transfer, per review):** per-step forward/backward still re-reads + re-H2D the
   (frozen) layer weights per-8-byte (~40 min/step). Cache all dequantized WEIGHTS device-side ONCE
   into per-layer f32/direct slabs before the loop (weights never change in LoRA training; only
   U/V adapters do), so forward/backward index slabs instead of streaming — target <1 min/step.
   That turns M2e-3 into an iterative-dogfoodable milestone. THEN swap MSE-overfit for corpus
   next-token CE (target shift, causal mask, final RMSNorm, tied lm_head) for real training.
   **CORPUS-CE ORACLE DONE (2026-08-29, commit f5045e7):** `train_corpus_ce_ref.py` = the PRODUCTION
   objective in numpy: embed a fixed tiny record-0 assistant slice (S=4) -> 36-layer LoRA forward ->
   output_norm -> TIED lm_head (h @ real 151936-vocab token_embd.T) -> target-shifted causal CE
   (input [4913 15314 3252 73874] -> labels [15314 3252 73874 2198]) -> CE backward (head grads +
   output_norm + 36-layer residual-fixed chain) -> AdamW. **CE loss DESCENDS 15.49 -> 0.697 over 8
   steps.** This is what the GPU driver must reproduce (m2e3ce_loss_ref.txt + input ids committed).
   The MSE-overfit loop was ONLY the correctness gate; corpus-CE is the real loss.
   **GPU LOOP WIRING BUG FOUND + FIXED (2026-08-29, commit 419a4d2):** the monolithic GPU loop
   wrote the loss gradient sub(XI,TGT) AND the residual chain to buffer 'dG', but the entire
   backward block reads its upstream gradient from a SEPARATE never-written buffer 'DG' -> ALL
   backward grads were exactly 0 (in-loop L35 dU_q2 norm 0 vs oracle 52300), AdamW never updated,
   loop diverged. Fix: loss + chain now write DG (verified single-step in-loop: dU_q2=52300=oracle,
   dV_q2=22364, dCtx2/dq2/dXN2/DX1/dX1la all non-zero).
   **OPEN (5-step GPU run gone 319.27->984.33->... still diverging):** top-layer grads are now
   correct (52300=oracle) but the monolithic loop STILL does not descend. The remaining bug is in
   chained lower-layer gradients or the multi-layer AdamW/moment offsets, NOT the top-layer backward
   or the analytic math (numpy is FD-verified and descends). Do NOT keep patching the streaming
   monolith (~45 min/step makes every probe painful): build the packed-weight-cache rewrite so steps
   are <1s, then bisect the descent there (compare GPU L0/L17/L35 chained dU/dV + epoch adapters to
   the oracle's to localize which layer diverges). numpy-side (analytic fix + MSE + corpus-CE oracles
   all FD-verified and DESCENDING) is the solid ground truth; GPU-side full-loop descent is the
   remaining blocker.
   **ROOT CAUSE FOUND + FIXED (2026-08-30): THE 'DIVERGENCE' WAS INPUT-ALIASING, NOT THE BACKWARD.**
   Every training step re-seeded the forward with `XI = XA`, but XA is the layer-alternation SCRATCH
   buffer (the forward loop swaps XI<->XOcur each layer, so layer 1's output OVERWRITES XA). XA held
   pristine m2e_input only for step 1; every later step fed layer-35's output back in -> step 1
   matched, steps 2+ diverged (GPU 319.27->984.33). This hit the packed (train_full_loop_pkd.vyb) AND
   old monolithic (train_full_loop.vyb) drivers identically. FIX: added a pristine XIN buffer
   (m2e_input downloaded ONCE into XIN), and at the top of EACH step `dcopy XIN->XA` (S*D f64) before
   `XI=XA`. Now GPU matches the oracle per-step AND DESCENDS: 319.27->313.42 (2-step) /
   319.27->313.42->292.00->266.50->248.95 (5-step), with output/adapter/moment parity ~4e-6 at every
   step (previously step-2 output corr 0.45). Makefile target `train-full-loop-pkd` gates it.
   **GOTCHA (reusable): in a kernel-mode Vyb training loop, any buffer ALSO used as a layer
   input/output scratch MUST NOT double as the pristine per-step input source. `XI = XA` is a
   POINTER alias, not a copy — re-source from a dedicated XIN (or re-download) each step.**
   **NEXT = swap the MSE-overfit target for the corpus next-token CE slice (train_corpus_ce_ref.py,
   committed f5045e7) — the production objective with a non-trivial head gradient -> actual descent +
   real LoRA fine-tuning on GPU, fully Vyb-native (as tested on an RTX 3090). Then whole-stack interview.**
   **CORPUS-CE ON GPU: WIRED + step-1 gate GREEN (2026-08-30).** New head kernels in
   kernels/train.vyb: `embed` (token_embd[ids] gather), `hgemm` (logits = h @ emb^T, S=4 x V=151936),
   `cefwd` (softmax CE + dlog=(soft-onehot)/S + plab[s]), `hback` (dh = dlog @ emb) — validated
   standalone (head_probe.vyb + head_ref.py + verify_head.py, HEAD_VERIFY: OK at ~4-5e-6 incl. loss
   4.5e-7). New driver `native/train/train_full_ce.vyb` = the packed loop lifted to S=4
   (train_corpus_ce ref objective): generated from train_full_loop_pkd.vyb by `_gen_ce.py`
   context-aware S=2->4 transform (PROTECTS the S-independent LoRA/AdamW numels like Vq=8192/Uo=8192
   from being rescaled; only S-dependent working buffers + the full 36-layer ASLB slab offsets are
   recomputed, verified programmatically with _check_ce.py). Wired in: token_embd (q6k, off=5966784)
   + output_norm (f32, off=5956544) loaded (TSV parser extended to include non-blk. tensors), ids/
   labels from committed m2e3ce_*.txt->.bin, pristine input = embed() into XIN, head runs right after
   the forward (rmsnorm->hgemm->cefwd->host -log CE loss->hback->drmsnorm=DG), then the SAME
   backward+AdamW. **GPU step-1 CE loss = 15.4907 == oracle 15.49067 (exact).** Full 8-step descent
   run (15.49->0.697) validates per-step loss-match + descent; `make train-ce-loop` gates it.
   GOTCHAS: token_embd/output_norm are NOT in the driver's nmv (TSV parser only collected "blk."
   names) -> extend the `if nm.substring(0,4)=="blk."` filter. cuMemAlloc EMB f64 = 388,956,160*8 =
   3.1GB (fits, 24GB card). `log()` (natural log) is a built-in Vyb host intrinsic — used for the
   host-side CE loss from plab[s], avoiding a device vlog. CE loss is computed via plab (softmax at
   label) + host log, not logsumexp, to dodge device-log.
   **S=4 LIFT BUG + FIXED (step-2+ drift): the S*R LoRA-s buffers were NOT rescaled by the literal
   transform (S*R=4 at S=2 -> must be 8 at S=4). sq..sd allocs were `4*8` (32B, under-alloc) and the
   ASLB activation-cache dcopy counts were literal `4` -> only 4-of-8 s-intermediates cached, so the
   BACKWARD read wrong s -> wrong dV/s-grads from step 1. Signature: step-1 loss matched (forward
   fused-gemmf computes s internally), steps 2+ drifted (corr 0.99, final adapters ~40% off). FIX:
   sq..sd + ds_* allocs `4*8`->`S*R*8`, ASLB dcopy counts `4`->`S*R` (fwd+bwd). After fix: 2-step
   per-step loss matches oracle to 2e-6 AND step-1 dU_q matches at L0/L17/L35 to ~1e-4 (verified).
   LESSON: gate an S-lift on step-1 GRADIENTS (dU/dV L0/17/35), not step-1 loss; and only rescale
   TRUE S-dependent sizes (adapter numels Uq=D*R=5120, Vq=R*NQ=8192 stay).**
   **8-step full gate = `make train-ce-loop` (M2E3CE_VERIFY): per-step CE loss matches the committed
   oracle 15.49->0.697 + descends.**
   **WHOLE-STACK RECORD TRAINING (2026-08-30): moved OFF the 4-token toy to the real goal-desktop
   record.** target = train rec 6 'I want to run a graphical desktop workstation' -> manifest-v<3 SUMMARY
   (gates desktop as open decision; 84 tokens). `_gen_ce.py` now parameterized by TS (rescale every
   S-proportional literal + ASLB offsets by TS/4 from the S=4 CE base; protects adapter numels; s-
   intermediates already SYMBOLIC S*R so they auto-scale). Regenerated train_full_ce.vyb at S=83
   (input=response[0:83], labels=response[1:84]); oracle train_corpus_ce_ref.py reads goaldesk ids
   (S derived), writes m2e3ce_*.bin for the GPU; NSTP=4. Oracle at S=83: CE 15.164->14.096 over 4
   steps (descends). GPU step-1 loss 15.147 vs oracle 15.164 (~1.1e-3 off — numerical-tails artifact
   of -log on p~3e-7, NOT a bug: step-1 GRADIENTS dU_q match oracle at L0/L17/L35 to ~7e-5 corr 1.0,
   so forward+head+backward are correct at S=83). Whole-stack NSTP=4 descent run CONFIRMED/COMMITTED
   (1ef9740): GPU 15.147->14.669->14.182->14.096 == oracle, M2E3CE_VERIFY: OK, corr 0.99996, step-2+
   match ~1e-7 rel. The loop now LEARNS the real goal-desktop whole-stack response.
   **NEXT after descent demo: context-prefix conditioning**
   (cache the fixed system+user context once, train only over the response -> real goal->response
   whole-stack interview), then decode to emit the agent-response JSON end-to-end.
   **GOAL-CONDITIONED CONTEXT-PREFIX TRAINING DONE (2026-08-30, committed 1e5b605):** the objective
   is now [user GOAL context | assistant response] with masked CE — loss/backprop ONLY over response
   rows (context rows LABEL==-1 ignored, normalized by NV=#valid). cefwd kernel masked (LAB==-1 rows
   get LOS=1.0/dlog=0; +48 NV param); host computes NV + divides loss by NV. `_gen_ce.py` made
   BASE-S-AGNOSTIC (reads current S, recomputes ASLB layout from dims, rescales current->target)
   + FIXED greedy `re.sub("S<Int> = \d+")` that corrupted LOS<Int>/IDS<Int> pointer inits (anchored
   to line start). oracle builds masked ctx+resp from the goal-desktop record (transformers), S=93
   (9-token goal ctx + 84-token SUMMARY resp), NV=83, descends 15.96->14.28 over 4 steps.
   VERIFIED: GPU step-1 15.93 == oracle 15.96 (numerical-tails artifact at high loss); step-1 dU_q
   gradients match oracle at L0/17/35 to ~3e-5-2.6e-4 corr 1.0 -> masked fwd+head+bwd CORRECT.
   `_gen_ce.py <TS> <NSTP>` regenerates any length from the current file. Goal-conditioned NSTP=4
   descent run CONFIRMED/COMMITTED (1e5b605): GPU 15.932->16.350->14.741->14.269 == oracle
   [15.957,16.352,14.741,14.276], M2E3CE_VERIFY: OK, corr 0.99994, steps 3-4 match ~1e-4 rel.
   The loop now LEARNS the whole-stack response CONDITIONED ON THE GOAL.
   **KV-CACHE CONTEXT (full-manifest): ARCHITECTED + numpy ref built (2026-08-30, f9d56a6).**
   `native/train/KVCACHE-CONTEXT.md` = authoritative design (cache frozen context per-layer
   roped-K/V once via LoRA run_layer -> CK/CV; KV-aware LoRA run_kv response forward; frozen-context
   backward; validation = reproduce the already-validated goal-context full-sequence S93, NO new
   oracle). `kvctx_ref.py` VERIFIED: numpy per-layer roped-K/V for the 9-token goal context
   (native/out/kvctx_L{L}_{DKr,DV}_ref.bin) = gold for the GPU cache. `kvctx.vyb` + `verify_kvctx.py`
   **VERIFIED (4b43568): GPU per-layer roped-K/V (CK/CV) matches numpy to ~3-4e-6 (corr 1.0) at
   L0/8/17/26/35** — the frozen context cache is correct. Fixes baked in: q4k/q6k direct-arg deq via
   `deq_w` (dispatch mixed Q4/Q6 by PTY), W buffers f64. NEXT: KV-aware LoRA run_kv response forward
   attending to cached CK/CV (== validated S93), then frozen-context backward.
   NEXT: decode to emit the agent-response JSON end-to-end.

## Reference checkpoint
`.venv/bin/python native/gguf/decode_ref.py` (~28 min) rewrites the target token
ids; `make -f native/Makefile prefill` still reproduces both prefill gates green.

---

## f32 resident engine: 2.777 -> 13.2 tok/s (4.7x), and the next 3-6x IDENTIFIED

`make -C native resident-bench CHAT_GEN=3 RESIDENT_RUNS=10` = **13.0-13.2 tok/s, gold stream MATCH**
(2.777 at the start of this line). Eight commits, each gated on identical greedy output:

| commit | change | tok/s |
|---|---|---|
| `e2e7ab0` | f32 weight path | 2.78 -> 4.06 |
| `49cfb9e` | f32 decode arithmetic | 4.06 -> 4.69 |
| `ebb29ed` | **expand all 36 layers to f32 ONCE at load** | 4.70 -> 9.43 |
| `a011af6` | one transfer per launch parameter block | 9.67 -> 10.73 |
| `58fa9e7` | fold widen into the reduce | neutral (correct) |
| `5e60914` | KSP 8 -> 32 + profiler bracket | 10.73 -> 11.34 |
| `241c6ea` | **M-block 4 rows/thread in gemm32_p** | 11.34 -> 12.71 |
| `5d1be56` | logits MACs in f32 | neutral (correct) |

### incremental decode is worth ~1.45x, NOT 3-6x -- CORRECTED BY MEASUREMENT (see the note at the end of this section)

**Measured** (`VYB_RUNS=5` at each length, per token):

| GEN | S range | avg S | ms/token | tok/s |
|---|---|---|---|---|
| 3 | 5..7 | 6.0 | 75.7 | 13.2 |
| 24 | 5..28 | 16.5 | 198.6 | 5.0 |
| 64 | 5..68 | 36.5 | 628.6 | 1.6 |

The first step is ~linear (S x2.75 -> time x2.62), but the second is SUPERLINEAR (S x2.21 -> time
x3.16). Fitting `t = a*S + b*S^2` gives **a ~ 7.8 ms, b ~ 0.26 ms**: the LINEAR term is the
full-sequence recompute through all 36 layers, and the QUADRATIC term is attention (S queries against S
keys). Both are exactly what incremental decode removes -- the gemms become O(1) and attention becomes O(S)
(1 query against S keys) -- so the case for it strengthens as the context grows, which is also when users
notice: at S~36 the engine is already down to 1.6 tok/s.

**Why:** the decode step is a FULL-SEQUENCE forward. Every generated token re-runs all 36 layers over
all `Scur` rows: `Scur = SEEDLEN + g` (driver ~758), `run_layer(Scur, ...)` (~782), the final norm is
launched over `Scur * D` elements (~800), and only the LAST row is used for logits (`s7 = Scur - 1`,
~802). Causal attention means rows < S-1 cannot change, so every earlier row is recomputed identically
each step. Gemm work per token is O(S) when it should be O(1).

**The path (kernels already half-ready):**
1. `qwen3rope` ALREADY supports incremental use: it takes `POS` (absolute position base; row r ->
   POS+r) and the driver only ever passes 0. For decode, pass POS = S-1 with one row.
2. `attn` does NOT: it takes one `S` used as BOTH query count and key count. It needs a query-row
   offset plus a separate key length, so 1 query row can attend over the S-row cache.
3. Feed the layer gemms M=1 (just the new row) while attention reads the persistent K/V cache -- the
   `DK`/`DV` buffers are already `MAXS * NKV * 8`, i.e. the cache already exists and is merely
   overwritten every step. There is prior art in this repo for a frozen context cache (`kvctx`, CK/CV).
4. Norms / silu / resid / o-proj / ffn then run on that one row.

**Expected:** per-token gemm work becomes O(1) and the floor becomes one pass over the f32 weights
(14.5 GB at ~936 GB/s = ~15.5 ms/token), instead of ~76 ms at S=6 growing with context. ~3-6x at
current lengths and, more importantly, FLAT instead of degrading as the context grows.

> **THIS 3-6x CLAIM WAS WRONG.** Measured by passing S=1 to `run_layer` in the decode step (output
> invalid by construction -- no cache -- but the WORK is the right shape): full-sequence layer 60.3
> ms/token vs one-row 36.7 ms/token, and the token only goes 75.7 -> 52.3 ms. That is **~1.45x**.
>
> The decomposition is the useful part: the layer phase is **~32 ms FIXED per token plus only ~4.7 ms
> per row**. Incremental decode deletes the per-row term (worth ~23 ms at S=6, growing with context) and
> leaves the 32 ms standing.
>
> **THE REAL TARGET IS THAT FIXED COST**, and the obvious candidate is re-streaming the f32 weights:
> 14.5 GB read once per row-block per token, so at S=6 (RB=4 -> 2 row-blocks) ~29 GB = ~31 ms at
> 936 GB/s -- which matches the ~32 ms almost exactly. So the next cheap test is not the decode rework
> but SHRINKING THE WEIGHTS: keep them packed (q4k/q6k, ~2.3 GB) with the dequant fused INTO the gemm
> (the llama.cpp approach), or as a cheaper intermediate store them f16/bf16 (7.25 GB) with a widening
> load in the gemm. Halving the weight bytes should roughly halve that fixed cost. NB the earlier
> `ebb29ed` win came from deleting a per-token dequant PASS, which is not the same as keeping weights
> small -- it left them f32 and therefore fat.
>
> **THAT WEIGHT-SHRINKING PLAN IS ALSO WRONG -- the harness says so directly.** Same gemm, same K, same
> KSP, 300 reps: **M=5 (124.5M MACs, 2 row-blocks) = 74 ms** vs **M=1 (24.9M MACs, 1 row-block) = 69 ms**.
> Five times less work for a 7% change in time. With RB=4 the per-THREAD k-loop is identical in both
> cases (each thread covers 4 rows regardless), and at these grid sizes (80-160 blocks vs 82 SMs) the
> kernel is LATENCY-bound: the k-loop walks B at stride N*4, i.e. a fresh cache line every iteration, so
> it is a serial chain of dependent memory accesses whose length is set by K/KSP, not by M or by the
> weight bytes. **The gemm's cost is set by the length of the k-chain and the warps available to hide it,
> not by FLOPs and not by weight traffic.**
>
> So: the next session should measure the gemm's occupancy/latency behaviour (warps resident vs KSP) and
> attack the CHAIN -- more independent loads in flight per thread, or a layout that makes the B walk
> sequential -- NOT the weight format. Two roadmap claims of mine have now been falsified by a cheap
> measurement before a rewrite; do that first, again.
>
> **AND THE TILED GEMM IS NOT BLOCKED ON THE COMPILER.** The Vyb kernel surface already exposes the
> primitives (found in `src/vre/llvm/cgen_expr_kernel.cpp`; the header note claiming "no shared memory /
> no barriers" is STALE):
>
> * `ld_shared_f64(off)` / `st_shared_f64(off, v)` -- a module-wide **32 KB addrspace(3) shared buffer**
>   (4096 f64; the compiler notes the sm_86 static shared cap is 48 KB). Caller bounds-checks.
> * `kernel_barrier()` -- block-wide `bar.sync 0`.
> * `lane_id()`, `warp_size()`, `tid_x`/`tid_y`/`tid_z`, `blk_x`/`blk_y`/`blk_z`, `dim_x`, `grid_x`.
> * `ld_f16` / `ld_bf16` (widening to f64).
>
> So the next real step is a SHARED-MEMORY TILED `gemm32_p`: cooperatively load a B tile coalesced,
> `kernel_barrier()`, then accumulate out of shared instead of walking global B with a 10 KB stride
> (which is the dependent chain that makes the kernel latency-bound -- see above). That attacks the
> actual bottleneck, and it is kernel work, not compiler work.
>
> **TRIED IT -- IT LOSES TOO (5th falsification).** `gemm32_tiled` written exactly as above (block of
> 256 = 256 output columns; stages a [16 k][256 col] B tile into the 4096-f64 shared buffer with fully
> coalesced loads, barrier, accumulates from shared, barrier; same param block and grid as gemm32_p so
> the driver needs no change). Result: **CORRECT (C00 verified) but 129 ms vs 74 ms in the harness --
> 1.8x SLOWER.** Most likely the barriers: 2 per chunk x 76 chunks = 152 block-wide barriers, which
> serialise the block's 8 warps, plus the staging traffic through shared on top of the loads that were
> already happening. Not pursued further: the shared cap (4096 f64 = 32 KB) forces TK=16 at TN=256, so
> the barrier count cannot be reduced without shrinking the column tile and changing the grid.
>
> **CONCLUSION on the gemm: it is at a local optimum for this kernel model, and every structural
> alternative has now been measured and lost** -- RB=2/8, a 4-way unroll, KSP=64, f16/packed weights
> (STILL OPEN -- the 'moot' claim was WRONG, see the correction below), [N][K] B layout, shared-memory tiling. The remaining gap to peak is real
> but is not reachable by any single axis I have been able to measure.

> **CORRECTION (the paragraph above is WRONG).** Every sweep that produced that conclusion was taken
> with `runk()`, which calls `cuModuleLoadData` + `cuModuleGetFunction` on EVERY call -- so each timed
> "launch" included a PTX JIT compile (~93 us/call). That is why M=1 vs M=5, K x4 and KSP x4 all looked
> work-independent. The harness now has `bench_n()` (module loaded ONCE, then reps launches).
> Re-measured there:
>
>   A) K=9728 KSP=8   176 us/gemm   124.5M MACs, 99.6 MB of B
>   B) K=2432 KSP=8    60 us/gemm    31.1M MACs, 24.9 MB     -> work+bytes /4 => time /2.9
>   C) K=9728 KSP=32  153 us/gemm   same bytes, chain /4     -> only -13%
>
> The gemm DOES scale with work and bytes, nearly linearly; the k-chain is NOT the driver. So the
> weight-format lever is BACK ON: keep the weights packed (q4k/q6k, ~2.3 GB vs 14.5 GB) with the dequant
> fused into the gemm, or f16 (7.25 GB) as the cheaper intermediate first. The RB=2/RB=8/unroll/KSP/
> tiled entries above are valid only as RELATIVE comparisons (all carry the same +93 us/call handicap).
> Anything about the gemm concluded from a per-call-module-load timing must be re-measured.

### Traps this line cost real time (do not repeat)

* **The f64 trap** (cost 3 rewrites, then won 12%): `ld_f32` has a Float64 SURFACE type, so writing the
  load INLINE (`acc + av * ld_f32(pb)`) makes the MAC an f64 multiply + 4 conversions, and f64 is 1/64
  rate on sm_86. The kernel gets ~10x slower while staying numerically CORRECT. **Bind every load to a
  `<Float32>` local before the multiply.** `ptxas -v` confirms 44 regs / 0 spills; grep any should-be-f32
  kernel's PTX for `mul.rn.f64`.
* **`native/host/gemm32_harness.vyb` is the fast loop** (~2 s: `total=` ms for 300 reps + CPU-checked
  C00). It had gone STALE (expecting the pre-`58fa9e7` f32 pass 2 + `widen32`, reporting `got=0`); it
  is revived and takes a `REP` count. Use it -- do not bench kernel ideas through the whole model.
* **nsys per-kernel TIMES are not trustworthy** (it serialises kernels; the reported total can
  understate the wall). Shares are usable; end-to-end tok/s is the only verdict. Bracket with
  `cuProfilerStart/Stop` + `--capture-range=cudaProfilerApi` to exclude the one-time load.
* **Bench variance is ~±2%** -- use `RESIDENT_RUNS>=8` before believing anything under ~5%.
* **A failing driver still prints "gold stream MATCH"** from the previous run's output FILE. Require
  the final `RESIDENT-BENCH: PASS`.
* **Never run two benches at once.** The engine wants ~20.5 GB of 24 GB, so a second run dies at the
  weight allocation (exit 11), which looks like a regression and is not one.
* `pkill -f` / `pgrep -f resident_driver` match their own shell; use `pkill -x vyb`.
* Measured optima: gemm row-blocking **RB=4** (harness: RB=1 72 ms, RB=2 191-279 ms, RB=4 70-75 ms,
  RB=8 95-98 ms); **KSP=32** (8->32 = +5.7%, 32->64 = -1.5%). GPU throttling is NOT a factor
  (1950-1965 of 2100 MHz, mem 9501 MHz, 188-296 W of 350 W, 88-99% util).
## Gate record: VYB_GTKW 1536 -> 3072 (f06957c) -- GATED, the "gate pending" note is closed

Re-ran the numerics gate on GB10 (lefty) at GEN=128, 10 runs per setting:

  VYB_GTKW=1536   median 48.39 tok/s   gold stream MATCH   ids 785 6722 315 9625 374 12095 11 323
  VYB_GTKW=3072   median 47.62 tok/s   gold stream MATCH   ids IDENTICAL

  cmp of native/out/chatgen_ids_vyb.txt between the two -> IDENTICAL

So "Numerics differ from 1536 -> gate pending" does NOT reproduce at GEN=128 on GB10: the generated
token stream is bit-identical and both settings match the gold stream. The change also buys no
measurable DECODE time (47.6 vs 48.4 tok/s, inside the run-to-run spread), consistent with GTKW being a
PREFILL-side knob (GB10 512-token prefill gemm16_t 514 -> 474 ms per the commit message). Kept.

---

# HANDOFF -- 27B Ridge bring-up (session write-up)

## State at handoff

- Branch `perf/ursela-lefty`, pushed to origin. Working tree CLEAN (only `.ubench.sh` untracked --
  that is this line's own bench harness, left alone).
- **The 27B runs and PASSES its oracle verification** on lefty (GB10):
  `make -C native ridge-forward` -> per-position top1 5/5, 1/1, 2/2, 3/3 agree; hidden
  cos(post-output_norm) 0.9992..0.9997; `RIDGE_FORWARD_VERIFY_DONE`. A whole 64-block hybrid
  attention+GDN forward reproduces llama.cpp's own hidden and top1.
- **Measured throughput: 0.32 tok/s** (`PREFILL_TIMING S=3 ms=9292 us_per_token=3097333`). It is
  streaming-bound, not GPU-bound -- see the bottleneck note below.
- Nothing is running; no model server holds the GPUs. `righty` is UNTOUCHED.

## How to run the 27B

```
# model (12.6 GB) is already staged on lefty; mmproj-*.gguf is the vision tower, not the model
export RIDGE_GGUF=$(ls ~/Models/qwen38-27b-ridge/*.gguf | grep -v mmproj | head -1)
cd ~/Projects/VybForge-perf
make -C native ridge-forward        # must PASS; add VYB_RIDGE_TRACE=1 for the embed-row dump
grep -E "PREFILL_TIMING|PREFILL_LAYER" native/out/ridge_forward_driver.log   # throughput + heartbeat
```
The verifier echoes only an ALLOWLIST of driver lines, so read the raw log for anything new.

## Immediate next step: the resident 27B path (target ~20 tok/s)

Both drivers each have HALF of what is needed:

- `resident_driver.vyb` (4B, 48 tok/s) -- the machinery to copy: all 36 layers' PACKED weights resident
  on-GPU with ONE upload (per-tensor `cuMemAlloc_v2` blocks, then one HtoD each), plus a fused quantized
  GEMV for decode straight from the resident blocks. BUT hardcoded to 4B dims, pure attention,
  Q4_K/Q6_K only -- no GDN, no qwen35, no untied head, no IQ2_S/IQ3_S.
- `model_driver.vyb` (27B) -- the correctness half, all VERIFIED: hybrid config, GDN, untied chunked
  head, IQ2_S/IQ3_S grid codecs. But it streams every weight from the file on every forward.

So this is a PORT, not new research: give model_driver resident allocations + one upload each (the
resident_driver pattern), keep the 3.7bpw weights PACKED (~12.6 GB fits the Spark's 121 GB easily; f16
would be 27 GB and is unnecessary), and dequantize fused at use. Do NOT touch the ridge config / GDN /
untied-head logic -- it is verified and passing.

Suggested FIRST SLICE (bounded, self-contained, verifiable): make ONE layer's tensors resident, leave
the other 64 streaming, then (1) `make -C native ridge-forward` must still PASS, and (2) compare
`PREFILL_TIMING` against 3.10 s/token. If one layer moves the number, extending to 65 is mechanical.
That seam lets a session stop with a working faster engine instead of a broken half-rewrite.

Why it will pay: the model is ALREADY page-cached (19 GB in buff/cache) yet streaming reaches only
12.6 GB / 9.29 s = 1.36 GB/s -- so the cost is the per-tensor read->String->HtoD plumbing, not I/O.
Residency removes the plumbing; the ceiling then becomes memory bandwidth:
12.6 GB / 273 GB/s ~= 46 ms/forward ~= 20 tok/s (vs 0.32 today).

There is NO residency switch to flip -- the full env-hook list for model_driver is prompts, model and
tokenizer paths, the four probe hooks, MTP and the trace flag. It is streaming-only by construction.

## What was fixed in this session (all pushed)

- `660ac8c` fix(gate): the oracle provenance check compared the whole llama-server version string
  verbatim, which is UNPASSABLE for anyone who rebuilds the oracle at the pinned commit (git
  abbreviates the commit differently: 7 chars in the fixture, 9 in a fresh build). Now compares build
  number exactly, commit by PREFIX, base version exactly, gguf hash verbatim.
- `940de8c` fix(27b): migrate `model_driver`'s 12 two-arg `put_i(base + off, value)` calls to the 3-arg
  form (`put_i` is 3-arg, defined at gdn_layer_driver.vyb:67 -- the driver did not COMPILE before this).
  Also builds the kernel modules the ridge config needs (rope, q8_0, gdn, attn35, q5k, iq2s, iq3s --
  only layer/q4k/q6k/qwen3 ship) and generates the iq2s/iq3s grid images.
- `0d67d61` fix(inventory): **THE NaN ROOT CAUSE.** `ridge_inventory.py` wrote its TSV with extra
  columns and no `off=` prefix; the driver parses offsets from an `off=`-prefixed field, so `te_off`
  became garbage (it logged `token_embd off=6`) and the engine read token_embd from file offset 6 --
  the GGUF header -- as weights. Garbage embeddings -> NaN hidden -> top1 0. The tool now emits
  `name, dims, type, off=<n>, size=<n>`, so regenerating cannot reintroduce it.
- `9688cf2` perf(ridge): prefill timing (`PREFILL_TIMING`). First 27B throughput number.
- `a908d46` docs: closed the VYB_GTKW 1536->3072 gate (bit-identical at GEN=128 on GB10, gold MATCH
  both ways -- the "gate pending" note in f06957c does not reproduce).

## Also established about this line

- The perf line does ~48 tok/s on the 4B (`.ubench.sh gtw3072 GEN=128`), vs 15.5 tok/s on the 3090
  branch -- the SPARK line is the better one. `.ubench.sh <label> GEN=128` is the tool that settles a
  commit whose message says "gate pending" (it is hidden -- `./ubench.sh` fails).
- The line is 56 ahead / 10 behind `origin/main`; the 10 are the 3090 line's #23 commits, including an
  attention rework that COMPETES with this line's own `attn_sc5`/`attn_cx5`. Do not blind-merge.

## Traps that cost real time in this session (do not re-walk)

- Vyb CONSTANT-FOLDS `x != x` to false, so a NaN count built on it reports 0 while the printed value
  is literally `nan`. Print values via `to_string()`; never trust a comparison-based NaN count.
- A local declared at one point in `model_driver` is NOT in scope at another -- the file is deeply
  nested. Hoist timing/locals to the top-level init block (`DPK<Int> = 0; ...`).
- The driver cannot be invoked bare: `llm` resolves via `--module-path native/llm`. Use the make target.
  If a patch breaks the driver, `git checkout -- native/host/model_driver.vyb` restores the passing state.
- A PRETTY-PRINTED `vocab.json` encodes every token to id 0 (stdlib/vllm's `build_vocab_from` stops
  `read_int` at the first non-digit). The HF `Qwen/Qwen3.8-27B` vocab downloads pretty-printed.
  Symptom: correct-looking forward pass with `PROMPT_IDS=0 0 0 0 0`.
- The 27B vocab is 248044 tokens, NOT the 4B's 151936. `hf download Qwen/Qwen3.8-27B vocab.json
  merges.txt --local-dir artifacts/ridge-tokenizer` (needs huggingface-hub; the `hf` script lands in
  ~/.local/bin, often not on PATH).
- An independent gguf-py reference is the way to settle dequant questions, but read the tensor through
  the reader's OWN accessor (`r.tensors`, then `tensor.data`) -- do NOT hand-compute
  `reader.data_offset + tensor.data_offset`, which is what produced my own garbage readings and a
  wrong "the file is broken" conclusion.

## Rest of the queue

1. Resident 27B (above) -- the real performance work; target ~20 tok/s.
2. `ridge-mtp` gate -- the `blk.64` nextn MTP draft head is already identified and excluded from the
   main pass.
3. Merge this perf line to `main`, attention resolved toward `attn_sc5`/`attn_cx5`.
4. Port the two genuine wins from the 3090 line onto this one: the `bench_n` harness fix (the old
   `runk()` reloaded/re-JIT'd the PTX module per call, so its timings measured module loading) and the
   f64-binding rule (`ld_f32` carries a Float64 surface type; bind every load to a <Float32> local or
   every MAC silently becomes f64).
5. Stage the 27B on righty so both Sparks can benchmark in parallel.

## Reference

Skill `dgx-spark-two-node-serving` carries the full detail: the 27B bring-up steps, the root cause and
the format comparison, the verification result, and the eliminations (Q6_K kernel verified OK, offsets
and stride verified correct, `download` helper verified identical to the working 4B's).

---

# HANDOFF — the f16 port of the 27B (opened 2026-10-11, branch `perf/ursela-lefty`)

## Why the "resident path" in the earlier section was wrong, and what replaced it

That section's premise ("the cost is the per-read/HtoD plumbing — 12.6 GB / 9.29 s = 1.36 GB/s") is
FALSE, measured. `546d8d6` adds `VYB_RES_N` (keep the first N blocks' PACKED weights in one device
buffer) and `VYB_PHASE` (per-layer staging accumulator). S=3 forward, `PREFILL_TIMING ms`:

```
N=0 streaming   9558 / 9531 / 9607
N=1   172 MB    9351 / 9257 / 9445
N=4   698 MB    9544 / 9539
N=64  9692 MB   10701 / 12977     848/848 tensors staged from device, gate PASS
```
N=64 removes the read+HtoD path ENTIRELY and is 1.2-3.4 s SLOWER (the one-time upload is charged
inside the window; the streaming it removes costs less). So the read path is < ~1 s of 9.5 s.

`VYB_PHASE=1` prices the real target: `PHASE_SPLIT staging_ms=34707 total_ms=45616 staging_pct=76`.
Read as a WORK RATIO (the syncs serialize what normally overlaps): staging — read + HtoD + the
DEQUANT it launches — is ~3.2x the layer math. **The port target is the per-token dequant, not the
file read.** Note the syncs inflate the absolute total (45.6 s vs ~9.5 s unsynced).

## f16 is safe — measured on the model's own tensors before writing any kernel

`/tmp/f16_precision.py` (CPU only, gguf-py dequant, `blk.N.ffn_*` + `attn_qkv`, round to f16,
recompute the block output vs the exact f64 path):

```
blk.0  (FFN IQ3_S)  ffn_out rel_l2 9.5e-05  cos 1.00000000   residual cos 1.00000000
blk.17 (FFN IQ2_S)  ffn_out rel_l2 3.2e-04  cos 0.99999995   residual cos 0.99999999
blk.40 (FFN IQ2_S)  ffn_out rel_l2 3.4e-04  cos 0.99999994   residual cos 0.99999999
attn_qkv (Q4_K)     proj    rel_l2 2.1e-04  cos 0.99999998
```
The gate's bar is hidden cos >= 0.99 (met today at 0.9992..0.9997 in f64), so f16 has 3-4 orders of
magnitude of headroom. Nothing in the port's risk profile is numerics.

## What is ALREADY LANDED (compiles, gate green, inert)

`native/host/model_driver.vyb` (uncommitted at time of writing, on top of `546d8d6`):
- `cuMemcpyAsync(dst, src, bytes, stream)` added to the extern block (the 4B's param-block idiom).
- structs `GP16 {a,b,sk,m,n,k,nblk,ksp}`, `GP16R {sk,scr,m,n,p1,p2,ksp}`, `GP3C {src,dst,n}` —
  field order == kernel offsets, copied ONCE per launch instead of per field.
- `cast_elem(cfn, pb, src, dst, n)` — the 4B's `cast1` (f64->f32 with `cast32`, f32->f16 with `cast16`).
- `gemm_split(cfn, grn, pb, skb, ksp, a, b, c, m, n, k)` — the 4B's `gemm2`: pass 1 `gemm16_p`
  (grid `nblk*ksp`), pass 2 `gemm32_r` at **`pb + 64`** (with the base pointer it faults rc 700).
  `ksp` = split + 65536 * (cf - 1); gemm16_p is the 2-column kernel, so callers pass `split + 65536`.
  `nblk = ((m+3)/4 * (n/cf) + 255)/256`.
- Handles `g16p/g3r/c32n/c16n` loaded from `ml` (layer.ptx) with a loud `MODLOAD_FAIL` — verified
  present: `F16_PATH kernels gemm16_p/gemm32_r/cast32/cast16 loaded from layer.ptx`.
NO new kernel is needed: `cast32`, `cast16`, `gemm16_p`, `gemm32_r` all exist in `native/kernels/layer.vyb`
and are gated by the 4B driver. `gemm32_r` STORES f64, so every caller keeps the dtype it already had.

## The remaining build (in order; each step ends with `make -C native ridge-forward` GREEN or revert)

1. **FFN trio** (the biggest single chunk: 3 x 89,128,960 elements x 64 blocks = 34 GB of the ~49 GB
   resident). Allocate `WGA/WUA/WDA` f16 slabs, `T32` (f32 scratch D*FF*4), `SKX` (split-K partials,
   ksp*S*FF*4), `X32` (S*D f32), `H32` (S*FF f32), a 128-byte param buffer.
   One-time pass before the layer loop: per block, `stage_one(ffn_gate) -> DWg` (f64, already the
   TRANSPOSED [D,FF] layout the GEMM wants — `inz` does that) -> `cast_elem(c32n, DWg->T32)` ->
   `cast_elem(c16n, T32 -> WGA + L*D*FF*2)`; same for up/down.
   New `run_ffn16` = rmsnorm (unchanged, X1N f64) -> cast X1N->X32 -> `gemm_split` x2 (A=X32 f32,
   B=slab + L*D*FF*2, C=Gr/Up f64) -> silu (unchanged) -> cast Hu->H32 -> `gemm_split` (C=M f64) ->
   resid (unchanged). Dispatch behind `VYB_FFN16=1` (default OFF, so the default path stays
   byte-for-byte the streaming one) and SKIP the per-token FFN `stage_one` calls when it is on.
   Verify: gate green in BOTH modes, then `PREFILL_TIMING` with the rig down.
2. **Mixer families** (the GDN block's `attn_qkv`/`attn_gate`/`ssm_out`; the attention block's
   `attn_q`/`k`/`v`/`output`) — same pattern, per-kind slabs indexed by absolute block.
3. **The attention block's split/gate path** (`qg_split`, the per-head norms, `gate_mul`) — the one
   piece the 4B has no equivalent of; its A operands need the f32 copies too.
4. Then re-measure. The floor is the layer math (~25% of today's staged work), and the f16 GEMMs are
   faster than the f64 ones, so the target is well below the current 9.5 s/forward at S=3.

## Timings are only valid with the rig DOWN

The DSv4.1 rig holds 106.5 GiB of the GB10's 121 GiB per node and the 27B forward is memory-bandwidth
bound: the same binary that times 9391 ms on a free GPU times 42587-52932 ms while the rig serves
(measured, twice). Every timing claim needs `docker stop` on both nodes first; see the
`dgx-spark-two-node-serving` skill for the exact commands (the `~/ds41-serve.sh` wrapper is STALE —
its `stop-server.sh` is gone; the stack is `dsv41-exl3-head` on lefty + `dsv41-exl3-worker` on righty,
both must stop, and `dsv41-exl3-nfs` is NOT a GPU holder).
