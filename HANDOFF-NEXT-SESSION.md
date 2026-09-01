# HANDOFF — RESUME HERE (untracked; do not commit)

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
   on the 3090 (no torch) — the true no-Python path — full Qwen3-4B research-scale, PLUS an expanded
   interview corpus. Design doc `training/CORPUS-STRATEGY.md`: **facts live in context (a capabilities
   manifest in the system prompt), behavior lives in the weights**, so a new VybOS capability = edit the
   manifest, NOT a retrain. Also: goal-driven WHOLE-STACK interviews ("I want a Hyprland desktop" →
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
   training loop on the 3090 in pure Vyb (forward gemmf+addscaled -> MSE-grad sub+scalec -> LoRA backward
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
   real LoRA fine-tuning on the 3090, fully Vyb-native. Then whole-stack interview.**
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
   record.** target = train rec 6 'I want a Hyprland desktop workstation' -> manifest-v<3 SUMMARY
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
