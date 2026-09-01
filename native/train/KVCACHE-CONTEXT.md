# KV-Cache Context-Prefix Training (full-manifest conditioning)

Status: **response forward VERIFIED (2be728e); frozen-context training VERIFIED on GPU
(0d03267, KVRESP_TRAIN_VERIFY OK); NEXT = true per-token KV trainer (per-step cost = response
length) then full manifest (369) then decode.**
Parent context: HANDOFF-NEXT-SESSION.md, training/CORPUS-STRATEGY.md, native/train/train_full_ce.vyb,
native/host/kv_driver.vyb.

## Goal
Make the FULL manifest context (system+user, ~369 tokens) condition the whole-stack response in
the Vyb-native corpus-CE trainer WITHOUT re-forwarding the context every step. Full-sequence masked
CE ([ctx|resp]) works (committed 1e5b605, goal-context S=93) but costs ~S-proportional per step, so
the 369-token manifest (S~453) is infeasible for many steps. The KV-cache makes per-step cost =
response length (84) regardless of context length: **cache the frozen context's per-layer roped-K/V
once; each step attend the response to the cached prefix.**

## Approach (reuses the decode KV-cache in kv_driver.vyb)
1. **Context KV build (once)**: forward the context tokens via the LoRA `run_layer`
   (symbolic S), and after each layer L copy that layer's roped-K (`DKr`) and `V` (`DV`) into
   per-layer cache slabs `CK[L]`/`CV[L]` (each `[MAXS,NKV]` f64). Context is FROZEN.
2. **Response forward per token** (each step): for response token t (absolute position P=ctx+t),
   run a `run_kv`-style 1-token LoRA forward (like kv_driver.run_kv but with gemmf/addscaled on all
   7 projections) that (a) computes q/k/v/o/g/u/d, (b) ROPEs at absolute position P, (c) stores this
   position's roped-K/V into `CK[L]`/`CV[L]` row P, (d) runs `attn` with `Sctx = P` (dial `Sctx*H`)
   so the q attends rows 0..P (cached context + response-so-far), (e) residual+ffn. Cache the
   response's per-layer activations (ASLB-style) for backward.
3. **Masked CE** on the response positions (existing `cefwd`, LABEL==-1 context rows ignored).
4. **Backward (frozen context)**: run the response-position backward (the same backward kernels),
   but the response `dattn` must accumulate q/k/v gradients into RESPONSE positions only —
   context rows' k/v grads are DROPPED (context frozen). Then AdamW on the response (or all) LoRA.
   This requires `dattn` to know the context boundary and zero the context-region grads.

## Verification (no new heavy oracle needed)
The goal-context case (S=9 ctx + 84 resp, S=93) is ALREADY validated full-sequence vs the numpy
oracle (M2E3CE_VERIFY OK, committed 1e5b605). So the KV-cache response forward for that SAME
case must reproduce the full-sequence response hidden/logits/loss (and per-layer roped-K/V of the
context must equal the numpy forward of those 9 tokens): `kvctx_ref.py` computes the 9-token
context's per-layer DKr/DV (verified, native/out/kvctx_L*_ref.bin) — the gold for the context
cache. This cheap reference validates the mechanism before scaling the context to 369 tokens.

## Concrete state
- `native/train/kvctx_ref.py` — numpy per-layer roped-K/V for the 9-token goal context (VERIFIED,
  writes kvctx_L{L}_{DKr,DV}_ref.bin; 9*1024 f64 per layer). Also kvctx_ctx_ids (goal token ids).
- `native/train/kvctx.vyb` — GPU context-KV-cache build. **VERIFIED (KVCTX_VERIFY: OK)**: per-layer
  roped-K/DKr and V/DV match numpy to ~3-4e-6 (corr 1.0) at L0/8/17/26/35. Fixes baked in:
  q4kdeq/q6kdeq take 4 DIRECT args (a0=packed,a1=out,a2=numel,a3=0) via `deq_w` which also
  dispatches the mixed Q4/Q6 for attn_v/ffn_down by PTY; W buffers are f64 (wq_nl*8).
- NEXT (from here): the KV-aware LoRA response forward (per-token, run_kv-style attending to the
  cached CK/CV) then the frozen-context backward.

## Response forward — implementation notes (found by drafting it; READ BEFORE CODING)
**LoRA IS A 3-GEMM + ADDSCALED PATTERN (critical, was my root-cause of HALL corr ~0):** the
validated run_layer does NOT do `gemmf(a,W,o)` then `addscaled(o,U,V,AR)`. It does, per projection:
  (1) `gemmf(a, W, out)` -> out = a@W
  (2) `gemmf(a, U, s)`   -> s  = a@U            (LoRA low-rank A; s is [S,R])
  (3) `gemmf(s, V, e)`   -> e  = s@V            (LoRA low-rank B; e is [S,dim])
  (4) `addscaled(out, e, AR, n)` -> out[i] += AR*e[i]   (addscaled(A,B,c,n) = elementwise A+=c*B,
      params +0=A,+8=B,+16=c(Float),+24=n — NOT a matmul!)
My first resp_layer_kv wrongly did `addscaled(C, U, V, AR, S)` (treated V as a vector with n=1) ->
every projection corrupted -> response hiddens corr ~0 vs numpy. FIXED in kvrespfwd via `proj_lor`
helper (the 3-gemm+addscaled, per-token row-P with slo [S,R]/shi [S,FF] scratch buffers).
**STATUS: still FAILING (final HALL corr ~0). Localizing with per-stage layer-0 dumps.**
DIAGNOSTIC TRAP (cost a session): the layer-0 stage dumps (XN/DQN/DQr/DCtx) were read from the
CONTEXT working buffers (same names, S=9 sized) instead of the RESPONSE buffers (rXN/rDQN/rDQr/
rDCtx <- what resp_layer_kv actually writes). So they showed garbage regardless of correctness;
emb (correct, corr 1.0) and N1 (correct) confirmed the inputs, but the stage values must be read
from the r* buffers. FIXED the dumps to rXN/rDQN/rDQr/rDCtx.
**Status: RESPONSE FORWARD VERIFIED (2026-08-30) — final HALL corr 1.0 / |g|=|r| / max|g-r| 1.66e-4.**
**Resolution — TWO real driver bugs (NOT a kernel bug):**
1. **DF (RoPE invfreq) was never loaded** into the 64-slot device buffer -> the qwen3rope read
   all-zero FR -> angle 0 -> identity (DQr==DQN). Fixed by loading native/out/layer0_invfreq.bin
   into DF BEFORE the context build. A standalone probe (kvrope_probe.vyb) proved the kernel+call
   pattern (row-offset + S=1 + POS=9) rotates correctly with a real FR — no rickenator/Vyb issue.
2. **Cache-slab STRIDE mismatch**: the context build wrote each layer’s roped-K/V at
   `CK + L*S*NKV*8` (S=9 rows/layer) but the response reads them at `CK + L*96*NKV*8`
   (96 rows/layer) — coincident only at layer 0, so layers >=1 read garbage context K/V (layer-0
   corr 1.0, layer-1 corr 0.16). Fixed the context build to the same 96-row stride.
With both fixes, layer-0..35 response hiddens all match numpy (final corr 1.0, max|g-r| 1.66e-4).
The KV-aware LoRA response forward is VERIFIED. NEXT: wrap it into the full context-manifest
training (response attended forward is done; add the frozen-context backward), then decode.
The per-token forward must get these RIGHT or it silently corrupts:
1. **Row-P addressing, not row 0.** The per-token q/k/v/o/g/u/d projections and roped activations
   must write to ROW P of each buffer (e.g. `DQ + P*NQ*8`, `DK + P*NKV*8`), like run_kv does. The
   context-build working buffers (sized S=ctx) are TOO SMALL for response positions P=ctx+t — use
   dedicated per-token buffers sized [ctx+resp, ...] (or [MAXS,...]) for HB (hidden base), DQ/DK/
   DV/DQN/DKN/DQr/DKr/DCtx/DO/X1/X1N/Gr/Up/Hu/M2.
2. **residual x1 = input-hidden-row-P + DO** (the skip from the token's OWN previous hidden), NOT
   the q buffer. (A draft bug used DQr row P as the residual input — wrong.)
3. **Attention over the combined cache**: attn(Q=DQr, K=CK[L], V=CV[L], Sctx=P+1) with grid
   Sctx*H; only DCtx row P is read (other rows recompute garbage — discard). The response q must
   live at row P of DQr so output row P = that token's attention over rows 0..P. Before attn, store
   this position's roped-K (DKr row P) -> CK[L] row P and V (DV row P) -> CV[L] row P.
4. **Absolute rope position P** (qwen3rope +224 = P, S=1), matching how the context rows were roped
   at their positions and how run_kv / the full-sequence forward rope response tokens.
5. Loop order: token t (P=ctx+t, Sctx=ctx+t+1) outside, LAYER inside (dequant that layer's
   weights/norms once per L, reuse across the 84 tokens for that layer), then move to next L. Cache
   the response hidden h[t] after layer 36.
6. Head/validation: rmsnorm(h)->HO, hgemm(HO,EMB)->logits, cefwd(masked, NV=resp)->CE loss. The
   goal-context S=93 full-sequence forward is ALREADY validated (committed 1e5b605) — the KV-cache
   response forward MUST reproduce its response hidden + masked CE loss (GPU 15.93 / oracle 15.96)
   — no new oracle needed. Optionally dump per-position response logits to diff directly.

## Next steps
1. [DONE] GPU context-KV-cache build VERIFIED (4b43568).
2. [DONE] KV-aware LoRA `run_kv` per-token response forward (attend to cached CK/CV + response-so-far) VERIFIED (RESPONSE FORWARD, commit 2be728e): final HALL corr 1.0, loss 15.93 == full-seq.
3. [numpy] FROZEN-CONTEXT backward + AdamW = full-seq backward with context-row (0..NCTX-1)
   attention q/k/v grads ZEROED and context LoRA frozen. **numpy ORACLE DONE & VERIFIED
   (commit 3435de0): `native/train/kvresp_train_ref.py`** — masked CE on response only,
   per-step loss 15.957->15.495->14.841->14.095 (DESCENDS), response-row grads provably
   identical to the FD-validated full-seq chain (context queries never attend response rows).
   `make kvresp-train-oracle` reproduces it.
   **[GPU] FROZEN-CONTEXT KV TRAINING VERIFIED (commit 0d03267): `native/train/kvresp_train.vyb`**
   = the validated batched S=93 forward+backward+AdamW engine (train_full_ce.vyb) + ZERO the
   context-region rows of dQr2/dKr2/dVu2 right after dattn (zbuf, NCTX*NQ/NKV). KVRESP_TRAIN_VERIFY:
   OK — step-1 dU_q/dV_q (L0/17/35) corr 1.0 / norm-rel ~2e-6; per-step CE 15.932->14.081 == oracle
   (maxrel 2.4e-3) and descends; L0 Uq parity 9.7e-5. `make kvresp-train` gates it. This validates the
   frozen-context BACKWARD + AdamW descent (the new math). The per-step victory is that context grads
   never touch LoRA.
   **NEXT = TRUE KV TRAINER (per-step cost = response length):** combine the VERIFIED per-token KV-cache
   forward (kvrespfwd.vyb, context cached once in CK/CV) with the VERIFIED frozen-context backward here —
   cache per-layer RESPONSE activations (all 84 rows, not just the last token), run the batched backward
   restricted to response rows with dattn over the combined cache + context-region zeroing. Gate = same
   kvresp_train_ref.py oracle. Then scale the context to the full 369-token manifest, then decode to emit
   the agent-response JSON end-to-end.
4. Scale the context to the full manifest (369 tokens); confirm per-step cost stays ~response-only.
