# KV-Cache Context-Prefix Training (full-manifest conditioning)

Status: **architected; numpy reference built & verified; GPU build in progress (kvctx.vyb WIP).**
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
- `native/train/kvctx.vyb` — WIP GPU context-build (reuses run_layer, captures CK/CV, dumps) —
  UNVERIFIED; do not treat as correct. Known TODO: dispatch the mixed Q4/Q6 dequant for `attn_v`
  and `ffn_down` by PTY (layers 0-3 & 31-35 are Q6_K) — my draft hardcoded q4fn/q6fn.

## Next steps (next session, order)
1. Fix kvctx.vyb: type-dispatch dequant (q4fn/q6fn by PTY) for all 7 weights + f32expand norms;
   verify the GPU context-build CK/CV == kvctx_ref per-layer (maxrel gate).
2. Add the KV-aware LoRA `run_kv` per-token response forward; verify response logits/loss == the
   validated full-sequence S=93 result (no new oracle).
3. Add the frozen-context backward (zero/adjust context-region dattn grads) + AdamW; verify descent
   == the masked goal-conditioned oracle (committed ref).
4. Scale the context to the full manifest (369 tokens); confirm per-step cost stays ~response-only.
